"""候选文件工具；候选 Python 只在 macOS sandbox-exec 子进程执行。"""
import ast
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tomllib
import uuid

from mini_agentx.agents.brainstorm_contracts import TaskBoundary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DevelopingTools:
    SPECS = {
        "read_candidate_file": {"arguments": {"path": "model.py 或 training.toml"}, "description": "读取候选文件"},
        "write_candidate_file": {"arguments": {"path": "model.py 或 training.toml", "content": "完整文件文本"}, "description": "写入候选文件，修改后须重新检查和训练"},
        "check_candidate": {"arguments": {}, "description": "检查配置、模型接口、有限分数与反向传播"},
        "train_candidate": {"arguments": {}, "description": "检查通过后运行固定 BPR 训练及验证"},
        "read_training_result": {"arguments": {}, "description": "读取当前版本训练结果及父指标差值"},
        "get_candidate_diff": {"arguments": {}, "description": "查看相对父文件的差异"},
    }

    def __init__(self, root, runs, proposal_path, directory=None):
        self.root, self.runs = Path(root).resolve(), Path(runs).resolve()
        envelope = json.loads(Path(proposal_path).read_text())
        if envelope.get("schema_version") != 1:
            raise ValueError("不支持提案版本")
        self.task = TaskBoundary.from_dict(envelope["task"])
        self.proposal = envelope["proposal"]
        if self.proposal.get("maturity") != "ready":
            raise ValueError("Developing 只接收 ready 提案")
        if not re.fullmatch(r"baseline-[a-f0-9]{12}", envelope["parent_run_id"]):
            raise ValueError("首版只支持 baseline 父实验")
        parent = self.runs / envelope["parent_run_id"]
        if parent.is_symlink() or (parent / "report.json").is_symlink():
            raise ValueError("父实验不允许符号链接")
        self.parent = json.loads((parent / "report.json").read_text())
        source = self.root / "src/mini_agentx/recommender/model.py"
        if digest(parent / "model.pt") != self.parent["checkpoint_sha256"]:
            raise ValueError("父权重摘要不匹配")
        if digest(source) != self.parent["model_source_sha256"]:
            raise ValueError("父代码与当前基线不同，不能复制")
        if envelope["dataset_version"] != self.parent["dataset_version"]:
            raise ValueError("提案与父实验数据版本不同")
        self.directory = Path(directory) if directory is not None else self.runs / ("developing-" + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True)
        self.candidate = self.directory / "candidate"
        self.candidate.mkdir()
        settings = self.parent["config"]["training"]
        config_text = "[training]\n" + "".join(f"{key} = {value}\n" for key, value in settings.items())
        self.original = {"model.py": source.read_text(), "training.toml": config_text}
        for name, content in self.original.items():
            (self.candidate / name).write_text(content)
        (self.directory / "proposal.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2))
        self.checked = self.trained = None
        self.report = None
        self.process_calls = 0

    def fingerprint(self):
        return {name: digest(self.path(name)) for name in self.original}

    def path(self, name):
        if name not in self.original:
            raise ValueError("只允许候选 model.py 和 training.toml")
        path = self.candidate / name
        if path.is_symlink() or self.candidate.is_symlink():
            raise ValueError("不允许符号链接")
        return path

    def settings(self):
        value = tomllib.loads(self.path("training.toml").read_text())
        if set(value) != {"training"} or set(value["training"]) != set(self.parent["config"]["training"]):
            raise ValueError("只支持现有 training 字段，不能改种子或路径")
        settings = value["training"]
        limits = {"dimensions": 256, "epochs": 100, "batch_size": 8192, "threads": 4}
        for key, limit in limits.items():
            if type(settings[key]) is not int or not 1 <= settings[key] <= limit:
                raise ValueError(f"{key} 超过候选资源限制 1..{limit}")
        for key in ("learning_rate", "l2"):
            if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or not 0 <= settings[key] <= 1:
                raise ValueError(f"{key} 必须是 0..1 的有限数")
        ast.parse(self.path("model.py").read_text())
        return settings

    def run_worker(self, mode):
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
            raise ValueError("首版候选执行要求 macOS sandbox-exec；不降级为无隔离执行")
        self.process_calls += 1
        if self.process_calls > 5:
            raise ValueError("检查和训练子进程总预算最多5次")
        settings = self.settings()
        config = json.loads(json.dumps(self.parent["config"]))
        config["training"] = settings
        config["paths"]["processed_data"] = str(Path(self.parent["data_directory"]).parent)
        config["paths"]["runs"] = str(self.directory / "training")
        self.config_path = self.directory / "worker-config.json"
        self.config_path.write_text(json.dumps(config))
        # Runtime reads are broad for Python compatibility; known credentials are denied.
        quote = lambda value: json.dumps(str(value))
        profile = '(version 1)(allow default)(deny network*)(deny file-write*)'
        private = [self.root / ".env", self.root / ".git", self.root / ".aws", self.root / ".codex",
                   Path.home() / ".ssh", Path.home() / ".aws", Path.home() / ".codex",
                   Path.home() / ".config"]
        profile += '(deny file-read* ' + ' '.join('(subpath ' + quote(p) + ')' for p in private) + ')'
        profile += '(allow file-write* (subpath ' + quote(self.directory) + '))'
        profile += '(deny process-exec (with no-log))'
        # Allow the initial Python interpreter; subprocess spawning remains denied for other executables.
        profile += '(allow process-exec (subpath ' + quote(sys.prefix) + '))'
        worker = self.root / "src/mini_agentx/tools/developing_worker.py"
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(self.root / "src"),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        timeout = min(self.task.max_training_seconds, 60) if mode == "train" else 20
        log = self.directory / f"{mode}-{self.process_calls}.log"
        with log.open("w") as output:
            proc = subprocess.Popen(["/usr/bin/sandbox-exec", "-p", profile, sys.executable,
                                     str(worker), mode, str(self.candidate), str(self.config_path)],
                                    env=env, cwd=self.candidate, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise ValueError(f"{mode} 超时（{timeout}秒）；日志 {log.name}") from None
        if proc.returncode:
            raise ValueError(f"{mode} 失败（退出码 {proc.returncode}）：" + log.read_text()[-5000:])
        return json.loads((self.directory / f"{mode}-result.json").read_text())

    def execute(self, name, arguments):
        if name not in self.SPECS or not isinstance(arguments, dict) or set(arguments) != set(self.SPECS[name]["arguments"]):
            raise ValueError("工具名或参数不符合定义")
        if name == "read_candidate_file":
            return {"path": arguments["path"], "content": self.path(arguments["path"]).read_text()}
        if name == "write_candidate_file":
            if arguments["path"] not in self.task.allowed_files:
                raise ValueError("文件不在任务允许修改范围")
            content = arguments["content"]
            if not isinstance(content, str) or not content.strip() or len(content.encode()) > 24000:
                raise ValueError("文件须为非空文本且不超过24000字节")
            self.path(arguments["path"]).write_text(content)
            self.checked = self.trained = self.report = None
            return {"written": arguments["path"], "sha256": digest(self.path(arguments["path"]))}
        if name == "get_candidate_diff":
            return {key: "".join(difflib.unified_diff(value.splitlines(True), self.path(key).read_text().splitlines(True),
                                                     fromfile="parent/" + key, tofile="candidate/" + key))
                    for key, value in self.original.items()}
        if name == "check_candidate":
            self.checked = self.trained = self.report = None
            result = self.run_worker("check")
            self.checked = self.fingerprint()
            return result
        if name == "train_candidate":
            self.trained = self.report = None
            if self.checked != self.fingerprint():
                raise ValueError("当前文件尚未通过检查，请先 check_candidate")
            self.report = self.run_worker("train")
            if self.report["dataset_version"] != self.parent["dataset_version"]:
                self.report = None
                raise ValueError("训练数据版本不一致")
            self.trained = self.fingerprint()
        if self.report is None or self.trained != self.fingerprint():
            raise ValueError("当前版本没有有效训练结果")
        return {"run_id": self.report["run_id"], "validation_metrics": self.report["validation_metrics"],
                "parent_validation_metrics": self.parent["validation_metrics"],
                "ndcg_delta": self.report["validation_metrics"]["ndcg@10"] - self.parent["validation_metrics"]["ndcg@10"],
                "history": self.report["history"], "training_seconds": self.report["training_seconds"],
                "dataset_version": self.report["dataset_version"]}

    def finish(self):
        if self.report is None or self.trained != self.fingerprint() or self.checked != self.trained:
            raise ValueError("当前代码必须检查通过并完成真实训练")
        if all(self.path(key).read_text() == text for key, text in self.original.items()):
            raise ValueError("没有候选修改，不能交接开发完成")
        run = self.directory / "training" / self.report["run_id"]
        if digest(run / "model.pt") != self.report["checkpoint_sha256"]:
            raise ValueError("候选权重摘要不匹配")
        return {"status": "developed", "directory": str(self.directory), "training_run": str(run),
                "parent_run_id": self.parent["run_id"], "proposal_id": self.proposal["proposal_id"],
                "files": self.fingerprint(), "result": self.execute("read_training_result", {}),
                "diff": self.execute("get_candidate_diff", {}),
                "limitations": ["开发完成不等于方案有效；尚未自动接入A/B和晋升", "语义是否忠实于原假设仍需审查"]}
