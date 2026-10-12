import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
from statistics import median


class BrainstormTools:
    FILES = (
        "src/mini_agentx/recommender/model.py",
        "src/mini_agentx/recommender/train.py",
        "src/mini_agentx/recommender/evaluate.py",
        "configs/baseline.toml",
        "configs/brainstorm-task.json",
    )
    READ_ROOTS = ("src", "configs", "docs", "tests", "examples")
    TEXT_SUFFIXES = {".py", ".toml", ".md", ".json"}
    SPECS = {
        "list_directory": {"arguments": {"path": "项目相对目录，例如 src/mini_agentx；. 列出可调查根目录"}, "description": "浏览可调查目录，最多200条，不执行终端命令"},
        "search_code": {"arguments": {"path": "可调查目录，例如 src/mini_agentx/recommender", "query": "非空文本，按字面匹配"}, "description": "搜索代码或文档，返回文件路径、行号与匹配行，最多50处"},
        "list_files": {"arguments": {}, "description": "列出常用入口文件与可调查根目录"},
        "read_file": {"arguments": {"path": "项目相对文件路径，先浏览或搜索发现"}, "description": "读取发现的代码、配置或文档；只限可调查范围，模型入口读取当前父模型快照"},
        "list_experiments": {"arguments": {}, "description": "列出真实训练与模型模式模拟 A/B 的摘要，排除注入场景"},
        "read_experiment": {"arguments": {"run_id": "已列出的实验 ID"}, "description": "读取验证指标或模型模式 A/B 观测，不读取测试结果及模拟真值"},
        "list_feedback": {"arguments": {}, "description": "列出当前父实验和数据版本的反馈摘要，包括失败与待复查记录"},
        "read_feedback": {"arguments": {"feedback_id": "反馈ID"}, "description": "读取观测、方案与有限结论；不回传未经验证的LLM解释"},
        "training_data_summary": {"arguments": {}, "description": "统计父实验训练集的用户交互与物品流行度，无留出标签"},
    }

    def __init__(self, root, runs, context):
        self.root = Path(root).resolve()
        self.runs = Path(runs).resolve()
        self.parent = context["parent_run_id"]
        self.dataset_version = context.get("dataset_version")

    def safe_file(self, base, relative):
        target = base / relative
        # Reject symlinks even if they currently point within the allowed root.
        current = base
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("不允许读取符号链接")
        resolved = target.resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("文件越界")
        return resolved

    def readable(self, relative, directory=False):
        if not isinstance(relative, str) or not relative or len(relative) > 300:
            raise ValueError('路径必须是项目相对路径')
        parts = Path(relative).parts
        if Path(relative).is_absolute() or any(part in ('.', '..') or part.startswith('.') for part in parts):
            raise ValueError('不允许绝对路径、隐藏路径或上级目录')
        if not parts or (parts[0] not in self.READ_ROOTS and relative != 'README.md'):
            raise ValueError('路径不在可调查范围：src/configs/docs/tests/examples/README.md')
        target = self.safe_file(self.root, relative)
        if directory:
            if not target.is_dir():
                raise ValueError('不是可调查目录')
        elif target.suffix not in self.TEXT_SUFFIXES or not target.is_file():
            raise ValueError('只读取代码、配置和文档文本')
        return target

    def experiment(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"(?:baseline|ab)-[a-f0-9]{12}", identifier):
            raise ValueError("无效或不支持的实验 ID")
        path = self.safe_file(self.runs, f"{identifier}/report.json")
        report = json.loads(path.read_text())
        if identifier.startswith("baseline-"):
            return {"run_id": identifier, "model": report["model"],
                    "dataset_version": report["dataset_version"],
                    "validation_metrics": report["validation_metrics"],
                    "history": [{key: row[key] for key in ("epoch", "loss", "ndcg@10", "recall@10", "evaluated_users", "candidate_items") if key in row}
                                for row in report["history"]],
                    "training_seconds": report["training_seconds"],
                    "model_source_sha256": report["model_source_sha256"],
                    "training_seed": report["config"]["training"].get("seed", report["config"]["seed"]),
                    "training_strategy": report.get("training_strategy", {"loss": "bpr", "negative_sampling": "uniform"}),
                    "training_config": {key: report["config"]["training"][key] for key in
                                        ("dimensions", "learning_rate", "batch_size", "epochs", "l2", "threads")}}
        if report.get("mode") != "models" or report.get("simulated") is not True:
            raise ValueError("注入场景不作为模型优化证据开放")
        return {"run_id": identifier, "simulated": True, "models": report["models"],
                "analysis": {key: value for key, value in report["analysis"].items() if key != "rule_decision"},
                "caveats": report["caveats"]}

    def execute(self, name, arguments):
        if name not in self.SPECS:
            raise ValueError("未知或禁止的工具")
        if not isinstance(arguments, dict) or set(arguments) != set(self.SPECS[name]["arguments"]):
            raise ValueError("工具参数字段不符合定义")
        if name in ("list_feedback", "read_feedback"):
            from mini_agentx.memory.store import FeedbackStore
            store = FeedbackStore(self.runs)
            if name == "read_feedback":
                return "experiment_kb", store.read(arguments["feedback_id"], self.parent, self.dataset_version)
            records = store.list(self.parent, self.dataset_version)
            summaries = [{key: record[key] for key in ("feedback_id", "mechanism_key", "proposal_id", "final_verdict", "conclusions")}
                         for record in records]
            return "experiment_kb", {"feedback": summaries, "count": len(summaries)}
        if name == 'list_directory':
            relative = arguments['path']
            if relative == '.':
                return 'system_kb', {'path': '.', 'entries': [{'path': name, 'type': 'directory'} for name in self.READ_ROOTS if (self.root / name).is_dir()], 'truncated': False}
            target = self.readable(relative, directory=True)
            entries = []
            for child in sorted(target.iterdir()):
                if child.is_symlink() or child.name.startswith('.') or child.name == '__pycache__':
                    continue
                if child.is_dir() or child.suffix in self.TEXT_SUFFIXES:
                    entries.append({'path': child.relative_to(self.root).as_posix(), 'type': 'directory' if child.is_dir() else 'file'})
            return 'system_kb', {'path': relative, 'entries': entries[:200], 'truncated': len(entries) > 200}
        if name == 'search_code':
            target = self.readable(arguments['path'], directory=True)
            query = arguments['query']
            if not isinstance(query, str) or not query.strip() or len(query) > 200:
                raise ValueError('query 须为1..200字符的非空文本')
            matches, inspected, truncated = [], 0, False
            for file in sorted(target.rglob('*')):
                if not file.is_file() or file.suffix not in self.TEXT_SUFFIXES:
                    continue
                try:
                    file = self.readable(file.relative_to(self.root).as_posix())
                    if file.stat().st_size > 24000:
                        continue
                    inspected += 1
                    if inspected > 200:
                        truncated = True
                        break
                    for number, line in enumerate(file.read_text().splitlines(), 1):
                        if query in line:
                            matches.append({'path': file.relative_to(self.root).as_posix(), 'line': number, 'text': line[:500]})
                            if len(matches) >= 50:
                                truncated = True
                                break
                    if truncated:
                        break
                except (ValueError, OSError, UnicodeError):
                    continue
            return 'system_kb', {'query': query, 'matches': matches, 'truncated': truncated}
        if name == "list_files":
            return "system_kb", {"paths": list(self.FILES), 'read_roots': list(self.READ_ROOTS)}
        if name == "read_file":
            path = arguments["path"]
            target = self.readable(path)
            if path == "src/mini_agentx/recommender/model.py":
                parent_path = self.safe_file(self.runs, f"{self.parent}/report.json")
                parent_report = json.loads(parent_path.read_text()) if parent_path.exists() else {}
                if parent_report.get("candidate_source") == "candidate_model.py":
                    target = self.safe_file(self.runs, f"{self.parent}/candidate_model.py")
                    if hashlib.sha256(target.read_bytes()).hexdigest() != parent_report["model_source_sha256"]:
                        raise ValueError("父模型源码摘要不匹配")
            if target.stat().st_size > 24000:
                raise ValueError("文件超过读取大小上限")
            return "system_kb", {"path": path, "content": target.read_text(),
                                  "actual_source": str(target), "parent_run_id": self.parent,
                                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        if name == "read_experiment":
            return "experiment_kb", self.experiment(arguments["run_id"])
        if name == "list_experiments":
            entries = []
            for folder in sorted(self.runs.iterdir(), key=lambda path: path.name):
                if not re.fullmatch(r"(?:baseline|ab)-[a-f0-9]{12}", folder.name):
                    continue
                try:
                    item = self.experiment(folder.name)
                except (ValueError, OSError, KeyError):
                    continue
                entries.append({"run_id": folder.name, "kind": "simulated_model_ab" if item.get("simulated") else "training",
                                "dataset_version": item.get("dataset_version", item.get("models", {}).get("dataset_version"))})
            return "experiment_kb", {"experiments": entries[-30:]}
        parent = self.safe_file(self.runs, f"{self.parent}/report.json")
        if not re.fullmatch(r"baseline-[a-f0-9]{12}", self.parent):
            raise ValueError("训练统计仅支持已记录的基线实验")
        report = json.loads(parent.read_text())
        data = Path(report["data_directory"]).resolve()
        allowed_data = self.safe_file(self.root, "data/processed")
        if not data.is_relative_to(allowed_data.resolve()):
            raise ValueError("数据目录不在项目 processed 白名单")
        manifest = json.loads(self.safe_file(data, "manifest.json").read_text())
        training = self.safe_file(data, "train.csv")
        if manifest["dataset_version"] != report["dataset_version"] or hashlib.sha256(training.read_bytes()).hexdigest() != manifest["file_sha256"]["train.csv"]:
            raise ValueError("训练数据版本或摘要不匹配")
        with training.open() as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError("训练数据为空")
        users = Counter(row["user_id"] for row in rows)
        items = Counter(row["item_id"] for row in rows)
        return "data_analysis", {
            "scope": "training_only", "dataset_version": report["dataset_version"],
            "interactions": len(rows), "users": len(users), "items": len(items),
            "min_user_interactions": min(users.values()), "max_user_interactions": max(users.values()),
            "median_user_interactions": median(users.values()),
            "top10_item_interaction_share": sum(sorted(items.values(), reverse=True)[:10]) / len(rows),
            "items_with_at_most_5_interactions": sum(value <= 5 for value in items.values()),
        }
