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
    SPECS = {
        "list_files": {"arguments": {}, "description": "列出可读代码与配置白名单"},
        "read_file": {"arguments": {"path": "白名单中的项目相对路径"}, "description": "读取代码或配置，不读取其他文件"},
        "list_experiments": {"arguments": {}, "description": "列出真实训练与模型模式模拟 A/B 的摘要，排除注入场景"},
        "read_experiment": {"arguments": {"run_id": "已列出的实验 ID"}, "description": "读取验证指标或模型模式 A/B 观测，不读取测试结果及模拟真值"},
        "training_data_summary": {"arguments": {}, "description": "统计父实验训练集的用户交互与物品流行度，无留出标签"},
    }

    def __init__(self, root, runs, context):
        self.root = Path(root).resolve()
        self.runs = Path(runs).resolve()
        self.parent = context["parent_run_id"]

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
        if name == "list_files":
            return "system_kb", {"paths": list(self.FILES)}
        if name == "read_file":
            path = arguments["path"]
            if not isinstance(path, str) or path not in self.FILES:
                raise ValueError("文件不在只读白名单")
            target = self.safe_file(self.root, path)
            if target.stat().st_size > 24000:
                raise ValueError("文件超过读取大小上限")
            return "system_kb", {"path": path, "content": target.read_text(),
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
