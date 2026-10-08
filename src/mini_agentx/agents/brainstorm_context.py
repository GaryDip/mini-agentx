"""从基线报告按白名单选取证据，不传整个运行报告给 LLM。"""
import hashlib
import json
from pathlib import Path
import uuid

from .brainstorm_contracts import TaskBoundary, validate_context


def build_context(baseline, task_path, runs):
    task = json.loads(Path(task_path).read_text())
    TaskBoundary.from_dict(task)
    baseline = Path(baseline)
    report = json.loads((baseline / "report.json").read_text())
    checkpoint_hash = hashlib.sha256((baseline / "model.pt").read_bytes()).hexdigest()
    if checkpoint_hash != report["checkpoint_sha256"]:
        raise ValueError("基线权重摘要不匹配")
    context = {
        "schema_version": 1, "task": task,
        "parent_run_id": report["run_id"], "dataset_version": report["dataset_version"],
        "evidence": [
            {"evidence_id": "system-model", "source": "system_kb", "status": "documented",
             "facts": {"model": report["model"], "dimensions": report["config"]["training"]["dimensions"],
                       "scoring": "user/item embedding dot product", "loss": "BPR with L2 regularization",
                       "features": "user and item IDs only", "metric": "ndcg@10",
                       "candidate_file_scope": "paths relative to isolated candidate directory"}},
            {"evidence_id": "baseline-validation", "source": "experiment_kb", "status": "observed",
             "facts": {"ndcg@10": report["validation_metrics"]["ndcg@10"],
                       "recall@10": report["validation_metrics"]["recall@10"],
                       "evaluated_users": report["validation_metrics"]["evaluated_users"],
                       "best_epoch": report["best_epoch"], "training_seconds": report["training_seconds"]}},
            {"evidence_id": "popularity-validation", "source": "experiment_kb", "status": "observed",
             "facts": {"ndcg@10": report["popularity_validation_metrics"]["ndcg@10"],
                       "recall@10": report["popularity_validation_metrics"]["recall@10"]}},
        ], "avoid_set": [],
    }
    validate_context(context)
    output = Path(runs) / ("brainstorm-input-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    (output / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    return {"context_path": str(output / "context.json"), "parent_run_id": report["run_id"],
            "evidence_ids": [item["evidence_id"] for item in context["evidence"]]}
