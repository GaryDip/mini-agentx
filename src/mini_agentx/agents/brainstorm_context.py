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
                       "scoring": "读取当前父模型源码确认实际打分方法", "loss": ("读取当前模型源码确认自有损失与正则" if report.get("model_protocol") == "model_owned_training_v1" else report.get("training_strategy", {}).get("loss", "bpr") + " with embedding L2"),
                       "features": "user/item IDs and training positive interaction pairs; no content features or holdout labels", "metric": "ndcg@10",
                       "model_interface": "Recommender(num_users,num_items,settings): fit_epoch(train_pairs,settings,rng), score(user_ids,item_ids), score_all(), nn.Module state_dict; legacy MatrixFactorization compatible",
                       "learning_rate": report["config"]["training"]["learning_rate"],
                       "l2": report["config"]["training"]["l2"],
                       "epochs": report["config"]["training"]["epochs"],
                       "sampling": report.get("training_strategy", {}).get("negative_sampling", "uniform"),
                       "change_surface": "model.py: arbitrary model-owned representation/training via Recommender; no required embedding names; training.toml retains existing numeric/resource/strategy fields",
                       "frozen_training_logic": "data split, evaluation, resource limits, epoch scheduling and checkpoint selection fixed; fit_epoch owns optimizer/loss/sampling; train-only input",
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
    from mini_agentx.memory.store import FeedbackStore
    records = FeedbackStore(runs).list(context["parent_run_id"], context["dataset_version"])
    if records:
        context["task"]["unknowns"] = [value for value in context["task"]["unknowns"] if value != "当前没有历史候选实验记忆"]
        context["evidence"].append({"evidence_id": "experiment-memory", "source": "experiment_kb", "status": "observed",
                                    "facts": {"count": len(records), "scope": "baseline_lineage_and_dataset", "tool": "list_feedback"}})
    validate_context(context)
    output = Path(runs) / ("brainstorm-input-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    (output / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    return {"context_path": str(output / "context.json"), "parent_run_id": report["run_id"],
            "evidence_ids": [item["evidence_id"] for item in context["evidence"]]}
