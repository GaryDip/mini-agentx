import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def load_data(directory: Path, include_test=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"数据摘要不匹配：{name}")
    splits = {}
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    for name in names:
        with (directory / f"{name}.csv").open() as file:
            splits[name] = [(int(row["user_id"]), int(row["item_id"]))
                            for row in csv.DictReader(file)]
    return manifest, splits


def ranking_metrics(scores, targets, seen, k=10):
    """每个用户一个留出正例；稳定排序用物品 ID 打破分数相同的情况。"""
    if not targets:
        raise ValueError("没有可评估用户")
    if not torch.isfinite(scores).all():
        raise ValueError("推荐分数包含非有限值")
    masked = scores.clone()
    for user, item in seen:
        masked[user, item] = -torch.inf
    top = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :k]
    recall = ndcg = 0.0
    for user, item in targets:
        hits = (top[user] == item).nonzero().flatten()
        if len(hits):
            recall += 1
            ndcg += 1 / math.log2(int(hits[0]) + 2)
    return {f"ndcg@{k}": ndcg / len(targets),
            f"recall@{k}": recall / len(targets),
            "evaluated_users": len(targets), "candidate_items": scores.shape[1]}


def evaluate_run(run: Path, split="validation"):
    from .model import MatrixFactorization
    metadata = json.loads((run / "report.json").read_text())
    manifest, splits = load_data(Path(metadata["data_directory"]), include_test=split == "test")
    if manifest["dataset_version"] != metadata["dataset_version"]:
        raise ValueError("训练和评估数据版本不一致")
    model = MatrixFactorization(manifest["users"], manifest["items"],
                                metadata["config"]["training"]["dimensions"])
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    seen = splits["train"] + (splits["validation"] if split == "test" else [])
    with torch.no_grad():
        metrics = ranking_metrics(model.score_all(), splits[split], seen)
    (run / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
