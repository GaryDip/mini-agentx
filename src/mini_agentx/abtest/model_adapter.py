"""模型提供排名，独立合成偏好提供点击概率，不使用验证或测试标签。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from mini_agentx.recommender.evaluate import load_data


def model_feedback(model_a, model_b, seed):
    metadata = [json.loads((Path(run) / "report.json").read_text()) for run in (model_a, model_b)]
    if metadata[0]["dataset_version"] != metadata[1]["dataset_version"]:
        raise ValueError("A/B 模型数据版本不同")
    manifest, splits = load_data(Path(metadata[0]["data_directory"]), include_validation=False)
    if manifest["dataset_version"] != metadata[0]["dataset_version"]:
        raise ValueError("模型与当前数据版本不一致")
    rankings = {}
    from .isolated_rankings import isolated_rankings
    for arm, run, info in zip(("A", "B"), (model_a, model_b), metadata):
        rankings[arm] = isolated_rankings(run, info, manifest, splits["train"])
    rng = np.random.default_rng(seed + 907)
    factors = rng.normal(size=(manifest["items"], 16))
    factors /= np.linalg.norm(factors, axis=1, keepdims=True)
    preferences = np.zeros((manifest["users"], 16))
    for user, item in splits["train"]:
        preferences[user] += factors[item]
    norms = np.linalg.norm(preferences, axis=1, keepdims=True)
    preferences /= np.maximum(norms, 1e-12)

    def feedback(user, arm, session):
        slate = rankings[arm][user]
        item = slate[session % len(slate)]
        similarity = float(preferences[user] @ factors[item])
        probability = 0.05 + 0.3 / (1 + np.exp(-3 * similarity))
        return item, float(probability)

    return feedback, manifest["users"], {
        "a_run_id": metadata[0]["run_id"], "b_run_id": metadata[1]["run_id"],
        "dataset_version": manifest["dataset_version"],
        "ranking_execution": "macOS sandbox-exec; actual source and checkpoint verified",
        "a_source_sha256": metadata[0]["model_source_sha256"],
        "b_source_sha256": metadata[1]["model_source_sha256"],
        "a_checkpoint_sha256": metadata[0]["checkpoint_sha256"],
        "b_checkpoint_sha256": metadata[1]["checkpoint_sha256"],
        "feedback_oracle": "fixed synthetic item factors and training-history preference",
        "caveat": "Oracle assumptions may not match MovieLens preferences; outcomes are illustrative.",
    }
