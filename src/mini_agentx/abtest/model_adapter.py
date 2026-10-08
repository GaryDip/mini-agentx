"""模型提供排名，独立合成偏好提供点击概率，不使用验证或测试标签。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mini_agentx.recommender.evaluate import load_data
from mini_agentx.recommender.model import MatrixFactorization


def model_feedback(model_a, model_b, seed):
    metadata = [json.loads((Path(run) / "report.json").read_text()) for run in (model_a, model_b)]
    if metadata[0]["dataset_version"] != metadata[1]["dataset_version"]:
        raise ValueError("A/B 模型数据版本不同")
    manifest, splits = load_data(Path(metadata[0]["data_directory"]), include_validation=False)
    if manifest["dataset_version"] != metadata[0]["dataset_version"]:
        raise ValueError("模型与当前数据版本不一致")
    rankings = {}
    for arm, run, info in zip(("A", "B"), (model_a, model_b), metadata):
        checkpoint = Path(run) / "model.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != info["checkpoint_sha256"]:
            raise ValueError("模型权重摘要不匹配")
        model = MatrixFactorization(manifest["users"], manifest["items"],
                                    info["config"]["training"]["dimensions"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with torch.no_grad():
            scores = model.score_all()
        if not torch.isfinite(scores).all():
            raise ValueError("模型分数非有限")
        for user, item in splits["train"]:
            scores[user, item] = -torch.inf
        ordered = torch.argsort(scores, dim=1, descending=True, stable=True).tolist()
        rankings[arm] = [[item for item in row if torch.isfinite(scores[user, item])][:10]
                         for user, row in enumerate(ordered)]
        if any(not row for row in rankings[arm]):
            raise ValueError("用户没有可推荐的未见电影")
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
        "feedback_oracle": "fixed synthetic item factors and training-history preference",
        "caveat": "Oracle assumptions may not match MovieLens preferences; outcomes are illustrative.",
    }
