"""只由固定工具在隔离子进程中执行，不读取密钥。"""
import importlib.util
import json
from pathlib import Path
import resource
import sys

import torch


def main():
    mode, directory, config_path = sys.argv[1:]
    candidate = Path(directory)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(config["training"]["threads"])
    spec = importlib.util.spec_from_file_location("candidate_model", candidate / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.MatrixFactorization
    if mode == "check":
        model = cls(3, 5, config["training"]["dimensions"])
        users, items = torch.tensor([0, 1]), torch.tensor([1, 2])
        score, all_scores = model.score(users, items), model.score_all()
        if score.shape != (2,) or all_scores.shape != (3, 5):
            raise ValueError("score/score_all 输出形状错误")
        if not torch.isfinite(score).all() or not torch.isfinite(all_scores).all():
            raise ValueError("分数非有限")
        if not torch.allclose(score, all_scores[users, items], atol=1e-6):
            raise ValueError("score 与 score_all 不一致")
        # Fixed trainer needs both embedding lookups for regularization.
        loss = score.sum() + model.users(users).square().sum() + model.items(items).square().sum()
        loss.backward()
        result = {"valid": True, "score_shape": list(score.shape), "score_all_shape": list(all_scores.shape)}
    else:
        from mini_agentx.recommender.train import train
        summary = train(config, model_class=cls, model_source=candidate / "model.py")
        run = Path(config["paths"]["runs"]) / summary["run_id"]
        result = json.loads((run / "report.json").read_text())
        (run / "candidate_model.py").write_text((candidate / "model.py").read_text())
        result["model"] = "candidate-BPR"
        result["candidate_source"] = "candidate_model.py"
        (run / "report.json").write_text(json.dumps(result, indent=2))
    (candidate.parent / f"{mode}-result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
