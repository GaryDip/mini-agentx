import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch
from torch.nn import functional as F

from .evaluate import load_data, ranking_metrics
from .model import MatrixFactorization


def train(config):
    settings = config["training"]
    for name in ("dimensions", "epochs", "batch_size", "threads"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"training.{name} 必须为正整数")
    for name in ("learning_rate", "l2"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} 必须为非负有限数")
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    data = Path(config["paths"]["processed_data"]) / "movielens-100k-v1"
    manifest, splits = load_data(data)
    model = MatrixFactorization(manifest["users"], manifest["items"], settings["dimensions"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    positives = [set() for _ in range(manifest["users"])]
    for user, item in splits["train"]:
        positives[user].add(item)
    negatives = [[item for item in range(manifest["items"]) if item not in history]
                 for history in positives]
    if any(not pool for pool in negatives):
        raise ValueError("用户没有可采样负例")
    run = Path(config["paths"]["runs"]) / ("baseline-" + uuid.uuid4().hex[:12])
    run.mkdir(parents=True)
    started = time.monotonic()
    popularity = torch.zeros(manifest["items"])
    for _, item in splits["train"]:
        popularity[item] += 1
    popular_metrics = ranking_metrics(popularity.expand(manifest["users"], -1),
                                     splits["validation"], splits["train"])
    history = []
    best = -1.0
    best_epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(splits["train"])).tolist()
        total = 0.0
        for offset in range(0, len(order), settings["batch_size"]):
            batch = [splits["train"][index] for index in order[offset:offset + settings["batch_size"]]]
            users = torch.tensor([row[0] for row in batch])
            items = torch.tensor([row[1] for row in batch])
            sampled = torch.tensor([rng.choice(negatives[user]) for user, _ in batch])
            loss = F.softplus(model.score(users, sampled) - model.score(users, items)).mean()
            regularizer = (model.users(users).square().sum() + model.items(items).square().sum()
                           + model.items(sampled).square().sum()) / len(batch)
            loss = loss + settings["l2"] * regularizer
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            metrics = ranking_metrics(model.score_all(), splits["validation"], splits["train"])
        history.append({"epoch": epoch, "loss": total / len(order), **metrics})
        if metrics["ndcg@10"] > best:
            best = metrics["ndcg@10"]
            best_epoch = epoch
            torch.save(model.state_dict(), run / "model.pt")
    report = {
        "run_id": run.name, "model": "BPR-MF", "config": config,
        "data_directory": str(data.resolve()), "dataset_version": manifest["dataset_version"],
        "best_epoch": best_epoch, "validation_metrics": history[best_epoch - 1],
        "popularity_validation_metrics": popular_metrics, "history": history,
        "training_seconds": time.monotonic() - started,
        "torch_version": torch.__version__, "device": "cpu",
        "checkpoint_sha256": hashlib.sha256((run / "model.pt").read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
        "test_evaluated": False,
    }
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"run_id": run.name, "best_epoch": best_epoch,
            "validation_metrics": report["validation_metrics"],
            "popularity_validation_metrics": popular_metrics,
            "training_seconds": report["training_seconds"]}
