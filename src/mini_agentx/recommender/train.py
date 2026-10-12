import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

import torch

from .evaluate import load_data, ranking_metrics
from .model import MatrixFactorization
from .interface import create_model, LegacyTraining, checked_loss
from .strategy import options, experiment_signature


def train(config, model_class=None, model_source=None):
    settings = config["training"]
    for name in ("dimensions", "epochs", "batch_size", "threads"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"training.{name} 必须为正整数")
    for name in ("learning_rate", "l2"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} 必须为非负有限数")
    strategy = options(config)
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(strategy["seed"])
    rng = random.Random(strategy["seed"])
    data = Path(config["paths"]["processed_data"]) / "movielens-100k-v1"
    manifest, splits = load_data(data)
    cls = model_class or MatrixFactorization
    model = create_model(cls, manifest['users'], manifest['items'], settings)
    owns_training = callable(getattr(model, 'fit_epoch', None))
    legacy = None if owns_training else LegacyTraining(model, splits['train'], settings, strategy['seed'], manifest['items'])
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
        loss_value = checked_loss(model.fit_epoch(list(splits['train']), dict(settings), rng)) if owns_training else checked_loss(legacy.fit_epoch())
        model.eval()
        with torch.no_grad():
            metrics = ranking_metrics(model.score_all(), splits["validation"], splits["train"])
        history.append({"epoch": epoch, "loss": loss_value, **metrics})
        if metrics["ndcg@10"] > best:
            best = metrics["ndcg@10"]
            best_epoch = epoch
            torch.save(model.state_dict(), run / "model.pt")
    report = {
        "run_id": run.name, "model": cls.__name__ if owns_training else "BPR-MF", "config": config,
        "model_protocol": "model_owned_training_v1" if owns_training else "legacy_mf",
        "data_directory": str(data.resolve()), "dataset_version": manifest["dataset_version"],
        "best_epoch": best_epoch, "validation_metrics": history[best_epoch - 1],
        "popularity_validation_metrics": popular_metrics, "history": history,
        "training_seconds": time.monotonic() - started,
        "torch_version": torch.__version__, "device": "cpu",
        "checkpoint_sha256": hashlib.sha256((run / "model.pt").read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(Path(model_source or Path(__file__).with_name("model.py")).read_bytes()).hexdigest(),
        "test_evaluated": False,
    }
    report["training_strategy"] = {"seed": strategy["seed"], "loss": "model_owned", "negative_sampling": "model_owned"} if owns_training else strategy
    report["experiment_signature"] = experiment_signature(report["model_source_sha256"], config, manifest["dataset_version"])
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"run_id": run.name, "best_epoch": best_epoch,
            "validation_metrics": report["validation_metrics"],
            "popularity_validation_metrics": popular_metrics,
            "training_seconds": report["training_seconds"]}
