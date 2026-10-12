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
    from mini_agentx.recommender.interface import model_class, check_contract
    cls = model_class(module)
    if mode == "check":
        from mini_agentx.recommender.strategy import options
        result = check_contract(cls, config['training'], options(config)['seed'])
    else:
        from mini_agentx.recommender.train import train
        summary = train(config, model_class=cls, model_source=candidate / "model.py")
        run = Path(config["paths"]["runs"]) / summary["run_id"]
        result = json.loads((run / "report.json").read_text())
        (run / "candidate_model.py").write_text((candidate / "model.py").read_text())
        result["model"] = cls.__name__
        result["candidate_source"] = "candidate_model.py"
        (run / "report.json").write_text(json.dumps(result, indent=2))
    (candidate.parent / f"{mode}-result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
