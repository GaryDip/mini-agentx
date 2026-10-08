import argparse
import json

from mini_agentx.config import load_config
from mini_agentx.data.prepare import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini AgentX：推荐系统自动优化学习项目"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="mini-agentx 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command")
    check_parser = subparsers.add_parser(
        "check-config", help="读取并校验实验配置"
    )
    check_parser.add_argument(
        "--config", default="configs/baseline.toml",
        help="配置文件路径，默认 configs/baseline.toml",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-data", help="转换正反馈并保存固定数据划分"
    )
    prepare_parser.add_argument("--config", default="configs/baseline.toml")
    train_parser = subparsers.add_parser("train", help="训练 BPR-MF 基线")
    train_parser.add_argument("--config", default="configs/baseline.toml")
    eval_parser = subparsers.add_parser("evaluate", help="评估已保存的模型")
    eval_parser.add_argument("--run", required=True)
    eval_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    eval_parser.add_argument("--config", default="configs/baseline.toml")
    args = parser.parse_args()
    if args.command in ("check-config", "prepare-data"):
        try:
            config = load_config(args.config)
            if args.command == "prepare-data":
                config = prepare_data(config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(config, ensure_ascii=False, indent=2))
    elif args.command in ("train", "evaluate"):
        from pathlib import Path
        from mini_agentx.recommender.train import train
        from mini_agentx.recommender.evaluate import evaluate_run
        try:
            config = load_config(args.config)
            if args.command == "train":
                result = train(config)
            else:
                result = evaluate_run(Path(config["paths"]["runs"]) / args.run, args.split)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
