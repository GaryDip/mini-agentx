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
    args = parser.parse_args()
    if args.command in ("check-config", "prepare-data"):
        try:
            config = load_config(args.config)
            if args.command == "prepare-data":
                config = prepare_data(config)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
