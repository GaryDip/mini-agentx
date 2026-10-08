import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini AgentX：推荐系统自动优化学习项目"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="mini-agentx 0.1.0",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
