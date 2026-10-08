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
    ab_parser = subparsers.add_parser("ab-run", help="运行固定窗口的模拟 A/B 实验")
    ab_parser.add_argument("--config", default="configs/baseline.toml")
    ab_parser.add_argument("--ab-config", default="configs/abtest.toml")
    ab_parser.add_argument("--scenario", choices=("improvement", "regression", "guardrail", "inconclusive", "null"), default="improvement")
    ab_parser.add_argument("--model-a", help="A 模型 run ID；模型模式须使用 null 场景")
    ab_parser.add_argument("--model-b", help="B 模型 run ID")
    report_parser = subparsers.add_parser("ab-report", help="读取可交给 Evaluation Agent 的模拟实验报告")
    report_parser.add_argument("--config", default="configs/baseline.toml")
    report_parser.add_argument("--run", required=True)
    report_parser.add_argument("--evidence-only", action="store_true", help="移除程序判决，输出 LLM 盲评证据")
    agent_parser = subparsers.add_parser("ab-evaluate", help="调用 DeepSeek 对模拟 A/B 证据进行盲评")
    agent_parser.add_argument("--run", required=True)
    agent_parser.add_argument("--config", default="configs/baseline.toml")
    agent_parser.add_argument("--llm-config", default="configs/llm.toml")
    context_parser = subparsers.add_parser("brainstorm-context", help="从真实基线构造 Brainstorm 输入，不调用 LLM")
    context_parser.add_argument("--baseline-run", required=True)
    context_parser.add_argument("--task", default="configs/brainstorm-task.json")
    context_parser.add_argument("--config", default="configs/baseline.toml")
    proposals_parser = subparsers.add_parser("check-proposals", help="校验候选提案及交接，不执行实验")
    proposals_parser.add_argument("--context", required=True)
    proposals_parser.add_argument("--proposals", required=True)
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
    elif args.command in ("ab-run", "ab-report"):
        from pathlib import Path
        from mini_agentx.abtest.simulator import load_ab_config, simulate, evaluation_evidence
        try:
            runs = Path(load_config(args.config)["paths"]["runs"])
            if args.command == "ab-run":
                report = simulate(load_ab_config(args.ab_config), args.scenario, runs,
                                  runs / args.model_a if args.model_a else None,
                                  runs / args.model_b if args.model_b else None)
                result = {"experiment_id": report["experiment_id"], "simulated": True,
                          "stage": report["stage"], "primary": report["analysis"]["primary"],
                          "rule_decision": report["analysis"]["rule_decision"]}
            else:
                result = json.loads((runs / args.run / "report.json").read_text())
                if result.get("simulated") is not True:
                    raise ValueError("该产物不是模拟 A/B 报告")
                if args.evidence_only:
                    result = evaluation_evidence(result)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "ab-evaluate":
        from pathlib import Path
        from mini_agentx.llm.client import DeepSeekClient
        from mini_agentx.agents.evaluation import evaluate_experiment
        try:
            runs = Path(load_config(args.config)["paths"]["runs"])
            result = evaluate_experiment(runs / args.run, DeepSeekClient(args.llm_config))
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command in ("brainstorm-context", "check-proposals"):
        from pathlib import Path
        from mini_agentx.agents.brainstorm_context import build_context
        from mini_agentx.agents.brainstorm_contracts import validate_batch
        try:
            if args.command == "brainstorm-context":
                runs = Path(load_config(args.config)["paths"]["runs"])
                result = build_context(runs / args.baseline_run, args.task, runs)
            else:
                context = json.loads(Path(args.context).read_text())
                batch = json.loads(Path(args.proposals).read_text())
                result = validate_batch(batch, context)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
