from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.abtest.simulator import load_ab_config, simulate, evaluation_evidence
from mini_agentx.agents.evaluation import validate_output, evaluate_experiment, resolve_path


def assessment(evidence, verdict="KEEP"):
    paths = ["analysis.primary.absolute_effect_b_minus_a", "analysis.primary.ci95",
             "analysis.checks.guardrails_pass", "analysis.checks.enough_users",
             "analysis.checks.traffic_balanced"]
    return {"simulated": True, "verdict": verdict, "rationale": "模拟证据分析",
            "citations": [{"path": path, "value": resolve_path(evidence, path)} for path in paths],
            "risks": ["合成反馈"], "next_steps": ["预先定义后续窗口"], "caveats": ["非真实收益"]}


class FakeClient:
    config = {"max_attempts": 2}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return next(self.responses), {"model": "mock", "usage": {"total_tokens": 0}}


class EvaluationAgentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = load_ab_config(Path(__file__).resolve().parents[1] / "configs/abtest.toml")
        self.report = simulate(config, "improvement", self.directory.name)
        self.run = Path(self.directory.name) / self.report["experiment_id"]
        self.evidence = evaluation_evidence(self.report)

    def test_numeric_hallucination_is_rejected(self):
        output = assessment(self.evidence)
        output["citations"][0]["value"] += 0.1
        with self.assertRaisesRegex(ValueError, "不匹配"):
            validate_output(output, self.evidence)

    def test_retry_and_blind_context(self):
        bad = assessment(self.evidence)
        bad["verdict"] = "INVALID"
        client = FakeClient([bad, assessment(self.evidence)])
        result = evaluate_experiment(self.run, client)
        self.assertEqual(result["final"]["verdict"], "KEEP")
        self.assertEqual(len(client.calls), 2)
        context = client.calls[0][1]["content"]
        self.assertNotIn("rule_decision", context)
        self.assertNotIn("ctr_delta", context)
        self.assertNotIn("simulator_private", context)

    def test_disagreement_requires_review_and_tamper_is_rejected(self):
        result = evaluate_experiment(self.run, FakeClient([assessment(self.evidence, "DISCARD")]))
        self.assertEqual(result["final"]["verdict"], "REVIEW_REQUIRED")
        with (self.run / "events.csv").open("a") as file:
            file.write("tampered\n")
        client = FakeClient([])
        with self.assertRaisesRegex(ValueError, "摘要"):
            evaluate_experiment(self.run, client)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
