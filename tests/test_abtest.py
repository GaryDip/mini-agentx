from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.abtest.simulator import assign_arm, load_ab_config, simulate, evaluation_evidence
from mini_agentx.abtest.statistics import analyze


class ABTest(unittest.TestCase):
    def setUp(self):
        self.config = load_ab_config(Path(__file__).resolve().parents[1] / "configs/abtest.toml")

    def test_stable_assignment(self):
        self.assertEqual([assign_arm(i, 42) for i in range(100)],
                         [assign_arm(i, 42) for i in range(100)])
        self.assertEqual(set(assign_arm(i, 42) for i in range(100)), {"A", "B"})

    def test_scenarios_and_reproducible_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {"improvement": "KEEP", "regression": "DISCARD",
                        "guardrail": "DISCARD", "inconclusive": "EXTEND"}
            for name, verdict in expected.items():
                report = simulate(self.config, name, directory)
                self.assertEqual(report["analysis"]["rule_decision"]["verdict"], verdict)
                if name == "guardrail":
                    self.assertEqual(report["stage"], "canary_stopped")
                private = json.loads((Path(directory) / report["experiment_id"] / "simulator_private.json").read_text())
                self.assertIn("scenario", private)
                self.assertNotIn("scenario", report)
            a = simulate(self.config, "improvement", directory)
            b = simulate(self.config, "improvement", directory)
            self.assertEqual(a["analysis"], b["analysis"])
            self.assertEqual(a["events_sha256"], b["events_sha256"])
            evidence = json.dumps(evaluation_evidence(a))
            self.assertNotIn('rule_decision', evidence)
            self.assertNotIn('ctr_delta', evidence)

    def test_clustered_uncertainty_and_unbalanced_traffic(self):
        assignments = {user: "A" if user < 20 else "B" for user in range(100)}
        events = [{"user_id": user, "arm": arm, "click": user % 2,
                   "error": 0, "latency_ms": 60} for user, arm in assignments.items()]
        policy = self.config["policy"]
        one = analyze(events, assignments, policy, 300, 42)
        repeated = analyze(events * 10, assignments, policy, 300, 42)
        self.assertEqual(one["primary"]["ci95"], repeated["primary"]["ci95"])
        self.assertEqual(one["rule_decision"]["reason"], "traffic_mismatch_investigate_before_more_traffic")

    def test_invalid_configuration_and_partial_model_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                simulate(self.config, "null", directory, model_a=Path(directory))
            config = deepcopy(self.config)
            config["scenarios"]["improvement"]["ctr_delta"] = float('nan')
            with self.assertRaises(ValueError):
                simulate(config, "improvement", directory)


if __name__ == "__main__":
    unittest.main()
