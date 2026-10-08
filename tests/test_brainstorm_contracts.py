from copy import deepcopy
import json
from pathlib import Path
import unittest

from mini_agentx.agents.brainstorm_contracts import TaskBoundary, validate_batch


class BrainstormContractsTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "examples/brainstorm"
        self.context = json.loads((root / "context.json").read_text())
        self.batch = json.loads((root / "proposals.json").read_text())

    def test_selects_ready_only(self):
        result = validate_batch(self.batch, self.context)
        self.assertEqual(result["selected_proposal"]["proposal_id"], "example-p1")
        batch = deepcopy(self.batch)
        batch["proposals"][0]["maturity"] = "backlog"
        result = validate_batch(batch, self.context)
        self.assertIsNone(result["selected_proposal"])
        self.assertEqual(result["handoff_status"], "needs_evidence")

    def test_invalid_handoffs_are_rejected(self):
        mutations = [
            lambda batch: batch.update(parent_run_id="wrong-parent"),
            lambda batch: batch["proposals"][0]["changes"][0].update(file="../model.py"),
            lambda batch: batch["proposals"][0]["changes"][0].update(file="evaluate.py"),
            lambda batch: batch["proposals"][0]["evidence_refs"][0].update(evidence_id="invented"),
            lambda batch: batch["proposals"][0]["evidence_refs"][0].update(value="invented fact"),
            lambda batch: batch["proposals"][0]["validation"].update(metric="test_ndcg"),
            lambda batch: batch["proposals"][1].update(probes=[]),
            lambda batch: batch["proposals"][1].update(mechanism_key="item_bias"),
            lambda batch: batch["proposals"][0].update(priority=True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                batch = deepcopy(self.batch)
                mutate(batch)
                with self.assertRaises(ValueError):
                    validate_batch(batch, self.context)

    def test_avoid_set_and_task_boundary(self):
        context = deepcopy(self.context)
        context["avoid_set"] = ["item_bias"]
        with self.assertRaisesRegex(ValueError, "已拒绝"):
            validate_batch(self.batch, context)
        task = deepcopy(self.context["task"])
        task["allowed_files"] = [".env"]
        with self.assertRaises(ValueError):
            TaskBoundary.from_dict(task)


if __name__ == "__main__":
    unittest.main()
