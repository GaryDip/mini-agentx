from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.agents.brainstorm import generate_proposals
from mini_agentx.llm.client import LLMError


class FakeClient:
    config = {"max_attempts": 2}

    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, {"model": "mock", "usage": {"total_tokens": 0}}


class BrainstormAgentTest(unittest.TestCase):
    def setUp(self):
        self.example = Path(__file__).resolve().parents[1] / "examples/brainstorm"
        self.batch = json.loads((self.example / "proposals.json").read_text())

    def test_repair_then_persist_handoff(self):
        bad = deepcopy(self.batch)
        bad["proposals"][0]["changes"][0]["file"] = "../evaluate.py"
        with tempfile.TemporaryDirectory() as directory:
            result = generate_proposals(self.example / "context.json", directory, FakeClient([bad, self.batch]))
            self.assertEqual(result["attempts"], 2)
            output = Path(result["directory"])
            self.assertTrue((output / "proposal.json").exists())
            trace = json.loads((output / "trace.json").read_text())
            self.assertEqual([item["status"] for item in trace["attempts"]], ["invalid_output", "valid"])

    def test_no_ready_has_no_executable_proposal(self):
        batch = deepcopy(self.batch)
        batch["proposals"][0]["maturity"] = "backlog"
        with tempfile.TemporaryDirectory() as directory:
            result = generate_proposals(self.example / "context.json", directory, FakeClient([batch]))
            self.assertEqual(result["handoff_status"], "needs_evidence")
            self.assertFalse((Path(result["directory"]) / "proposal.json").exists())

    def test_budget_failure_preserves_trace_without_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "生成失败"):
                generate_proposals(self.example / "context.json", directory,
                                   FakeClient([LLMError("网络失败"), LLMError("网络失败")]))
            output = next(Path(directory).iterdir())
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "proposal.json").exists())


if __name__ == "__main__":
    unittest.main()
