import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_agentx.agents.developing import develop
from mini_agentx.tools.developing import DevelopingTools


class FakeClient:
    workflow = {"max_model_calls": 12, "max_tool_calls": 10}
    def __init__(self, responses):
        self.responses = iter(responses)
    def complete(self, messages):
        return next(self.responses), {"model": "mock"}


def action(name, arguments=None):
    return {"action": "tool", "name": name, "arguments": arguments or {}, "reason": "教学测试动作"}


class DevelopingTest(unittest.TestCase):
    def make_tools(self, root):
        source = Path(__file__).resolve().parents[1] / "src/mini_agentx/recommender/model.py"
        target = root / "src/mini_agentx/recommender/model.py"
        target.parent.mkdir(parents=True)
        target.write_text(source.read_text())
        parent = root / "runs/baseline-0123456789ab"
        parent.mkdir(parents=True)
        (parent / "model.pt").write_bytes(b"mock")
        from mini_agentx.tools.developing import digest
        report = {"run_id": parent.name, "checkpoint_sha256": digest(parent / "model.pt"),
                  "model_source_sha256": digest(target), "dataset_version": "v1",
                  "config": {"seed": 42, "paths": {}, "training": {"dimensions": 32, "learning_rate": .001,
                             "batch_size": 1024, "epochs": 20, "l2": .0001, "threads": 1}},
                  "validation_metrics": {"ndcg@10": .04}}
        (parent / "report.json").write_text(json.dumps(report))
        task = json.loads((source.parents[3] / "configs/brainstorm-task.json").read_text())
        proposal = {"schema_version": 1, "task": task, "parent_run_id": parent.name,
                    "dataset_version": "v1", "proposal": {"proposal_id": "p1", "maturity": "ready"}}
        path = root / "proposal.json"
        path.write_text(json.dumps(proposal))
        return DevelopingTools(root, root / "runs", path)

    def test_revision_requires_recheck_and_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = self.make_tools(Path(directory))
            with self.assertRaises(ValueError):
                tools.execute("write_candidate_file", {"path": "../model.py", "content": "x"})
            with self.assertRaises(ValueError):
                tools.execute("train_candidate", {})
            with patch.object(tools, "run_worker", return_value={"valid": True}):
                tools.execute("check_candidate", {})
            tools.execute("write_candidate_file", {"path": "training.toml", "content": tools.original["training.toml"].replace("32", "64")})
            self.assertIsNone(tools.checked)
            with self.assertRaises(ValueError):
                tools.execute("train_candidate", {})
            with self.assertRaises(ValueError):
                tools.finish()

    def test_repair_exact_requests_and_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = self.make_tools(Path(directory))
            content = tools.original["training.toml"].replace("32", "64")
            calls = [action("read_candidate_file", {"path": "model.py"}),
                     action("read_candidate_file", {"path": "training.toml"}),
                     action("write_candidate_file", {"path": "../invalid", "content": "x"}),
                     action("write_candidate_file", {"path": "training.toml", "content": content}),
                     action("check_candidate"), action("train_candidate"),
                     action("read_training_result"), action("get_candidate_diff"),
                     {"action": "finish", "reason": "模拟训练已完成；没有声称提高"}]
            def worker(mode):
                if mode == "check": return {"valid": True}
                return {"dataset_version": "v1", "run_id": "baseline-ffffffffffff", "validation_metrics": {"ndcg@10": .03},
                        "history": [], "training_seconds": 1}
            with patch.object(tools, "run_worker", side_effect=worker), patch.object(tools, "finish", return_value={"status": "developed"}):
                result = develop(tools, FakeClient(calls))
            self.assertEqual(result["status"], "developed")
            messages = json.loads((tools.directory / "request-04.json").read_text())
            self.assertIn("tool_error", messages[-1]["content"])
            self.assertTrue((tools.directory / "activity.md").exists())

    def test_finish_without_training_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = self.make_tools(Path(directory))
            client = FakeClient([{"action": "finish", "reason": "声称完成"}] * 3)
            with self.assertRaisesRegex(ValueError, "Developing 失败"):
                develop(tools, client)
            self.assertFalse((tools.directory / "handoff.json").exists())


if __name__ == "__main__":
    unittest.main()
