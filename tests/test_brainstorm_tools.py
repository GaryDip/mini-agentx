from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.agents.brainstorm import generate_proposals
from mini_agentx.tools.brainstorm import BrainstormTools
from test_brainstorm_agent import FakeClient


class ToolLoopTest(unittest.TestCase):
    def test_investigation_results_become_citable_evidence(self):
        example = Path(__file__).resolve().parents[1] / "examples/brainstorm"
        batch = json.loads((example / "proposals.json").read_text())
        parent = json.loads((example / "context.json").read_text())["parent_run_id"]

        class StubTools:
            SPECS = BrainstormTools.SPECS

            def execute(self, name, arguments):
                if name == "read_file":
                    return "system_kb", {"path": arguments["path"], "content": "model code"}
                if name == "read_experiment":
                    return "experiment_kb", {"run_id": parent}
                return "data_analysis", {"scope": "training_only", "users": 2}

        batch["proposals"][0]["evidence_refs"].append({
            "evidence_id": "tool-3", "field": "users", "value": 2})
        client = FakeClient([
            {"action": "tool", "name": "read_file", "arguments": {"path": "src/mini_agentx/recommender/model.py"}, "reason": "确认模型接口与表示结构"},
            {"action": "tool", "name": "read_experiment", "arguments": {"run_id": parent}},
            {"action": "tool", "name": "training_data_summary", "arguments": {}},
            {"action": "propose", "batch": batch},
        ])
        client.workflow = {"max_model_calls": 5, "max_tool_calls": 3}
        with tempfile.TemporaryDirectory() as directory:
            result = generate_proposals(example / "context.json", directory, client, StubTools())
            self.assertEqual(result["tool_calls"], 3)
            saved = json.loads((Path(result["directory"]) / "context.json").read_text())
            self.assertEqual(saved["evidence"][-1]["evidence_id"], "tool-3")
            activity = (Path(result["directory"]) / "activity.md").read_text()
            self.assertIn("确认模型接口与表示结构", activity)
            self.assertLess(activity.index("请求工具：read_file"), activity.index("工具结果：read_file"))
            self.assertIn("校验通过并完成交接", activity)

    def test_propose_without_investigation_fails(self):
        example = Path(__file__).resolve().parents[1] / "examples/brainstorm"
        batch = json.loads((example / "proposals.json").read_text())
        client = FakeClient([{"action": "propose", "batch": batch}] * 2)
        client.workflow = {"max_model_calls": 2, "max_tool_calls": 3}
        with tempfile.TemporaryDirectory() as directory:
            tools = BrainstormTools(Path(directory), Path(directory), {"parent_run_id": "unused"})
            with self.assertRaisesRegex(ValueError, "生成失败"):
                generate_proposals(example / "context.json", directory, client, tools)


class ToolBoundaryTest(unittest.TestCase):
    def test_read_scope_symlinks_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            parent = "baseline-0123456789ab"
            data = root / "data/processed/example"
            data.mkdir(parents=True)
            (runs / parent).mkdir(parents=True)
            train = data / "train.csv"
            train.write_text("user_id,item_id,timestamp\n0,0,1\n0,1,2\n1,0,1\n")
            (data / "test.csv").write_text("PRIVATE_TEST_LABELS")
            (data / "manifest.json").write_text(json.dumps({"dataset_version": "v1", "file_sha256": {
                "train.csv": hashlib.sha256(train.read_bytes()).hexdigest()}}))
            (runs / parent / "report.json").write_text(json.dumps({"dataset_version": "v1", "data_directory": str(data)}))
            tools = BrainstormTools(root, runs, {"parent_run_id": parent})
            for path in (".env", "../.env", "data/processed/example/test.csv", "runs/ab-x/simulator_private.json"):
                with self.assertRaises(ValueError):
                    tools.execute("read_file", {"path": path})
            model = root / tools.FILES[0]
            model.parent.mkdir(parents=True)
            model.symlink_to(data / "test.csv")
            with self.assertRaises(ValueError):
                tools.execute("read_file", {"path": tools.FILES[0]})
            source, facts = tools.execute("training_data_summary", {})
            self.assertEqual(source, "data_analysis")
            self.assertEqual(facts["interactions"], 3)
            self.assertNotIn("PRIVATE_TEST_LABELS", json.dumps(facts))
            with self.assertRaises(ValueError):
                tools.execute("shell", {"command": "anything"})


if __name__ == "__main__":
    unittest.main()
