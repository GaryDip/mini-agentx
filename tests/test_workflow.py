import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_agentx.agents.activity_log import ActivityLog
from mini_agentx.workflow.runner import iterate


class Client:
    config = {'max_attempts': 2}
    workflow = {}
    def complete(self, messages):
        return {}, {'model': 'mock', 'usage': {'total_tokens': 1}}


class WorkflowTest(unittest.TestCase):
    def fixture(self, root):
        runs = root / 'runs'
        parent = runs / 'baseline-0123456789ab'
        parent.mkdir(parents=True)
        import hashlib
        code = root / 'src/mini_agentx/recommender/model.py'
        code.parent.mkdir(parents=True)
        code.write_text('# mock model')
        (parent / 'model.pt').write_bytes(b'mock weights')
        (parent / 'report.json').write_text(json.dumps({'run_id': parent.name, 'model_source_sha256': hashlib.sha256(code.read_bytes()).hexdigest(), 'checkpoint_sha256': hashlib.sha256(b'mock weights').hexdigest(), 'config': {'training': {}}}))
        context = root / 'context.json'
        context.write_text(json.dumps({'task': {}, 'evidence': [], 'parent_run_id': parent.name}))
        return runs, parent.name, context

    def test_live_child_log_has_round_agent_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'child').mkdir()
            parent = ActivityLog(root, '完整工作流')
            parent.round_number = 2
            child = ActivityLog(root / 'child', 'Developing', parent=parent)
            child.record('请求工具：check_candidate', reason='验证修改后的接口', result={'valid': True})
            text = (root / 'activity.md').read_text()
            self.assertIn('第2轮 · Developing', text)
            self.assertIn('验证修改后的接口', text)
            event = json.loads((root / 'activity.jsonl').read_text())
            self.assertEqual(event['agent'], 'Developing')
            self.assertEqual(event['round'], 2)
            self.assertTrue(event['result']['valid'])

    def test_two_round_order_requests_and_fail_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, parent, context = self.fixture(root)
            order = []
            def brainstorm(*args, **kwargs):
                order.append('brainstorm')
                args[2].complete([{'role': 'user', 'content': 'mock context'}])
                return {'handoff_status': 'ready'}
            def development(tools, client, activity=None):
                order.append('developing')
                folder = Path(tools.directory)
                folder.mkdir(exist_ok=True)
                (folder / 'handoff.json').write_text(json.dumps({'training_run': str(runs / parent)}))
                client.complete([{'role': 'user', 'content': 'mock proposal'}])
                return {'result': {'ndcg_delta': .01}}
            def evaluation(*args, **kwargs):
                order.append('evaluation')
                args[3].complete([{'role': 'user', 'content': 'mock evidence'}])
                return {'ab_run_id': 'ab-111111111111', 'assessment': {'final': {'verdict': 'EXTEND'}}, 'feedback_id': 'feedback-222222222222'}
            clients = {name: Client() for name in ('brainstorm', 'developing', 'evaluation')}
            with patch('mini_agentx.workflow.runner.build_context', return_value={'context_path': str(context)}) as build, patch('mini_agentx.workflow.runner.BrainstormTools'), patch('mini_agentx.workflow.runner.DevelopingTools', side_effect=lambda *args, **kwargs: type('Tools', (), {'directory': kwargs['directory']})()), patch('mini_agentx.workflow.runner.generate_proposals', side_effect=brainstorm), patch('mini_agentx.workflow.runner.develop', side_effect=development), patch('mini_agentx.workflow.runner.run_candidate_evaluation', side_effect=evaluation):
                result = iterate(root, runs, parent, 2, clients)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(order, ['brainstorm', 'developing', 'evaluation'] * 2)
                self.assertEqual(build.call_count, 2)
                self.assertEqual(result['model_calls'], 6)
                self.assertEqual(len(list(Path(result['directory']).glob('*request-*.json'))), 6)
                self.assertIn('第2轮', Path(result['activity_log']).read_text())
                order.clear()
                result = iterate(root, runs, parent, 2, clients, max_model_calls=1)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['model_calls'], 1)
                self.assertEqual(order, ['brainstorm', 'developing'])
                self.assertEqual(result['rounds'][0]['stage'], 'developing')

    def test_needs_evidence_does_not_develop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, parent, context = self.fixture(root)
            with patch('mini_agentx.workflow.runner.build_context', return_value={'context_path': str(context)}), patch('mini_agentx.workflow.runner.BrainstormTools'), patch('mini_agentx.workflow.runner.generate_proposals', return_value={'handoff_status': 'needs_evidence'}), patch('mini_agentx.workflow.runner.develop') as develop:
                result = iterate(root, runs, parent, 2, {name: Client() for name in ('brainstorm', 'developing', 'evaluation')})
                self.assertEqual(result['status'], 'needs_evidence')
                develop.assert_not_called()
