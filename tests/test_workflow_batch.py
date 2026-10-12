import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_workflow
from mini_agentx.workflow.runner import iterate


class BatchTest(unittest.TestCase):
    def test_all_ready_attempted_failure_isolated_probe_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs, parent, context = test_workflow.WorkflowTest().fixture(root)
            data = json.loads(context.read_text()); data['dataset_version'] = 'v1'; context.write_text(json.dumps(data))
            attempted = []
            def brainstorm(*args, **kwargs):
                path = kwargs['destination']; path.mkdir()
                (path / 'proposals.json').write_text(json.dumps({'proposals': [{'proposal_id': 'a', 'priority': 1, 'maturity': 'ready'}, {'proposal_id': 'b', 'priority': 2, 'maturity': 'ready'}, {'proposal_id': 'c', 'priority': 3, 'maturity': 'probe_first'}]}))
                return {'handoff_status': 'ready'}
            def tools(*args, **kwargs):
                envelope = json.loads(Path(args[2]).read_text())
                self.assertEqual(envelope['parent_run_id'], parent)
                return type('Tools', (), {'directory': kwargs['directory'], 'proposal': envelope['proposal']})()
            def develop(tool, *args, **kwargs):
                attempted.append(tool.proposal['proposal_id'])
                if tool.proposal['proposal_id'] == 'a': raise ValueError('candidate a failed')
                path = tool.directory; path.mkdir()
                (path / 'handoff.json').write_text(json.dumps({'training_run': str(runs / parent)}))
                return {'result': {'ndcg_delta': .01}}
            assessment = {'ab_run_id': 'ab-111111111111', 'assessment': {'final': {'verdict': 'EXTEND'}}, 'feedback_id': 'feedback-222222222222'}
            with patch('mini_agentx.workflow.runner.build_context', return_value={'context_path': str(context)}), patch('mini_agentx.workflow.runner.BrainstormTools'), patch('mini_agentx.workflow.runner.generate_proposals', side_effect=brainstorm), patch('mini_agentx.workflow.runner.DevelopingTools', side_effect=tools), patch('mini_agentx.workflow.runner.develop', side_effect=develop), patch('mini_agentx.workflow.runner.run_candidate_evaluation', return_value=assessment):
                result = iterate(root, runs, parent, 1, {name: test_workflow.Client() for name in ('brainstorm', 'developing', 'evaluation')})
            self.assertEqual(attempted, ['a', 'b'])
            self.assertEqual([item['status'] for item in result['rounds'][0]['candidates']], ['failed', 'completed'])
            self.assertTrue((Path(result['rounds'][0]['candidates'][1]['candidate_archive']) / 'decision.json').exists())
