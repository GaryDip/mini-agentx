import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_agentx.workflow.runner import iterate
from mini_agentx.workflow.models import archive_model
from mini_agentx.tools.brainstorm import BrainstormTools
from mini_agentx.memory.store import FeedbackStore
from test_workflow import Client
import test_workflow


class PromotionTest(unittest.TestCase):
    def test_five_rounds_two_promotions_keep_artifacts_and_lineage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs, original, context = test_workflow.WorkflowTest().fixture(root)
            report_path = runs / original / 'report.json'
            report = json.loads(report_path.read_text())
            report['dataset_version'] = 'v1'
            report_path.write_text(json.dumps(report))
            parents = []
            def build(parent, *args):
                parents.append(parent.name)
                return {'context_path': str(context)}
            def development(tools, *args, **kwargs):
                directory = Path(tools.directory)
                training = directory / 'training/baseline-abcdefabcdef'
                training.mkdir(parents=True)
                source = f'# candidate {len(parents)}'
                (training / 'candidate_model.py').write_text(source)
                (training / 'model.pt').write_bytes(b'candidate weights')
                report = json.loads((runs / parents[-1] / 'report.json').read_text())
                report.update(run_id=training.name, candidate_source='candidate_model.py',
                              checkpoint_sha256=hashlib.sha256(b'candidate weights').hexdigest(),
                              model_source_sha256=hashlib.sha256(source.encode()).hexdigest())
                (training / 'report.json').write_text(json.dumps(report))
                (directory / 'handoff.json').write_text(json.dumps({'training_run': str(training)}))
                return {'result': {'ndcg_delta': .01}}
            verdicts = iter(['KEEP', 'EXTEND', 'KEEP', 'DISCARD', 'REVIEW_REQUIRED'])
            def evaluation(*args, **kwargs):
                verdict = next(verdicts)
                return {'ab_run_id': 'ab-111111111111', 'feedback_id': 'feedback-222222222222',
                        'assessment': {'final': {'verdict': verdict}, 'rule_decision': {'verdict': verdict},
                        'validation': {'citations_verified': True, 'agrees_with_rule': True, 'interpretation_warnings': []}}}
            with patch('mini_agentx.workflow.runner.build_context', side_effect=build), patch('mini_agentx.workflow.runner.BrainstormTools'), patch('mini_agentx.workflow.runner.DevelopingTools', side_effect=lambda *args, **kwargs: type('Tools', (), {'directory': kwargs['directory']})()), patch('mini_agentx.workflow.runner.generate_proposals', return_value={'handoff_status': 'ready'}), patch('mini_agentx.workflow.runner.develop', side_effect=development), patch('mini_agentx.workflow.runner.run_candidate_evaluation', side_effect=evaluation):
                result = iterate(root, runs, original, 5, {key: Client() for key in ('brainstorm', 'developing', 'evaluation')})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(result['promotions']), 2)
            self.assertEqual(len(result['baseline_versions']), 3)
            self.assertEqual(parents[0], original)
            self.assertEqual(parents[1], parents[2])
            self.assertEqual(parents[3], parents[4])
            self.assertNotEqual(parents[0], parents[1])
            self.assertNotEqual(parents[1], parents[3])
            for item in result['rounds']:
                archive = Path(item['candidate_archive'])
                for name in ('model.pt', 'candidate_model.py', 'training.toml', 'report.json', 'decision.json'):
                    self.assertTrue((archive / name).exists())
            latest = runs / result['current_baseline_run']
            self.assertEqual((latest / 'candidate_model.py').read_text(), '# candidate 3')
            tools = BrainstormTools(root, runs, {'parent_run_id': latest.name, 'dataset_version': 'v1'})
            _, observed = tools.execute('read_file', {'path': 'src/mini_agentx/recommender/model.py'})
            self.assertEqual(observed['content'], '# candidate 3')
            self.assertEqual(observed['parent_run_id'], latest.name)
            (latest / 'candidate_model.py').write_text('# tampered')
            with self.assertRaises(ValueError):
                archive_model(root, runs, latest, root / 'bad-archive')

            self.assertEqual(FeedbackStore(runs).lineage(latest.name, 'v1'), [parents[3], parents[1], original])
            store = FeedbackStore(runs)
            store.save({'feedback_id': 'feedback-123456789abc', 'source_run': 'candidate-evaluation-123456789abc', 'parent_run_id': original, 'dataset_version': 'v1'})
            self.assertEqual(len(store.list(latest.name, 'v1')), 1)
            self.assertEqual(store.read('feedback-123456789abc', latest.name, 'v1')['parent_run_id'], original)
