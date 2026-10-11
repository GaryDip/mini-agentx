import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_agentx.abtest.isolated_rankings import validate_rankings, isolated_rankings
from mini_agentx.workflow.candidate_evaluation import run_candidate_evaluation


class RankingTest(unittest.TestCase):
    def test_invalid_rankings_rejected(self):
        manifest = {'users': 1, 'items': 4}
        validate_rankings([[1, 2, 3]], manifest, [(0, 0)])
        for bad in ([[0, 1, 2]], [[1, 1, 2]], [[1, 2, 9]], [[True, 2, 3]], [[1]], []):
            with self.assertRaises(ValueError):
                validate_rankings(bad, manifest, [(0, 0)])

    def test_custom_source_tamper_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'candidate_model.py').write_text('never execute')
            (run / 'model.pt').write_bytes(b'weights')
            info = {'candidate_source': 'candidate_model.py', 'model_source_sha256': 'incorrect', 'checkpoint_sha256': 'incorrect'}
            with self.assertRaisesRegex(ValueError, '摘要'):
                isolated_rankings(run, info, {}, [])
            info['candidate_source'] = '../model.py'
            with self.assertRaisesRegex(ValueError, '路径'):
                isolated_rankings(run, info, {}, [])


class HandoffTest(unittest.TestCase):
    def fixture(self, root):
        development = root / 'developing-0123456789ab'
        candidate = development / 'candidate'
        candidate.mkdir(parents=True)
        hashes = {}
        for name in ('model.py', 'training.toml'):
            (candidate / name).write_text(name)
            hashes[name] = hashlib.sha256(name.encode()).hexdigest()
        train = development / 'training/baseline-ffffffffffff'
        train.mkdir(parents=True)
        (train / 'report.json').write_text(json.dumps({'model_source_sha256': hashes['model.py'], 'dataset_version': 'v1', 'run_id': train.name}))
        handoff = {'status': 'developed', 'parent_run_id': 'baseline-111111111111', 'training_run': str(train),
                   'files': hashes, 'result': {'dataset_version': 'v1'}, 'proposal_id': 'p1'}
        (development / 'handoff.json').write_text(json.dumps(handoff))
        return development

    def test_prepare_and_evaluate_stages(self):
        report = {'experiment_id': 'ab-222222222222', 'analysis': {'primary': {'absolute_effect_b_minus_a': 0.0}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self.fixture(root)
            with patch('mini_agentx.workflow.candidate_evaluation.load_ab_config', return_value={}), patch('mini_agentx.workflow.candidate_evaluation.simulate', return_value=report) as simulator:
                result = run_candidate_evaluation(development, root, 'unused')
                self.assertEqual(result['status'], 'ab_completed')
                self.assertEqual(simulator.call_args.args[1], 'null')
                with patch('mini_agentx.workflow.candidate_evaluation.evaluate_experiment', return_value={'final': {'verdict': 'REVIEW_REQUIRED'}}):
                    result = run_candidate_evaluation(development, root, 'unused', object())
                self.assertEqual(result['assessment']['final']['verdict'], 'REVIEW_REQUIRED')
                self.assertFalse(result['promoted'])
            (development / 'candidate/model.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, '交接不一致'):
                run_candidate_evaluation(development, root, 'unused')

    def test_failure_preserves_ab_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self.fixture(root)
            with patch('mini_agentx.workflow.candidate_evaluation.load_ab_config', return_value={}), patch('mini_agentx.workflow.candidate_evaluation.simulate', return_value={'experiment_id': 'ab-222222222222', 'analysis': {'primary': {}}}), patch('mini_agentx.workflow.candidate_evaluation.evaluate_experiment', side_effect=ValueError('mock error')):
                with self.assertRaisesRegex(ValueError, '候选评估失败'):
                    run_candidate_evaluation(development, root, 'unused', object())
            result = json.loads(next(root.glob('candidate-evaluation-*/status.json')).read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['ab_run_id'], 'ab-222222222222')
