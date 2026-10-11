from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.memory.store import FeedbackStore, import_feedback
from mini_agentx.tools.brainstorm import BrainstormTools
from mini_agentx.agents.brainstorm import generate_proposals
from test_brainstorm_agent import FakeClient


class MemoryTest(unittest.TestCase):
    def fixture(self, root):
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        parent_id = 'baseline-111111111111'
        development = root / 'developing-222222222222'
        training = development / 'training/baseline-333333333333'
        metrics = {'ndcg@10': .04, 'recall@10': .08}
        save(root / parent_id / 'report.json', {'run_id': parent_id, 'dataset_version': 'v1', 'validation_metrics': metrics})
        files = {}
        for name in ('model.py', 'training.toml'):
            path = development / 'candidate' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('example')
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        training.mkdir(parents=True)
        (training / 'model.pt').write_bytes(b'weights')
        report = {'run_id': training.name, 'dataset_version': 'v1', 'validation_metrics': {'ndcg@10': .05, 'recall@10': .09},
                  'model_source_sha256': files['model.py'], 'training_seconds': 1, 'config': {'training': {'dimensions': 64}},
                  'checkpoint_sha256': hashlib.sha256(b'weights').hexdigest()}
        save(training / 'report.json', report)
        proposal = {'proposal_id': 'p1', 'mechanism_key': 'capacity', 'title': 'capacity', 'hypothesis': '待验证', 'changes': [], 'validation': {}}
        save(development / 'proposal.json', {'parent_run_id': parent_id, 'dataset_version': 'v1', 'proposal': proposal})
        save(development / 'handoff.json', {'parent_run_id': parent_id, 'files': files})
        ab = root / 'ab-444444444444'
        ab.mkdir()
        (ab / 'events.csv').write_text('example events')
        analysis = {'primary': {'absolute_effect_b_minus_a': -.002, 'ci95': [-.01, .01]},
                    'checks': {'traffic_balanced': True, 'enough_users': True, 'guardrails_pass': True},
                    'rule_decision': {'verdict': 'EXTEND'}}
        save(ab / 'report.json', {'experiment_id': ab.name, 'mode': 'models', 'simulated': True,
                                 'events_sha256': hashlib.sha256((ab / 'events.csv').read_bytes()).hexdigest(),
                                 'models': {'a_run_id': parent_id, 'b_run_id': training.name}, 'analysis': analysis})
        paths = {'analysis.primary.absolute_effect_b_minus_a': -.002, 'analysis.primary.ci95': [-.01, .01],
                 'analysis.checks.traffic_balanced': True, 'analysis.checks.enough_users': True, 'analysis.checks.guardrails_pass': True}
        llm = {'verdict': 'EXTEND', 'simulated': True, 'rationale': '存在流量不平衡信号，UNVERIFIED_TEXT',
               'citations': [{'path': k, 'value': v} for k, v in paths.items()],
               'risks': ['限制'], 'next_steps': ['调查'], 'caveats': ['模拟']}
        evaluation = ab / 'evaluations/e1'
        save(evaluation / 'assessment.json', {'experiment_id': ab.name, 'llm_assessment': llm, 'final': {'verdict': 'EXTEND'}})
        pipeline = root / 'candidate-evaluation-555555555555'
        save(pipeline / 'status.json', {'status': 'evaluated', 'parent_run_id': parent_id, 'development_directory': str(development),
                                        'candidate_training_run': str(training), 'ab_run_id': ab.name,
                                        'assessment': {'evaluation_directory': str(evaluation)}})
        return pipeline, evaluation

    def test_import_idempotent_filtering_and_unverified_text_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline, evaluation = self.fixture(root)
            feedback = import_feedback(root, pipeline.name)
            self.assertEqual(feedback['final_verdict'], 'REVIEW_REQUIRED')
            self.assertTrue(feedback['observed']['checks']['traffic_balanced'])
            self.assertNotIn('UNVERIFIED_TEXT', json.dumps(feedback))
            store = FeedbackStore(root)
            import_feedback(root, pipeline.name)
            self.assertEqual(len(store.list('baseline-111111111111', 'v1')), 1)
            self.assertEqual(store.list('baseline-111111111111', 'another-version'), [])
            self.assertEqual(store.list('another-parent', 'v1'), [])
            full = json.loads((pipeline / 'feedback.json').read_text())
            self.assertIn('UNVERIFIED_TEXT', full['llm_unverified']['rationale'])
            # Re-import corrected assessment updates the same record instead of adding a success.
            assessment = json.loads((evaluation / 'assessment.json').read_text())
            assessment['llm_assessment']['rationale'] = '证据不足'
            (evaluation / 'assessment.json').write_text(json.dumps(assessment))
            self.assertEqual(import_feedback(root, pipeline.name)['final_verdict'], 'EXTEND')
            self.assertEqual(len(store.list('baseline-111111111111', 'v1')), 1)
            (root / 'ab-444444444444/events.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, '摘要'):
                import_feedback(root, pipeline.name)

    def test_feedback_tool_result_enters_next_model_messages(self):
        example = Path(__file__).resolve().parents[1] / 'examples/brainstorm'
        context = json.loads((example / 'context.json').read_text())
        batch = json.loads((example / 'proposals.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {'feedback_id': 'feedback-123456789abc', 'source_run': 'candidate-evaluation-123456789abc',
                      'parent_run_id': context['parent_run_id'], 'dataset_version': context['dataset_version'],
                      'mechanism_key': 'old_capacity', 'proposal_id': 'old', 'final_verdict': 'REVIEW_REQUIRED',
                      'conclusions': ['证据需要复查'], 'llm_unverified': {'rationale': 'BAD_RAW_TEXT'}}
            FeedbackStore(root).save(record)
            actual = BrainstormTools(Path.cwd(), root, context)
            class Tools:
                SPECS = BrainstormTools.SPECS
                def execute(self, name, arguments):
                    if name in ('list_feedback', 'read_feedback'):
                        return actual.execute(name, arguments)
                    if name == 'read_file':return 'system_kb', {'path': arguments['path']}
                    if name == 'read_experiment':return 'experiment_kb', {'run_id': context['parent_run_id']}
                    return 'data_analysis', {'users': 2}
            context['evidence'].append({'evidence_id': 'experiment-memory', 'source': 'experiment_kb', 'status': 'observed', 'facts': {'count': 1}})
            path = root / 'context.json'
            path.write_text(json.dumps(context))
            batch['proposals'][0]['evidence_refs'].append({'evidence_id': 'tool-5', 'field': 'final_verdict', 'value': 'REVIEW_REQUIRED'})
            responses = [{'action': 'tool', 'name': name, 'arguments': args} for name, args in (
                ('read_file', {'path': 'src/mini_agentx/recommender/model.py'}),
                ('read_experiment', {'run_id': context['parent_run_id']}), ('training_data_summary', {}),
                ('list_feedback', {}), ('read_feedback', {'feedback_id': record['feedback_id']}))]
            responses.append({'action': 'propose', 'batch': batch})
            class CapturingClient(FakeClient):
                def complete(self, messages):
                    self.last_messages = deepcopy(messages)
                    return super().complete(messages)
            client = CapturingClient(responses)
            client.workflow = {'max_model_calls': 8, 'max_tool_calls': 6}
            result = generate_proposals(path, root, client, Tools())
            self.assertEqual(result['handoff_status'], 'ready')
            text = json.dumps(client.last_messages, ensure_ascii=False)
            self.assertIn('REVIEW_REQUIRED', text)
            self.assertNotIn('BAD_RAW_TEXT', text)
