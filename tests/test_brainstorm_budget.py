from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mini_agentx.agents.brainstorm import generate_proposals
from mini_agentx.tools.brainstorm import BrainstormTools
from mini_agentx.llm.client import LLMError
from test_brainstorm_agent import FakeClient


class BudgetTest(unittest.TestCase):
    def test_api_failure_still_reserves_proposal_and_repair(self):
        example = Path(__file__).resolve().parents[1] / 'examples/brainstorm'
        parent = json.loads((example / 'context.json').read_text())['parent_run_id']
        batch = json.loads((example / 'proposals.json').read_text())
        batch['proposals'][0]['evidence_refs'].append({'evidence_id': 'tool-3', 'field': 'users', 'value': 2})
        bad = deepcopy(batch)
        bad['proposals'][0]['changes'][0]['file'] = '../evaluate.py'
        class Tools:
            SPECS = BrainstormTools.SPECS
            def execute(self, name, args):
                if name == 'read_file': return 'system_kb', {'path': args['path']}
                if name == 'read_experiment': return 'experiment_kb', {'run_id': parent}
                return 'data_analysis', {'users': 2}
        client = FakeClient([LLMError('网络失败'),
            {'action': 'tool', 'name': 'read_file', 'arguments': {'path': 'src/mini_agentx/recommender/model.py'}},
            {'action': 'tool', 'name': 'read_experiment', 'arguments': {'run_id': parent}},
            {'action': 'tool', 'name': 'training_data_summary', 'arguments': {}},
            {'action': 'propose', 'batch': bad}, {'action': 'propose', 'batch': batch}])
        client.workflow = {'max_model_calls': 6, 'max_tool_calls': 8}
        with tempfile.TemporaryDirectory() as folder:
            result = generate_proposals(example / 'context.json', folder, client, Tools())
            self.assertEqual(result['handoff_status'], 'ready')
            self.assertEqual(result['tool_calls'], 3)
            requests = Path(result['directory'])
            for number, remaining in ((5, 2), (6, 1)):
                messages = json.loads((requests / f'request-{number:02}.json').read_text())
                budget = json.loads(messages[-1]['content'])['runtime_budget']
                self.assertFalse(budget['tool_allowed'])
                self.assertEqual(budget['model_calls_remaining'], remaining)
                self.assertEqual(budget['tool_calls_remaining'], 5)
