import json
import unittest
from unittest.mock import patch

from mini_agentx.llm.client import DeepSeekClient, LLMError
from mini_agentx.llm.function_calling import FunctionCallingClient


class NativeStub:
    config = {'max_attempts': 2}
    workflow = {}
    def __init__(self): self.requests = []
    def complete_tools(self, messages, tools, tool_choice='auto'):
        self.requests.append((messages, tools, tool_choice))
        name = 'read' if len(self.requests) == 1 else 'finish'
        args = {'path': 'model.py', 'reason': '查看模型'} if name == 'read' else {'reason': '完成'}
        return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call-{len(self.requests)}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, {}


class CallingTest(unittest.TestCase):
    def test_tool_result_role_id_and_budget_forces_finish(self):
        stub = NativeStub()
        client = FunctionCallingClient(stub, {'read': {'arguments': {'path': '路径'}, 'description': '读取'}}, 'finish')
        messages = [{'role': 'system', 'content': '使用工具'}]
        action, _ = client.complete(messages)
        messages.extend([{'role': 'assistant', 'content': json.dumps(action, ensure_ascii=False)},
                         {'role': 'user', 'content': json.dumps({'tool_result': {'content': 'model code'}})},
                         {'role': 'user', 'content': json.dumps({'runtime_budget': {'tool_allowed': False}})}])
        client.complete(messages)
        actual, definitions, choice = stub.requests[-1]
        self.assertEqual(actual[1]['tool_calls'][0]['id'], 'call-1')
        self.assertEqual(actual[2]['role'], 'tool')
        self.assertEqual(actual[2]['tool_call_id'], 'call-1')
        self.assertEqual(choice['function']['name'], 'finish')
        self.assertEqual(definitions[0]['function']['parameters']['additionalProperties'], False)

    def test_native_http_payload_and_null_content_tool_calls(self):
        client = object.__new__(DeepSeekClient)
        client.model, client._key = 'mock', 'fake-test-key'
        client.config = {'base_url': 'https://api.deepseek.com', 'max_tokens': 100, 'timeout_seconds': 1}
        body = {'choices': [{'finish_reason': 'tool_calls', 'message': {'content': None, 'tool_calls': [{'id': 'id1', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{"reason":"完成"}'}}]}}]}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(body).encode()
        with patch('mini_agentx.llm.client.urlopen', return_value=Response()) as http:
            response, _ = client.complete_tools([{'role': 'user', 'content': 'task'}], [{'type': 'function'}])
            payload = json.loads(http.call_args.args[0].data)
            self.assertIn('tools', payload)
            self.assertNotIn('response_format', payload)
            self.assertIsNone(response['content'])
            body['choices'][0]['finish_reason'] = 'length'
            with self.assertRaises(LLMError): client.complete_tools([], [])

    def test_extra_tool_arguments_not_executed(self):
        stub = NativeStub()
        original = stub.complete_tools
        def malformed(*args, **kwargs):
            response, metadata = original(*args, **kwargs)
            response['tool_calls'][0]['function']['arguments'] = '{"path":"x","reason":"test","extra":"bad"}'
            return response, metadata
        stub.complete_tools = malformed
        client = FunctionCallingClient(stub, {'read': {'arguments': {'path': '路径'}, 'description': '读取'}}, 'finish')
        with self.assertRaisesRegex(LLMError, '额外字段'): client.complete([])

    def test_brainstorm_native_loop_repairs_terminal_call_with_tool_result(self):
        from copy import deepcopy
        from pathlib import Path
        import tempfile
        from mini_agentx.agents.brainstorm import generate_proposals
        from mini_agentx.tools.brainstorm import BrainstormTools
        example = Path(__file__).resolve().parents[1] / 'examples/brainstorm'
        context = json.loads((example / 'context.json').read_text())
        batch = json.loads((example / 'proposals.json').read_text())
        batch['proposals'][0]['evidence_refs'].append({'evidence_id': 'tool-3', 'field': 'users', 'value': 2})
        bad = deepcopy(batch)
        bad['proposals'][0]['changes'][0]['file'] = '../evaluate.py'
        class Tools:
            SPECS = BrainstormTools.SPECS
            def execute(self, name, args):
                if name == 'read_file': return 'system_kb', {'path': args['path']}
                if name == 'read_experiment': return 'experiment_kb', {'run_id': context['parent_run_id']}
                return 'data_analysis', {'users': 2}
        actions = iter([('read_file', {'path': 'src/mini_agentx/recommender/model.py'}),
                        ('read_experiment', {'run_id': context['parent_run_id']}),
                        ('training_data_summary', {}), ('propose', {'batch': bad}), ('propose', {'batch': batch})])
        requests = []
        class Client:
            config = {'max_attempts': 2}
            workflow = {'max_model_calls': 5, 'max_tool_calls': 3}
            def complete_tools(self, messages, tools, tool_choice='auto'):
                requests.append(deepcopy(messages))
                name, args = next(actions)
                args['reason'] = '测试调查或提交'
                return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call{len(requests)}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, {}
        with tempfile.TemporaryDirectory() as folder:
            result = generate_proposals(example / 'context.json', folder, Client(), Tools())
            self.assertEqual(result['handoff_status'], 'ready')
            messages = requests[-1]
            feedback = next(m for m in messages if m.get('tool_call_id') == 'call4')
            self.assertEqual(feedback['role'], 'tool')
            self.assertIn('校验失败', feedback['content'])
            actual = json.loads((Path(result['directory']) / 'request-05.json').read_text())
            self.assertIn('tools', actual)
            self.assertEqual(actual['tool_choice']['function']['name'], 'propose')
