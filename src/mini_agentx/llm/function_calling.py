"""原生工具调用传输；保留现有 agent 的动作校验与状态机。"""
import json
from pathlib import Path
from collections import defaultdict
from .client import LLMError


def schemas(specs, terminal):
    definitions = []
    for name, spec in specs.items():
        properties = {key: {'type': 'string', 'description': description} for key, description in spec['arguments'].items()}
        if name in ('read_candidate_file', 'write_candidate_file'):
            properties['path']['enum'] = ['model.py', 'training.toml']
        if name == 'read_file':
            from mini_agentx.tools.brainstorm import BrainstormTools
            properties['path']['enum'] = list(BrainstormTools.FILES)
        properties['reason'] = {'type': 'string', 'description': '简短公开行动说明'}
        definitions.append({'type': 'function', 'function': {'name': name, 'description': spec['description'],
            'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}})
    properties = {'reason': {'type': 'string'}}
    if terminal == 'propose':
        properties['batch'] = {'type': 'object', 'description': '符合任务和提案契约的完整候选批次，程序另行校验'}
    definitions.append({'type': 'function', 'function': {'name': terminal,
        'description': '提交提案，接受程序校验' if terminal == 'propose' else '完成开发交接，接受程序校验',
        'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}})
    return definitions


class FunctionCallingClient:
    def __init__(self, client, specs, terminal):
        self.client, self.terminal = client, terminal
        self.config, self.workflow = client.config, getattr(client, 'workflow', {})
        self.definitions = schemas(specs, terminal)
        self.responses = defaultdict(list)
        self.allowed = set(specs) | {terminal}
        self.request_directory = None
        self.calls = 0

    def native_messages(self, messages):
        result, used = [], defaultdict(int)
        pending = None
        for message in messages:
            if message['role'] == 'assistant':
                key = message['content']
                position = used[key]
                if position >= len(self.responses[key]):
                    raise LLMError('工具消息缺少对应的原生调用记录')
                native = self.responses[key][position]
                used[key] += 1
                result.append(native)
                pending = native['tool_calls'][0]['id']
            elif pending is not None:
                result.append({'role': 'tool', 'tool_call_id': pending, 'content': message['content']})
                pending = None
            else:
                result.append(message)
        if pending is not None:
            raise LLMError('工具调用缺少返回，拒绝发送不完整消息链')
        return result

    def complete(self, messages):
        native = self.native_messages(messages)
        choice = 'required'
        # 最后的预算控制可强制提交工具，而非继续调查。
        try:
            control = json.loads(messages[-1]['content']).get('runtime_budget', {})
            if control.get('tool_allowed') is False:
                choice = {'type': 'function', 'function': {'name': self.terminal}}
        except (ValueError, TypeError, AttributeError, IndexError):
            pass
        self.calls += 1
        if self.request_directory is not None:
            path = Path(self.request_directory) / f'request-{self.calls:02}.json'
            path.write_text(json.dumps({'messages': native, 'tools': self.definitions, 'tool_choice': choice}, ensure_ascii=False, indent=2))
        response, metadata = self.client.complete_tools(native, self.definitions, tool_choice=choice)
        calls = response.get('tool_calls', [])
        if len(calls) != 1:
            raise LLMError('每轮要求一个原生工具调用；本次没有执行任何工具')
        call = calls[0]
        name = call['function']['name']
        if name not in self.allowed:
            raise LLMError('模型请求未注册工具；没有执行')
        try:
            arguments = json.loads(call['function']['arguments'])
            if not isinstance(arguments, dict):
                raise ValueError()
        except (ValueError, TypeError):
            raise LLMError('工具参数不是 JSON 对象；没有执行') from None
        schema = next(item['function']['parameters'] for item in self.definitions if item['function']['name'] == name)
        if set(arguments) != set(schema['required']):
            raise LLMError('工具参数缺失或包含额外字段；没有执行')
        for key, value in arguments.items():
            definition = schema['properties'][key]
            if definition['type'] == 'string' and (not isinstance(value, str) or not value.strip()):
                raise LLMError('工具参数必须为非空字符串；没有执行')
            if definition['type'] == 'object' and not isinstance(value, dict):
                raise LLMError('工具参数必须为对象；没有执行')
            if 'enum' in definition and value not in definition['enum']:
                raise LLMError('工具参数不在允许范围；没有执行')
        reason = arguments.get('reason', '')
        if name == self.terminal:
            action = {'action': name, **arguments}
        else:
            action = {'action': 'tool', 'name': name, 'arguments': {key: value for key, value in arguments.items() if key != 'reason'}, 'reason': reason}
        key = json.dumps(action, ensure_ascii=False)
        self.responses[key].append(response)
        return action, {**metadata, 'tool_call_id': call['id'], 'function_name': name, 'protocol': 'native_function_calling', 'native_response': response}


def native_agent(client, specs, terminal):
    # 离线测试桩仍支持旧 action 接口；生产客户端始终使用原生调用。
    return FunctionCallingClient(client, specs, terminal) if callable(getattr(client, 'complete_tools', None)) else client
