import unittest
from mini_agentx.agents.activity_view import render_activity


class ActivityViewTest(unittest.TestCase):
    def test_hides_transport_preserves_actions_results_and_repair(self):
        def event(action, result=None, reason=None):
            return {'agent': 'Developing', 'round': 1, 'action': action, 'result': result, 'reason': reason}
        events = [event('发送LLM请求', {'request_file': 'secret-noise'}), event('LLM调用用量', {'total_tokens': 123}),
                  event('调用模型：第 1 次'), event('请求工具：check_candidate', {}, '检查新损失接口'),
                  event('工具结果：check_candidate', {'valid': False, 'error': '输出形状错误'}),
                  event('工具失败', {'error': '输出形状错误'}),
                  event('请求工具：write_candidate_file', {'path': 'model.py', 'content': 'full code'}, '修正 shape'),
                  event('工具结果：write_candidate_file', {'written': 'model.py'})]
        text = render_activity(events)
        self.assertNotIn('发送LLM请求', text)
        self.assertNotIn('total_tokens', text)
        self.assertNotIn('secret-noise', text)
        self.assertNotIn('full code', text)
        for expected in ('Developing', 'check_candidate', '输出形状错误', '修正 shape', 'model.py', '已写入文件'):
            self.assertIn(expected, text)
