"""从原始事件生成 agent 行动阅读版，不增加模型调用。"""
import json
from pathlib import Path


HIDDEN = ('发送LLM请求', 'LLM调用用量', '调用模型：', '准备任务与证据', '准备评估证据',
          '启动工作流', '工作流结束', '核对开发交接完成', '调用Evaluation Agent',
          'Evaluation完成', '本轮完成：修改、观测与反馈', '开始任务', '开始开发')


def brief(value):
    if isinstance(value, float):
        return f'{value:.6g}'
    text = str(value)
    return text[:1000] + '…（完整内容见 activity.jsonl）' if len(text) > 1000 else text


def facts_summary(value, tool=None):
    if not isinstance(value, dict):
        return brief(value) if value is not None else ''
    facts = value.get('facts', value)
    lines = []
    if 'content' in facts:
        text = facts['content']
        lines.append(f"读取 `{facts.get('path', '文件')}`（{len(text.splitlines())} 行）。")
        if facts.get('path', '').endswith('.toml'):
            lines.append('```toml\n' + text + '\n```')
    if 'feedback' in facts:
        lines.append(f"找到 {facts.get('count', len(facts['feedback']))} 条历史反馈。")
        for row in facts['feedback']:
            lines.append(f"- `{row.get('feedback_id')}`：{row.get('mechanism_key')}，结论 {row.get('final_verdict')}。")
    if 'proposal' in facts and 'observed' in facts:
        lines.append(f"历史方案：{facts['proposal'].get('title')}；结论：{facts.get('final_verdict')}。")
        observed = facts['observed']
        for key in ('parent_ndcg', 'candidate_ndcg', 'ndcg_delta', 'ab_effect'):
            if key in observed:
                lines.append(f"{key}：{brief(observed[key])}。")
        lines.extend(facts.get('conclusions', []))
    metrics = facts.get('validation_metrics')
    if metrics:
        lines.append('验证结果：' + '；'.join(f'{key}={brief(metrics[key])}' for key in ('ndcg@10', 'recall@10') if key in metrics) + '。')
    for key, label in (('ndcg_delta', '相对基线 NDCG 差值'), ('valid', '检查通过'), ('error', '错误'), ('tool_error', '工具错误'), ('errors', '累计错误次数'),
                       ('interactions', '训练交互数'), ('users', '用户数'), ('items', '物品数'),
                       ('written', '已写入文件'), ('status', '状态')):
        if key in facts:
            lines.append(f'{label}：{brief(facts[key])}。')
    primary = facts.get('primary')
    if primary:
        lines.append(f"模拟 CTR 差值：{brief(primary.get('absolute_effect_b_minus_a'))}；95% 区间：{primary.get('ci95')}。")
    if tool == 'get_candidate_diff':
        for name, diff in facts.items():
            if isinstance(diff, str) and diff:
                lines.append(f'修改 `{name}`：\n```diff\n{diff}\n```')
    if 'experiments' in facts:
        lines.append(f"找到 {len(facts['experiments'])} 个实验。")
    if 'entries' in facts:
        lines.extend(f"- `{entry['path']}`（{entry['type']}）" for entry in facts['entries'])
    if 'matches' in facts:
        lines.append(f"搜索 `{facts.get('query')}`：{len(facts['matches'])} 处匹配。")
        lines.extend(f"- `{match['path']}:{match['line']}`：{match['text']}" for match in facts['matches'])
    if facts.get('truncated'):
        lines.append('结果达到上限，已截断。')
    if 'paths' in facts:
        lines.append('可读文件：' + '、'.join(facts['paths']) + '。')
    if 'training_config' in facts:
        lines.append('训练配置：' + '；'.join(f'{k}={v}' for k, v in facts['training_config'].items()) + '。')
    return '\n\n'.join(lines) or '工具执行完成；完整返回保存在 activity.jsonl。'


def render_activity(events, title='Agent'):
    lines = [f'# {title} 行动日志', '', '按轮次查看 agent 的工具调用、返回、修正和决定。模型的说明是公开理由，效果以实际观测为准。', '']
    group = None
    step = 0
    for event in events:
        action, actor = event['action'], event.get('agent', title)
        if action.startswith(HIDDEN):
            continue
        current = (event.get('round'), actor)
        if current != group:
            if current[0] != (group[0] if group else None):
                lines.extend([f'## 第 {current[0]} 轮', ''])
            lines.extend([f'### {actor}', ''])
            group = current
        result = event.get('result') or {}
        reason = event.get('reason')
        if action.startswith('工具结果：'):
            tool = action.split('：', 1)[1]
            lines.extend(['**返回：**', '', facts_summary(result, tool), ''])
            continue
        step += 1
        label = action.replace('请求工具：', '调用工具：')
        lines.extend([f'**{step}. {label}**', ''])
        if reason:
            lines.extend([reason, ''])
        if action.startswith('请求工具：'):
            arguments = result.get('arguments', result)
            params = {key: value for key, value in arguments.items() if key != 'content'}
            if params:
                lines.extend(['参数：' + '；'.join(f'`{key}={value}`' for key, value in params.items()), ''])
            if 'content' in arguments:
                lines.extend([f"写入 `{arguments.get('path')}`（{len(arguments['content'].splitlines())} 行）；具体修改见后续差异。", ''])
        elif action == '开始本轮':
            lines.extend([f"本轮从基线 `{result.get('parent')}` 开始。", ''])
        elif action == '模型提交候选批次':
            for proposal in result.get('proposals', []):
                lines.extend([f"- **{proposal['title']}**（{proposal['maturity']}，优先级 {proposal['priority']}）：{proposal['hypothesis']}"])
            lines.append('')
        elif action == '校验通过并完成交接':
            selected = result.get('selected_proposal')
            lines.extend([f"选中：**{selected['title']}**。" if selected else f"交接状态：{result.get('handoff_status')}。", ''])
        elif action == '模型返回评估':
            lines.extend([f"Agent 判断：**{result.get('verdict')}**。", ''])
        elif action == '程序复核后的最终判断':
            lines.extend([f"最终判断：**{result['final']['verdict']}**；原因：{result['final'].get('reason')}。", ''])
            validation = result.get('validation', {})
            if validation.get('agrees_with_rule') is False:
                lines.extend(['Agent 的判断与预定规则不一致，不能据此采纳。', ''])
            lines.extend([*validation.get('interpretation_warnings', []), ''])
        elif action in ('保留当前基线', '采纳候选并升级基线'):
            lines.extend([f"当轮基线：`{result.get('baseline_run')}`；采纳：{result.get('adopted')}。", ''])
            if result.get('new_baseline_run'):
                lines.extend([f"下一轮使用：`{result['new_baseline_run']}`。", ''])
        elif action == '实验反馈已保存':
            lines.extend([f"保存反馈 `{result.get('feedback_id')}`，供下一轮查询。", ''])
        elif action == '开发交接完成':
            # reason 已解释修改与观测，避免再复制整份报告。
            if not reason:
                lines.extend([facts_summary(result.get('result', result)), ''])
        else:
            lines.extend([facts_summary(result), ''])
    return '\n'.join(lines)


def rebuild(directory, title='完整工作流'):
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / 'activity.jsonl').read_text().splitlines() if line.strip()]
    (directory / 'activity.md').write_text(render_activity(events, title))
