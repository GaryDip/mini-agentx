"""三个agent的有界多轮编排；动态基线比较，可信KEEP后晋升。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import re
import uuid

from mini_agentx.agents.activity_log import ActivityLog
from mini_agentx.agents.brainstorm_context import build_context
from mini_agentx.agents.brainstorm import generate_proposals
from mini_agentx.agents.developing import develop
from mini_agentx.tools.brainstorm import BrainstormTools
from mini_agentx.tools.developing import DevelopingTools
from mini_agentx.workflow.candidate_evaluation import run_candidate_evaluation
from mini_agentx.llm.client import LLMError
from mini_agentx.workflow.models import archive_model, promote


class BudgetClient:
    def __init__(self, client, state, save, activity, stage, directory):
        self.client, self.state, self.save = client, state, save
        self.activity, self.stage, self.directory = activity, stage, directory
        self.config = client.config
        self.workflow = getattr(client, 'workflow', {})
        self.calls = 0

    def complete(self, messages):
        if self.state['model_calls'] >= self.state['max_model_calls']:
            raise LLMError('工作流总模型调用预算耗尽')
        self.calls += 1
        self.state['model_calls'] += 1
        self.save()
        request = self.directory / f"round-{self.activity.round_number:02}-{self.stage}-request-{self.calls:02}.json"
        request.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
        self.activity.record('发送LLM请求', agent=self.stage,
                             result={'message_count': len(messages), 'request_file': str(request),
                                     'total_model_calls': self.state['model_calls'], 'max_model_calls': self.state['max_model_calls']})
        output, metadata = self.client.complete(messages)
        self.activity.record('LLM调用用量', agent=self.stage, result=metadata)
        return output, metadata


def iterate(root, runs, baseline_run, rounds, clients, task='configs/brainstorm-task.json',
            ab_config='configs/abtest.toml', max_model_calls=None):
    if type(rounds) is not int or not 1 <= rounds <= 5:
        raise ValueError('轮数必须在1..5之间')
    if not isinstance(baseline_run, str) or not re.fullmatch(r'baseline-[a-f0-9]{12}', baseline_run):
        raise ValueError('父模型run ID无效')
    limit = max_model_calls if max_model_calls is not None else rounds * 24
    if type(limit) is not int or not 1 <= limit <= 120:
        raise ValueError('总模型调用预算必须在1..120之间')
    root, runs = Path(root).resolve(), Path(runs).resolve()
    parent = runs / baseline_run
    if parent.is_symlink() or not (parent / 'report.json').is_file():
        raise ValueError('父实验不存在或为符号链接')
    started_at = datetime.now(ZoneInfo('Asia/Singapore'))
    directory = runs / ('workflow-' + started_at.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    activity = ActivityLog(directory, '完整工作流')
    state = {'run_id': directory.name, 'directory': str(directory), 'status': 'running',
             'created_at': started_at.isoformat(), 'timezone': 'Asia/Singapore',
             'baseline_run': baseline_run, 'requested_rounds': rounds, 'rounds': [],
             'model_calls': 0, 'max_model_calls': limit, 'promoted': False,
             'activity_log': str(directory / 'activity.md'), 'current_baseline_run': baseline_run,
             'baseline_versions': [], 'promotions': []}
    def save():
        path = directory / 'status.json'
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(path)
    initial_archive = directory / 'models/baselines/v00'
    archive_model(root, runs, parent, initial_archive)
    state['baseline_versions'].append({'version': 0, 'run_id': baseline_run, 'archive': str(initial_archive)})
    save()
    activity.record('启动工作流', reason='逐轮提出、实现、评估和保存反馈；每轮与当前基线比较，只有可信KEEP采纳升级', result=state.copy())
    for index in range(1, rounds + 1):
        activity.round_number = index
        baseline_run = state['current_baseline_run']
        parent = runs / baseline_run
        round_state = {'round': index, 'status': 'running', 'stage': 'context', 'baseline_run': baseline_run, 'adopted': False}
        state['rounds'].append(round_state)
        save()
        activity.record('开始本轮', result={'parent': baseline_run, 'round': index})
        try:
            context_path = build_context(parent, task, runs)['context_path']
            round_state['context_path'] = context_path
            context = json.loads(Path(context_path).read_text())
            activity.record('准备任务与证据', result={'task': context['task'], 'evidence': context['evidence'], 'context_file': context_path})
            round_state['stage'] = 'brainstorm'
            brainstorm_directory = runs / ('brainstorm-' + uuid.uuid4().hex[:12])
            round_state['brainstorm_directory'] = str(brainstorm_directory)
            save()
            brainstorm = generate_proposals(context_path, runs,
                BudgetClient(clients['brainstorm'], state, save, activity, 'Brainstorm', directory),
                BrainstormTools(root, runs, context), activity=activity, destination=brainstorm_directory)
            round_state['brainstorm_status'] = brainstorm['handoff_status']
            if brainstorm['handoff_status'] != 'ready':
                round_state['status'] = state['status'] = 'needs_evidence'
                activity.record('本轮没有可开发方案，停止后续阶段', result=brainstorm)
                save()
                break
            round_state['stage'] = 'developing'
            developing_directory = runs / ('developing-' + uuid.uuid4().hex[:12])
            round_state['developing_directory'] = str(developing_directory)
            save()
            tools = DevelopingTools(root, runs, brainstorm_directory / 'proposal.json', directory=developing_directory)
            developed = develop(tools, BudgetClient(clients['developing'], state, save, activity, 'Developing', directory), activity=activity)
            candidate_archive = directory / f'models/candidates/round-{index:02}'
            handoff = json.loads((developing_directory / 'handoff.json').read_text())
            archive_model(root, runs, Path(handoff['training_run']), candidate_archive)
            round_state['candidate_archive'] = str(candidate_archive)
            round_state['stage'] = 'candidate_evaluation'
            evaluation_directory = runs / ('candidate-evaluation-' + uuid.uuid4().hex[:12])
            round_state['candidate_evaluation_directory'] = str(evaluation_directory)
            save()
            evaluation = run_candidate_evaluation(developing_directory, runs, ab_config,
                BudgetClient(clients['evaluation'], state, save, activity, 'Evaluation', directory),
                activity=activity, directory=evaluation_directory)
            round_state['ab_run_id'] = evaluation['ab_run_id']
            round_state['final'] = evaluation['assessment']['final']
            if not evaluation.get('feedback_id'):
                raise ValueError('实验已评估，但反馈保存失败；停止避免下一轮丢失经验：' + evaluation.get('feedback_error', '缺少反馈ID'))
            round_state['feedback_id'] = evaluation['feedback_id']
            decision = {'round': index, 'baseline_run': baseline_run, 'adopted': False,
                        'final': round_state['final'], 'evaluation_directory': str(evaluation_directory)}
            if round_state['final']['verdict'] == 'KEEP':
                assessment = evaluation['assessment']
                if not (assessment['validation']['citations_verified'] is True and assessment['validation']['agrees_with_rule'] is True
                        and not assessment['validation']['interpretation_warnings'] and assessment['rule_decision']['verdict'] == 'KEEP'):
                    raise ValueError('KEEP缺少可信评估校验，拒绝升级')
                upgraded = promote(root, runs, candidate_archive, baseline_run, evaluation_directory.name)
                version = len(state['baseline_versions'])
                baseline_archive = directory / f'models/baselines/v{version:02}'
                archive_model(root, runs, upgraded, baseline_archive)
                decision.update(adopted=True, new_baseline_run=upgraded.name, baseline_archive=str(baseline_archive))
                state['baseline_versions'].append({'version': version, 'run_id': upgraded.name, 'archive': str(baseline_archive)})
                state['promotions'].append(decision.copy())
                state['current_baseline_run'] = upgraded.name
                state['promoted'] = round_state['adopted'] = True
                round_state['new_baseline_run'] = upgraded.name
            (candidate_archive / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))
            activity.record('采纳候选并升级基线' if decision['adopted'] else '保留当前基线', result=decision)
            round_state['status'] = 'completed'
            round_state['stage'] = 'done'
            activity.record('本轮完成：修改、观测与反馈', result={'ndcg_delta': developed['result']['ndcg_delta'],
                             'final': round_state['final'], 'feedback_id': round_state['feedback_id'],
                             'next': '下一轮查询基线链历史反馈，并使用当前已采纳基线'})
            save()
        except (OSError, ValueError, KeyError, TypeError) as error:
            round_state['status'] = state['status'] = 'failed'
            round_state['error'] = str(error)
            activity.record('本轮失败，停止工作流', result={'stage': round_state['stage'], 'error': str(error),
                                                         'artifacts': round_state.copy()})
            save()
            break
    if state['status'] == 'running':
        state['status'] = 'completed'
    activity.round_number = None
    activity.record('工作流结束', result={'status': state['status'], 'model_calls': state['model_calls'], 'rounds': state['rounds']})
    with (directory / 'activity.md').open('a') as file:
        file.write("\n## 按轮查看结果\n\n| 轮次 | 状态 | 最终判断 | 当轮基线 | 是否采纳 | 反馈 |\n| --- | --- | --- | --- | --- | --- |\n")
        for item in state['rounds']:
            file.write(f"| {item['round']} | {item['status']} | {item.get('final', {}).get('verdict', '未评估')} | {item['baseline_run']} | {item['adopted']} | {item.get('feedback_id', '未产生')} |\n")
        file.write("\n各阶段原始文件：\n\n")
        for item in state['rounds']:
            for stage in ('brainstorm_directory', 'developing_directory', 'candidate_evaluation_directory'):
                if item.get(stage):
                    file.write(f"- 第{item['round']}轮 {stage}: {item[stage]}\n")
    save()
    (directory / 'report.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    return state
