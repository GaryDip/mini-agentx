"""开发交接 → 模型模式模拟A/B → LLM评估；不自动晋升。"""
import hashlib
import json
from pathlib import Path
import re
import uuid

from mini_agentx.abtest.simulator import load_ab_config, simulate
from mini_agentx.agents.activity_log import ActivityLog
from mini_agentx.agents.evaluation import evaluate_experiment


def run_candidate_evaluation(development, runs, ab_config, client=None):
    runs = Path(runs).resolve()
    development = Path(development).resolve()
    if not development.is_relative_to(runs):
        raise ValueError('开发目录必须在 runs 下')
    handoff = json.loads((development / 'handoff.json').read_text())
    if handoff.get('status') != 'developed':
        raise ValueError('只评估完成开发的候选')
    if not re.fullmatch(r'baseline-[a-f0-9]{12}', handoff['parent_run_id']):
        raise ValueError('父实验ID无效')
    candidate = Path(handoff['training_run']).resolve()
    if not candidate.is_relative_to(development / 'training'):
        raise ValueError('候选训练目录越界')
    for name in ('model.py', 'training.toml'):
        path = development / 'candidate' / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != handoff['files'][name]:
            raise ValueError('候选文件与开发交接不一致')
    report = json.loads((candidate / 'report.json').read_text())
    if report['model_source_sha256'] != handoff['files']['model.py']:
        raise ValueError('实际训练模型与开发源码不同')
    if report['dataset_version'] != handoff['result']['dataset_version']:
        raise ValueError('训练数据版本与交接不一致')
    directory = runs / ('candidate-evaluation-' + uuid.uuid4().hex[:12])
    directory.mkdir()
    activity = ActivityLog(directory, 'Candidate Evaluation')
    state = {'status': 'running', 'development_directory': str(development),
             'parent_run_id': handoff['parent_run_id'], 'candidate_training_run': str(candidate),
             'proposal_id': handoff['proposal_id'], 'simulated': True, 'promoted': False}
    def save():
        (directory / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    save()
    activity.record('核对开发交接完成', result=state.copy())
    try:
        activity.record('运行模型模式A/B', result={'a': handoff['parent_run_id'], 'b': report['run_id']})
        ab = simulate(load_ab_config(ab_config), 'null', runs, runs / handoff['parent_run_id'], candidate)
        state['ab_run_id'] = ab['experiment_id']
        state['status'] = 'ab_completed'
        save()
        activity.record('A/B观测完成', result={'primary': ab['analysis']['primary'], 'simulated': True})
        if client is not None:
            activity.record('调用Evaluation Agent', reason='分析观测；移除规则答案，不读取私有场景参数')
            assessment = evaluate_experiment(runs / ab['experiment_id'], client)
            state['assessment'] = assessment
            state['status'] = 'evaluated'
            activity.record('Evaluation完成', result=assessment['final'])
        else:
            activity.record('A/B已准备，尚未调用LLM', result={'ab_run_id': ab['experiment_id']})
    except (OSError, ValueError, KeyError, TypeError) as error:
        state['status'] = 'failed'
        state['error'] = str(error)
        activity.record('阶段失败', result={'error': str(error)})
        save()
        raise ValueError(f'候选评估失败，见 {directory}/activity.md：{error}') from None
    state['directory'] = str(directory)
    state['activity_log'] = str(directory / 'activity.md')
    save()
    (directory / 'handoff.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    return state
