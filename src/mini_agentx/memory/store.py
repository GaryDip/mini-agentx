"""SQLite 保存结构化观测；LLM 原始解释单独标记，不作为事实回传。"""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from mini_agentx.agents.evaluation import interpretation_warnings


def safe_path(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError('反馈来源必须在 runs 内')
    for component in [path, *path.parents]:
        if component.resolve() == root:
            break
        if component.is_symlink():
            raise ValueError('反馈来源不允许符号链接')
    return path.resolve()


class FeedbackStore:
    def __init__(self, runs):
        self.runs = Path(runs).resolve()
        self.path = safe_path(self.runs, self.runs / 'memory.sqlite3')

    @contextmanager
    def _connect(self, write=False):
        if write:
            self.runs.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
            connection.execute('''CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY, source_run TEXT UNIQUE NOT NULL,
                parent_run TEXT NOT NULL, dataset_version TEXT NOT NULL,
                updated_at TEXT NOT NULL, payload TEXT NOT NULL)''')
        else:
            connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, record):
        with self._connect(write=True) as connection:
            connection.execute('''INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_run) DO UPDATE SET
                parent_run=excluded.parent_run, dataset_version=excluded.dataset_version,
                updated_at=excluded.updated_at, payload=excluded.payload''',
                (record['feedback_id'], record['source_run'], record['parent_run_id'], record['dataset_version'],
                 datetime.now(timezone.utc).isoformat(), json.dumps(record, ensure_ascii=False, allow_nan=False)))
        return record

    def lineage(self, parent_run, dataset_version):
        parents = []
        while parent_run not in parents:
            if not isinstance(parent_run, str) or not re.fullmatch(r'baseline-[a-f0-9]{12}', parent_run):
                break
            parents.append(parent_run)
            report_path = safe_path(self.runs, self.runs / parent_run / 'report.json')
            if not report_path.exists():
                break
            report = json.loads(report_path.read_text())
            if report.get('dataset_version') != dataset_version:
                break
            ancestor = report.get('promotion', {}).get('parent_run_id')
            if not ancestor:
                break
            parent_run = ancestor
        return parents or [parent_run]

    def list(self, parent_run, dataset_version):
        if not self.path.exists():
            return []
        parents = self.lineage(parent_run, dataset_version)
        placeholders = ','.join('?' for _ in parents)
        with self._connect() as connection:
            rows = connection.execute(f'SELECT payload FROM feedback WHERE parent_run IN ({placeholders}) AND dataset_version=? ORDER BY updated_at DESC, feedback_id LIMIT 20',
                                      (*parents, dataset_version)).fetchall()
        return [public_feedback(json.loads(row['payload'])) for row in rows]

    def read(self, identifier, parent_run, dataset_version):
        if not isinstance(identifier, str) or not re.fullmatch(r'feedback-[a-f0-9]{12}', identifier):
            raise ValueError('反馈ID无效')
        if not self.path.exists():
            raise ValueError('反馈记录不存在')
        parents = self.lineage(parent_run, dataset_version)
        placeholders = ','.join('?' for _ in parents)
        with self._connect() as connection:
            row = connection.execute(f'SELECT payload FROM feedback WHERE feedback_id=? AND parent_run IN ({placeholders}) AND dataset_version=?',
                                     (identifier, *parents, dataset_version)).fetchone()
        if row is None:
            raise ValueError('反馈不存在或不属于当前基线链及数据版本')
        return public_feedback(json.loads(row['payload']))


def public_feedback(record):
    """只回传程序选取的观测和有限结论，不传 LLM 自由文本。"""
    return {key: value for key, value in record.items() if key not in ('llm_unverified', 'source_paths')}


def import_feedback(runs, source_run):
    runs = Path(runs).resolve()
    if not isinstance(source_run, str) or not re.fullmatch(r'candidate-evaluation-[a-f0-9]{12}', source_run):
        raise ValueError('反馈来源必须是 candidate-evaluation run ID')
    pipeline = safe_path(runs, runs / source_run)
    status_path = safe_path(runs, pipeline / 'status.json')
    state = json.loads(status_path.read_text())
    if state['status'] not in ('evaluated', 'ab_completed', 'failed'):
        raise ValueError('运行尚未结束，不能导入反馈')
    development = safe_path(runs, state['development_directory'])
    proposal = json.loads(safe_path(runs, development / 'proposal.json').read_text())
    handoff = json.loads(safe_path(runs, development / 'handoff.json').read_text())
    training = safe_path(runs, state['candidate_training_run'])
    report = json.loads(safe_path(runs, training / 'report.json').read_text())
    parent = json.loads(safe_path(runs, runs / state['parent_run_id'] / 'report.json').read_text())
    if not (state['parent_run_id'] == handoff['parent_run_id'] == proposal['parent_run_id'] == parent['run_id']):
        raise ValueError('父实验交接不一致')
    if report['dataset_version'] != parent['dataset_version'] or report['dataset_version'] != proposal['dataset_version']:
        raise ValueError('反馈数据版本不一致')
    for name, expected in handoff['files'].items():
        if name not in ('model.py', 'training.toml'):
            raise ValueError('候选文件字段无效')
        file = safe_path(runs, development / 'candidate' / name)
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('候选代码在交接后改变')
    checkpoint = safe_path(runs, training / 'model.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('训练权重摘要不匹配')
    if report['model_source_sha256'] != handoff['files']['model.py']:
        raise ValueError('训练源码与开发交接不一致')
    metric = report['validation_metrics']['ndcg@10']
    parent_metric = parent['validation_metrics']['ndcg@10']
    observed = {'parent_ndcg': parent_metric, 'candidate_ndcg': metric, 'ndcg_delta': metric - parent_metric,
                'training_seed': report['config']['training'].get('seed', report['config'].get('seed')),
                'model_source_sha256': report['model_source_sha256'],
                'training_strategy': report.get('training_strategy', {'loss': 'bpr', 'negative_sampling': 'uniform'}),
                'parent_recall': parent['validation_metrics']['recall@10'],
                'candidate_recall': report['validation_metrics']['recall@10'],
                'training_seconds': report['training_seconds'], 'training_config': report['config']['training']}
    ab = assessment = None
    sources = {'status': str(status_path), 'development': str(development), 'training': str(training)}
    if state.get('ab_run_id'):
        ab_path = safe_path(runs, runs / state['ab_run_id'] / 'report.json')
        ab = json.loads(ab_path.read_text())
        events = safe_path(runs, ab_path.parent / 'events.csv')
        if ab.get('mode') != 'models' or ab.get('simulated') is not True:
            raise ValueError('只导入模型模式模拟A/B')
        if hashlib.sha256(events.read_bytes()).hexdigest() != ab['events_sha256']:
            raise ValueError('A/B事件摘要不匹配')
        if ab['models']['a_run_id'] != parent['run_id'] or ab['models']['b_run_id'] != report['run_id']:
            raise ValueError('A/B模型不属于本次实验')
        observed.update(ab_effect=ab['analysis']['primary']['absolute_effect_b_minus_a'],
                        ab_ci95=ab['analysis']['primary']['ci95'], checks=ab['analysis']['checks'])
        sources['ab'] = str(ab_path.parent)
    final = 'NOT_EVALUATED'
    warnings = []
    if state.get('assessment'):
        # Read original assessment artifact, not only its copy in the pipeline.
        assessment_path = safe_path(runs, Path(state['assessment']['evaluation_directory']) / 'assessment.json')
        assessment = json.loads(assessment_path.read_text())
        if ab is None or assessment['experiment_id'] != ab['experiment_id']:
            raise ValueError('评估不属于本次A/B')
        from mini_agentx.agents.evaluation import validate_output
        from mini_agentx.abtest.simulator import evaluation_evidence
        evidence = evaluation_evidence(ab)
        validate_output(assessment['llm_assessment'], evidence)
        warnings = interpretation_warnings(assessment['llm_assessment'], evidence)
        verdict = assessment['llm_assessment']['verdict']
        rule = ab['analysis']['rule_decision']['verdict']
        final = 'REVIEW_REQUIRED' if warnings or verdict != rule or assessment['final']['verdict'] == 'REVIEW_REQUIRED' else verdict
        sources['assessment'] = str(assessment_path)
    if state['status'] == 'failed':
        final = 'FAILED'
    conclusions, suggestions = feedback_conclusions(observed, final)
    selected = proposal['proposal']
    record = {'schema_version': 1, 'feedback_id': 'feedback-' + source_run.rsplit('-', 1)[1],
              'source_run': source_run, 'parent_run_id': parent['run_id'], 'dataset_version': report['dataset_version'],
              'proposal_id': selected['proposal_id'], 'mechanism_key': selected['mechanism_key'],
              'proposal': {key: selected[key] for key in ('title', 'hypothesis', 'changes', 'validation')},
              'candidate_run_id': report['run_id'], 'files': handoff['files'],
              'actual_diff': handoff.get('diff', {}),
              'observed': observed, 'final_verdict': final, 'simulated': True,
              'conclusions': conclusions, 'next_steps': suggestions,
              'review_warnings': warnings, 'stage_status': state['status'],
              'failure': state.get('error') if state['status'] == 'failed' else None,
              'llm_unverified': assessment['llm_assessment'] if assessment else None,
              'source_paths': sources}
    FeedbackStore(runs).save(record)
    (pipeline / 'feedback.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return public_feedback(record)


def feedback_conclusions(observed, verdict):
    conclusions = ['单种子离线验证提高。' if observed['ndcg_delta'] > 0 else '本次单种子离线验证没有提高。',
                   '离线观测不证明具体误差原因；模拟反馈不等价于真实业务收益。']
    suggestions = ['对照同一机制与配置的历史尝试，避免将相同实验当作新机制。']
    if verdict == 'REVIEW_REQUIRED':
        conclusions.append('评估需要复查，不能将LLM解释作为已证实经验。')
        suggestions.append('先核对解释与观测，不根据未经确认的诊断改变模型。')
    elif verdict == 'EXTEND':
        conclusions.append('当前模拟观测未达到保留条件。')
        suggestions.append('先调查证据不足的原因，再预先规划新的验证窗口。')
    elif verdict == 'DISCARD':
        conclusions.append('按当前模拟政策不保留候选；不自动永久禁止该机制。')
        suggestions.append('查看观测与护栏，区分效果问题和实验条件问题。')
    elif verdict == 'KEEP':
        conclusions.append('通过当前模拟判断条件，不代表真实线上有效或已晋升。')
        suggestions.append('复核多种子离线表现及模拟假设，保留结论的适用范围。')
    elif verdict == 'FAILED':
        conclusions.append('流程失败不能记为模型无效。')
        suggestions.append('先修复失败阶段，已有A/B时复用其观测。')
    else:
        conclusions.append('尚未完成LLM评估，不能给出保留判断。')
        suggestions.append('评估已保存的观测，不为了得到好结果反复生成实验。')
    return conclusions, suggestions
