"""保存模型实物及可追溯的基线版本；不执行候选代码。"""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from mini_agentx.memory.store import safe_path


def archive_model(root, runs, source, destination):
    source = safe_path(runs, source)
    report = json.loads(safe_path(runs, source / 'report.json').read_text())
    code = safe_path(runs, source / 'candidate_model.py') if report.get('candidate_source') else Path(root) / 'src/mini_agentx/recommender/model.py'
    for path, expected in ((safe_path(runs, source / 'model.pt'), report['checkpoint_sha256']), (code, report['model_source_sha256'])):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('模型归档摘要不匹配')
    destination = Path(destination)
    destination.mkdir(parents=True)
    shutil.copyfile(source / 'model.pt', destination / 'model.pt')
    shutil.copyfile(code, destination / 'candidate_model.py')
    # 所有归档都自带代码，读取时不依赖项目当前代码。
    report['candidate_source'] = 'candidate_model.py'
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (destination / 'training.toml').write_text('[training]\n' + ''.join(f'{key} = {json.dumps(value)}\n' for key, value in report['config']['training'].items()))
    return report


def promote(root, runs, candidate_archive, parent_run, evaluation_run):
    identifier = 'baseline-' + uuid.uuid4().hex[:12]
    destination = Path(runs) / identifier
    report = archive_model(root, runs, candidate_archive, destination)
    original = report['run_id']
    report.update(run_id=identifier, promotion={'parent_run_id': parent_run,
                  'candidate_run_id': original, 'evaluation_run': evaluation_run, 'criterion': 'validated_KEEP'})
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return destination
