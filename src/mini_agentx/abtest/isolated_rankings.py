"""验证模型来源，在受限子进程生成排名，再检查返回结构。"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolated_rankings(run, info, manifest, training):
    run = Path(run).resolve()
    if info.get('candidate_source'):
        if info['candidate_source'] != 'candidate_model.py':
            raise ValueError('候选源码路径必须为 candidate_model.py')
        source = run / 'candidate_model.py'
    else:
        source = Path(__file__).parents[1] / 'recommender/model.py'
    checkpoint = run / 'model.pt'
    if source.is_symlink() or checkpoint.is_symlink():
        raise ValueError('模型文件不允许符号链接')
    if digest(source) != info['model_source_sha256'] or digest(checkpoint) != info['checkpoint_sha256']:
        raise ValueError('模型代码或权重摘要不匹配')
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise ValueError('模型A/B加载要求 macOS sandbox-exec；不降级执行')
    payload = {'source': str(source.resolve()), 'checkpoint': str(checkpoint),
               'source_sha256': info['model_source_sha256'], 'checkpoint_sha256': info['checkpoint_sha256'],
               'users': manifest['users'], 'items': manifest['items'],
               'dimensions': info['config']['training']['dimensions'], 'train': training}
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix='agentx-ranking-') as temporary:
        directory = Path(temporary).resolve()
        request, output = directory / 'request.json', directory / 'rankings.json'
        request.write_text(json.dumps(payload))
        quote = lambda value: json.dumps(str(value))
        profile = '(version 1)(allow default)(deny network*)(deny file-write*)'
        private = [root / '.env', root / '.git', root / '.aws', root / '.codex',
                   Path.home() / '.ssh', Path.home() / '.aws', Path.home() / '.codex', Path.home() / '.config']
        profile += '(deny file-read* ' + ' '.join('(subpath ' + quote(p) + ')' for p in private) + ')'
        profile += '(allow file-write* (subpath ' + quote(directory) + '))'
        profile += '(deny process-exec)(allow process-exec (subpath ' + quote(sys.prefix) + '))'
        environment = {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (directory / 'worker.log').open('w') as log:
            proc = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', profile, sys.executable,
                                     str(Path(__file__).with_name('ranking_worker.py')), str(request), str(output)],
                                    env=environment, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise ValueError('模型排名生成超时') from None
        if proc.returncode:
            raise ValueError(f'模型排名生成失败（{proc.returncode}）：' + (directory / 'worker.log').read_text()[-3000:])
        rankings = json.loads(output.read_text())['rankings']
    validate_rankings(rankings, manifest, training)
    if digest(source) != payload['source_sha256'] or digest(checkpoint) != payload['checkpoint_sha256']:
        raise ValueError('评分过程中模型文件发生变化')
    return rankings


def validate_rankings(rankings, manifest, training):
    if not isinstance(rankings, list) or len(rankings) != manifest['users']:
        raise ValueError('排名用户数量错误')
    seen = [set() for _ in range(manifest['users'])]
    for user, item in training:
        seen[user].add(item)
    for user, row in enumerate(rankings):
        expected = min(10, manifest['items'] - len(seen[user]))
        if not isinstance(row, list) or not expected or len(row) != expected:
            raise ValueError('排名数量错误或无候选物品')
        if any(type(item) is not int or not 0 <= item < manifest['items'] or item in seen[user] for item in row):
            raise ValueError('排名包含非法或已见物品')
        if len(set(row)) != len(row):
            raise ValueError('排名物品重复')
