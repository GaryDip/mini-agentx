"""隔离进程加载实际模型，输出排名；不返回代码或评分矩阵。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys

import torch


def main():
    input_path, output_path = map(Path, sys.argv[1:])
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    payload = json.loads(input_path.read_text())
    torch.set_num_threads(1)
    source, checkpoint = Path(payload['source']), Path(payload['checkpoint'])
    for path, key in ((source, 'source_sha256'), (checkpoint, 'checkpoint_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != payload[key]:
            raise ValueError('模型文件在启动前发生变化')
    spec = importlib.util.spec_from_file_location('ab_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from mini_agentx.recommender.interface import model_class, create_model
    settings = payload.get('training_settings', {'dimensions': payload['dimensions']})
    model = create_model(model_class(module), payload['users'], payload['items'], settings)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model.score_all().clone()
    if scores.shape != (payload['users'], payload['items']) or not torch.isfinite(scores).all():
        raise ValueError('评分矩阵形状错误或分数非有限')
    for user, item in payload['train']:
        scores[user, item] = -torch.inf
    order = torch.argsort(scores, dim=1, descending=True, stable=True).tolist()
    rankings = [[item for item in row if torch.isfinite(scores[user, item])][:10]
                for user, row in enumerate(order)]
    output_path.write_text(json.dumps({'rankings': rankings}))


if __name__ == '__main__':
    main()
