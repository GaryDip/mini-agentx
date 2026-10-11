"""训练策略接口：只接收训练负例池，不接触留出标签。"""
import hashlib
import json
import torch
from torch.nn import functional as F


def options(config):
    settings = config['training']
    result = {'seed': settings.get('seed', config['seed']), 'loss': settings.get('loss', 'bpr'),
              'negative_sampling': settings.get('negative_sampling', 'uniform')}
    if type(result['seed']) is not int or not 0 <= result['seed'] <= 2147483647:
        raise ValueError('seed 必须是 0..2147483647 的整数')
    if result['loss'] not in ('bpr', 'hinge', 'bce'):
        raise ValueError('loss 只支持 bpr/hinge/bce')
    if result['negative_sampling'] not in ('uniform', 'popularity', 'hard'):
        raise ValueError('negative_sampling 只支持 uniform/popularity/hard')
    return result


def objective(positive, negative, name):
    if name == 'bpr':
        return F.softplus(negative - positive).mean()
    if name == 'hinge':
        return F.relu(1 - positive + negative).mean()
    if name == 'bce':
        return (F.softplus(-positive).mean() + F.softplus(negative).mean()) / 2
    raise ValueError('未知损失')


def sample_negatives(model, users, pools, popularity, rng, name):
    sampled = []
    for user in users.tolist():
        pool = pools[user]
        if name == 'uniform':
            item = rng.choice(pool)
        elif name == 'popularity':
            item = rng.choices(pool, weights=[float(popularity[i] + 1) ** .75 for i in pool], k=1)[0]
        elif name == 'hard':
            choices = rng.choices(pool, k=5)
            with torch.no_grad():
                scores = model.score(torch.full((5,), user, dtype=torch.long), torch.tensor(choices))
            item = choices[int(scores.argmax())]
        else:
            raise ValueError('未知负采样')
        sampled.append(item)
    return torch.tensor(sampled)


def experiment_signature(source_hash, config, dataset):
    settings = dict(config['training'])
    settings.update(options(config))
    payload = {'model_source': source_hash, 'settings': settings, 'dataset': dataset}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
