"""训练策略接口：只接收训练负例池，不接触留出标签。"""
import hashlib
import json
from itertools import accumulate
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


def popularity_cumulative(pools, popularity):
    weights = [(float(count) + 1) ** .75 for count in popularity.tolist()]
    return [list(accumulate(weights[item] for item in pool)) for pool in pools]


def sample_negatives(model, users, pools, popularity, rng, name, cumulative=None):
    user_list = users.tolist()
    if name == 'uniform':
        return torch.tensor([rng.choice(pools[user]) for user in user_list])
    if name == 'popularity':
        cumulative = cumulative if cumulative is not None else popularity_cumulative(pools, popularity)
        return torch.tensor([rng.choices(pools[user], cum_weights=cumulative[user], k=1)[0] for user in user_list])
    if name == 'hard':
        choices = torch.tensor([rng.choices(pools[user], k=5) for user in user_list])
        with torch.no_grad():
            scores = model.score(users.repeat_interleave(5), choices.flatten()).reshape(-1, 5)
        return choices.gather(1, scores.argmax(dim=1, keepdim=True)).flatten()
    raise ValueError('未知负采样')


def experiment_signature(source_hash, config, dataset):
    settings = dict(config['training'])
    settings.update(options(config))
    payload = {'model_source': source_hash, 'settings': settings, 'dataset': dataset}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
