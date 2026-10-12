"""模型协议与旧矩阵分解兼容桥；框架不依赖模型内部参数命名。"""
import math
import io
import random
import torch
from .strategy import options, objective, sample_negatives, popularity_cumulative


def model_class(module):
    if hasattr(module, 'Recommender'):
        if not callable(getattr(module.Recommender, 'fit_epoch', None)):
            raise ValueError('新Recommender必须实现fit_epoch；旧接口使用MatrixFactorization')
        return module.Recommender
    if hasattr(module, 'MatrixFactorization'):
        return module.MatrixFactorization
    raise ValueError('候选须提供 Recommender，旧模型可提供 MatrixFactorization')


def create_model(cls, users, items, settings):
    model = cls(users, items, settings) if hasattr(cls, 'fit_epoch') else cls(users, items, settings['dimensions'])
    if not isinstance(model, torch.nn.Module):
        raise ValueError('模型须为nn.Module，以支持统一安全权重保存与恢复')
    for name in ('score', 'score_all'):
        if not callable(getattr(model, name, None)):
            raise ValueError('模型缺少打分接口：' + name)
    return model


class LegacyTraining:
    """仅用于旧源码兼容；新模型自行实现fit_epoch，不要求users/items属性。"""
    def __init__(self, model, training, settings, seed, items):
        self.model, self.training, self.settings = model, training, settings
        self.rng = random.Random(seed)
        self.strategy = options({'seed': seed, 'training': settings})
        self.optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        positives = [set() for _ in range(model.score_all().shape[0])]
        self.popularity = torch.zeros(items)
        for user, item in training:
            positives[user].add(item);self.popularity[item] += 1
        self.negatives = [[item for item in range(items) if item not in history] for history in positives]
        if any(not pool for pool in self.negatives):
            raise ValueError('用户没有可采样负例')
        self.cumulative = popularity_cumulative(self.negatives, self.popularity) if self.strategy['negative_sampling'] == 'popularity' else None

    def fit_epoch(self):
        model, settings = self.model, self.settings
        total = 0.
        order = torch.randperm(len(self.training)).tolist()
        for offset in range(0, len(order), settings['batch_size']):
            batch = [self.training[index] for index in order[offset:offset + settings['batch_size']]]
            users = torch.tensor([row[0] for row in batch]);items = torch.tensor([row[1] for row in batch])
            negatives = sample_negatives(model, users, self.negatives, self.popularity, self.rng, self.strategy['negative_sampling'], self.cumulative)
            loss = objective(model.score(users, items), model.score(users, negatives), self.strategy['loss'])
            regularizer = (model.users(users).square().sum() + model.items(items).square().sum() + model.items(negatives).square().sum()) / len(batch)
            loss = loss + settings['l2'] * regularizer
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限')
            self.optimizer.zero_grad();loss.backward();self.optimizer.step()
            total += loss.item() * len(batch)
        return total / len(order)


def checked_loss(value):
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.detach().item()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('fit_epoch必须返回有限的标量训练损失或诊断值')
    return float(value)


def check_contract(cls, settings, seed):
    torch.manual_seed(seed)
    model = create_model(cls, 3, 5, settings)
    training = [(0, 1), (0, 2), (1, 2), (2, 3)]
    if callable(getattr(model, 'fit_epoch', None)):
        model.train();checked_loss(model.fit_epoch(training, dict(settings), random.Random(seed)))
        protocol = 'model_owned_training_v1'
    else:
        model.train();checked_loss(LegacyTraining(model, training, settings, seed, 5).fit_epoch())
        protocol = 'legacy_mf'
    model.eval()
    users, items = torch.tensor([0, 1]), torch.tensor([1, 2])
    with torch.no_grad():
        scores, all_scores = model.score(users, items), model.score_all()
    if not isinstance(scores, torch.Tensor) or not isinstance(all_scores, torch.Tensor) or scores.shape != (2,) or all_scores.shape != (3, 5):
        raise ValueError('score/score_all输出形状错误')
    if not torch.isfinite(scores).all() or not torch.isfinite(all_scores).all() or not torch.allclose(scores, all_scores[users, items], atol=1e-6):
        raise ValueError('分数非有限或两种打分不一致')
    restored = create_model(cls, 3, 5, settings)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer);buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True));restored.eval()
    with torch.no_grad():
        if not torch.allclose(restored.score_all(), all_scores, atol=1e-6):
            raise ValueError('保存加载后打分不一致；所有推理状态应进入state_dict')
    return {'valid': True, 'model_protocol': protocol, 'score_shape': [2], 'score_all_shape': [3, 5]}
