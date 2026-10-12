"""ID输入，模型自有embedding、MLP打分与BPR训练的示例。"""
import torch
from torch import nn
from torch.nn import functional as F


class Recommender(nn.Module):
    def __init__(self, num_users, num_items, settings):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        dimensions = settings['dimensions']
        self.user_latent = nn.Embedding(num_users, dimensions)
        self.item_latent = nn.Embedding(num_items, dimensions)
        self.network = nn.Sequential(nn.Linear(dimensions * 2, dimensions), nn.ReLU(), nn.Linear(dimensions, 1))
        self.optimizer = None
        self.negative_pools = None

    def fit_epoch(self, train_pairs, settings, rng):
        # 该示例明确使用BPR和均匀负采样，不将未实现的配置解释为已有功能。
        if settings.get('loss', 'bpr') != 'bpr' or settings.get('negative_sampling', 'uniform') != 'uniform':
            raise ValueError('此MLP示例只实现bpr/uniform')
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['learning_rate'])
            history = [set() for _ in range(self.num_users)]
            for user, item in train_pairs: history[user].add(item)
            self.negative_pools = [[item for item in range(self.num_items) if item not in seen] for seen in history]
        order = list(range(len(train_pairs)));rng.shuffle(order)
        total = 0.
        for offset in range(0, len(order), settings['batch_size']):
            batch = [train_pairs[index] for index in order[offset:offset + settings['batch_size']]]
            users = torch.tensor([user for user, _ in batch]);items = torch.tensor([item for _, item in batch])
            negatives = torch.tensor([rng.choice(self.negative_pools[user]) for user, _ in batch])
            loss = F.softplus(self.score(users, negatives) - self.score(users, items)).mean()
            loss = loss + settings['l2'] * sum(parameter.square().mean() for parameter in self.parameters())
            self.optimizer.zero_grad();loss.backward();self.optimizer.step()
            total += loss.item() * len(batch)
        return total / len(order)

    def score(self, user_ids, item_ids):
        return self.network(torch.cat([self.user_latent(user_ids), self.item_latent(item_ids)], dim=-1)).squeeze(-1)

    def score_all(self):
        rows = []
        for offset in range(0, self.num_users, 16):
            users = torch.arange(offset, min(offset + 16, self.num_users))
            items = torch.arange(self.num_items)
            rows.append(self.score(users.repeat_interleave(self.num_items), items.repeat(len(users))).reshape(len(users), -1))
        return torch.cat(rows)
