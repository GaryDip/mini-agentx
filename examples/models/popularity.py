"""没有embedding、没有梯度优化器的模型协议示例。"""
import torch
from torch import nn


class Recommender(nn.Module):
    def __init__(self, num_users, num_items, settings):
        super().__init__()
        self.num_users = num_users
        self.register_buffer('popularity', torch.zeros(num_items))

    def fit_epoch(self, train_pairs, settings, rng):
        counts = torch.bincount(torch.tensor([item for _, item in train_pairs]), minlength=len(self.popularity)).float()
        self.popularity.copy_(counts)
        return 0.0  # 非梯度方法的有限诊断值，不声称这是BPR损失。

    def score(self, user_ids, item_ids):
        return self.popularity[item_ids]

    def score_all(self):
        return self.popularity.unsqueeze(0).expand(self.num_users, -1)
