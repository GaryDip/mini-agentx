import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dimensions: int = 32):
        super().__init__()
        self.users = nn.Embedding(users, dimensions)
        self.items = nn.Embedding(items, dimensions)
        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        return (self.users(user_ids) * self.items(item_ids)).sum(dim=-1)

    def score_all(self):
        return self.users.weight @ self.items.weight.T
