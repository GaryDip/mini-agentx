import unittest

import torch
from torch.nn import functional as F

from mini_agentx.recommender.model import MatrixFactorization


class ModelTest(unittest.TestCase):
    def test_bpr_update_increases_positive_margin(self):
        model = MatrixFactorization(1, 2, 2)
        with torch.no_grad():
            model.users.weight.fill_(0.2)
            model.items.weight.copy_(torch.tensor([[0.1, 0.1], [-0.1, -0.1]]))
        user, positive, negative = torch.tensor([0]), torch.tensor([0]), torch.tensor([1])
        before = model.score(user, positive) - model.score(user, negative)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        F.softplus(-before).mean().backward()
        optimizer.step()
        after = model.score(user, positive) - model.score(user, negative)
        self.assertGreater(after.item(), before.item())


if __name__ == "__main__":
    unittest.main()
