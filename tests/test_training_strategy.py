import random
from pathlib import Path
import tempfile
import unittest
import torch
from mini_agentx.recommender.model import MatrixFactorization
from mini_agentx.recommender.strategy import objective, sample_negatives, experiment_signature, options, popularity_cumulative
import test_developing


class StrategyTest(unittest.TestCase):
    def test_loss_gradient_and_training_only_sampler(self):
        pools = [[2, 3], [0, 3]]
        users = torch.tensor([0, 1])
        for sampling in ('uniform', 'popularity', 'hard'):
            for loss in ('bpr', 'hinge', 'bce'):
                model = MatrixFactorization(2, 4, 3)
                negatives = sample_negatives(model, users, pools, torch.tensor([1., 2., 8., 3.]), random.Random(42), sampling)
                self.assertTrue(all(item in pools[user] for user, item in zip(users.tolist(), negatives.tolist())))
                value = objective(model.score(users, torch.tensor([0, 1])), model.score(users, negatives), loss)
                self.assertEqual(value.ndim, 0)
                value.backward()
                self.assertTrue(torch.isfinite(model.items.weight.grad).all())

    def test_old_config_defaults_and_signature(self):
        config = {'seed': 42, 'training': {'dimensions': 32}}
        signature = experiment_signature('source', config, 'data')
        self.assertEqual(signature, experiment_signature('source', {'seed': 999, 'training': {'dimensions': 32, 'seed': 42, 'loss': 'bpr', 'negative_sampling': 'uniform'}}, 'data'))
        self.assertNotEqual(signature, experiment_signature('source', {'seed': 43, 'training': {'dimensions': 32}}, 'data'))
        with self.assertRaises(ValueError):
            options({'seed': 42, 'training': {'loss': 'invalid'}})

    def test_duplicate_rejected_new_seed_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            tools = test_developing.DevelopingTest().make_tools(Path(folder))
            with self.assertRaisesRegex(ValueError, '重复实验'):
                tools.reject_duplicate()
            tools.path('training.toml').write_text(tools.original['training.toml'].replace('seed = 42', 'seed = 43'))
            tools.reject_duplicate()
            self.assertEqual(tools.settings()['seed'], 43)
            tools.path('training.toml').write_text(tools.original['training.toml'].replace('"bpr"', '"bce"').replace('"uniform"', '"hard"'))
            tools.reject_duplicate()

    def test_cached_popularity_preserves_sampling_distribution_and_seed(self):
        pools = [[1, 2, 3], [0, 2]]
        counts = torch.tensor([4., 20., 2., 9.])
        users = torch.tensor([0, 1] * 50)
        model = MatrixFactorization(2, 4, 3)
        rng = random.Random(42)
        expected = torch.tensor([rng.choices(pools[user], weights=[float(counts[item] + 1) ** .75 for item in pools[user]], k=1)[0] for user in users.tolist()])
        cumulative = popularity_cumulative(pools, counts)
        actual = sample_negatives(model, users, pools, counts, random.Random(42), 'popularity', cumulative)
        self.assertTrue(torch.equal(expected, actual))
