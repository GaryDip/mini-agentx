import math
import unittest

import torch

from mini_agentx.recommender.evaluate import ranking_metrics


class MetricsTest(unittest.TestCase):
    def test_masking_and_hand_calculated_ranks(self):
        scores = torch.tensor([[100., 3., 2., 1.], [100., 3., 2., 1.]])
        metrics = ranking_metrics(scores, [(0, 1), (1, 2)], [(0, 0), (1, 0)], k=2)
        self.assertEqual(metrics["recall@2"], 1)
        self.assertAlmostEqual(metrics["ndcg@2"], (1 + 1 / math.log2(3)) / 2)

    def test_ties_and_misses(self):
        metrics = ranking_metrics(torch.zeros(1, 4), [(0, 3)], [], k=2)
        self.assertEqual(metrics["ndcg@2"], 0)
        self.assertEqual(metrics["recall@2"], 0)

    def test_invalid_scores(self):
        with self.assertRaises(ValueError):
            ranking_metrics(torch.tensor([[float('nan')]]), [(0, 0)], [])


if __name__ == "__main__":
    unittest.main()
