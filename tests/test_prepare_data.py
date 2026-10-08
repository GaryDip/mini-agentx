import csv
from pathlib import Path
import tempfile
import unittest

from mini_agentx.data.prepare import prepare_data


class PrepareDataTest(unittest.TestCase):
    def test_temporal_split_cold_items_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw/ml-100k/u.data"
            source.parent.mkdir(parents=True)
            # Deliberately unsorted input; verify temporal sorting.
            source.write_text("1 30 5 30\n1 20 4 20\n1 10 5 10\n"
                              "2 10 5 20\n2 30 4 30\n2 20 5 10\n"
                              "2 40 1 5\n3 10 5 10\n")
            config = {"paths": {"raw_data": str(root / "raw"),
                                 "processed_data": str(root / "out")}}
            first = prepare_data(config)
            self.assertEqual(first, prepare_data(config))
            self.assertEqual(first["split_counts"], {"train": 2, "validation": 2, "test": 0})
            self.assertEqual(first["cold_holdout_rows_excluded"]["test"], 2)
            self.assertEqual(first["excluded_users"], 1)
            output = root / "out/movielens-100k-v1"
            with (output / "train.csv").open() as file:
                train = list(csv.DictReader(file))
            self.assertEqual([row["item_id"] for row in train], ["0", "1"])
            source.write_text("1 10 5 1\n1 10 5 2\n")
            with self.assertRaisesRegex(ValueError, "重复交互"):
                prepare_data(config)


if __name__ == "__main__":
    unittest.main()
