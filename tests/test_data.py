import gzip
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from utils.data import prepare, records, eda


class PreparationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for t in range(100):
            rows.append((f'u{t % 7}', f'i{t % 11}', 1 + t % 5, 1000+t))
        return pd.DataFrame(rows, columns=['reviewerID', 'asin', 'overall', 'unixReviewTime'])

    def test_temporal_split_and_train_vocab(self):
        df = self.fixture()
        splits, report = prepare(df, max_items=5, min_interactions=1, min_entities=1)
        train, val, test = splits
        self.assertLess(train.unixReviewTime.max(), val.unixReviewTime.min())
        self.assertLess(val.unixReviewTime.max(), test.unixReviewTime.min())
        self.assertEqual(train.asin.nunique(), 5)
        for part in splits:
            self.assertFalse(part.duplicated(['reviewerID', 'asin']).any())
            self.assertFalse(part[['user_idx', 'item_idx']].isna().any().any())
            self.assertTrue(set(part.asin) <= set(train.asin))
        _, shuffled = prepare(df.sample(frac=1, random_state=3), max_items=5, min_interactions=1, min_entities=1)
        self.assertEqual(report, shuffled)

    def test_future_ratings_do_not_change_training(self):
        df = self.fixture()
        original, _ = prepare(df, min_interactions=1, min_entities=1)
        df.loc[df.unixReviewTime > original[0].unixReviewTime.max(), 'overall'] = 1
        changed, _ = prepare(df, min_interactions=1, min_entities=1)
        pd.testing.assert_frame_equal(original[0], changed[0])

    def test_constraints_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Infeasible'):
            prepare(self.fixture())

    def test_duplicate_earliest_and_missing(self):
        df = self.fixture()
        duplicate = df.iloc[[0]].copy()
        duplicate.unixReviewTime = 9999
        duplicate.overall = 5
        invalid = df.iloc[[1]].copy()
        invalid.overall = None
        parts, _ = prepare(pd.concat([df, duplicate, invalid]), min_interactions=1, min_entities=1)
        row = parts[0].query("reviewerID == 'u0' and asin == 'i0'")
        self.assertEqual(row.unixReviewTime.tolist(), [1000])

    def test_gzip_reader_and_eda(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample.gz'
            with gzip.open(path, 'wt') as f:
                f.write(json.dumps({'asin': 'x'}) + '\n')
            self.assertEqual(list(records(path)), [{'asin': 'x'}])
        report = eda(self.fixture(), pd.DataFrame({'asin': ['i0'], 'brand': [''], 'category': [[]]}))
        self.assertEqual(report['users'], 7)
        self.assertEqual(report['unique_pairs'], 77)
        self.assertEqual(report['sparsity_unique_pairs'], 0.0)
        self.assertEqual(report['empty_metadata']['brand'], 1)


if __name__ == '__main__':
    unittest.main()
