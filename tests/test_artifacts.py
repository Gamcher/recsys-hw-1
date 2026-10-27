"""Integration checks of prepared real data; skipped before download."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'data/processed'


@unittest.skipUnless((ROOT / 'preparation.json').exists(), 'Official data not prepared')
class ArtifactTests(unittest.TestCase):
    def test_saved_splits(self):
        report = json.loads((ROOT / 'preparation.json').read_text())
        parts = [pd.read_csv(ROOT / f'{name}.csv.gz', keep_default_na=False)
                 for name in ['train', 'validation', 'test']]
        self.assertEqual([len(x) for x in parts], report['prepared_split_rows'])
        self.assertGreaterEqual(sum(map(len, parts)), 100000)
        self.assertLessEqual(sum(map(len, parts)), 2000000)
        self.assertGreaterEqual(parts[0].reviewerID.nunique(), 1000)
        self.assertGreaterEqual(parts[0].asin.nunique(), 1000)
        self.assertLessEqual(parts[0].asin.nunique(), 5000)
        seen = set()
        for part in parts:
            pairs = set(zip(part.reviewerID, part.asin))
            self.assertEqual(len(pairs), len(part))
            self.assertFalse(seen.intersection(pairs))
            seen.update(pairs)
            self.assertTrue(part.overall.between(1, 5).all())
            self.assertTrue(part.user_idx.eq(part.reviewerID.map(report['user_mapping'])).all())
            self.assertTrue(part.item_idx.eq(part.asin.map(report['item_mapping'])).all())
        self.assertLess(parts[0].unixReviewTime.max(), parts[1].unixReviewTime.min())
        self.assertLess(parts[1].unixReviewTime.max(), parts[2].unixReviewTime.min())
        metadata = pd.read_json(ROOT / 'item_metadata.jsonl', lines=True)
        self.assertTrue(metadata.asin.is_unique)
        self.assertTrue(set(metadata.asin) <= set(parts[0].asin))
