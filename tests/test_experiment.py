"""A failed refresh must not present an old test table as a fresh result."""
import json
import tempfile
import unittest
from pathlib import Path
from utils.experiment import run, verify_results, sha256


class ExperimentStatusTests(unittest.TestCase):
    def test_failed_run_invalidates_previous_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'results'
            out.mkdir()
            old = out / 'test.csv'
            old.write_text('previous results')
            (out / 'status.json').write_text(json.dumps({'state':'completed','run_id':'old'}))
            with self.assertRaises(FileNotFoundError):
                run(data_dir=root/'missing_data',output_dir=out)
            status = json.loads((out/'status.json').read_text())
            self.assertEqual(status['state'],'running')
            self.assertNotEqual(status['run_id'],'old')
            self.assertEqual(old.read_text(),'previous results')
            with self.assertRaises(ValueError):
                verify_results(out)

    def test_completed_results_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            names=['protocol.json','selected.json','validation.csv','test.csv','recommendations.json']
            for name in names:
                (out/name).write_text('current artifact')
            status={'state':'completed','artifact_sha256':{name:sha256(out/name) for name in names}}
            (out/'status.json').write_text(json.dumps(status))
            self.assertEqual(verify_results(out),status)
            (out/'test.csv').write_text('stale or modified table')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                verify_results(out)
