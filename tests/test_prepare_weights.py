import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PrepareWeightsTests(unittest.TestCase):
    def test_originals_cannot_be_overwritten_by_output_or_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / 'config.json'
            original.write_text('original config\n')
            (directory / 'alias.bin.json').symlink_to(original)
            for name in ('config.json', 'config', 'model.safetensors.index', 'alias.bin'):
                with self.subTest(output=name):
                    run = subprocess.run([sys.executable, str(ROOT / 'tools/prepare_weights.py'), '--model', '8b',
                                          '--directory', str(directory), '--output', str(directory / name)],
                                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertIn('output and report must differ', run.stderr)
                    self.assertEqual(original.read_text(), 'original config\n')
