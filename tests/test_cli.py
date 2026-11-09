"""Native CLI argument validation must not load weights or require a GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.binary = Path(os.environ.get('PLAMO_TEST_CLI', ROOT / 'build/plamo')).resolve()

    def run_cli(self, *args):
        return subprocess.run([str(self.binary), *args], cwd=self.temp.name, text=True, capture_output=True)

    def test_help_without_model_or_gpu(self):
        for args in ((), ('--help',), ('-h',)):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('FP32', result.stdout)

    def test_invalid_arguments_are_rejected_before_loading(self):
        for args in (('--unknown', 'x'), ('x', 'y'), ('-j',), ('-j', 'x', '-e', 'y'), ('--',),
                     ('--backend', 'cuda', 'x'), ('--memory', 'unknown', 'x'),
                     ('--threads', '0', 'x'), ('--threads', '65', 'x'), ('--threads', '+4', 'x'),
                     ('--threads', '4x', 'x'), ('--max-tokens', '0', 'x'), ('--max-tokens', '4097', 'x'),
                     ('--max-tokens', '18446744073709551616000', 'x')):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('usage:', result.stderr)

    def test_quoted_and_empty_text_are_arguments(self):
        for args in (('--', '-j'), ('-j', '--quoted text'), ('',), ('--backend', 'cpu', '-g', 'x'),
                     ('--threads', 'auto', '--max-tokens', '4096', '-e', 'こんにちは')):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('tokenizer is missing', result.stderr)


if __name__ == '__main__':
    unittest.main()
