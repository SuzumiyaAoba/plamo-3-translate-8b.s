"""Exercise installed command behavior without model data or network access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from tools.model_catalog import MODELS, profile
from tools.generate_launchers import render

ROOT = Path(__file__).resolve().parents[1]
REVISION = "19ecab3fb31424ee937b7503730318b50e55771c"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="plamo launcher ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "model data"
        self.model = self.data / "models/plamo"
        self.model.mkdir(parents=True)
        self.environment = dict(os.environ, PLAMO_DATA_DIR=str(self.data))
        self.environment.pop('PLAMO_MODEL', None)
        self.engine = self.root / "plamo-8b"
        self.engine.write_text(f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]], ensure_ascii=False))\n")
        self.engine.chmod(0o755)
        self.launcher = self.root / "plamo"
        self.launcher.write_text((ROOT / "nix/plamo.sh.in").read_text().replace("@engines@", str(self.root)).replace("@modelProfiles@", render()))
        self.shell = shutil.which("bash")
        self.assertIsNotNone(self.shell)

    def run_launcher(self, *args):
        return subprocess.run([self.shell, str(self.launcher), *args], env=self.environment, cwd=self.root, text=True, capture_output=True)

    def vocabulary(self):
        (self.model / "tokenizer.jsonl").write_text("fixture only")
        (self.model / "REVISION").write_text(REVISION + "\n")

    def test_help_needs_no_model(self):
        for args in ((), ("--help",), ("-h",)):
            result = self.run_launcher(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("download-model", result.stdout)

    def test_missing_files_explain_download(self):
        result = self.run_launcher("Hello")
        self.assertEqual(result.returncode, 1)
        self.assertIn("plamo-download --weights", result.stderr)

    def test_tokenizer_preserves_unicode_and_empty_input(self):
        self.vocabulary()
        for text in ("こんにちは 😀 'quoted'", ""):
            result = self.run_launcher(text)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [str(self.data.resolve()), [text]])

    def test_generation_requires_all_shards(self):
        self.vocabulary()
        self.assertEqual(self.run_launcher("-j", "Hello.").returncode, 1)
        for i in range(1, 5):
            (self.model / f"model-{i:05d}-of-00004.safetensors").touch()
        for option in ("-g", "-j", "-e"):
            result = self.run_launcher(option, "Hello.")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [str(self.data.resolve()), [option, "Hello."]])

    def test_mismatched_revision_is_rejected(self):
        self.vocabulary()
        (self.model / "REVISION").write_text("wrong\n")
        result = self.run_launcher("Hello")
        self.assertEqual(result.returncode, 1)
        self.assertIn("revision", result.stderr)

    def test_xdg_data_directory(self):
        self.environment.pop("PLAMO_DATA_DIR")
        self.environment["XDG_DATA_HOME"] = str(self.root / "xdg")
        model = self.root / "xdg/plamo-3-translate-8b/models/plamo"
        model.mkdir(parents=True)
        (model / "tokenizer.jsonl").touch()
        result = self.run_launcher("Hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0], str(model.parents[1].resolve()))

    def test_each_model_selects_its_own_engine_and_files(self):
        for size in MODELS:
            with self.subTest(model=size):
                p = profile(size)
                model_dir = self.data / p.directory
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "tokenizer.jsonl").touch()
                (model_dir / "REVISION").write_text(p.revision + "\n")
                if size != "8b":
                    shutil.copy2(self.engine, self.root / ("plamo-" + size))
                for weight in p.weights:
                    (model_dir / weight).touch()
                result = self.run_launcher("--model", size, "-j", "Hello.")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [str(self.data.resolve()), ["-j", "Hello."]])
                (model_dir / p.weights[-1]).unlink()
                missing = self.run_launcher("--model=" + size, "-j", "Hello.")
                self.assertEqual(missing.returncode, 1)
                self.assertIn('--model ' + size, missing.stderr)

    def test_invalid_arguments(self):
        result = self.run_launcher("--unknown", "text")
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_downloader_uses_same_directory_and_preserves_arguments(self):
        script = self.root / "download probe.py"
        script.write_text("import json, os, sys\nprint(json.dumps([sys.argv[1:], os.environ['SSL_CERT_FILE']]))\n")
        wrapper = self.root / "plamo-download"
        wrapper.write_text((ROOT / "nix/plamo-download.sh.in").read_text().replace("@python@", sys.executable).replace("@script@", str(script)).replace("@caBundle@", "/fixture/ca-bundle.crt"))
        self.environment.pop("SSL_CERT_FILE", None)
        result = subprocess.run([self.shell, str(wrapper), "--weights", "--verify"], env=self.environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [["--cache", "--weights", "--verify"], "/fixture/ca-bundle.crt"])


if __name__ == "__main__":
    unittest.main()
