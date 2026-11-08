"""Offline checks for atomic, pinned model downloads."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("download", Path(__file__).resolve().parents[1] / "tools/download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.payload = b"original upstream bytes\x00\xff"
        self.expected = {"size": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}

    def fetch(self, name, payload=None, **kwargs):
        with patch.object(download.urllib.request, "urlopen", return_value=io.BytesIO(self.payload if payload is None else payload)) as opener:
            download.download_file(name, self.directory, self.expected, **kwargs)
        return opener

    def test_verified_download_and_nested_license_path(self):
        opener = self.fetch("LICENSE/en")
        self.assertEqual((self.directory / "LICENSE/en").read_bytes(), self.payload)
        self.assertFalse((self.directory / "LICENSE/en.part").exists())
        self.assertIn(download.REVISION, opener.call_args.args[0])

    def test_cache_checked_without_network(self):
        target = self.directory / "tokenizer.jsonl"
        target.write_bytes(self.payload)
        self.fetch(target.name).assert_not_called()

    def test_bad_cached_content_is_replaced_even_at_same_size(self):
        target = self.directory / "tokenizer.jsonl"
        target.write_bytes(b"x" * len(self.payload))
        self.fetch(target.name).assert_called_once()
        self.assertEqual(target.read_bytes(), self.payload)

    def test_truncated_response_does_not_publish_partial_file(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.fetch("new.safetensors", payload=self.payload[:-1])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_checksum_failure_keeps_old_target(self):
        target = self.directory / "model.safetensors"
        old = b"a" * len(self.payload)
        target.write_bytes(old)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.fetch(target.name, payload=b"b" * len(self.payload), verify_existing=True)
        self.assertEqual(target.read_bytes(), old)
        self.assertFalse(target.with_suffix(".safetensors.part").exists())

    def test_network_failure_leaves_no_partial_file(self):
        with patch.object(download.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                download.download_file("tokenizer.jsonl", self.directory, self.expected)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_explicit_verification_checks_cached_weights(self):
        target = self.directory / "model.safetensors"
        target.write_bytes(b"x" * len(self.payload))
        self.fetch(target.name, verify_existing=True).assert_called_once()
        self.assertEqual(target.read_bytes(), self.payload)

    def test_another_revision_is_rejected_before_download(self):
        marker = self.directory / "REVISION"
        marker.write_text("another-revision\n")
        with patch("sys.argv", ["download.py", "--directory", str(self.directory)]), patch.object(download.urllib.request, "urlopen") as opener:
            with self.assertRaises(SystemExit) as error:
                download.main()
        self.assertEqual(error.exception.code, 1)
        opener.assert_not_called()
        self.assertEqual(marker.read_text(), "another-revision\n")


if __name__ == "__main__":
    unittest.main()
