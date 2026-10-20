"""Offline checks for corruption, resuming downloads and path containment."""

import importlib.util
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

SPEC = importlib.util.spec_from_file_location(
    "fetch_assets", Path(__file__).resolve().parents[1] / "scripts/fetch_assets.py"
)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = {"base_url": "https://example.invalid", "revision": "a" * 40}
        self.content = b"hello\n"
        self.entry = {
            "path": "models/test.xml",
            "size": 6,
            "git_blob_sha1": "ce013625030ba8dba906f756967f9e9ca394464a",
        }

    def test_download_then_check_uses_no_network(self):
        with patch.object(
            assets, "urlopen", return_value=BytesIO(self.content)
        ) as request:
            digest = assets.fetch_file(self.source, self.entry, self.root, False)
            request.assert_called_once()
        with patch.object(
            assets, "urlopen", side_effect=AssertionError("Network accessed")
        ):
            self.assertEqual(
                digest, assets.fetch_file(self.source, self.entry, self.root, True)
            )

    def test_corrupt_download_is_not_installed(self):
        with patch.object(assets, "urlopen", return_value=BytesIO(b"wrong\n")):
            with self.assertRaisesRegex(ValueError, "Integrity"):
                assets.fetch_file(self.source, self.entry, self.root, False)
        self.assertFalse((self.root / self.entry["path"]).exists())

    def test_corrupt_local_file_is_preserved_and_rejected(self):
        target = self.root / self.entry["path"]
        target.parent.mkdir()
        target.write_bytes(b"wrong\n")
        with patch.object(
            assets, "urlopen", side_effect=AssertionError("Network accessed")
        ):
            with self.assertRaisesRegex(ValueError, "Integrity"):
                assets.fetch_file(self.source, self.entry, self.root, False)
        self.assertEqual(target.read_bytes(), b"wrong\n")

    def test_check_missing_file_does_not_download(self):
        with patch.object(
            assets, "urlopen", side_effect=AssertionError("Network accessed")
        ):
            with self.assertRaises(FileNotFoundError):
                assets.fetch_file(self.source, self.entry, self.root, True)

    def test_connection_reset_retries_then_resumes(self):
        reset = URLError(ConnectionResetError(54, "Connection reset by peer"))
        with patch.object(
            assets, "urlopen", side_effect=[reset, BytesIO(self.content)]
        ) as request:
            with patch.object(assets.time, "sleep") as sleep:
                assets.fetch_file(self.source, self.entry, self.root, False)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual((self.root / self.entry["path"]).read_bytes(), self.content)

    def test_short_read_retries_without_saving_partial_file(self):
        with patch.object(
            assets, "urlopen", side_effect=[BytesIO(b"he"), BytesIO(self.content)]
        ):
            with patch.object(assets.time, "sleep") as sleep:
                assets.fetch_file(self.source, self.entry, self.root, False)
        sleep.assert_called_once_with(1)
        self.assertEqual((self.root / self.entry["path"]).read_bytes(), self.content)

    def test_read_connection_reset_is_retried(self):
        class BrokenResponse(BytesIO):
            def read(self, size):
                raise ConnectionResetError(54, "Connection reset by peer")

        with patch.object(
            assets, "urlopen", side_effect=[BrokenResponse(), BytesIO(self.content)]
        ):
            with patch.object(assets.time, "sleep") as sleep:
                assets.fetch_file(self.source, self.entry, self.root, False)
        sleep.assert_called_once_with(1)
        self.assertEqual((self.root / self.entry["path"]).read_bytes(), self.content)

    def test_retries_are_bounded_and_leave_no_partial_file(self):
        with patch.object(
            assets, "urlopen", side_effect=TimeoutError("timed out")
        ) as request:
            with patch.object(assets.time, "sleep") as sleep:
                with self.assertRaisesRegex(OSError, "models/test.xml.*5 attempts"):
                    assets.fetch_file(self.source, self.entry, self.root, False)
        self.assertEqual(request.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4, 8])
        self.assertFalse((self.root / self.entry["path"]).exists())

    def test_http_404_is_not_retried(self):
        error = HTTPError("https://example.invalid", 404, "Not Found", {}, None)
        with patch.object(assets, "urlopen", side_effect=error) as request:
            with patch.object(assets.time, "sleep") as sleep:
                with self.assertRaisesRegex(OSError, "models/test.xml.*404"):
                    assets.fetch_file(self.source, self.entry, self.root, False)
        request.assert_called_once()
        sleep.assert_not_called()

    def test_http_503_is_retried(self):
        error = HTTPError("https://example.invalid", 503, "Unavailable", {}, None)
        with patch.object(
            assets, "urlopen", side_effect=[error, BytesIO(self.content)]
        ):
            with patch.object(assets.time, "sleep") as sleep:
                assets.fetch_file(self.source, self.entry, self.root, False)
        sleep.assert_called_once_with(1)

    def test_rejects_parent_absolute_and_symlink_escape(self):
        for name in ("../outside.xml", "/outside.xml", "models/../../outside.xml"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                assets.asset_path(self.root, name)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            assets.asset_path(self.root, "escape/outside.xml")


if __name__ == "__main__":
    unittest.main()
