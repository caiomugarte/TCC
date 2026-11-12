"""Hash-checked download cache: reuse, integrity, source failures, offline mode (SRC-01..SRC-04, CLI-02).

Every test runs against FakeOpener, so no request leaves the process.
"""

import hashlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild.cache import Cache  # noqa: E402
from pit_rebuild.errors import IntegrityError, SourceError  # noqa: E402
from pit_rebuild_support import FakeOpener, FakeResponse, make_zip  # noqa: E402

DFP_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2025.zip"
ITR_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2025.zip"
ZIP_BODY = make_zip({"dfp_cia_aberta_2025.csv": "CNPJ_CIA;DT_REFER\n1;2025-12-31\n"})


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def manifest(self):
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def names(self):
        return sorted(path.name for path in self.root.iterdir())


class DownloadAndReuseTests(CacheTestCase):
    def test_miss_downloads_once_and_records_the_manifest_entry(self):
        opener = FakeOpener({DFP_URL: FakeResponse(ZIP_BODY)})
        cache = Cache(self.root, offline=False, opener=opener)

        path = cache.get(DFP_URL, "dfp.zip")

        self.assertEqual(path.read_bytes(), ZIP_BODY)
        self.assertEqual(opener.requests, [("GET", DFP_URL)])
        entry = self.manifest()["files"]["dfp.zip"]
        self.assertEqual(entry["url"], DFP_URL)
        self.assertEqual(entry["sha256"], hashlib.sha256(ZIP_BODY).hexdigest())
        self.assertEqual(entry["bytes"], len(ZIP_BODY))
        self.assertRegex(entry["retrieved_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertEqual([item["sha256"] for item in cache.used], [entry["sha256"]])

    def test_second_get_in_a_new_run_makes_zero_requests(self):
        Cache(self.root, False, FakeOpener({DFP_URL: FakeResponse(ZIP_BODY)})).get(DFP_URL, "dfp.zip")
        opener = FakeOpener()

        path = Cache(self.root, False, opener).get(DFP_URL, "dfp.zip")

        self.assertEqual(opener.requests, [])
        self.assertEqual(path.read_bytes(), ZIP_BODY)

    def test_optional_download_failure_returns_none_and_writes_nothing(self):
        opener = FakeOpener({ITR_URL: TimeoutError("timed out")})
        cache = Cache(self.root, offline=False, opener=opener)

        self.assertIsNone(cache.get(ITR_URL, "b3-company.json", required=False))
        self.assertFalse((self.root / "b3-company.json").exists())


class IntegrityTests(CacheTestCase):
    def test_flipped_byte_makes_verify_and_get_exit_4_naming_the_file(self):
        Cache(self.root, False, FakeOpener({DFP_URL: FakeResponse(ZIP_BODY)})).get(DFP_URL, "dfp.zip")
        cached = self.root / "dfp.zip"
        damaged = bytearray(cached.read_bytes())
        damaged[0] ^= 0x01
        cached.write_bytes(bytes(damaged))
        opener = FakeOpener()

        with self.assertRaises(IntegrityError) as verify_ctx:
            Cache(self.root, False, opener).verify()
        with self.assertRaises(IntegrityError) as get_ctx:
            Cache(self.root, False, opener).get(DFP_URL, "dfp.zip")

        self.assertEqual(verify_ctx.exception.exit_code, 4)
        self.assertIn("dfp.zip", str(verify_ctx.exception))
        self.assertEqual(get_ctx.exception.exit_code, 4)
        self.assertEqual(opener.requests, [])


class SourceFailureTests(CacheTestCase):
    def failing_download_keeps_cached_files(self, failure):
        ok = Cache(self.root, False, FakeOpener({DFP_URL: FakeResponse(ZIP_BODY)}))
        kept = ok.get(DFP_URL, "dfp.zip")
        cache = Cache(self.root, False, FakeOpener({ITR_URL: failure}))

        with self.assertRaises(SourceError) as ctx:
            cache.get(ITR_URL, "itr.zip")

        self.assertEqual(ctx.exception.exit_code, 5)
        self.assertIn(ITR_URL, str(ctx.exception))
        self.assertEqual(kept.read_bytes(), ZIP_BODY)
        self.assertEqual(self.names(), ["dfp.zip", "manifest.json"])
        self.assertEqual(set(self.manifest()["files"]), {"dfp.zip"})

    def test_http_error_on_a_required_zip_exits_5_and_keeps_cached_files(self):
        failure = urllib.error.HTTPError(ITR_URL, 500, "Internal Server Error", {}, io.BytesIO(b""))

        self.failing_download_keeps_cached_files(failure)

    def test_timeout_on_a_required_zip_exits_5_and_keeps_cached_files(self):
        self.failing_download_keeps_cached_files(TimeoutError("timed out"))

    def test_non_zip_body_on_a_required_zip_exits_5_and_keeps_cached_files(self):
        self.failing_download_keeps_cached_files(FakeResponse(b"<html>service unavailable</html>"))


class OfflineTests(CacheTestCase):
    def test_offline_miss_exits_5_when_required_and_is_none_when_optional(self):
        cache = Cache(self.root, offline=True, opener=FakeOpener())

        with self.assertRaises(SourceError) as ctx:
            cache.get(DFP_URL, "dfp.zip", required=True)
        self.assertEqual(ctx.exception.exit_code, 5)
        self.assertIsNone(cache.get(ITR_URL, "itr.zip", required=False))

    def test_offline_mode_answers_from_the_cache_with_zero_requests(self):
        Cache(self.root, False, FakeOpener({DFP_URL: FakeResponse(ZIP_BODY)})).get(DFP_URL, "dfp.zip")
        opener = FakeOpener()
        offline = Cache(self.root, offline=True, opener=opener)

        cached = offline.get(DFP_URL, "dfp.zip")
        missing = offline.get(ITR_URL, "itr.zip", required=False)
        size = offline.head_size(ITR_URL)

        self.assertEqual(cached.read_bytes(), ZIP_BODY)
        self.assertIsNone(missing)
        self.assertIsNone(size)
        self.assertEqual(opener.requests, [])


class HeadSizeTests(CacheTestCase):
    def test_head_size_records_content_length_and_offline_reads_it_back(self):
        opener = FakeOpener({DFP_URL: FakeResponse(headers={"Content-Length": "1234"})})
        cache = Cache(self.root, offline=False, opener=opener)

        self.assertEqual(cache.head_size(DFP_URL), 1234)
        self.assertEqual(opener.requests, [("HEAD", DFP_URL)])
        self.assertEqual(self.manifest()["sizes"], {DFP_URL: 1234})

        offline_opener = FakeOpener()
        self.assertEqual(Cache(self.root, offline=True, opener=offline_opener).head_size(DFP_URL), 1234)
        self.assertEqual(offline_opener.requests, [])


class RetrievedAtTests(CacheTestCase):
    """SHR-08 (B-3): the manifest's retrieved_at of a cached file is the moment R it was read; a name the manifest lacks has none."""

    def test_retrieved_at_is_the_value_the_manifest_holds_and_none_for_a_name_it_lacks(self):
        self.root.mkdir(parents=True)
        body = b"[]"
        entry = {
            "bytes": len(body),
            "name": "b3_company_SBSP.json",
            "retrieved_at": "2026-11-10T00:34:49Z",
            "sha256": hashlib.sha256(body).hexdigest(),
            "url": "https://example.test/b3_company_SBSP",
        }
        (self.root / entry["name"]).write_bytes(body)
        (self.root / "manifest.json").write_text(json.dumps({"files": {entry["name"]: entry}, "sizes": {}}), encoding="utf-8")
        cache = Cache(self.root, offline=True, opener=FakeOpener())

        self.assertEqual(cache.retrieved_at("b3_company_SBSP.json"), "2026-11-10T00:34:49Z")
        self.assertIsNone(cache.retrieved_at("b3_company_ALFA.json"))


if __name__ == "__main__":
    unittest.main()
