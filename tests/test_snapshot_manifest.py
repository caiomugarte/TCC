import json
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from allocation_config import ASSET_CLASSES  # noqa: E402
from snapshot_manifest import (  # noqa: E402
    SnapshotManifest,
    SnapshotManifestError,
    SnapshotRegistry,
    latest_compatible_manifest,
    sha256_path,
    validate_manifest,
)


def _write_levels(path: Path, columns: list[str], dates: list[str]) -> None:
    lines = ["date," + ",".join(columns)]
    for index, current_date in enumerate(dates, start=1):
        values = [str(100.0 + index + column_index) for column_index in range(len(columns))]
        lines.append(",".join([current_date, *values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_fixture(root: Path) -> dict[str, object]:
    allocation = root / "allocation"
    allocation.mkdir(parents=True, exist_ok=True)
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    _write_levels(allocation / "caio_stocks.csv", ["AAA3", "BBB3"], dates)
    _write_levels(allocation / "caio_fiis.csv", ["AAA11", "BBB11"], dates)
    _write_levels(allocation / "sp500_total_return_usd.csv", ["value"], dates)
    _write_levels(allocation / "di.csv", ["value"], dates)
    _write_levels(allocation / "btc_usd.csv", ["value"], dates)
    _write_levels(allocation / "ptax.csv", ["value"], dates)

    stock_source = root / "stocks.csv"
    fii_source = root / "fiis.csv"
    stock_source.write_text("TICKER\nAAA3\n", encoding="utf-8")
    fii_source.write_text("TICKER\nAAA11\n", encoding="utf-8")

    return {
        "manifest_id": "fixture-manifest",
        "manifest_version": "1",
        "created_at": "2020-01-04T12:00:00+00:00",
        "cutoff_date": dates[-1],
        "common_dates": dates,
        "supported_classes": list(ASSET_CLASSES),
        "allocation_source": {
            "path": "allocation",
            "sha256": sha256_path(allocation),
            "provider": "fixture-allocation",
        },
        "stock_source": {
            "path": "stocks.csv",
            "sha256": sha256_path(stock_source),
            "provider": "fixture-stock",
        },
        "fii_source": {
            "path": "fiis.csv",
            "sha256": sha256_path(fii_source),
            "provider": "fixture-fii",
        },
    }


class SnapshotManifestTests(unittest.TestCase):
    def test_registry_selects_latest_market_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir = Path(directory)
            payload = build_manifest_fixture(registry_dir)
            older = dict(payload)
            older["manifest_id"] = "older"
            older["common_dates"] = ["2020-01-01", "2020-01-02"]
            older["cutoff_date"] = "2020-01-02"
            newer = dict(payload)
            newer["manifest_id"] = "newer"
            (registry_dir / "older.json").write_text(
                json.dumps(older), encoding="utf-8"
            )
            (registry_dir / "newer.json").write_text(
                json.dumps(newer), encoding="utf-8"
            )

            selected = latest_compatible_manifest(registry_dir)

            self.assertEqual(selected.manifest_id, "newer")
            self.assertEqual(selected.cutoff_date, date(2020, 1, 3))
            self.assertEqual(
                SnapshotRegistry(registry_dir).latest_compatible().manifest_id,
                "newer",
            )

    def test_changed_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "stocks.csv").write_text("TICKER\nCHANGED\n", encoding="utf-8")

            with self.assertRaisesRegex(SnapshotManifestError, "SHA-256 mismatch"):
                validate_manifest(manifest_path)

    def test_unsupported_class_is_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            payload["supported_classes"] = list(ASSET_CLASSES[:-1])

            with self.assertRaisesRegex(SnapshotManifestError, "does not support classes"):
                validate_manifest(payload, verify_hashes=False)

    def test_selector_sources_cannot_be_derived_from_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            del payload["stock_source"]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                SnapshotManifestError, "explicit stock selector source"
            ):
                validate_manifest(manifest_path, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
