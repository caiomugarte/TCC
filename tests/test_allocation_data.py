import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(Path(__file__).parent))

from allocation_data import SnapshotError, load_premium_snapshot_bundle  # noqa: E402
from snapshot_manifest import (  # noqa: E402
    ALLOCATION_FILES,
    SnapshotManifest,
    sha256_path,
)
from test_snapshot_manifest import build_manifest_fixture  # noqa: E402


def _write_series(path: Path, dates: list, columns: dict) -> None:
    names = list(columns)
    lines = ["date," + ",".join(names)]
    for index, current_date in enumerate(dates):
        lines.append(
            ",".join([current_date, *(str(columns[name][index]) for name in names)])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PremiumAllocationDataTests(unittest.TestCase):
    def test_premium_loader_returns_brl_rows_and_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            bundle = load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))

            self.assertEqual(bundle.manifest_id, "fixture-manifest")
            self.assertEqual(bundle.start_date.isoformat(), "2020-01-01")
            self.assertEqual(bundle.end_date.isoformat(), "2020-01-03")
            self.assertEqual(
                tuple(row.date.isoformat() for row in bundle.rows),
                tuple(payload["common_dates"]),
            )
            self.assertEqual(
                set(bundle.rows[0].returns),
                {
                    "brazilian_stocks",
                    "fiis",
                    "international_equity",
                    "fixed_income",
                    "crypto",
                },
            )
            self.assertIn("allocation", bundle.source_metadata)

    def test_missing_premium_index_input_raises_without_substitution(self):
        cases = (
            ("ifix.csv", "caio_fiis.csv", "fiis"),
            ("ibovespa.csv", "caio_stocks.csv", "brazilian_stocks"),
        )
        for missing, basket, asset_class in cases:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = build_manifest_fixture(root)
                allocation = root / "allocation"
                (allocation / missing).unlink()
                self.assertTrue((allocation / basket).is_file())
                # Hash taken after the deletion: the hash matches, so the loader
                # can only fail on the missing index, not substitute the basket.
                payload["allocation_source"]["sha256"] = sha256_path(allocation)
                manifest_path = root / "manifest.json"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(SnapshotError) as caught:
                    load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))

                self.assertIn(f"source {asset_class} not found", str(caught.exception))

    def test_premium_returns_follow_index_and_class_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            allocation = root / "allocation"
            dates = payload["common_dates"]
            _write_series(allocation / "ibovespa.csv", dates, {"value": (100, 110, 99)})
            _write_series(allocation / "ifix.csv", dates, {"value": (50, 55, 44)})
            _write_series(
                allocation / "sp500_total_return_usd.csv",
                dates,
                {"value": (100, 102, 102)},
            )
            _write_series(allocation / "ptax.csv", dates, {"value": (5.0, 5.0, 5.5)})
            _write_series(allocation / "di.csv", dates, {"value": (1.0, 1.0004, 1.0008)})
            _write_series(allocation / "btc_usd.csv", dates, {"value": (1000, 900, 900)})
            # Baskets that would give other numbers if the loader read them.
            _write_series(
                allocation / "caio_stocks.csv",
                dates,
                {"AAA3": (10, 30, 30), "BBB3": (10, 10, 40)},
            )
            _write_series(
                allocation / "caio_fiis.csv",
                dates,
                {"AAA11": (10, 20, 20), "BBB11": (10, 10, 5)},
            )
            payload["allocation_source"]["sha256"] = sha256_path(allocation)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            bundle = load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))

            self.assertEqual(
                tuple(row.date.isoformat() for row in bundle.rows), tuple(dates)
            )
            expected = {
                "brazilian_stocks": (0.10, -0.10),
                "fiis": (0.10, -0.20),
                "international_equity": (0.02, 0.10),
                "fixed_income": (0.0004, 0.0004 / 1.0004),
                "crypto": (-0.10, 0.10),
            }
            for key, (first, second) in expected.items():
                with self.subTest(key=key):
                    self.assertAlmostEqual(bundle.rows[1].returns[key], first, delta=1e-12)
                    self.assertAlmostEqual(bundle.rows[2].returns[key], second, delta=1e-12)

    def test_premium_loader_rejects_caio_series_manifest(self):
        cases = (
            ("fiis", "ifix.csv", "caio_fiis.csv"),
            ("brazilian_stocks", "ibovespa.csv", "caio_stocks.csv"),
        )
        for key, expected, declared in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = build_manifest_fixture(root)
                payload["allocation_source"]["metadata"] = {
                    "files": {**ALLOCATION_FILES, key: declared}
                }
                manifest_path = root / "manifest.json"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(SnapshotError) as caught:
                    load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))

                self.assertIn(
                    f"source {key} must be {expected}; manifest declares {declared}",
                    str(caught.exception),
                )

    def test_premium_loader_rejects_multi_series_index_file(self):
        cases = (
            ("ibovespa.csv", {"AAA3": (100, 101, 102), "BBB3": (100, 99, 98)}),
            ("ifix.csv", {"AAA11": (100, 101, 102), "BBB11": (100, 99, 98)}),
        )
        for filename, columns in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = build_manifest_fixture(root)
                allocation = root / "allocation"
                _write_series(allocation / filename, payload["common_dates"], columns)
                payload["allocation_source"]["sha256"] = sha256_path(allocation)
                manifest_path = root / "manifest.json"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(SnapshotError):
                    load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))


if __name__ == "__main__":
    unittest.main()
