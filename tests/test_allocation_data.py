import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(Path(__file__).parent))

from allocation_data import (  # noqa: E402
    SnapshotError,
    load_premium_snapshot_bundle,
    load_snapshot_bundle,
)
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


BASIC_DATES = ["2020-01-01", "2020-01-02", "2020-01-03"]


def _write_basic_snapshot(directory: Path) -> None:
    (directory / "metadata.json").write_text(
        json.dumps({
            "source": "fixture",
            "retrieved_at": "2026-07-21T12:00:00-03:00",
            "cutoff_date": BASIC_DATES[-1],
        }),
        encoding="utf-8",
    )
    levels = {
        "brazilian_stocks": (100, 120, 90),
        "fiis": (10, 11, 12.1),
        "international_equity": (100, 101, 102),
        "fixed_income": (100, 100.1, 100.2),
        "crypto": (100, 110, 120),
        "ptax": (5, 5, 5),
    }
    for key, values in levels.items():
        _write_series(directory / ALLOCATION_FILES[key], BASIC_DATES, {"value": values})


class BasicAllocationDataTests(unittest.TestCase):
    def test_basic_loader_reads_ibovespa_and_ifix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_basic_snapshot(root)

            bundle = load_snapshot_bundle(root)

        self.assertEqual([row.date.isoformat() for row in bundle.rows], BASIC_DATES)
        self.assertAlmostEqual(bundle.rows[1].returns["brazilian_stocks"], 0.20)
        self.assertAlmostEqual(bundle.rows[2].returns["brazilian_stocks"], -0.25)
        self.assertAlmostEqual(bundle.rows[1].returns["fiis"], 0.10)
        self.assertAlmostEqual(bundle.rows[2].returns["fiis"], 0.10)
        self.assertEqual(bundle.metadata["source"], "fixture")

    def test_basic_loader_ignores_caio_stocks_even_if_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_basic_snapshot(root)
            _write_series(root / "caio_stocks.csv", BASIC_DATES, {"AAA3": (1, 50, 2)})
            _write_series(root / "caio_fiis.csv", BASIC_DATES, {"AAA11": (1, 50, 2)})
            portfolio = root / "portfolio.json"
            portfolio.write_text(json.dumps([{"TICKER": "AAA3"}]), encoding="utf-8")

            bundle = load_snapshot_bundle(root, portfolio)

        self.assertAlmostEqual(bundle.rows[1].returns["brazilian_stocks"], 0.20)
        self.assertAlmostEqual(bundle.rows[1].returns["fiis"], 0.10)

    def test_basic_loader_missing_ibovespa_raises_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_basic_snapshot(root)
            _write_series(root / "caio_stocks.csv", BASIC_DATES, {"AAA3": (100, 101, 102)})
            (root / "ibovespa.csv").unlink()

            with self.assertRaisesRegex(SnapshotError, "ibovespa.csv"):
                load_snapshot_bundle(root)


if __name__ == "__main__":
    unittest.main()
