import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(Path(__file__).parent))

from allocation_data import SnapshotError, load_premium_snapshot_bundle  # noqa: E402
from snapshot_manifest import SnapshotManifest  # noqa: E402
from test_snapshot_manifest import build_manifest_fixture  # noqa: E402


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

    def test_missing_premium_fii_input_does_not_use_ifix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_manifest_fixture(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            (root / "allocation" / "caio_fiis.csv").unlink()
            (root / "allocation" / "ifix.csv").write_text(
                "date,value\n2020-01-01,100\n2020-01-02,101\n2020-01-03,102\n",
                encoding="utf-8",
            )

            with self.assertRaises(SnapshotError):
                load_premium_snapshot_bundle(SnapshotManifest.load(manifest_path))


if __name__ == "__main__":
    unittest.main()
