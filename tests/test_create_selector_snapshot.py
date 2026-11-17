"""Snapshot creation of the selector CSVs (issue #80, spec #75): stocks from CVM and B3, FIIs from Status Invest.

The stock CSV comes from the CVM/B3 collector over the rebuild's fixture universe (tests/fixtures/pit_rebuild/universe
with the overlay tests/fixtures/stock_collector), cached once by an online run against a fake opener. Status Invest is
never called for stocks. The FII CSV still comes from Status Invest (a fake collector here) and is filtered by the B3
recent-trade rule of #63 at the cutoff date, since Premium runs no longer filter (#17). No test touches the network.
"""

import csv
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "tests", PROJECT_ROOT / "api", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import create_selector_snapshot as creation  # noqa: E402
from app.services.b3_trading import CotahistCache  # noqa: E402
from fetch_status_invest_fii import OUTPUT_COLUMNS as FII_COLUMNS  # noqa: E402
from pit_rebuild import collector  # noqa: E402
from pit_rebuild_support import UNIVERSE, FakeOpener  # noqa: E402
from test_stock_collector import OVERLAY, fixture_routes  # noqa: E402

_spec = importlib.util.spec_from_file_location("cotahist_fixture", PROJECT_ROOT / "api" / "tests" / "cotahist_fixture.py")
cotahist_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cotahist_fixture)

D1 = date(2026, 8, 7)


def no_network(request, timeout=None):
    raise AssertionError(f"a request was made while --offline is set: {request}")


def fii_writer(tickers):
    def collect(*, output_path, timeout):
        with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FII_COLUMNS)
            writer.writeheader()
            for ticker in tickers:
                writer.writerow({column: ticker if column == "TICKER" else "1" for column in FII_COLUMNS})

    return collect


def tickers_of(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["TICKER"] for row in csv.DictReader(handle)]


class CreationCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        universe = root / "universe"
        shutil.copytree(UNIVERSE, universe)
        shutil.copytree(OVERLAY, universe, dirs_exist_ok=True)
        cls.cache = root / "cache"
        code = collector.main(
            ["--date", D1.isoformat(), "--out", str(root / "warm" / "stocks.csv"), "--cache-dir", str(cls.cache)],
            opener=FakeOpener(fixture_routes(universe)),
        )
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.out = self.root / "selectors"

    def fii_cotahist(self, lines):
        data = cotahist_fixture.cotahist_zip(lines)
        return CotahistCache(self.root / "cotahist", downloader=lambda _url: data)

    def create(self, *, fiis=("FUND11",), lines=None, extra=()):
        lines = lines if lines is not None else [cotahist_fixture.trade("FUND11", D1, codbdi="12")]
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = creation.main(
                ["--date", D1.isoformat(), "--out-dir", str(self.out), "--cache-dir", str(self.cache), "--offline", *extra],
                opener=no_network,
                fii_collector=fii_writer(fiis),
                cotahist_cache=self.fii_cotahist(lines),
            )
        return code, err.getvalue()


class StockSourceTests(CreationCase):
    def test_the_stock_csv_is_the_cvm_b3_collector_csv_and_status_invest_is_never_called_for_stocks(self):
        import fetch_status_invest

        live = AssertionError("Status Invest called for stocks")
        with mock.patch.object(fetch_status_invest, "refresh", side_effect=live) as status_invest, \
                mock.patch.object(fetch_status_invest, "fetch_sector", side_effect=live) as sector:
            code, err = self.create()

        self.assertEqual(code, 0, err)
        status_invest.assert_not_called()
        sector.assert_not_called()
        expected = self.root / "direct" / "stocks.csv"
        self.assertEqual(
            collector.main(["--date", D1.isoformat(), "--out", str(expected), "--cache-dir", str(self.cache), "--offline"],
                           opener=no_network),
            0,
        )
        self.assertEqual((self.out / "stocks.csv").read_bytes(), expected.read_bytes())
        provenance = json.loads((self.out / "stocks.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["provider"], "cvm-b3")
        self.assertEqual(provenance["as_of"], D1.isoformat())

    def test_a_stock_collection_failure_writes_nothing_and_never_falls_back_to_status_invest(self):
        import fetch_status_invest

        with mock.patch.object(fetch_status_invest, "refresh") as status_invest:
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = creation.main(
                    ["--date", D1.isoformat(), "--out-dir", str(self.out), "--cache-dir", str(self.root / "absent"), "--offline"],
                    opener=no_network,
                    fii_collector=fii_writer(["FUND11"]),
                    cotahist_cache=self.fii_cotahist([]),
                )

        self.assertEqual(code, 3, err.getvalue())
        status_invest.assert_not_called()
        self.assertFalse(self.out.exists())


class FiiFilterTests(CreationCase):
    def test_a_fii_without_a_session_in_the_30_days_up_to_the_cutoff_is_left_out_and_recorded(self):
        code, err = self.create(
            fiis=("FUND11", "EDGE11", "OLD11", "DEAD11"),
            lines=[
                cotahist_fixture.trade("FUND11", D1, codbdi="12"),
                cotahist_fixture.trade("EDGE11", D1 - timedelta(days=30), codbdi="12"),
                cotahist_fixture.trade("OLD11", D1 - timedelta(days=31), codbdi="12"),
            ],
        )

        self.assertEqual(code, 0, err)
        self.assertEqual(tickers_of(self.out / "fiis.csv"), ["FUND11", "EDGE11"])
        provenance = json.loads((self.out / "fiis.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["provider"], "statusinvest")
        record = provenance["recent_trade_filter"]
        self.assertEqual(record["collected_on"], D1.isoformat())
        self.assertEqual(
            [(item["ticker"], item["last_trade"]) for item in record["removed"]["fiis"]],
            [("DEAD11", "nenhum"), ("OLD11", (D1 - timedelta(days=31)).isoformat())],
        )
        self.assertNotIn("stocks", record["removed"])

    def test_a_status_invest_fii_failure_exits_6_and_writes_nothing(self):
        def broken(*, output_path, timeout):
            raise RuntimeError("Status Invest is down")

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = creation.main(
                ["--date", D1.isoformat(), "--out-dir", str(self.out), "--cache-dir", str(self.cache), "--offline"],
                opener=no_network,
                fii_collector=broken,
                cotahist_cache=self.fii_cotahist([]),
            )

        self.assertEqual(code, 6, err.getvalue())
        self.assertFalse(self.out.exists())


class RegistryTests(CreationCase):
    def test_the_created_csvs_build_a_manifest_with_cvm_b3_stocks_and_status_invest_fiis(self):
        import snapshot_registry
        from snapshot_manifest import SnapshotManifest, validate_manifest
        from test_snapshot_registry import CREATED_AT, build_fetch_inputs

        registry = self.root
        snapshot_dir, _, _ = build_fetch_inputs(registry)
        code, err = self.create()
        self.assertEqual(code, 0, err)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = snapshot_registry.main([
                "build", "--registry", str(registry), "--manifest-id", "created", "--created-at", CREATED_AT,
                "--snapshot-dir", str(snapshot_dir), "--stock-universe", str(self.out / "stocks.csv"),
                "--fii-universe", str(self.out / "fiis.csv"),
                "--stock-provenance", str(self.out / "stocks.provenance.json"),
                "--fii-provenance", str(self.out / "fiis.provenance.json"),
            ])
        self.assertEqual(code, 0)
        manifest = json.loads((registry / "created.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sources"]["stock"]["provider"], "cvm-b3")
        self.assertIn("b3_detail_1001.json", manifest["sources"]["stock"]["metadata"]["raw_files"]["classificacao_setorial"])
        fii = manifest["sources"]["fii"]
        self.assertEqual(fii["provider"], "statusinvest")
        self.assertEqual(fii["metadata"]["recent_trade_filter"]["collected_on"], D1.isoformat())
        self.assertTrue(fii["metadata"]["retrieved_at"])
        validate_manifest(SnapshotManifest.load(registry / "created.json"))

    def test_build_refuses_a_fii_provenance_written_for_another_csv(self):
        import snapshot_registry
        from test_snapshot_registry import CREATED_AT, build_fetch_inputs

        registry = self.root
        snapshot_dir, _, _ = build_fetch_inputs(registry)
        code, err = self.create()
        self.assertEqual(code, 0, err)
        with (self.out / "fiis.csv").open("a", encoding="utf-8") as handle:
            handle.write("XTRA11" + "," * (len(FII_COLUMNS) - 1) + "\n")
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = snapshot_registry.main([
                "build", "--registry", str(registry), "--manifest-id", "created", "--created-at", CREATED_AT,
                "--snapshot-dir", str(snapshot_dir), "--stock-universe", str(self.out / "stocks.csv"),
                "--fii-universe", str(self.out / "fiis.csv"),
                "--fii-provenance", str(self.out / "fiis.provenance.json"),
            ])
        self.assertEqual(code, 4)
        self.assertIn("FII provenance", err.getvalue())
        self.assertFalse((registry / "created.json").exists())


if __name__ == "__main__":
    unittest.main()
