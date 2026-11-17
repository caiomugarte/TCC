"""The stock CSV collector from CVM and B3 end to end on the rebuild's universe fixture (issue #78).

The universe is tests/fixtures/pit_rebuild/universe with the overlay tests/fixtures/stock_collector on top:

- b3/detail: the B3 sector classification of each company (GetDetail); the record of COBR (1004) has no classification.
- b3/company_AZUL.json: AZUL4 split 1:2 (DESDOBRAMENTO 100 %) with ex-date in June 2026; the B3 total is 40 M.
  The DFP 2025 files 20 M PN shares, LL 30 M and PL 400 M, so at 2026-08-07 the count is 40 M:
  LPA 0.75, VPA 10.00, P/L 15.00 / 0.75 = 20.00 and VALOR DE MERCADO 40 M x 15.00 = 600 M.
- dfp/2025: the DRE and BPP of the universe with BANC (1007) in the bank layout (net income 3.09, equity 2.08, #80).
- dfp/2020: the fiscal-year 2020 net income of SHRT (937 500) and NOBX (80 M), for CAGR LUCROS 5 ANOS.
  SHRT: LL 30 M / 937 500 = 32 = 2^5, so the CAGR is 100 % and PEG = P/L 10.00 / 100 = 0.10.
  NOBX: LL 40 M < 80 M, so the CAGR is negative and PEG is empty.

ALFA (DFP 2025): EBIT 200 M, PL 2 000 M, debt 300 M + 200 M, cash 100 M + 50 M, so
ROIC = 100 x 200 M x 0.66 / (2 000 M + 500 M - 150 M) = 5.617 %.

At 2026-08-07 the newest filing (2025-12-31) is 219 days old; at 2026-11-06 it is 310 days old, past the
270-day limit, so every accounting field is empty while the market fields and DY keep their value.

A cache is filled once per class by an online run against a fake opener; every test then runs offline.
"""

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from fetch_status_invest import OUTPUT_COLUMNS  # noqa: E402
from pit_rebuild import collector, sources  # noqa: E402
from pit_rebuild_support import UNIVERSE, FakeOpener, FakeResponse, universe_routes  # noqa: E402

OVERLAY = PROJECT_ROOT / "tests" / "fixtures" / "stock_collector"
D1 = date(2026, 8, 7)
D2 = date(2026, 11, 6)


def no_network(request, timeout=None):
    raise AssertionError(f"a request was made while --offline is set: {request}")


def fixture_routes(universe: Path) -> dict:
    routes = universe_routes([D1, D2], universe)
    for path in sorted((universe / "b3" / "detail").glob("detail_*.json")):
        code = path.stem[len("detail_"):]
        body = path.read_bytes()
        routes[sources.b3_detail_url(code)] = FakeResponse(body, {"Content-Length": str(len(body))})
    return routes


def run(argv, opener=no_network) -> tuple[int, str]:
    err = io.StringIO()
    with redirect_stderr(err):
        code = collector.main(argv, opener=opener)
    return code, err.getvalue()


def read_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), {row["TICKER"]: row for row in reader}


class CollectorCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        universe = root / "universe"
        shutil.copytree(UNIVERSE, universe)
        shutil.copytree(OVERLAY, universe, dirs_exist_ok=True)
        cls.cache = root / "cache"
        cls.out = root / "out"
        cls.out.mkdir()
        fake = FakeOpener(fixture_routes(universe))
        for day in (D1, D2):
            code, err = run(
                ["--date", day.isoformat(), "--out", str(cls.out / f"online-{day}.csv"), "--cache-dir", str(cls.cache)],
                opener=fake,
            )
            assert code == 0, err

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def collect(self, day: date) -> tuple[list[str], dict[str, dict[str, str]], Path]:
        target = self.out / f"stocks-{day}.csv"
        code, err = run(["--date", day.isoformat(), "--out", str(target), "--cache-dir", str(self.cache), "--offline"])
        self.assertEqual(code, 0, err)
        header, rows = read_csv(target)
        return header, rows, target


class ColumnsAndScaleTests(CollectorCase):
    def test_the_csv_has_the_status_invest_columns_plus_subsetor_in_the_same_scale(self):
        header, rows, _ = self.collect(D1)

        self.assertEqual(header, [*OUTPUT_COLUMNS, "SUBSETOR"])
        alfa = rows["ALFA3"]
        self.assertAlmostEqual(float(alfa["PRECO"]), 10.00)
        self.assertAlmostEqual(float(alfa["VALOR DE MERCADO"]), 1_100_000_000)
        self.assertAlmostEqual(float(alfa["LIQUIDEZ MEDIA DIARIA"]), 1_000_000)
        self.assertAlmostEqual(float(alfa["P/L"]), 10.00)
        self.assertAlmostEqual(float(alfa["ROE"]), 5.00)  # percent: 100 M / 2 000 M
        self.assertAlmostEqual(float(alfa["MARGEM EBIT"]), 20.00)  # percent: 200 M / 1 000 M
        self.assertAlmostEqual(float(alfa["LIQ. CORRENTE"]), 2.00)  # ratio: 1 200 M / 600 M
        self.assertAlmostEqual(float(alfa["PATRIMONIO / ATIVOS"]), 2 / 3, places=4)  # ratio, as Status Invest
        self.assertAlmostEqual(float(rows["SOLO3"]["DY"]), 5.00)  # percent
        self.assertEqual(alfa["SETOR"], "Materiais Básicos")
        self.assertEqual(alfa["SUBSETOR"], "Siderurgia e Metalurgia")


class FormulaTests(CollectorCase):
    def test_roic_is_ebit_after_34_percent_over_controllers_equity_plus_debt_minus_cash(self):
        _, rows, _ = self.collect(D1)

        self.assertAlmostEqual(float(rows["ALFA3"]["ROIC"]), 5.617021, places=5)

    def test_peg_is_pl_over_the_5_year_earnings_cagr_and_empty_when_the_cagr_is_not_positive(self):
        _, rows, _ = self.collect(D1)

        self.assertAlmostEqual(float(rows["SHRT3"]["CAGR LUCROS 5 ANOS"]), 100.0, places=6)
        self.assertAlmostEqual(float(rows["SHRT3"]["PEG RATIO"]), 0.10, places=6)
        self.assertLess(float(rows["NOBX3"]["CAGR LUCROS 5 ANOS"]), 0)
        self.assertEqual(rows["NOBX3"]["PEG RATIO"], "")
        self.assertEqual(rows["ALFA3"]["PEG RATIO"], "")  # no fiscal year 2020: no CAGR


class PegUnitTests(unittest.TestCase):
    """The P/L guard of PEG RATIO: a loss-making company has no PEG even with a positive earnings CAGR (spec #75).

    The fixture universe cannot hold that row (a negative LL gives no positive CAGR LUCROS of the same LL), so the
    guard is checked on the function the rows use."""

    @staticmethod
    def peg(p_l, cagr):
        from pit_rebuild.indicators import Value

        return collector._peg({"P/L": Value(p_l), "CAGR LUCROS 5 ANOS": Value(cagr)})

    def test_a_negative_pl_with_a_positive_cagr_has_no_peg(self):
        self.assertIsNone(self.peg(-8.0, 25.0))

    def test_a_positive_pl_with_a_positive_cagr_is_pl_over_cagr(self):
        self.assertAlmostEqual(self.peg(10.0, 25.0), 0.4)

    def test_a_zero_or_negative_cagr_has_no_peg(self):
        self.assertIsNone(self.peg(10.0, 0.0))
        self.assertIsNone(self.peg(10.0, -5.0))


NOT_APPLICABLE_TO_FINANCIALS = (
    "EV/EBIT", "MARGEM EBIT", "ROIC", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.", "LIQ. CORRENTE", "PSR",
)


class FinancialTests(CollectorCase):
    def test_a_bank_has_no_indicator_that_does_not_apply_and_has_equity_over_assets(self):
        # BANC (1007) is "Financeiro / Intermediários Financeiros / Bancos"; its DFP 2025 has AT 9 000 M,
        # PL 1 500 M, EBIT and current assets, so every one of these columns would have a value.
        _, rows, _ = self.collect(D1)

        banc = rows["BANC3"]
        self.assertEqual({column: banc[column] for column in NOT_APPLICABLE_TO_FINANCIALS},
                         dict.fromkeys(NOT_APPLICABLE_TO_FINANCIALS, ""))
        self.assertAlmostEqual(float(banc["PATRIMONIO / ATIVOS"]), 1_500 / 9_000, places=6)
        self.assertAlmostEqual(float(banc["P/L"]), 10.00)
        self.assertNotEqual(banc["ROE"], "")
        self.assertNotEqual(banc["MARG. LIQUIDA"], "")

    def test_a_bank_filing_in_the_bank_layout_is_read_from_its_own_accounts(self):
        # The overlay files BANC's DFP 2025 in the bank layout of ITUB and SANB (#80): net income in 3.09 (controllers
        # 3.09.01 = 100 M of 120 M), equity in 2.08 (1 500 M, 100 M of minority in 2.08.09), and 2.03 holding the
        # amortised-cost liabilities (5 000 M). Price 10.00 x 100 M shares = 1 000 M of market value.
        _, rows, _ = self.collect(D1)

        banc = rows["BANC3"]
        self.assertAlmostEqual(float(banc["P/L"]), 10.00)  # 1 000 M / 100 M
        self.assertAlmostEqual(float(banc["ROE"]), 100 * 100 / 1_400, places=5)  # LL / PL of the controllers
        self.assertAlmostEqual(float(banc["ROA"]), 100 * 100 / 9_000, places=5)
        self.assertAlmostEqual(float(banc["MARG. LIQUIDA"]), 100 * 100 / 900, places=5)
        self.assertAlmostEqual(float(banc["P/VP"]), 1_000 / 1_400, places=5)
        self.assertAlmostEqual(float(banc["PATRIMONIO / ATIVOS"]), 1_500 / 9_000, places=6)

    def test_a_common_stock_keeps_every_indicator(self):
        _, rows, _ = self.collect(D1)

        alfa = rows["ALFA3"]
        for column in NOT_APPLICABLE_TO_FINANCIALS:
            self.assertNotEqual(alfa[column], "", column)


class ShareCountTests(CollectorCase):
    def test_the_filed_count_moves_with_the_b3_split_between_the_period_end_and_the_cutoff(self):
        _, rows, _ = self.collect(D1)

        azul = rows["AZUL4"]
        self.assertAlmostEqual(float(azul["LPA"]), 0.75)  # 30 M / (20 M x 2)
        self.assertAlmostEqual(float(azul["VPA"]), 10.00)  # 400 M / 40 M
        self.assertAlmostEqual(float(azul["P/L"]), 20.00)
        self.assertAlmostEqual(float(azul["VALOR DE MERCADO"]), 600_000_000)


MARKET_COLUMNS =("PRECO", "DY", "LIQUIDEZ MEDIA DIARIA", "VALOR DE MERCADO")


class StaleFilingTests(CollectorCase):
    def test_a_filing_older_than_270_days_leaves_every_accounting_column_empty(self):
        header, rows, _ = self.collect(D2)

        shrt = rows["SHRT3"]
        accounting = [column for column in header if column not in ("TICKER", "SETOR", "SUBSETOR", *MARKET_COLUMNS)]
        self.assertEqual({column: shrt[column] for column in accounting}, dict.fromkeys(accounting, ""))
        self.assertAlmostEqual(float(shrt["PRECO"]), 10.00)
        self.assertAlmostEqual(float(shrt["VALOR DE MERCADO"]), 300_000_000)
        self.assertAlmostEqual(float(shrt["DY"]), 8.00)
        self.assertEqual(shrt["SETOR"], "Consumo Cíclico")


class SelectorTests(CollectorCase):
    def test_the_stock_selector_ranks_the_generated_csv_without_code_changes(self):
        from config import PROFILE_WEIGHTS
        from pipelines.single_run import run_stock_selection

        _, rows, target = self.collect(D1)
        config = {
            "selection_preset": "fixture",
            "factor_weights": dict(PROFILE_WEIGHTS["caio_new"]),
            "liquidity_and_size_filters": {"cap_min": 200_000_000, "liq_min": 50_000},
            "system_ga_config": {"n_assets": 2, "lambda": 0.0, "generations": 3, "pop_size": 4},
        }
        with redirect_stdout(io.StringIO()):
            result = run_stock_selection(input_path=target, config=config, seed=7)

        # Only SHRT and NOBX have the 5-year history of the growth group (the DFP 2020 overlay); the other
        # tickers leave by the selector's own rules (#52), so the ranking is exactly these two.
        ranked = result["ranked"]
        self.assertEqual(sorted(ranked["TICKER"]), ["NOBX3", "SHRT3"])
        self.assertFalse(ranked["SCORE"].isna().any())
        self.assertEqual(result["selected_tickers"], ["NOBX3", "SHRT3"])
        self.assertTrue(set(result["exclusion_reasons"]) <= set(rows))


class CommandLineTests(CollectorCase):
    def test_the_package_entry_point_collects_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "stocks.csv"
            done = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "py" / "pit_rebuild"), "collect", "--date", D1.isoformat(),
                 "--out", str(target), "--cache-dir", str(self.cache), "--offline"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(read_csv(target)[0], [*OUTPUT_COLUMNS, "SUBSETOR"])
            self.assertTrue((Path(directory) / "stocks.provenance.json").is_file())

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = str(root / "stocks.csv")
            tampered = root / "tampered"
            shutil.copytree(self.cache, tampered)
            (tampered / "cad_cia_aberta.csv").write_bytes(b"changed")
            (root / "empty").mkdir()
            (root / "blocker").write_bytes(b"a file where the output directory should go")
            cases = {
                1: ["--date", D1.isoformat(), "--out", str(root / "blocker" / "stocks.csv"), "--cache-dir", str(self.cache), "--offline"],
                2: ["--date", "2026-02-30", "--out", out, "--cache-dir", str(self.cache), "--offline"],
                3: ["--date", D1.isoformat(), "--out", out, "--cache-dir", str(root / "absent"), "--offline"],
                4: ["--date", D1.isoformat(), "--out", out, "--cache-dir", str(tampered), "--offline"],
                5: ["--date", D1.isoformat(), "--out", out, "--cache-dir", str(root / "empty"), "--offline"],
            }
            for expected, argv in cases.items():
                with self.subTest(code=expected):
                    code, err = run(argv)
                    self.assertEqual(code, expected, err)
                    self.assertEqual(len(err.strip().splitlines()), 1, err)
            self.assertFalse((root / "stocks.csv").exists())

    def test_offline_without_the_b3_classification_of_a_company_exits_5_and_writes_nothing(self):
        # #80: an offline run must not leave SETOR and SUBSETOR empty because the classification was never fetched.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            shutil.copytree(self.cache, cache)
            (cache / collector.detail_name("1001")).unlink()

            code, err = run(["--date", D1.isoformat(), "--out", str(root / "stocks.csv"), "--cache-dir", str(cache), "--offline"])

            self.assertEqual(code, 5, err)
            self.assertIn("GetDetail", err)
            self.assertFalse((root / "stocks.csv").exists())


class UniverseTests(CollectorCase):
    def test_a_ticker_without_a_session_in_the_30_days_up_to_the_cutoff_is_not_in_the_csv(self):
        # The B3 recent-trade filter of #63 holds by construction: SEMP4's last session is 2026-05-29, 70 days
        # before D1, while SEMP3 traded on 2026-11-06. Premium runs no longer filter (#17), so the CSV must not carry it.
        _, rows, target = self.collect(D1)

        self.assertIn("SEMP3", rows)
        self.assertNotIn("SEMP4", rows)
        provenance = json.loads(collector.provenance_path(target).read_text(encoding="utf-8"))
        self.assertNotIn("SEMP4", provenance["excluded"])

    def test_a_company_that_b3_does_not_classify_has_setor_and_subsetor_empty(self):
        _, rows, _ = self.collect(D1)

        self.assertEqual((rows["COBR3"]["SETOR"], rows["COBR3"]["SUBSETOR"]), ("", ""))

    def test_every_row_without_a_classification_is_listed_in_the_provenance_with_its_reason(self):
        _, rows, target = self.collect(D1)

        provenance = json.loads(collector.provenance_path(target).read_text(encoding="utf-8"))
        self.assertEqual(provenance["unclassified"], {"COBR3": "no_classification"})
        self.assertEqual(
            sorted(provenance["unclassified"]), sorted(ticker for ticker, row in rows.items() if row["SETOR"] == ""),
        )


RAW_KINDS = {"dfp_itr", "cotahist", "proventos", "eventos", "classificacao_setorial"}


class ManifestTests(CollectorCase):
    def build(self, registry: Path, *extra: str) -> tuple[int, dict | None, str]:
        import snapshot_registry
        from test_snapshot_registry import CREATED_AT, build_fetch_inputs

        snapshot_dir, _, fii_universe = build_fetch_inputs(registry)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = snapshot_registry.main([
                "build", "--registry", str(registry), "--manifest-id", "cvm-b3", "--created-at", CREATED_AT,
                "--snapshot-dir", str(snapshot_dir), "--stock-universe", str(registry / "selectors" / "stocks.csv"),
                "--fii-universe", str(fii_universe), *extra,
            ])
        target = registry / "cvm-b3.json"
        return code, json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None, err.getvalue()

    def test_the_manifest_records_the_cvm_b3_provider_and_the_hash_of_every_raw_file_used(self):
        from snapshot_manifest import SnapshotManifest, validate_manifest

        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory)
            code, err = run(["--date", D1.isoformat(), "--out", str(registry / "selectors" / "stocks.csv"), "--cache-dir", str(self.cache), "--offline"])
            self.assertEqual(code, 0, err)
            code, manifest, err = self.build(registry, "--stock-provenance", str(registry / "selectors" / "stocks.provenance.json"))
            self.assertEqual(code, 0, err)
            stock = manifest["sources"]["stock"]
            self.assertEqual(stock["provider"], "cvm-b3")
            self.assertEqual(stock["metadata"]["as_of"], D1.isoformat())
            raw = stock["metadata"]["raw_files"]
            self.assertTrue(RAW_KINDS <= set(raw), raw.keys())
            cached = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))["files"]
            for kind in RAW_KINDS:
                for name, sha256 in raw[kind].items():
                    self.assertEqual(sha256, cached[name]["sha256"], name)
            self.assertIn("b3_detail_1001.json", raw["classificacao_setorial"])
            self.assertIn("b3_company_AZUL.json", raw["eventos"])
            validate_manifest(SnapshotManifest.load(registry / "cvm-b3.json"))

    def test_build_refuses_a_provenance_written_for_another_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory)
            code, err = run(["--date", D1.isoformat(), "--out", str(registry / "selectors" / "stocks.csv"), "--cache-dir", str(self.cache), "--offline"])
            self.assertEqual(code, 0, err)
            with (registry / "selectors" / "stocks.csv").open("a", encoding="utf-8") as handle:
                handle.write("XXXX3" + "," * 31 + "\n")
            code, manifest, err = self.build(registry, "--stock-provenance", str(registry / "selectors" / "stocks.provenance.json"))
            self.assertEqual(code, 4)
            self.assertIsNone(manifest)
            self.assertIn("provenance", err)


if __name__ == "__main__":
    unittest.main()
