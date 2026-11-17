"""The FII measurement end to end on the fixture universe (issue #60): files, values, verdicts, exit codes. No network.

tests/fixtures/pit_rebuild_fii holds four eligible funds, each with round numbers:

- ALFA11: every indicator from the reports and COTAHIST; B3 starts 2025-11, so DY and the dividend CAGR fall back to
  Yahoo (marked). On 2026-11-05 the reference CAGR DIVIDENDOS 3 ANOS (6.27) differs from the rebuilt 7.72.
- BETA11: its 2026-06 report (version 2) arrives 2026-09-01, after the first date; empty B3 page and no Yahoo.
- AMBI11: two funds report its ISIN; the B3 quota count picks the right one. Segment "Outros" has no counterpart.
- NOCV11: no report gives its ISIN (no_cvm_match); Yahoo only.
"""

import base64
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from pit_rebuild import fii_cli  # noqa: E402
from pit_rebuild_support import FakeOpener, FakeResponse, make_zip  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild_fii"
D1, D2 = date(2026, 8, 10), date(2026, 11, 5)
SI1, SI2 = FIXTURES / "si_2026-08-10.csv", FIXTURES / "si_2026-11-05.csv"
COLLECTED2 = date(2026, 11, 6)
ISINS = {"ALFA11": "BRALFACTF001", "BETA11": "BRBETACTF002", "AMBI11": "BRAMBICTF003", "NOCV11": "BRNOCVCTF004"}


def cotahist_line(day: date, ticker: str, close: float, voltot: float, isin: str) -> str:
    line = [" "] * 245
    def put(start: int, text: str) -> None:
        line[start - 1 : start - 1 + len(text)] = list(text)
    put(1, "01")
    put(3, day.strftime("%Y%m%d"))
    put(11, "12")
    put(13, ticker.ljust(12))
    put(25, "010")
    put(40, "CI".ljust(10))
    put(109, f"{round(close * 100):013d}")
    put(171, f"{round(voltot * 100):018d}")
    put(211, "0000001")
    put(231, isin)
    return "".join(line)


def close_of(ticker: str, day: date) -> tuple[float, float]:
    if ticker == "ALFA11":
        close = 121.0 if day == D1 else 125.0 if day == D2 else 120.0
        return close, 1_100_000.0 if day >= date(2026, 10, 15) else 1_000_000.0
    if ticker == "BETA11":
        return (50.0 if day <= D1 else 55.0), 200_000.0
    if ticker == "AMBI11":
        return 80.0, 300_000.0
    return 10.0, 100_000.0


def cotahist_2026() -> bytes:
    lines = []
    day = date(2026, 7, 1)
    while day <= date(2026, 11, 6):
        for ticker, isin in ISINS.items():
            close, voltot = close_of(ticker, day)
            lines.append(cotahist_line(day, ticker, close, voltot, isin))
        day += timedelta(days=1)
    return make_zip({"COTAHIST_A2026.TXT": "\n".join(lines) + "\n"})


def folder_zip(folder: Path) -> bytes:
    return make_zip({path.name: path.read_bytes() for path in sorted(folder.iterdir())})


def routes() -> dict:
    from pit_rebuild import fii_sources, sources

    served = {}
    def serve(url: str, body: bytes) -> None:
        served[url] = FakeResponse(body, {"Content-Length": str(len(body))})
    years = (2023, 2026)
    listing = "".join(f'<a href="inf_mensal_fii_{year}.zip">inf_mensal_fii_{year}.zip</a>\n' for year in years)
    serve(fii_sources.INFORME_LISTING, f"<html><pre>{listing}</pre></html>".encode())
    for year in years:
        serve(fii_sources.informe_url(year), folder_zip(FIXTURES / "informes" / str(year)))
    serve(sources.cotahist_url(2026), cotahist_2026())
    serve(sources.cotahist_url(2025), make_zip({"COTAHIST_A2025.TXT": ""}))
    for path in sorted((FIXTURES / "b3").glob("fund_*.json")):
        serve(fii_sources.b3_fund_url(path.stem[len("fund_"):]), path.read_bytes())
    for path in sorted((FIXTURES / "yahoo").glob("*.json")):
        serve(sources.yahoo_url(path.stem, COLLECTED2 + timedelta(days=1)), path.read_bytes())
    return served


def args(cache: Path, out: Path, *extra: str) -> list[str]:
    return [
        "fii-report", "--date", D1.isoformat(), "--si", str(SI1), "--date", D2.isoformat(), "--si", str(SI2),
        "--collected-on", D1.isoformat(), "--collected-on", COLLECTED2.isoformat(),
        "--cache-dir", str(cache), "--out-dir", str(out), *extra,
    ]


class FiiReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.cache, cls.out = root / "cache", root / "out"
        cls.opener = FakeOpener(routes())
        cls.code = fii_cli.main(args(cls.cache, cls.out), opener=cls.opener)
        cls.report = json.loads((cls.out / fii_cli.REPORT_JSON).read_text(encoding="utf-8"))
        cls.markdown = (cls.out / fii_cli.REPORT_MD).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ticker(self, day: date, ticker: str) -> dict:
        return self.report["dates"][day.isoformat()]["tickers"][ticker]

    def test_the_report_is_written(self):
        self.assertEqual(self.code, 0)
        self.assertEqual(self.report["schema"], fii_cli.SCHEMA)
        self.assertEqual(self.report["dates"][D1.isoformat()]["eligible"], 4)

    def test_alfa_is_rebuilt_from_the_report_known_at_each_date(self):
        first, second = self.ticker(D1, "ALFA11"), self.ticker(D2, "ALFA11")
        self.assertEqual(first["informe"], "informe 2026-06 v1, entregue em 2026-07-15")
        self.assertEqual(second["informe"], "informe 2026-09 v1, entregue em 2026-10-15")
        self.assertAlmostEqual(first["values"]["P/VP"], 121 / 133.1)
        self.assertAlmostEqual(second["values"]["PERCENTUAL EM CAIXA"], 10.0)
        self.assertEqual(second["values"]["LIQUIDEZ MEDIA DIARIA"], 1_100_000.0)
        self.assertEqual(first["sources"]["DY"], ["yahoo", "b3_short_history"])

    def test_beta_waits_for_the_report_delivered_after_the_first_date(self):
        self.assertEqual(self.ticker(D1, "BETA11")["informe"], "informe 2026-05 v1, entregue em 2026-06-15")
        self.assertEqual(self.ticker(D1, "BETA11")["delivered_after"], ["2026-06-01"])
        self.assertEqual(self.ticker(D1, "BETA11")["reasons"]["DY"], "b3_empty_yahoo_failed")

    def test_the_isin_reported_by_two_funds_goes_to_the_one_with_the_b3_quota_count(self):
        self.assertEqual(self.ticker(D1, "AMBI11")["cnpj"], "33.333.333/0001-33")
        self.assertEqual(self.ticker(D1, "AMBI11")["sources"]["DY"], ["b3", None])
        self.assertAlmostEqual(self.ticker(D1, "AMBI11")["values"]["DY"], 12.0)
        self.assertEqual(self.ticker(D1, "NOCV11")["reasons"]["PATRIMONIO"], "no_cvm_match")

    def test_statistics_and_verdicts_per_indicator_and_date(self):
        fields = self.report["dates"][D1.isoformat()]["fields"]
        # N COTISTAS: ALFA, BETA and AMBI within; NOCV has no report
        self.assertEqual((fields["N COTISTAS"]["denominator"], fields["N COTISTAS"]["within"]), (4, 3))
        self.assertEqual(fields["N COTISTAS"]["verdict"], "com ressalvas")
        # DY: ALFA (Yahoo), AMBI (B3) and NOCV (Yahoo) within; BETA has no source
        self.assertEqual(fields["DY"]["within"], 3)
        cagr = self.report["dates"][D2.isoformat()]["fields"]["CAGR DIVIDENDOS 3 ANOS"]
        self.assertEqual((cagr["denominator"], cagr["within"]), (1, 0))
        self.assertEqual(cagr["worst"][0]["ticker"], "ALFA11")
        self.assertEqual(cagr["worst"][0]["class"], "sem classificação")
        self.assertEqual(self.report["verdicts"]["CAGR DIVIDENDOS 3 ANOS"]["overall"], "não reconstruível")
        self.assertEqual(self.report["class_verdict"], "não reconstruível")

    def test_categories_show_the_mapping_and_the_crosstab(self):
        setor = self.report["dates"][D1.isoformat()]["categories"]["SETOR"]
        self.assertEqual((setor["denominator"], setor["within"]), (4, 2))  # ALFA, BETA; AMBI is Outros, NOCV unmatched
        self.assertIn({"cvm": "Outros", "status_invest": "Papéis", "tickers": 1, "same_after_mapping": False}, setor["crosstab"])
        gestao = self.report["dates"][D2.isoformat()]["categories"]["GESTAO"]
        self.assertEqual((gestao["denominator"], gestao["within"]), (2, 2))  # ALFA Ativa, BETA Definida -> Passiva
        self.assertIn("Definida", self.report["category_maps"]["GESTAO"])

    def test_criterion_part_one_needs_70_percent_on_both_dates(self):
        criterion = self.report["criterion"]
        self.assertFalse(criterion["part1"]["met"])
        self.assertIn("CAGR DIVIDENDOS 3 ANOS", criterion["part1"]["failing"])
        self.assertFalse(criterion["met"])

    def test_coverage_gives_the_first_report_year_of_each_indicator(self):
        coverage = self.report["coverage"]["informes"]
        self.assertEqual(coverage["N COTISTAS"]["first_year"], 2023)
        self.assertEqual(coverage["PERCENTUAL EM CAIXA"]["column"], "Total_Necessidades_Liquidez")

    def test_yahoo_is_marked_wherever_it_is_used(self):
        counts = self.report["dividends"][D1.isoformat()]["DY"]
        self.assertEqual(counts["sources"], {"b3": 1, "none": 1, "yahoo": 2})
        self.assertIn("Yahoo", self.markdown)

    def test_the_markdown_states_the_verdict(self):
        self.assertIn("**Veredito da classe:** `não reconstruível`", self.markdown)
        self.assertIn("| `SETOR` |", self.markdown)

    def test_offline_rerun_reads_only_the_cache_and_gives_the_same_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            offline = FakeOpener({})
            code = fii_cli.main(args(self.cache, Path(tmp), "--offline"), opener=offline)
            self.assertEqual(code, 0)
            self.assertEqual(offline.requests, [])
            again = json.loads((Path(tmp) / fii_cli.REPORT_JSON).read_text(encoding="utf-8"))
            self.assertEqual(again["dates"], self.report["dates"])

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(fii_cli.main(args(root / "empty", root / "o", "--offline"), opener=FakeOpener({})), 3)
            (root / "empty").mkdir()
            self.assertEqual(fii_cli.main(args(root / "empty", root / "o", "--offline"), opener=FakeOpener({})), 5)
            self.assertEqual(fii_cli.main(args(root / "c", PROJECT_ROOT / "data" / "x"), opener=FakeOpener({})), 2)
            bad = ["fii-report", "--date", D1.isoformat(), "--si", str(root / "absent.csv"), "--cache-dir", str(root / "c"), "--out-dir", str(root / "o")]
            self.assertEqual(fii_cli.main(bad, opener=FakeOpener({})), 3)


    def test_the_package_entry_point_routes_fii_commands(self):
        import subprocess

        done = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "py" / "pit_rebuild"), "fii-report", "--date", "2026-13-01", "--si", str(SI1)],
            capture_output=True, text=True, cwd=PROJECT_ROOT, env={"PYTHONDONTWRITEBYTECODE": "1"}, check=False,
        )
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("--date", done.stderr)


if __name__ == "__main__":
    unittest.main()
