"""The stock CSV collector from CVM and B3 end to end on the rebuild's universe fixture (issue #78).

The universe is tests/fixtures/pit_rebuild/universe with the overlay tests/fixtures/stock_collector on top:

- b3/detail: the B3 sector classification of each company (GetDetail), except COBR (1004), which has none.
- b3/company_AZUL.json: AZUL4 split 1:2 (DESDOBRAMENTO 100 %) with ex-date in June 2026; the B3 total is 40 M.
  The DFP 2025 files 20 M PN shares, LL 30 M and PL 400 M, so at 2026-08-07 the count is 40 M:
  LPA 0.75, VPA 10.00, P/L 15.00 / 0.75 = 20.00 and VALOR DE MERCADO 40 M x 15.00 = 600 M.
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
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
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


if __name__ == "__main__":
    unittest.main()
