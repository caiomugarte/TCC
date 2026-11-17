"""The 13 FII selector indicators rebuilt at d from the CVM reports, COTAHIST and the distributions (issue #60).

Every expected value is worked out by hand from the inputs written here. No network.
"""

from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import fii_indicators  # noqa: E402
from pit_rebuild.cotahist import Quotes, Session  # noqa: E402
from pit_rebuild.dividends import Dividend  # noqa: E402
from pit_rebuild.fii_dividends import B3Fund, B3Payment  # noqa: E402
from pit_rebuild.fii_informes import Informe, Informes  # noqa: E402

D = date(2026, 8, 10)
ISIN = "BRALFACTF001"
A = "11.111.111/0001-11"
DECOY = "22.222.222/0001-22"


def informe(cnpj, ref, delivered, nav, equity, shares, holders=50000, liquidity=None, segment="Shoppings", management="Ativa", isin=ISIN):
    return Informe(cnpj=cnpj, ref=ref, version=1, delivered=delivered, name="ALFA FII", isin=isin, segment=segment,
                   management=management, holders=holders, assets=equity, equity=equity, shares=shares,
                   nav_per_share=nav, liquidity=liquidity, cash=None)


def informes(*reports):
    by_fund, by_isin = {}, {}
    for item in sorted(reports, key=lambda r: (r.delivered, r.ref)):
        by_fund.setdefault(item.cnpj, []).append(item)
        by_isin.setdefault(item.isin, []).append((item.delivered, item.cnpj))
    return Informes(by_fund=by_fund, first_year={}, by_isin=by_isin)


def quotes(ticker="ALFA11", closes=None):
    # 25 sessions to D: volume 1 000 000 on each of the last 21, 5 000 000 before; close 121.00 on D
    days = [D - timedelta(days=offset) for offset in range(24, -1, -1)]
    rows = [Session(day=day, close=121.0 if day == D else 120.0, voltot=1_000_000.0 if i >= 4 else 5_000_000.0,
                    isin=ISIN, especi="CI", codbdi="12") for i, day in enumerate(days)]
    return Quotes(sessions=days, rows={ticker: rows}, dates={ticker: days})


REPORTS = (
    informe(A, date(2023, 6, 1), date(2023, 7, 14), nav=100.0, equity=1_000_000_000, shares=10_000_000),
    informe(A, date(2026, 6, 1), date(2026, 7, 15), nav=133.1, equity=1_331_000_000, shares=10_000_000, liquidity=66_550_000),
    informe(A, date(2026, 7, 1), date(2026, 8, 14), nav=140.0, equity=1_400_000_000, shares=10_000_000),
)
YAHOO = [Dividend(0.8 if (y, m) < (2024, 1) else 1.0, date(y, m, 15)) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 7)]
B3 = B3Fund(name="FII ALFA", quantity=10_000_000, labels=("RENDIMENTO",), payments=tuple(
    B3Payment(isin=ISIN, last_date_prior=date(y, m, 14), rate=1.0, label="RENDIMENTO") for y, m in [(2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 8)]
))


class RebuildTest(unittest.TestCase):
    def rebuild(self, reports=REPORTS, b3=B3, yahoo=YAHOO, q=None):
        return fii_indicators.rebuild("ALFA11", D, q or quotes(), informes(*reports), b3, yahoo)

    def test_informe_fields_come_from_the_newest_report_delivered_by_d(self):
        result = self.rebuild()
        self.assertEqual(result.cnpj, A)
        self.assertEqual(result.informe.ref, date(2026, 6, 1))
        self.assertEqual(result.values["N COTISTAS"], 50000)
        self.assertEqual(result.values["PATRIMONIO"], 1_331_000_000)
        self.assertEqual(result.values["N COTAS"], 10_000_000)
        self.assertAlmostEqual(result.values["PERCENTUAL EM CAIXA"], 5.0)  # 66.55 M / 1 331 M
        self.assertAlmostEqual(result.values["P/VP"], 121.0 / 133.1)
        self.assertAlmostEqual(result.values["CAGR VALOR COTA 3 ANOS"], 10.0)  # (133.1 / 100) ** (1/3) - 1
        self.assertEqual(result.values["SETOR"], "Shoppings")
        self.assertEqual(result.values["GESTAO"], "Ativa")
        self.assertEqual(result.delivered_after, [date(2026, 7, 1)])

    def test_price_and_liquidity_come_from_cotahist_up_to_d(self):
        result = self.rebuild()
        self.assertEqual(result.values["PRECO"], 121.0)
        self.assertEqual(result.values["LIQUIDEZ MEDIA DIARIA"], 1_000_000.0)

    def test_distributions_fall_back_to_yahoo_when_b3_does_not_reach_the_window(self):
        result = self.rebuild()
        # B3 starts 2025-11-15, after D - 365 days: DY from Yahoo, 12 x 1.00 over (2025-08-10, 2026-08-10]
        self.assertAlmostEqual(result.values["DY"], 12 / 121 * 100)
        self.assertEqual(result.sources["DY"], ("yahoo", "b3_short_history"))
        # the last distribution is in B3: 2026-07-14 + 1 day
        self.assertEqual(result.values["ULTIMO DIVIDENDO"], 1.0)
        self.assertEqual(result.sources["ULTIMO DIVIDENDO"], ("b3", None))
        self.assertAlmostEqual(result.values["CAGR DIVIDENDOS 3 ANOS"], ((12 / 9.6) ** (1 / 3) - 1) * 100)
        self.assertEqual(result.sources["CAGR DIVIDENDOS 3 ANOS"], ("yahoo", "b3_short_history"))
        self.assertAlmostEqual(result.b3_only_dy, 9 / 121 * 100)  # the nine B3 records in the window

    def test_an_isin_reported_by_two_funds_takes_the_one_closest_to_the_b3_quota_count(self):
        decoy = informe(DECOY, date(2026, 5, 1), date(2026, 6, 15), nav=22000.0, equity=5e9, shares=224_952, segment="Multicategoria")
        self.assertEqual(self.rebuild(reports=REPORTS + (decoy,)).cnpj, A)
        result = self.rebuild(reports=REPORTS + (decoy,), b3=None)
        self.assertIsNone(result.cnpj)
        self.assertEqual(result.reasons["N COTISTAS"], "ambiguous_isin")

    def test_a_ticker_whose_isin_no_report_gives_has_no_informe_value(self):
        result = self.rebuild(reports=(informe(A, date(2026, 6, 1), date(2026, 7, 15), 1, 1, 1, isin="BROTHRCTF001"),))
        self.assertIsNone(result.values["PATRIMONIO"])
        self.assertEqual(result.reasons["PATRIMONIO"], "no_cvm_match")
        self.assertEqual(result.values["PRECO"], 121.0)

    def test_a_report_more_than_270_days_old_gives_no_informe_value(self):
        old = (informe(A, date(2025, 10, 1), date(2025, 11, 14), nav=100.0, equity=1e9, shares=1e7),)
        result = self.rebuild(reports=old)
        self.assertIsNone(result.values["N COTISTAS"])
        self.assertEqual(result.reasons["N COTISTAS"], "stale_informe")

    def test_no_session_in_the_30_days_to_d_gives_no_price(self):
        q = quotes()
        late = D + timedelta(days=40)
        result = fii_indicators.rebuild("ALFA11", late, q, informes(*REPORTS), B3, YAHOO)
        self.assertIsNone(result.values["PRECO"])
        self.assertEqual(result.reasons["PRECO"], "no_quote")
        self.assertIsNone(result.values["DY"])


if __name__ == "__main__":
    unittest.main()
