"""FII distributions from the B3 fund page, with Yahoo only as a marked fallback (issue #60). No network."""

from datetime import date, timedelta
import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import fii_dividends  # noqa: E402
from pit_rebuild.dividends import Dividend  # noqa: E402

ISIN = "BRALFACTF001"


def b3_payload(records, quantity="10.000.000"):
    return json.dumps({"fund": "FII ALFA", "quantity": quantity, "code": "ALFA", "cashDividends": records, "stockDividends": []}).encode()


def record(last_date_prior: str, rate: str, isin: str = ISIN, label: str = "RENDIMENTO"):
    return {"isinCode": isin, "assetIssued": isin, "lastDatePrior": last_date_prior, "rate": rate, "label": label, "paymentDate": "", "relatedTo": "", "approvedOn": "", "remarks": ""}


def next_day(day: date) -> date:
    return day + timedelta(days=1)


class B3FundTest(unittest.TestCase):
    def test_the_fund_page_gives_the_quota_count_and_the_cash_records(self):
        fund = fii_dividends.parse_b3_fund(b3_payload([record("14/11/2025", "1,00000000000")]))
        self.assertEqual(fund.quantity, 10_000_000)
        self.assertEqual(len(fund.payments), 1)

    def test_distributions_of_the_quota_isin_go_ex_on_the_session_after_the_last_date_with_rights(self):
        fund = fii_dividends.parse_b3_fund(b3_payload([
            record("14/11/2025", "1,00000000000"),
            record("14/11/2025", "0,30000000000", isin="BRALFAR01M11"),  # a subscription receipt, another ISIN
            record("15/12/2025", "2,00000000000", label="AMORTIZACAO"),
        ]))
        self.assertEqual(fii_dividends.b3_dividends(fund, ISIN, next_day), [Dividend(amount=1.0, ex_date=date(2025, 11, 15))])


class SourceTest(unittest.TestCase):
    B3 = [Dividend(1.0, date(2025, 11, 17)), Dividend(1.0, date(2025, 12, 15))]
    YAHOO = [Dividend(0.9, date(2025, 10, 15)), Dividend(1.0, date(2025, 11, 17))]

    def test_b3_is_the_source_when_its_history_reaches_the_window_start(self):
        self.assertEqual(fii_dividends.choose(self.B3, self.YAHOO, date(2025, 11, 17)), ("b3", None, self.B3))

    def test_yahoo_is_a_marked_fallback_when_b3_starts_after_the_window(self):
        self.assertEqual(fii_dividends.choose(self.B3, self.YAHOO, date(2025, 11, 5)), ("yahoo", "b3_short_history", self.YAHOO))
        self.assertEqual(fii_dividends.choose(None, self.YAHOO, date(2025, 11, 5)), ("yahoo", "b3_failed", self.YAHOO))
        self.assertEqual(fii_dividends.choose([], self.YAHOO, date(2025, 11, 5)), ("yahoo", "b3_empty", self.YAHOO))

    def test_no_source_when_b3_falls_short_and_yahoo_failed(self):
        self.assertEqual(fii_dividends.choose(self.B3, None, date(2025, 11, 5)), ("none", "b3_short_history_yahoo_failed", None))


class MeasuresTest(unittest.TestCase):
    # 0.80 a month from 2022-07-15 to 2023-12-15, then 1.00 to 2026-10-15
    MONTHLY = [
        Dividend(0.8 if (year, month) < (2024, 1) else 1.0, date(year, month, 15))
        for year in range(2022, 2027)
        for month in range(1, 13)
        if (2022, 7) <= (year, month) <= (2026, 10)
    ]

    def test_last_dividend_is_the_newest_with_ex_date_on_or_before_d(self):
        self.assertEqual(fii_dividends.last_dividend(self.MONTHLY, date(2026, 8, 10)), 1.0)
        self.assertEqual(fii_dividends.last_dividend(self.MONTHLY, date(2023, 12, 14)), 0.8)
        self.assertIsNone(fii_dividends.last_dividend(self.MONTHLY, date(2022, 7, 14)))

    def test_dividend_cagr_compares_the_twelve_months_to_d_with_those_three_years_before(self):
        # (2025-08-10, 2026-08-10]: 12 x 1.00; (2022-08-10, 2023-08-10]: 12 x 0.80; (12 / 9.6) ** (1/3) - 1 = 7.7217 %
        self.assertAlmostEqual(fii_dividends.cagr_3y(self.MONTHLY, date(2026, 8, 10)), 7.7217, places=4)

    def test_dividend_cagr_needs_the_history_of_the_old_window(self):
        self.assertIsNone(fii_dividends.cagr_3y(self.MONTHLY, date(2025, 8, 10)))

    def test_dy_sums_the_window_d_minus_365_days_exclusive_to_d_inclusive(self):
        self.assertAlmostEqual(fii_dividends.sum_12m(self.MONTHLY, date(2026, 8, 15)), 12.0)
        self.assertAlmostEqual(fii_dividends.sum_12m(self.MONTHLY, date(2024, 1, 14)), 9.6)


if __name__ == "__main__":
    unittest.main()
