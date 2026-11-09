"""Dividends per share over the 12 months to d: B3 or Yahoo, the window and the leak guard (D-4, A-15, A-16, SRC-05, PIT-02).

The fixtures in tests/fixtures/pit_rebuild/ are trimmed copies of the response shapes in design.md
"Sources": the company record and the cash-dividend pages of PETROBRAS (two pages of two records,
pageSize 2 instead of 120), the empty AZUL history, and a Yahoo chart with two PETR4 dividends at
the ex-dates 2025-08-22 and 2026-08-24 (the amounts are placeholders). Inline payloads cover the
cases no fixture holds: the EUCATEX share count, a unit with UNT records, and the short history.
The calendar is a dense list of weekdays, so each ex-date maps to the next weekday.
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import cotahist, dividends  # noqa: E402
from pit_rebuild.dividends import B3Dividend, B3History, Dividend  # noqa: E402
from pit_rebuild.errors import PITLeakError  # noqa: E402

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def weekdays(start: date, end: date) -> list[date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


CALENDAR = cotahist.Quotes(sessions=weekdays(date(2025, 1, 2), date(2027, 12, 31)))


def next_session(day: date) -> date | None:
    return cotahist.next_session(CALENDAR, day)


class B3ParserTests(unittest.TestCase):
    def test_company_record_strips_the_trading_name_and_reads_the_code_cvm(self):
        company = dividends.parse_b3_company(fixture("b3_company_petr.json"))
        self.assertEqual(company.trading_name, "PETROBRAS")
        self.assertEqual(company.code_cvm, "9512")

    def test_total_number_shares_drops_the_grouping_dots(self):
        payload = b'[{"tradingName": "EUCATEX   ", "totalNumberShares": "92.619.256"}]'
        self.assertEqual(dividends.parse_b3_company(payload).total_shares, 92_619_256)

    def test_an_empty_company_list_is_none(self):
        self.assertIsNone(dividends.parse_b3_company(b"[]"))

    def test_comma_decimals_and_dd_mm_yyyy_dates_are_read(self):
        history = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        self.assertEqual(history.records[0], B3Dividend("PN", 0.4, date(2026, 8, 21)))
        self.assertEqual(history.records[1], B3Dividend("ON", 0.3, date(2026, 8, 24)))

    def test_both_pages_present_make_the_history_complete(self):
        history = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        self.assertTrue(history.complete)
        self.assertEqual((len(history.records), history.total_records), (4, 4))

    def test_page_two_missing_or_totals_announcing_more_records_is_not_complete(self):
        page_one = fixture("b3_cash_petrobras_p1.json")
        self.assertFalse(dividends.parse_b3_cash([page_one]).complete)
        announced_more = page_one.replace(b'"totalRecords":4', b'"totalRecords":5')
        self.assertFalse(dividends.parse_b3_cash([announced_more, fixture("b3_cash_petrobras_p2.json")]).complete)


class WindowAndEventTests(unittest.TestCase):
    def test_in_window_counts_d_and_d_minus_364_but_not_d_minus_365_or_d_plus_1(self):
        d = date(2026, 8, 24)
        self.assertTrue(dividends.in_window(d, d))
        self.assertTrue(dividends.in_window(d - timedelta(days=364), d))
        self.assertFalse(dividends.in_window(d - timedelta(days=365), d))
        self.assertFalse(dividends.in_window(d + timedelta(days=1), d))
        self.assertFalse(dividends.in_window(None, d))

    def test_b3_ex_date_is_the_next_session_after_last_date_prior_ex(self):
        # Friday 2026-08-21 -> ex-date Monday 2026-08-24 = d, counted; lastDatePriorEx = d -> ex-date 2026-08-25, not counted.
        d = date(2026, 8, 24)
        history = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        friday_record, monday_record = dividends.b3_ex_dates(history.records[:2], next_session)
        self.assertEqual(friday_record.ex_date, date(2026, 8, 24))
        self.assertTrue(dividends.in_window(friday_record.ex_date, d))
        self.assertEqual(monday_record.ex_date, date(2026, 8, 25))
        self.assertFalse(dividends.in_window(monday_record.ex_date, d))

    def test_yahoo_ex_date_comes_from_the_epoch(self):
        yahoo = dividends.parse_yahoo(fixture("yahoo_petr4.json"))
        self.assertEqual(yahoo, [Dividend(0.4, date(2025, 8, 22)), Dividend(0.3, date(2026, 8, 24))])

    def test_a_b3_record_before_the_first_session_has_no_ex_date_and_adds_nothing(self):
        record = B3Dividend("ON", 0.5, date(2020, 3, 2))
        (dividend,) = dividends.b3_ex_dates([record], next_session)
        self.assertIsNone(dividend.ex_date)
        self.assertEqual(dividends.dps_12m([dividend], date(2026, 1, 1)), 0.0)


class SourceChoiceTests(unittest.TestCase):
    def test_choose_source_returns_each_reason_from_its_own_input(self):
        d = date(2026, 8, 24)
        complete_petr = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        empty_azul = dividends.parse_b3_cash([fixture("b3_cash_azul_empty.json")])
        incomplete = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json")])
        short = B3History([B3Dividend("ON", 0.1, date(2025, 8, 25))], total_records=1, complete=True)
        exact = B3History([B3Dividend("ON", 0.1, date(2025, 8, 24))], total_records=1, complete=True)
        self.assertEqual(dividends.choose_source(None, d), ("yahoo", "b3_failed"))
        self.assertEqual(dividends.choose_source(empty_azul, d), ("yahoo", "b3_empty"))
        self.assertEqual(dividends.choose_source(incomplete, d), ("yahoo", "b3_incomplete"))
        self.assertEqual(dividends.choose_source(short, d), ("yahoo", "b3_short_history"))
        self.assertEqual(dividends.choose_source(exact, d), ("b3", None))
        self.assertEqual(dividends.choose_source(complete_petr, d), ("b3", None))

    def test_records_are_matched_by_type_stock(self):
        history = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        self.assertEqual([r.value for r in dividends.class_records(history.records, "ON")], [0.3])
        self.assertEqual([r.value for r in dividends.class_records(history.records, "PN")], [0.4, 0.1, 0.25])

    def test_a_unit_uses_unt_records_when_present_and_class_records_times_the_composition_when_absent(self):
        d = date(2026, 8, 24)
        with_units = [B3Dividend("UNT", 0.9, date(2026, 8, 21)), B3Dividend("ON", 0.1, date(2026, 8, 21)), B3Dividend("PN", 0.2, date(2026, 8, 21))]
        without_units = [B3Dividend("ON", 0.1, date(2026, 8, 21)), B3Dividend("PN", 0.2, date(2026, 8, 21))]
        composition = {"ON": 1, "PN": 2}
        self.assertEqual(dividends.dps_12m(dividends.unit_dividends(with_units, composition, next_session), d), 0.9)
        self.assertAlmostEqual(dividends.dps_12m(dividends.unit_dividends(without_units, composition, next_session), d), 0.5)


class LeakAndResolveTests(unittest.TestCase):
    def test_a_record_with_ex_date_after_d_passed_to_the_sum_is_a_pit_leak(self):
        with self.assertRaises(PITLeakError) as ctx:
            dividends.sum_dividends([Dividend(0.3, date(2026, 8, 25))], date(2026, 8, 24), "PETR4")
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertEqual(ctx.exception.offending_date, date(2026, 8, 25))

    def test_resolve_after_b3_empty_with_yahoo_failed_is_a_confirmed_non_payer(self):
        empty_azul = dividends.parse_b3_cash([fixture("b3_cash_azul_empty.json")])
        choice = dividends.choose_source(empty_azul, date(2026, 8, 24))
        source, reason, divs = dividends.resolve(choice, None, None)
        self.assertEqual((source, reason), ("b3", "b3_empty_yahoo_failed"))
        self.assertEqual(dividends.dps_12m(divs, date(2026, 8, 24)), 0.0)

    def test_resolve_after_b3_failed_with_yahoo_failed_has_no_dividend_source(self):
        choice = dividends.choose_source(None, date(2026, 8, 24))
        self.assertEqual(dividends.resolve(choice, None, None), ("none", "no_dividend_source", None))

    def test_resolve_with_yahoo_parsed_keeps_the_original_reason_and_b3_keeps_its_records(self):
        yahoo = dividends.parse_yahoo(fixture("yahoo_petr4.json"))
        self.assertEqual(
            dividends.resolve(("yahoo", "b3_empty"), None, yahoo),
            ("yahoo", "b3_empty", yahoo),
        )
        self.assertEqual(
            dividends.resolve(("yahoo", "b3_short_history"), None, yahoo),
            ("yahoo", "b3_short_history", yahoo),
        )
        b3_divs = [Dividend(0.4, date(2026, 8, 24))]
        self.assertEqual(dividends.resolve(("b3", None), b3_divs, None), ("b3", None, b3_divs))

    def test_petr_pn_window_sums_the_records_in_the_12_months_to_d(self):
        # PN records: ex 2026-08-24 (0.4), 2025-12-23 (0.1, counted), 2025-03-11 (0.25, before 2025-08-24: not counted).
        d = date(2026, 8, 24)
        history = dividends.parse_b3_cash([fixture("b3_cash_petrobras_p1.json"), fixture("b3_cash_petrobras_p2.json")])
        pn = dividends.b3_ex_dates(dividends.class_records(history.records, "PN"), next_session)
        self.assertAlmostEqual(dividends.dps_12m(pn, d, "PETR4"), 0.5)


if __name__ == "__main__":
    unittest.main()
