"""Dividends per share over the 12 months to d: B3 or Yahoo, the window and the leak guard (D-4, A-15, A-16, SRC-05, PIT-02).

The fixtures in tests/fixtures/pit_rebuild/ are trimmed copies of the response shapes in design.md
"Sources": the company record and the cash-dividend pages of PETROBRAS (two pages of two records,
pageSize 2 instead of 120), the empty AZUL history, and a Yahoo chart with two PETR4 dividends at
the ex-dates 2025-08-22 and 2026-08-24 (the amounts are placeholders). Inline payloads cover the
cases no fixture holds: the EUCATEX share count, a unit with UNT records, and the short history.
The calendar is a dense list of weekdays, so each ex-date maps to the next weekday.
"""

import json
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


class StockEventTests(unittest.TestCase):
    def test_sbsp_stock_events_carry_the_r5_multipliers(self):
        # DESDOBRAMENTO 400 -> 1 + 400/100 = 5; BONIFICACAO 0,16098032200 -> 1.0016098032; BONIFICACAO 2,96469750000
        # -> 1.029646975; GRUPAMENTO 0,008 -> 0.008 (A-20). The cached record lists them in this order.
        events = dividends.parse_b3_company(fixture("b3_company_sbsp.json")).stock_events
        self.assertEqual([event.label for event in events], ["DESDOBRAMENTO", "BONIFICACAO", "BONIFICACAO", "GRUPAMENTO"])
        self.assertEqual({event.isin for event in events}, {"BRSBSPACNOR5"})
        for event, multiplier in zip(events, [5.0, 1.0016098032, 1.029646975, 0.008]):
            self.assertAlmostEqual(event.multiplier, multiplier, places=10)
        self.assertEqual(events[0].last_date_prior, date(2026, 4, 28))

    def test_espa_grouping_multiplier_is_its_factor(self):
        (event,) = dividends.parse_b3_company(fixture("b3_company_espa.json")).stock_events
        self.assertEqual((event.label, event.isin, event.last_date_prior), ("GRUPAMENTO", "BRESPAACNOR7", date(2026, 6, 12)))
        self.assertAlmostEqual(event.multiplier, 0.1, places=10)

    def test_labels_other_than_split_bonus_and_grouping_are_skipped(self):
        entries = [
            {"factor": "1,00", "isinCode": "BRXXXXACNOR1", "label": "CIS RED CAP", "lastDatePrior": "01/02/2026"},
            {"factor": "2,00", "isinCode": "BRXXXXACNOR1", "label": "RESG TOTAL RV", "lastDatePrior": "01/03/2026"},
            {"factor": "3,00", "isinCode": "BRXXXXACNOR1", "label": "INCORPORACAO", "lastDatePrior": "01/04/2026"},
        ]
        payload = json.dumps([{"tradingName": "X", "codeCVM": "1", "totalNumberShares": "1", "stockDividends": entries}])
        self.assertEqual(dividends.parse_b3_company(payload.encode()).stock_events, ())

    def test_two_identical_entries_give_one_event(self):
        entry = {"assetIssued": "BRESPAACNOR7", "factor": "0,10000000000", "approvedOn": "28/04/2026",
                 "isinCode": "BRESPAACNOR7", "label": "GRUPAMENTO", "lastDatePrior": "12/06/2026", "remarks": ""}
        payload = json.dumps([{"tradingName": "X", "codeCVM": "1", "totalNumberShares": "1", "stockDividends": [entry, dict(entry)]}])
        self.assertEqual(len(dividends.parse_b3_company(payload.encode()).stock_events), 1)

    def test_a_company_without_stock_dividends_has_no_events(self):
        for item in ({"tradingName": "X"}, {"tradingName": "X", "stockDividends": None}, {"tradingName": "X", "stockDividends": []}):
            with self.subTest(item=item):
                self.assertEqual(dividends.parse_b3_company(json.dumps([item]).encode()).stock_events, ())


class CashRecordFilterTests(unittest.TestCase):
    def test_a_record_repeated_on_its_page_and_on_the_next_counts_once(self):
        # p1 holds A, A and a REST CAP DIN record; p2 holds A and C. A is kept once; its two repeats are duplicates.
        history = dividends.parse_b3_cash([fixture("b3_cash_dupes_p1.json"), fixture("b3_cash_dupes_p2.json")])
        self.assertEqual(
            history.records,
            [B3Dividend("ON", 0.182219167, date(2025, 12, 30)), B3Dividend("PN", 0.08411356466, date(2026, 4, 29))],
        )
        self.assertEqual((history.raw_records, history.duplicates), (5, 2))

    def test_rest_cap_din_is_a_capital_return_and_is_not_a_record(self):
        history = dividends.parse_b3_cash([fixture("b3_cash_dupes_p1.json"), fixture("b3_cash_dupes_p2.json")])
        self.assertEqual(history.capital_returns, 1)
        self.assertNotIn(1.89816664941, [record.value for record in history.records])  # the REST CAP DIN amount

    def test_the_history_is_complete_on_the_raw_count_though_fewer_records_are_kept(self):
        history = dividends.parse_b3_cash([fixture("b3_cash_dupes_p1.json"), fixture("b3_cash_dupes_p2.json")])
        self.assertEqual((history.total_records, history.raw_records, len(history.records)), (5, 5, 2))
        self.assertTrue(history.complete)

    def test_records_that_differ_in_one_field_are_both_kept(self):
        page = {"page": {"pageNumber": 1, "pageSize": 2, "totalRecords": 2, "totalPages": 1}, "results": [
            {"typeStock": "ON", "valueCash": "0,1000", "lastDatePriorEx": "21/08/2026", "corporateAction": "DIVIDENDO"},
            {"typeStock": "ON", "valueCash": "0,1000", "lastDatePriorEx": "24/08/2026", "corporateAction": "DIVIDENDO"},
        ]}
        history = dividends.parse_b3_cash([json.dumps(page).encode()])
        self.assertEqual(len(history.records), 2)
        self.assertEqual(history.duplicates, 0)
        self.assertTrue(history.complete)


class SplitAdjustmentTests(unittest.TestCase):
    SBSP_ISIN = "BRSBSPACNOR5"

    def setUp(self):
        self.events = dividends.parse_b3_company(fixture("b3_company_sbsp.json")).stock_events

    def test_an_sbsp_dividend_with_ex_date_2026_03_02_at_d_2026_08_07_is_divided_by_both_later_events(self):
        # The 2026-04 split (ex 2026-04-29) and the 2026-03 bonus (ex 2026-03-20) follow the ex-date; the 2025-12
        # bonus (ex 2025-12-24) does not. Hand-computed: 5 x 1.0016098032.
        divisor = dividends.split_divisor(self.events, self.SBSP_ISIN, date(2026, 3, 2), date(2026, 8, 7), next_session)
        self.assertAlmostEqual(divisor, 5 * 1.0016098032, places=9)

    def test_at_d_2026_04_27_only_the_bonus_applies_because_the_split_is_after_d(self):
        divisor = dividends.split_divisor(self.events, self.SBSP_ISIN, date(2026, 3, 2), date(2026, 4, 27), next_session)
        self.assertAlmostEqual(divisor, 1.0016098032, places=9)

    def test_b3_ex_dates_divides_each_amount_by_the_split_divisor_at_d(self):
        (dividend,) = dividends.b3_ex_dates(
            [B3Dividend("ON", 0.5, date(2026, 2, 27))], next_session, events=self.events, isin=self.SBSP_ISIN, d=date(2026, 8, 7)
        )
        self.assertEqual(dividend.ex_date, date(2026, 3, 2))
        self.assertAlmostEqual(dividend.amount, 0.5 / (5 * 1.0016098032), places=9)

    def test_a_dividend_after_the_split_is_not_divided(self):
        # The split's ex-date is 2026-04-29; this dividend's ex-date is 2026-05-04.
        (dividend,) = dividends.b3_ex_dates(
            [B3Dividend("ON", 0.5, date(2026, 5, 1))], next_session, events=self.events, isin=self.SBSP_ISIN, d=date(2026, 8, 7)
        )
        self.assertEqual(dividend.ex_date, date(2026, 5, 4))
        self.assertEqual(dividend.amount, 0.5)

    def test_an_event_whose_ex_date_equals_the_dividends_is_not_applied(self):
        # lastDatePriorEx 2026-04-28 gives ex-date 2026-04-29, the same day as the split's ex-date.
        (dividend,) = dividends.b3_ex_dates(
            [B3Dividend("ON", 0.5, date(2026, 4, 28))], next_session, events=self.events, isin=self.SBSP_ISIN, d=date(2026, 8, 7)
        )
        self.assertEqual(dividend.ex_date, date(2026, 4, 29))
        self.assertEqual(dividend.amount, 0.5)

    def test_an_event_of_another_isin_is_not_applied(self):
        self.assertEqual(dividends.split_divisor(self.events, "BRSBSPACNPR9", date(2026, 3, 2), date(2026, 8, 7), next_session), 1.0)

    def test_isin_none_leaves_amounts_unchanged(self):
        (dividend,) = dividends.b3_ex_dates(
            [B3Dividend("ON", 0.5, date(2026, 2, 27))], next_session, events=self.events, isin=None, d=date(2026, 8, 7)
        )
        self.assertEqual(dividend.amount, 0.5)

    def test_an_espa_dividend_before_the_2026_06_12_grouping_is_multiplied_by_ten(self):
        espa = dividends.parse_b3_company(fixture("b3_company_espa.json")).stock_events
        (dividend,) = dividends.b3_ex_dates(
            [B3Dividend("ON", 0.5, date(2026, 3, 10))], next_session, events=espa, isin="BRESPAACNOR7", d=date(2026, 8, 7)
        )
        self.assertAlmostEqual(dividend.amount, 5.0, places=9)


class UnitAdjustmentTests(unittest.TestCase):
    def test_unit_and_class_records_are_adjusted_by_their_own_isin(self):
        # An ENGI-shaped 10 % bonus on the unit, ON and PN ISINs, ex-date 2026-04-02 (lastDatePrior 2026-04-01).
        events = [
            dividends.StockEvent("BRENGIACNUT1", "BONIFICACAO", 1.10, date(2026, 4, 1)),
            dividends.StockEvent("BRENGIACNOR1", "BONIFICACAO", 1.10, date(2026, 4, 1)),
            dividends.StockEvent("BRENGIACNPR1", "BONIFICACAO", 1.10, date(2026, 4, 1)),
        ]
        isins = {"UNT": "BRENGIACNUT1", "ON": "BRENGIACNOR1", "PN": "BRENGIACNPR1"}
        d = date(2026, 8, 7)
        # A unit "1 ON e 4 PN" without UNT records: each class record times its count, divided by its class ISIN.
        class_records = [B3Dividend("ON", 0.11, date(2026, 3, 10)), B3Dividend("PN", 0.22, date(2026, 3, 10))]
        out = dividends.unit_dividends(class_records, {"ON": 1, "PN": 4}, next_session, events=events, isins=isins, d=d)
        self.assertAlmostEqual(dividends.dps_12m(out, d), 0.11 / 1.1 + 0.22 * 4 / 1.1, places=9)
        # With UNT records, the unit's own ISIN divides them.
        (unit_out,) = dividends.unit_dividends(
            [B3Dividend("UNT", 1.1, date(2026, 3, 10))], {"ON": 1, "PN": 4}, next_session, events=events, isins=isins, d=d
        )
        self.assertAlmostEqual(unit_out.amount, 1.0, places=9)


class EarliestRawAndSourceEdgeTests(unittest.TestCase):
    def test_earliest_raw_date_is_the_day_after_the_earliest_last_date_prior(self):
        records = [B3Dividend("PN", 0.1, date(1996, 3, 21)), B3Dividend("PN", 0.1, date(2025, 12, 31))]
        self.assertEqual(dividends.b3_earliest_raw(records), date(1996, 3, 22))
        self.assertEqual(dividends.b3_earliest_raw([B3Dividend("ON", 0.1, date(2020, 12, 31))]), date(2021, 1, 1))
        self.assertIsNone(dividends.b3_earliest_raw([]))

    def test_a_complete_history_of_capital_returns_only_is_a_short_history_not_a_crash(self):
        page = {"page": {"pageNumber": 1, "pageSize": 1, "totalRecords": 1, "totalPages": 1}, "results": [
            {"typeStock": "ON", "valueCash": "1,0", "lastDatePriorEx": "21/08/2026", "corporateAction": "REST CAP DIN"},
        ]}
        history = dividends.parse_b3_cash([json.dumps(page).encode()])
        self.assertTrue(history.complete)
        self.assertEqual(history.records, [])
        self.assertEqual(dividends.choose_source(history, date(2026, 8, 24)), ("yahoo", "b3_short_history"))


if __name__ == "__main__":
    unittest.main()
