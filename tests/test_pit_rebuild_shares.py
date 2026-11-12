"""Share counts at d: the CVM count moved by the B3 events of its classes (AD-033, SHR-01..SHR-04, SHR-08, B-2, B-3).

The adjustment is pure. Filings are cvm.Shares values and events are dividends.StockEvent values. The S12 case builds
its ISIN sets from a COTAHIST quotes object in memory, so no file and no network are read.
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import cotahist, cvm, shares  # noqa: E402
from pit_rebuild.dividends import StockEvent  # noqa: E402

DT_REFER = date(2026, 6, 30)  # the period end of the filed count (a Tuesday)
D = date(2026, 8, 7)  # the decision date (a Friday)
ON_ISIN = "BRTESTACNOR1"
NEW_ISIN = "BRTESTBCNOR1"
PREF_ISIN = "BRTESTACNPR1"


def weekdays(start: date, end: date) -> list[date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


CALENDAR = cotahist.Quotes(sessions=weekdays(date(2026, 1, 2), date(2027, 12, 31)))


def next_session(day: date) -> date | None:
    return cotahist.next_session(CALENDAR, day)


def filed(on: int = 1_000_000, pref: int = 0) -> cvm.Shares:
    return cvm.Shares(on=on, pref=pref, total=on + pref, dt_refer=DT_REFER, dt_receb=date(2026, 8, 1))


def adjust(events=(), on=1_000_000, pref=0, b3_total=None, retrieved_on=None, d=D, isins_d=None, isins_all=None):
    """shares.adjust for the fixture company: class ON on ON_ISIN and PREF on PREF_ISIN unless a test gives its own sets."""
    isins_d = {"ON": (ON_ISIN,), "PREF": (PREF_ISIN,)} if isins_d is None else isins_d
    isins_all = isins_d if isins_all is None else isins_all
    return shares.adjust(filed(on, pref), isins_d, isins_all, list(events), b3_total, retrieved_on, d, next_session)


class ShareCountInWindowTests(unittest.TestCase):
    """SHR-01, SHR-02, SHR-03: the events of the class with ex-date in (DT_REFER, d] move the count."""

    BONUS = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, date(2026, 7, 14))  # ex 2026-07-15

    def test_a_bonus_in_the_window_doubles_the_count_and_is_listed(self):
        result = adjust(events=[self.BONUS], b3_total=2_000_000)
        self.assertEqual((result.n_on, result.factor), (2_000_000, 1))
        self.assertEqual(
            result.applied,
            [{
                "class": "ON",
                "isins": [ON_ISIN],
                "factor": 2.0,
                "events": [{"isin": ON_ISIN, "label": "BONIFICACAO", "ex_date": date(2026, 7, 15), "multiplier": 2.0}],
            }],
        )

    def test_a_bonus_with_ex_date_after_d_is_not_applied(self):
        result = adjust(events=[self.BONUS], b3_total=2_000_000, d=date(2026, 7, 14))
        self.assertEqual((result.n_on, result.factor, result.applied), (1_000_000, 1, []))

    def test_an_event_with_ex_date_equal_to_dt_refer_is_not_applied(self):
        on_dt_refer = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, date(2026, 6, 29))  # ex 2026-06-30 = DT_REFER
        result = adjust(events=[on_dt_refer], b3_total=1_000_000)
        self.assertEqual((result.n_on, result.applied), (1_000_000, []))

    def test_two_events_in_the_window_multiply_and_a_grouping_divides(self):
        split = StockEvent(ON_ISIN, "DESDOBRAMENTO", 3.0, date(2026, 7, 20))  # ex 2026-07-21
        grouping = StockEvent(ON_ISIN, "GRUPAMENTO", 0.1, date(2026, 7, 27))  # ex 2026-07-28
        self.assertEqual(adjust(events=[self.BONUS, split]).n_on, 6_000_000)
        self.assertEqual(adjust(events=[grouping]).n_on, 100_000)

    def test_a_split_and_a_grouping_that_cancel_leave_the_class_unlisted(self):
        # A 1:2 split and a 2:1 grouping in the window multiply to 1 (SHR-03, S10).
        split = StockEvent(ON_ISIN, "DESDOBRAMENTO", 2.0, date(2026, 7, 14))  # ex 2026-07-15
        grouping = StockEvent(ON_ISIN, "GRUPAMENTO", 0.5, date(2026, 7, 20))  # ex 2026-07-21
        result = adjust(events=[split, grouping], b3_total=1_000_000)
        self.assertEqual((result.n_on, result.applied), (1_000_000, []))


class ScaleAtRetrievalTests(unittest.TestCase):
    """B-3 and SHR-08: the A-18 factor is chosen on the count adjusted by the events up to the B3 retrieval R."""

    SPLIT = StockEvent(ON_ISIN, "DESDOBRAMENTO", 5.0, date(2026, 7, 28))  # ex 2026-07-29
    R_BEFORE_EX = date(2026, 7, 28)
    R_ON_EX = date(2026, 7, 29)

    def test_a_split_after_d_and_within_the_retrieval_does_not_double_count(self):
        # Ex-date 2026-07-28 is after d = 2026-07-20 and on or before R = 2026-08-05; the B3 total 5,000,000 already holds it.
        split = StockEvent(ON_ISIN, "DESDOBRAMENTO", 5.0, date(2026, 7, 27))  # ex 2026-07-28
        result = adjust(events=[split], b3_total=5_000_000, retrieved_on=date(2026, 8, 5), d=date(2026, 7, 20))
        self.assertEqual((result.factor, result.n_on), (1, 1_000_000))

    def test_an_event_with_ex_date_after_the_retrieval_is_out_of_the_scale(self):
        # Ex-date 2026-07-29 is after R = 2026-07-28: the scale sees 1,000,000 against 1,000,000 and picks factor 1.
        result = adjust(events=[self.SPLIT], b3_total=1_000_000, retrieved_on=self.R_BEFORE_EX)
        self.assertEqual(result.factor, 1)
        self.assertEqual(result.n_on, 5_000_000)  # d = 2026-08-07 is after the ex-date, so the count still takes the split

    def test_an_event_with_ex_date_on_the_retrieval_day_is_in_the_scale(self):
        result = adjust(events=[self.SPLIT], b3_total=1_000_000, retrieved_on=self.R_ON_EX)
        self.assertEqual(result.factor, 0.1)  # 5,000,000 against 1,000,000: the power of ten nearest the ratio 5 on the log scale is 10^-1
        self.assertEqual(result.n_on, 500_000)

    def test_without_a_retrieval_date_every_later_event_enters_the_scale(self):
        unbounded = adjust(events=[self.SPLIT], b3_total=5_000_000, retrieved_on=None, d=date(2026, 7, 20))
        self.assertEqual((unbounded.factor, unbounded.n_on), (1, 1_000_000))  # the scale counts the split; d precedes it
        bounded = adjust(events=[self.SPLIT], b3_total=5_000_000, retrieved_on=self.R_BEFORE_EX, d=date(2026, 7, 20))
        self.assertEqual(bounded.factor, 10)  # the split is out of the scale, so the factor absorbs it


class IsinSetTests(unittest.TestCase):
    """B-2, S4 and S12: a class's counts take the events of its ISINs at d; the scale takes those of every ISIN it had."""

    def test_an_isin_taken_after_d_adds_its_events_to_the_scale_only(self):
        # S4: NEW_ISIN is listed after d = 2026-07-20. Its 1:5 split (ex 2026-07-29) is in the scale, not in the count at d.
        split = StockEvent(NEW_ISIN, "DESDOBRAMENTO", 5.0, date(2026, 7, 28))
        with_new = adjust(events=[split], b3_total=5_000_000, retrieved_on=date(2026, 8, 5), d=date(2026, 7, 20),
                          isins_d={"ON": (ON_ISIN,)}, isins_all={"ON": (ON_ISIN, NEW_ISIN)})
        self.assertEqual((with_new.factor, with_new.n_on), (1, 1_000_000))
        without_new = adjust(events=[split], b3_total=5_000_000, retrieved_on=date(2026, 8, 5), d=date(2026, 7, 20),
                             isins_d={"ON": (ON_ISIN,)}, isins_all={"ON": (ON_ISIN,)})
        self.assertEqual(without_new.factor, 10)

    def test_an_event_of_an_isin_the_class_had_inside_the_window_is_applied(self):
        # B-2: the ISIN changed between DT_REFER and d; the bonus under the old ISIN counts at d.
        old_bonus = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, date(2026, 7, 14))
        result = adjust(events=[old_bonus], isins_d={"ON": (ON_ISIN, NEW_ISIN)})
        self.assertEqual(result.n_on, 2_000_000)
        self.assertEqual(result.applied[0]["isins"], sorted([ON_ISIN, NEW_ISIN]))

    def test_one_event_listed_under_two_isins_of_the_class_counts_once(self):
        under_old = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, date(2026, 7, 14))
        under_new = StockEvent(NEW_ISIN, "BONIFICACAO", 2.0, date(2026, 7, 14))
        result = adjust(events=[under_old, under_new], isins_d={"ON": (ON_ISIN, NEW_ISIN)})
        self.assertEqual(result.n_on, 2_000_000)

    def test_an_event_listed_under_the_isin_of_dt_refer_is_applied_when_the_isin_changed_after_it(self):
        # S12: the ISIN changes on the session after DT_REFER and B3 lists the bonus under the old ISIN, whose only
        # session is DT_REFER's. class_isins keeps that ISIN in I_d, so the bonus counts at d.
        quotes = cotahist.Quotes(
            sessions=[DT_REFER, date(2026, 7, 1), date(2026, 7, 2)],
            rows={"TEST3": [
                cotahist.Session(DT_REFER, 10.0, 1000.0, ON_ISIN, "ON", "02"),
                cotahist.Session(date(2026, 7, 1), 11.0, 1000.0, NEW_ISIN, "ON", "02"),
                cotahist.Session(date(2026, 7, 2), 12.0, 1000.0, NEW_ISIN, "ON", "02"),
            ]},
            dates={"TEST3": [DT_REFER, date(2026, 7, 1), date(2026, 7, 2)]},
        )
        isins_at_d = cotahist.class_isins(quotes, "TEST3", DT_REFER, D)
        self.assertEqual(isins_at_d, (ON_ISIN, NEW_ISIN))
        bonus = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, DT_REFER)  # lastDatePrior = DT_REFER: ex-date 2026-07-01
        result = adjust(events=[bonus], isins_d={"ON": isins_at_d}, isins_all={"ON": isins_at_d})
        self.assertEqual(result.n_on, 2_000_000)


class ClassWithoutIsinTests(unittest.TestCase):
    """SHR-04 and the optional B3 total: a class with shares and no ISIN at d gets no event; without a total f is 1."""

    BONUS = StockEvent(ON_ISIN, "BONIFICACAO", 2.0, date(2026, 7, 14))

    def test_a_class_with_shares_and_no_isin_at_d_is_not_adjusted_and_is_listed(self):
        result = adjust(events=[self.BONUS], pref=500_000, isins_d={"ON": (ON_ISIN,), "PREF": ()})
        self.assertEqual((result.n_on, result.n_pref, result.no_isin), (2_000_000, 500_000, ["PREF"]))

    def test_without_a_b3_total_the_factor_is_one_unverified_and_the_events_still_apply(self):
        result = adjust(events=[self.BONUS], b3_total=None)
        self.assertEqual((result.factor, result.verified, result.residual, result.n_on), (1, False, None, 2_000_000))
        self.assertEqual(len(result.applied), 1)

    def test_the_residual_is_the_event_adjusted_total_over_the_b3_total_and_scaled_total_leaves_events_out(self):
        result = adjust(events=[self.BONUS], b3_total=2_000_000)
        self.assertEqual(result.residual, 1.0)  # 1,000,000 x 2 at factor 1 over 2,000,000
        self.assertEqual(result.scaled_total, 1_000_000)  # the filed total times the factor


if __name__ == "__main__":
    unittest.main()
