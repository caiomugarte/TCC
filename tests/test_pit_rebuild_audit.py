"""scripts/pit_rebuild_audit.py: the reproduction rules on small fixtures (no network, no cache, no Status Invest file).

The audit document states these rules (Reproduzir, Ordem das regras, Proventos (DY)); the fixtures pin them.
"""

import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True  # AD-013: the in-process loader of this suite writes no bytecode
_SPEC = importlib.util.spec_from_file_location("pit_rebuild_audit", REPO / "scripts" / "pit_rebuild_audit.py")
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _line(amount, last=date(2025, 1, 1), ex=None):
    return {"amount": amount, "last": last, "ex": ex, "file": "b3_cash_FIXTURE_p1.json"}


class ReproductionTests(unittest.TestCase):
    def test_a_value_reproduces_only_within_the_report_tolerance(self):  # CMP-01: 5 % relative, 0.5 p.p. for percentages
        self.assertTrue(audit.reproduces("P/L", 10.0, 10.4))  # 4 % relative
        self.assertFalse(audit.reproduces("P/L", 10.0, 10.6))  # 6 % relative
        self.assertTrue(audit.reproduces("ROE", 15.91, 15.98))  # 0.07 p.p.
        self.assertFalse(audit.reproduces("ROE", 15.91, 16.50))  # 0.59 p.p.
        self.assertFalse(audit.reproduces("P/L", 10.0, None))  # a view without a value reproduces nothing

    def test_the_oldest_reproducer_is_by_receipt_then_period_end_then_version(self):  # the citation rule of B-10
        a = {"dt_receb": date(2024, 5, 9), "dt": date(2024, 3, 31), "versao": 1}
        b = {"dt_receb": date(2024, 5, 9), "dt": date(2023, 12, 31), "versao": 2}
        c = {"dt_receb": date(2023, 6, 1), "dt": date(2025, 12, 31), "versao": 1}
        self.assertIs(audit.oldest([a, b, c]), c)  # the earliest receipt wins, whatever the period
        self.assertIs(audit.oldest([a, b]), b)  # same receipt: the earlier period end wins
        d = {"dt_receb": date(2024, 5, 9), "dt": date(2023, 12, 31), "versao": 1}
        self.assertIs(audit.oldest([b, d]), d)  # same receipt and period: the lower version wins


class DySubsetTests(unittest.TestCase):
    LINES = [_line(0.5), _line(0.25), _line(0.1), _line(0.7)]

    def test_a_subset_that_reproduces_the_dy_at_two_decimals_is_found(self):  # 0.5 + 0.25 + 0.1 = 0.85 on a price of 100
        found = audit.dy_search(self.LINES, 0.85, [("d", 100.0), ("d − 1", None)])
        self.assertEqual(found["d"], [0, 1, 2])
        self.assertIsNone(found["d − 1"])  # no close of d − 1: nothing to test

    def test_the_same_subset_reproduces_on_the_other_close_when_the_price_changes(self):  # 0.85 / 50 = 1.70 %
        found = audit.dy_search(self.LINES, 1.7, [("d", 100.0), ("d − 1", 50.0)])
        self.assertIsNone(found["d"])
        self.assertEqual(found["d − 1"], [0, 1, 2])

    def test_no_subset_means_no_witness(self):  # 0.3 is not a sum of these lines (0.1, 0.25, 0.5, 0.7 and their sums)
        self.assertIsNone(audit.dy_search(self.LINES, 0.3, [("d", 100.0)])["d"])

    def test_identical_lines_are_separate_items_as_the_status_invest_sums_them(self):  # two lines of 0.4 make 0.8
        found = audit.dy_search([_line(0.4), _line(0.4)], 0.8, [("d", 100.0)])
        self.assertEqual(found["d"], [0, 1])


if __name__ == "__main__":
    unittest.main()
