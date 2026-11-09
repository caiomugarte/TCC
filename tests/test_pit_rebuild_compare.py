"""Tolerance, differences and per-indicator statistics (CMP-01, CMP-02, A-2, A-3, A-4, A-12).

The boundaries are the spec's: 5.00 % relative and 0.50 p.p. are within, 5.01 % and 0.51 p.p.
are not; the rebuilt value rounds to 2 decimals before the comparison; a blank DY is the
reference 0.00 and stays in the denominator; any other blank reference leaves it.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pit_rebuild import compare  # noqa: E402


class ToleranceTests(unittest.TestCase):
    def test_relative_tolerance_is_inclusive_at_5_percent_and_exclusive_above(self):
        self.assertTrue(compare.within("P/L", 100.0, 105.0))
        self.assertFalse(compare.within("P/L", 100.0, 105.01))
        self.assertTrue(compare.within("P/L", 100.0, 95.0))
        self.assertFalse(compare.within("P/L", 100.0, 94.99))

    def test_percent_tolerance_is_inclusive_at_0_50_points_and_exclusive_above(self):
        self.assertTrue(compare.within("ROE", 10.0, 10.5))
        self.assertFalse(compare.within("ROE", 10.0, 10.51))
        self.assertTrue(compare.within("DY", 3.0, 2.5))
        self.assertFalse(compare.within("DY", 3.0, 2.49))

    def test_the_rebuilt_value_is_rounded_to_two_decimals_before_comparing(self):
        # 10.504 is 0.504 p.p. away unrounded; rounded to 10.50 it is exactly 0.50 and within.
        self.assertTrue(compare.within("ROE", 10.0, 10.504))
        self.assertEqual(compare.difference("P/L", 10.0, 10.004), 0.0)

    def test_a_zero_reference_is_within_only_when_the_rounded_rebuilt_value_is_zero(self):
        self.assertTrue(compare.within("P/L", 0.0, 0.004))
        self.assertTrue(compare.within("P/L", 0.0, -0.004))
        self.assertFalse(compare.within("P/L", 0.0, 0.01))

    def test_difference_is_percent_of_the_reference_or_points_and_none_for_a_zero_reference(self):
        self.assertAlmostEqual(compare.difference("P/L", 100.0, 110.0), 10.0)
        self.assertAlmostEqual(compare.difference("ROE", 10.0, 10.5), 0.5)
        self.assertIsNone(compare.difference("P/L", 0.0, 3.0))
        self.assertIsNone(compare.difference("P/L", 5.0, None))

    def test_reference_is_zero_for_a_blank_dy_and_none_for_other_blanks(self):
        self.assertEqual(compare.reference("DY", None), (0.0, True))
        self.assertEqual(compare.reference("P/L", None), (None, False))
        self.assertEqual(compare.reference("P/L", 7.5), (7.5, False))


class StatisticsTests(unittest.TestCase):
    def test_a_missing_rebuilt_value_is_in_the_denominator_and_outside(self):
        stats = compare.summarize("P/L", [("A", 10.0, None, "no_quote")])
        self.assertEqual((stats.denominator, stats.within, stats.pct_within), (1, 0, 0.0))
        self.assertEqual(stats.missing, [{"ticker": "A", "reason": "no_quote"}])

    def test_a_blank_reference_of_a_non_dy_field_is_outside_the_denominator_and_counted_as_blank(self):
        stats = compare.summarize(
            "P/L",
            [("B", None, 5.0, None), ("C", None, None, "no_quote"), ("D", 10.0, 10.0, None)],
        )
        self.assertEqual((stats.denominator, stats.within), (1, 1))
        self.assertEqual((stats.reference_blank, stats.reference_blank_with_value), (2, 1))
        self.assertEqual(stats.pct_within, 100.0)

    def test_a_blank_dy_is_in_the_denominator_and_within_at_0_50_only(self):
        stats = compare.summarize(
            "DY",
            [
                ("E", None, 3.0, None),  # outside: 3.0 vs 0.00
                ("F", None, 0.50, None),  # within
                ("G", None, 0.51, None),  # outside
                ("H", None, None, "no_dividend_source"),  # no rebuilt value
            ],
        )
        self.assertEqual((stats.denominator, stats.within), (4, 1))
        self.assertEqual((stats.reference_blank, stats.reference_blank_as_zero), (0, 4))
        self.assertEqual(stats.pct_within, 25.0)
        self.assertEqual(stats.missing, [{"ticker": "H", "reason": "no_dividend_source"}])

    def test_p90_is_the_nearest_rank_value(self):
        self.assertEqual(compare.p90(range(1, 11)), 9)
        self.assertEqual(compare.p90(range(1, 12)), 10)
        self.assertIsNone(compare.p90([]))

    def test_median_and_p90_of_the_relative_differences(self):
        pairs = [(f"T{i}", 100.0, 100.0 + i, None) for i in range(1, 6)]  # differences 1..5 %
        stats = compare.summarize("P/L", pairs)
        self.assertEqual(stats.median_diff, 3.0)
        self.assertEqual(stats.p90_diff, 5.0)  # ceil(0.9 × 5) = 5th smallest
        self.assertEqual(stats.diff_unit, "relative_pct")
        self.assertEqual(stats.within, 5)

    def test_worst_cases_are_capped_at_10_and_ties_are_broken_by_ticker(self):
        diffs = {"A": 3, "B": 9, "C": 9, "D": 1, "E": 2, "F": 4, "G": 5, "H": 6, "I": 7, "J": 8, "K": 10, "L": 11}
        stats = compare.summarize("P/L", [(ticker, 100.0, 100.0 + delta, None) for ticker, delta in diffs.items()])
        self.assertEqual(len(stats.worst), 10)
        self.assertEqual(
            [item["ticker"] for item in stats.worst],
            ["L", "K", "B", "C", "J", "I", "H", "G", "F", "A"],
        )
        self.assertEqual(stats.worst[0], {"ticker": "L", "reference": 100.0, "rebuilt": 111.0, "diff": 11.0})

    def test_a_pp_indicator_reports_points_and_its_within_share(self):
        stats = compare.summarize("ROE", [("R1", 10.0, 10.2, None), ("R2", 10.0, 11.0, None)])
        self.assertEqual((stats.within, stats.denominator), (1, 2))
        self.assertEqual(stats.diff_unit, "pp")
        self.assertAlmostEqual(stats.median_diff, 0.6)


if __name__ == "__main__":
    unittest.main()
