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
from pit_rebuild.fields import SCORE_FIELDS  # noqa: E402


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


def uniform_verdicts(value: str) -> dict[str, dict[str, str]]:
    return {
        field: {"non_financial": value, "non_financial_without_unclassified": value, "financial": value}
        for field in SCORE_FIELDS
    }


class VerdictTests(unittest.TestCase):
    def test_verdict_boundaries_are_90_and_70_points_inclusive_from_below(self):
        self.assertEqual(compare.verdict(90.0, 10), compare.RECONSTRUIVEL)
        self.assertEqual(compare.verdict(89.99, 10), compare.COM_RESSALVAS)
        self.assertEqual(compare.verdict(70.0, 10), compare.COM_RESSALVAS)
        self.assertEqual(compare.verdict(69.99, 10), compare.NAO_RECONSTRUIVEL)

    def test_a_zero_denominator_is_sem_referencia(self):
        self.assertEqual(compare.verdict(None, 0), compare.SEM_REFERENCIA)
        self.assertEqual(compare.verdict(0.0, 0), compare.SEM_REFERENCIA)

    def test_overall_takes_the_worse_date_and_ignores_sem_referencia(self):
        self.assertEqual(compare.overall(compare.RECONSTRUIVEL, compare.COM_RESSALVAS), compare.COM_RESSALVAS)
        self.assertEqual(compare.overall(compare.NAO_RECONSTRUIVEL, compare.SEM_REFERENCIA), compare.NAO_RECONSTRUIVEL)
        self.assertEqual(compare.overall(compare.SEM_REFERENCIA, compare.RECONSTRUIVEL), compare.RECONSTRUIVEL)
        self.assertEqual(compare.overall(compare.SEM_REFERENCIA, compare.SEM_REFERENCIA), compare.SEM_REFERENCIA)

    def test_class_verdict_is_the_worst_of_the_16_non_financial_fields(self):
        verdicts = uniform_verdicts(compare.RECONSTRUIVEL)
        self.assertEqual(compare.class_verdict(verdicts)["class"], compare.RECONSTRUIVEL)
        verdicts[SCORE_FIELDS[0]]["non_financial"] = compare.COM_RESSALVAS
        self.assertEqual(compare.class_verdict(verdicts)["class"], compare.COM_RESSALVAS)
        verdicts[SCORE_FIELDS[1]]["non_financial"] = compare.NAO_RECONSTRUIVEL
        self.assertEqual(compare.class_verdict(verdicts)["class"], compare.NAO_RECONSTRUIVEL)

    def test_a_non_reconstruivel_financial_field_or_liquidity_does_not_change_the_class(self):
        verdicts = uniform_verdicts(compare.RECONSTRUIVEL)
        verdicts[SCORE_FIELDS[0]]["financial"] = compare.NAO_RECONSTRUIVEL
        result = compare.class_verdict(verdicts)
        self.assertEqual((result["class"], result["financial"]), (compare.RECONSTRUIVEL, compare.NAO_RECONSTRUIVEL))
        verdicts["LIQUIDEZ MEDIA DIARIA"] = {
            "non_financial": compare.NAO_RECONSTRUIVEL,
            "non_financial_without_unclassified": compare.NAO_RECONSTRUIVEL,
            "financial": compare.NAO_RECONSTRUIVEL,
        }
        self.assertEqual(compare.class_verdict(verdicts)["class"], compare.RECONSTRUIVEL)

    def test_an_unclassified_ticker_missing_every_field_lowers_the_class_and_differs_only_with_it(self):
        # Ten tickers within, plus two unclassified tickers with no rebuilt value: 10 of 12 is 83.3 %.
        within_tickers = [(f"M{i}", 10.0, 10.0, None) for i in range(10)]
        unclassified = [("U1", 10.0, None, "no_quote"), ("U2", 10.0, None, "no_quote")]
        with_unclassified, without_unclassified = {}, {}
        for field in SCORE_FIELDS:
            wide = compare.summarize(field, within_tickers + unclassified)
            narrow = compare.summarize(field, within_tickers)
            with_unclassified[field] = {
                "non_financial": compare.verdict(wide.pct_within, wide.denominator),
                "non_financial_without_unclassified": compare.verdict(narrow.pct_within, narrow.denominator),
                "financial": compare.RECONSTRUIVEL,
            }
            without_unclassified[field] = {
                "non_financial": compare.verdict(narrow.pct_within, narrow.denominator),
                "non_financial_without_unclassified": compare.verdict(narrow.pct_within, narrow.denominator),
                "financial": compare.RECONSTRUIVEL,
            }
        result = compare.class_verdict(with_unclassified)
        self.assertEqual((result["class"], result["class_without_unclassified"]), (compare.COM_RESSALVAS, compare.RECONSTRUIVEL))
        self.assertTrue(result["differs"])
        self.assertFalse(compare.class_verdict(without_unclassified)["differs"])


if __name__ == "__main__":
    unittest.main()
