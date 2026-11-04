"""Protection zone core (AD-022, issue #46): base company, ranking and input checks."""

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.protection_zone import (  # noqa: E402
    base_company,
    rank_candidates,
    validate_zone_inputs,
)


class BaseCompanyTests(unittest.TestCase):
    def test_strips_the_first_matching_class_suffix(self):  # PZ-02 (base company rule)
        cases = {"TAEE11": "TAEE", "POMO4": "POMO", "BBAS3": "BBAS", "ITSA4": "ITSA", "ABC": "ABC"}
        for ticker, company in cases.items():
            with self.subTest(ticker=ticker):
                self.assertEqual(base_company(ticker), company)


class RankCandidatesTests(unittest.TestCase):
    def test_orders_by_frequency_then_score(self):  # PZ-02
        frequencies = {"BBBB3": 0.5, "AAAA3": 0.5, "CCCC3": 0.7, "DDDD3": 0.5}
        scores = {"BBBB3": 1.0, "AAAA3": 1.0, "CCCC3": 0.1, "DDDD3": 2.0}
        self.assertEqual(
            rank_candidates(frequencies, scores, dedup_base_company=False),
            ["CCCC3", "DDDD3", "AAAA3", "BBBB3"],
        )

    def test_equal_frequency_and_score_break_by_ticker_ascending(self):  # PZ-02
        frequencies = {"DDDD3": 0.5, "CCCC3": 0.5}
        scores = {"DDDD3": 1.0, "CCCC3": 1.0}
        self.assertEqual(
            rank_candidates(frequencies, scores, dedup_base_company=False),
            ["CCCC3", "DDDD3"],
        )

    def test_dedup_keeps_the_best_ranked_ticker_of_each_company(self):  # PZ-02, PZ-09
        scores = {"POMO3": 1.0, "POMO4": 1.0, "ABCD3": 1.0}
        self.assertEqual(
            rank_candidates({"POMO3": 0.6, "POMO4": 0.5, "ABCD3": 0.1}, scores, dedup_base_company=True),
            ["POMO3", "ABCD3"],
        )
        self.assertEqual(
            rank_candidates({"POMO3": 0.5, "POMO4": 0.6, "ABCD3": 0.1}, scores, dedup_base_company=True),
            ["POMO4", "ABCD3"],
        )

    def test_without_dedup_every_ticker_of_a_company_is_kept(self):  # PZ-02
        frequencies = {"POMO3": 0.6, "POMO4": 0.5, "ABCD3": 0.1}
        scores = {"POMO3": 1.0, "POMO4": 1.0, "ABCD3": 1.0}
        self.assertEqual(
            rank_candidates(frequencies, scores, dedup_base_company=False),
            ["POMO3", "POMO4", "ABCD3"],
        )

    def test_nan_score_keeps_first_appearance_order_on_equal_counts(self):  # PZ-02, AD-025
        nan = float("nan")
        frequencies = {"AAAA3": 0.4, "BBBB3": 0.4, "CCCC3": 0.4}
        scores = {"AAAA3": 0.5, "BBBB3": nan, "CCCC3": 0.7}
        self.assertEqual(
            rank_candidates(frequencies, scores, dedup_base_company=False),
            ["AAAA3", "BBBB3", "CCCC3"],
        )
        without_bbbb3 = {"AAAA3": 0.4, "CCCC3": 0.4}
        self.assertEqual(
            rank_candidates(without_bbbb3, scores, dedup_base_company=False),
            ["CCCC3", "AAAA3"],
        )


def _production_selection(runs, scores, n_assets):
    """Tickers that build_consensus_portfolio selects on these runs and scores."""

    import pandas as pd
    from pipelines.multi_run import build_consensus_portfolio

    ranked = pd.DataFrame(
        {"TICKER": list(scores), "SCORE": list(scores.values()), "SETOR": ["S"] * len(scores)}
    )
    with contextlib.redirect_stdout(io.StringIO()):
        frame = build_consensus_portfolio(
            [{"tickers": list(run)} for run in runs], ranked, selection_config={"n_assets": n_assets}
        )
    return sorted(frame["TICKER"].tolist())


class ProductionOrderTests(unittest.TestCase):
    def test_nan_fixture_selects_what_build_consensus_portfolio_selects(self):  # PZ-02, AD-025
        fixtures = [
            ([["AAAA3", "BBBB3", "CCCC3"]] * 4, {"AAAA3": 0.5, "BBBB3": math.nan, "CCCC3": 0.7}),
            ([["AAAA3", "CCCC3"]] * 4, {"AAAA3": 0.5, "CCCC3": 0.7}),
        ]
        for runs, scores in fixtures:
            frequencies = {ticker: 1.0 for ticker in runs[0]}
            ranking = rank_candidates(frequencies, scores, dedup_base_company=True)
            for n_assets in range(1, len(runs[0]) + 1):
                with self.subTest(runs=runs, n_assets=n_assets):
                    self.assertEqual(
                        sorted(ranking[:n_assets]), _production_selection(runs, scores, n_assets)
                    )


class ZoneInputTests(unittest.TestCase):
    def test_accepts_the_boundary_values(self):  # PZ-11 (valid side)
        validate_zone_inputs(1, 0, 0.0, {"AAAA3": 0.0, "BBBB3": 1.0})
        validate_zone_inputs(13, 19, 0.11, {"AAAA3": 0.5})

    def test_rejects_each_invalid_field_naming_it(self):  # PZ-11
        cases = [
            ("K", (13, -1, 0.11, {"AAAA3": 0.5})),
            ("K", (13, 1.5, 0.11, {"AAAA3": 0.5})),
            ("K", (13, True, 0.11, {"AAAA3": 0.5})),
            ("delta", (13, 19, float("nan"), {"AAAA3": 0.5})),
            ("delta", (13, 19, -0.01, {"AAAA3": 0.5})),
            ("delta", (13, 19, float("inf"), {"AAAA3": 0.5})),
            ("n_assets", (0, 19, 0.11, {"AAAA3": 0.5})),
            ("n_assets", (13.0, 19, 0.11, {"AAAA3": 0.5})),
            ("frequency", (13, 19, 0.11, {"AAAA3": 1.2})),
            ("frequency", (13, 19, 0.11, {"AAAA3": -0.1})),
            ("frequency", (13, 19, 0.11, {"AAAA3": float("nan")})),
        ]
        for field, args in cases:
            with self.subTest(field=field, args=args):
                with self.assertRaisesRegex(ValueError, field):
                    validate_zone_inputs(*args)


if __name__ == "__main__":
    unittest.main()
