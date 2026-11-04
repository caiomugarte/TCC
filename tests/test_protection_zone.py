"""Protection zone core (AD-022, issue #46): base company, ranking and input checks."""

import contextlib
import io
import math
from pathlib import Path
import random
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.protection_zone import (  # noqa: E402
    apply_protection_zone,
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


def _zone(incumbents, frequencies, n_assets, K, delta, *, scores=None, eligible=None, dedup=True):
    """apply_protection_zone with every candidate eligible and score 1.0 unless given."""

    scores = {t: 1.0 for t in frequencies} if scores is None else scores
    eligible = set(frequencies) | set(incumbents) if eligible is None else eligible
    return apply_protection_zone(
        incumbents,
        frequencies,
        scores,
        n_assets,
        K,
        delta,
        eligible=eligible,
        dedup_base_company=dedup,
    )


def _percent(counts, run_count=100):
    return {ticker: count / run_count for ticker, count in counts.items()}


class ZoneRuleTests(unittest.TestCase):
    def test_no_incumbents_gives_the_first_n_of_the_ranking(self):  # PZ-10
        rng = random.Random(46)
        for case in range(40):
            with self.subTest(case=case):
                size = rng.randint(1, 20)
                tickers = [f"T{i:02d}3" for i in range(size)]
                frequencies = {t: rng.randint(1, 100) / 100 for t in tickers}
                scores = {t: rng.random() for t in tickers}
                n_assets = rng.randint(1, size)
                expected = sorted(tickers, key=lambda t: (-frequencies[t], -scores[t], t))[:n_assets]
                selection = apply_protection_zone(
                    [], frequencies, scores, n_assets, 0, 0.11,
                    eligible=tickers, dedup_base_company=True,
                )
                self.assertEqual(selection.tickers, tuple(expected))
                self.assertEqual(
                    (selection.turnover, selection.exits, selection.departures, selection.kept),
                    (n_assets, 0, {}, ()),
                )

    def test_k_zero_is_the_consensus_and_incumbents_inside_it_stay(self):  # PZ-10, PZ-04
        frequencies = {"ALFA3": 0.9, "BETA3": 0.8, "GAMA3": 0.7, "DELT3": 0.1}
        selection = _zone(["BETA3", "DELT3"], frequencies, 2, 0, 0.11)
        self.assertEqual(selection.tickers, ("ALFA3", "BETA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "novo", "BETA3": "mantido"})
        self.assertEqual(selection.departures, {"DELT3": "saiu: fora do top K"})
        self.assertEqual(selection.kept, ("BETA3",))
        self.assertEqual(selection.entered, ("ALFA3",))
        self.assertEqual((selection.turnover, selection.exits), (1, 1))

    def test_incumbent_at_rank_k_stays_unless_a_candidate_beats_it_by_delta(self):  # PZ-03, PZ-05
        frequencies = {"ALFA3": 0.9, "GAMA3": 0.85, "BETA3": 0.8, "DELT3": 0.1}
        selection = _zone(["ALFA3", "BETA3"], frequencies, 2, 3, 0.11)
        self.assertEqual(selection.tickers, ("ALFA3", "BETA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "mantido", "BETA3": "mantido"})
        self.assertEqual(selection.departures, {})
        self.assertEqual(selection.kept, ("ALFA3", "BETA3"))
        self.assertEqual((selection.turnover, selection.exits), (0, 0))

    def test_incumbent_below_k_leaves_as_fora_do_top_k(self):  # PZ-07
        frequencies = {"ALFA3": 0.9, "GAMA3": 0.8, "BETA3": 0.7, "DELT3": 0.1}
        selection = _zone(["ALFA3", "BETA3"], frequencies, 2, 1, 0.11)
        self.assertEqual(selection.tickers, ("ALFA3", "GAMA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "mantido", "GAMA3": "novo"})
        self.assertEqual(selection.departures, {"BETA3": "saiu: fora do top K"})
        self.assertEqual(selection.kept, ("ALFA3",))
        self.assertEqual(selection.entered, ("GAMA3",))
        self.assertEqual((selection.turnover, selection.exits), (1, 1))

    def test_delta_boundary_is_inclusive_in_exact_fractions(self):  # PZ-05, edge case
        incumbents = ["QUIN3", "PAUL3"]
        swap = _zone(incumbents, _percent({"QUIN3": 60, "CARL3": 21, "PAUL3": 10}), 2, 3, 0.11)
        self.assertEqual(swap.tickers, ("QUIN3", "CARL3"))
        self.assertEqual(swap.reasons, {"QUIN3": "mantido", "CARL3": "substituiu PAUL3"})
        self.assertEqual(swap.departures, {"PAUL3": "saiu: substituído por CARL3"})
        self.assertEqual(swap.replaced, (("PAUL3", "CARL3"),))
        no_swap = _zone(incumbents, _percent({"QUIN3": 60, "CARL3": 20, "PAUL3": 10}), 2, 3, 0.11)
        self.assertEqual(no_swap.tickers, ("QUIN3", "PAUL3"))
        self.assertEqual(no_swap.replaced, ())

    def test_swap_happens_exactly_at_the_tolerance_boundary(self):  # PZ-05, edge case
        # f(c) equals f(p) + delta - 1e-12 in floats; the comparison is inclusive.
        boundary = 0.1 + 0.11 - 1e-12
        selection = _zone(
            ["QUIN3", "PAUL3"], {"QUIN3": 0.6, "CARL3": boundary, "PAUL3": 0.1}, 2, 3, 0.11
        )
        self.assertEqual(selection.tickers, ("QUIN3", "CARL3"))
        self.assertEqual(selection.replaced, (("PAUL3", "CARL3"),))

    def test_delta_zero_tie_favors_the_incumbent_on_equal_frequency(self):  # PZ-05
        frequencies = {"QUIN3": 0.9, "CARL3": 0.5, "PAUL3": 0.5}
        candidate_ahead = _zone(
            ["QUIN3", "PAUL3"], frequencies, 2, 3, 0.0,
            scores={"QUIN3": 1.0, "CARL3": 0.9, "PAUL3": 0.5},
        )
        self.assertEqual(candidate_ahead.tickers, ("QUIN3", "CARL3"))
        self.assertEqual(candidate_ahead.departures, {"PAUL3": "saiu: substituído por CARL3"})
        incumbent_ahead = _zone(
            ["QUIN3", "PAUL3"], frequencies, 2, 3, 0.0,
            scores={"QUIN3": 1.0, "CARL3": 0.5, "PAUL3": 0.9},
        )
        self.assertEqual(incumbent_ahead.tickers, ("QUIN3", "PAUL3"))
        self.assertEqual(incumbent_ahead.departures, {})

    def test_repeated_swaps_stop_at_the_first_failure(self):  # PZ-05
        frequencies = {
            "ALFA3": 0.9, "BETA3": 0.8, "GAMA3": 0.6, "DELT3": 0.4, "EPSI3": 0.25, "ZETA3": 0.2,
        }
        selection = _zone(["GAMA3", "DELT3", "ZETA3"], frequencies, 3, 10, 0.1)
        self.assertEqual(selection.tickers, ("ALFA3", "BETA3", "GAMA3"))
        self.assertEqual(
            selection.reasons,
            {"ALFA3": "substituiu ZETA3", "BETA3": "substituiu DELT3", "GAMA3": "mantido"},
        )
        self.assertEqual(
            selection.departures,
            {"DELT3": "saiu: substituído por BETA3", "ZETA3": "saiu: substituído por ALFA3"},
        )
        self.assertEqual(selection.replaced, (("ZETA3", "ALFA3"), ("DELT3", "BETA3")))
        self.assertEqual(selection.kept, ("GAMA3",))
        self.assertEqual(selection.entered, ("ALFA3", "BETA3"))
        self.assertEqual((selection.turnover, selection.exits), (2, 2))

    def test_replaced_incumbent_is_never_readmitted(self):  # PZ-05
        selection = _zone(["BETA3", "GAMA3"], {"ALFA3": 0.9, "BETA3": 0.5, "GAMA3": 0.3}, 2, 3, 0.1)
        self.assertEqual(selection.tickers, ("ALFA3", "BETA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "substituiu GAMA3", "BETA3": "mantido"})
        self.assertEqual(selection.departures, {"GAMA3": "saiu: substituído por ALFA3"})
        self.assertNotIn("GAMA3", selection.tickers)

    def test_ineligible_incumbent_is_never_protected_and_leaves_as_inelegivel(self):  # PZ-06
        selection = _zone(
            ["ALFA3", "BETA3"],
            {"ALFA3": 0.9, "BETA3": 0.8, "GAMA3": 0.5, "DELT3": 0.1},
            2,
            2,
            0.11,
            eligible={"ALFA3", "GAMA3", "DELT3"},
        )
        self.assertEqual(selection.tickers, ("ALFA3", "GAMA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "mantido", "GAMA3": "novo"})
        self.assertEqual(selection.departures, {"BETA3": "saiu: inelegível"})
        self.assertEqual((selection.turnover, selection.exits), (1, 1))

    def test_pomo3_ranked_above_pomo4_leaves_pomo4_and_enters_pomo3_as_novo(self):  # PZ-09
        selection = _zone(["POMO4"], {"POMO3": 0.9, "POMO4": 0.8, "ZETA3": 0.5}, 2, 2, 0.11)
        self.assertEqual(selection.tickers, ("POMO3", "ZETA3"))
        self.assertEqual(selection.reasons, {"POMO3": "novo", "ZETA3": "novo"})
        self.assertEqual(selection.departures, {"POMO4": "saiu: fora do top K"})
        self.assertEqual((selection.turnover, selection.exits), (2, 1))

    def test_pomo4_ranked_above_pomo3_is_protected_and_pomo3_is_never_selected(self):  # PZ-09
        selection = _zone(
            ["POMO4"], {"POMO4": 0.9, "POMO3": 0.8, "ZETA3": 0.5, "GAMA3": 0.2}, 3, 3, 0.11
        )
        self.assertEqual(selection.tickers, ("POMO4", "ZETA3", "GAMA3"))
        self.assertEqual(selection.reasons, {"POMO4": "mantido", "ZETA3": "novo", "GAMA3": "novo"})
        self.assertEqual(selection.departures, {})
        self.assertNotIn("POMO3", selection.reasons)

    def test_legacy_incumbents_of_one_company_protect_only_the_ranked_one(self):  # PZ-02, PZ-07
        selection = _zone(["POMO3", "POMO4"], {"POMO3": 0.9, "POMO4": 0.8, "ZETA3": 0.5}, 2, 3, 0.11)
        self.assertEqual(selection.tickers, ("POMO3", "ZETA3"))
        self.assertEqual(selection.reasons, {"POMO3": "mantido", "ZETA3": "novo"})
        self.assertEqual(selection.departures, {"POMO4": "saiu: fora do top K"})

    def test_capacity_cap_keeps_the_best_ranked_protected_incumbents(self):  # PZ-08
        tickers = [f"ABC{i:02d}3" for i in range(1, 16)]
        frequencies = {ticker: (100 - index) / 100 for index, ticker in enumerate(tickers)}
        selection = _zone(tickers, frequencies, 13, 20, 0.11)
        self.assertEqual(selection.tickers, tuple(tickers[:13]))
        self.assertEqual(
            selection.departures,
            {tickers[13]: "saiu: vagas reduzidas", tickers[14]: "saiu: vagas reduzidas"},
        )
        self.assertEqual(selection.kept, tuple(sorted(tickers[:13])))
        self.assertEqual((selection.turnover, selection.exits), (0, 2))
        self.assertEqual(selection.replaced, ())


class ZoneInvariantTests(unittest.TestCase):
    def test_incumbent_order_case_and_duplicates_do_not_change_the_result(self):  # PZ-12
        frequencies = {
            "ALFA3": 0.9, "BETA3": 0.8, "GAMA3": 0.6, "DELT3": 0.4, "EPSI3": 0.25, "ZETA3": 0.2,
        }
        incumbents = ["GAMA3", "DELT3", "ZETA3"]
        canonical = _zone(incumbents, frequencies, 3, 10, 0.1)
        variants = [
            list(reversed(incumbents)),
            ["zeta3", "Gama3", " delt3 "],
            incumbents + ["zeta3"],
            ["DELT3", "ZETA3", "GAMA3", "gama3"],
        ]
        for variant in variants:
            with self.subTest(incumbents=variant):
                self.assertEqual(_zone(variant, frequencies, 3, 10, 0.1), canonical)

    def test_every_incumbent_is_kept_or_departed_exactly_once(self):  # PZ-12, PZ-01
        rng = random.Random(4611)
        universe = [f"G{i:02d}3" for i in range(14)]
        for case in range(60):
            with self.subTest(case=case):
                frequencies = {t: rng.randint(0, 100) / 100 for t in universe if rng.random() < 0.8}
                scores = {t: rng.random() for t in universe}
                eligible = {t for t in universe if rng.random() < 0.85}
                incumbents = rng.sample(universe + ["OUTX3", "OUTY3"], rng.randint(1, 6))
                incumbents += [t.lower() for t in incumbents[:2]]
                n_assets = rng.randint(1, 6)
                K = rng.randint(0, 8)
                delta = rng.choice([0.0, 0.05, 0.11])
                selection = apply_protection_zone(
                    incumbents, frequencies, scores, n_assets, K, delta,
                    eligible=eligible, dedup_base_company=True,
                )
                normalized = {t.strip().upper() for t in incumbents}
                kept, departed = set(selection.kept), set(selection.departures)
                self.assertEqual(kept | departed, normalized)
                self.assertFalse(kept & departed)
                self.assertEqual(selection.turnover, len(selection.entered))
                self.assertEqual(selection.turnover, len(set(selection.tickers) - normalized))
                self.assertEqual(selection.exits, len(selection.departures))
                self.assertLessEqual(len(selection.tickers), n_assets)
                self.assertEqual(set(selection.reasons), set(selection.tickers))
                for reason in selection.departures.values():
                    self.assertTrue(reason.startswith("saiu: "), reason)
                for reason in selection.reasons.values():
                    self.assertTrue(
                        reason in ("mantido", "novo") or reason.startswith("substituiu "), reason
                    )

    def test_fewer_candidates_than_slots_returns_every_admitted_ticker(self):  # PZ-04, edge
        selection = _zone([], {"ALFA3": 0.5, "BETA3": 0.2}, 5, 0, 0.11)
        self.assertEqual(selection.tickers, ("ALFA3", "BETA3"))
        self.assertEqual(selection.reasons, {"ALFA3": "novo", "BETA3": "novo"})
        self.assertEqual(selection.turnover, 2)

    def test_invalid_zone_inputs_are_rejected_before_ranking(self):  # PZ-11
        with self.assertRaisesRegex(ValueError, "K"):
            _zone(["ALFA3"], {"ALFA3": 0.5}, 2, -1, 0.11)

    def test_module_imports_neither_pandas_nor_scripts(self):  # T2 done-when
        import ast

        source = (PROJECT_ROOT / "py" / "core" / "protection_zone.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {"pandas", "numpy", "scripts", "convergence_analysis"}, imported)


if __name__ == "__main__":
    unittest.main()
