"""Segment cap and HHI skip in the protection-zone module (issue #56, AD-027..AD-029)."""

from pathlib import Path
import sys
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

import core.protection_zone as protection_zone  # noqa: E402
from core.protection_zone import (  # noqa: E402
    SEGMENT_CAP_DEPARTURE,
    SegmentCapInfeasibleError,
    apply_protection_zone,
    select_with_fallback,
    take_within_segment_cap,
)


def _segments(*tickers: str) -> dict[str, str]:
    """Segment of each synthetic ticker: its leading letter (P1 -> "P")."""

    return {ticker: ticker[0] for ticker in tickers}


class TakeWithinSegmentCapTests(unittest.TestCase):
    def test_take_without_cap_is_the_plain_prefix(self):  # SEG-15, SEG-18
        ordered = ["P1", "P2", "P3", "Q1", "P4"]
        self.assertEqual(
            take_within_segment_cap(ordered, 3, segment_of=_segments(*ordered), max_per_segment=None),
            ["P1", "P2", "P3"],
        )
        self.assertEqual(
            take_within_segment_cap(ordered, 3, segment_of=None, max_per_segment=None),
            ["P1", "P2", "P3"],
        )

    def test_take_skips_a_full_segment_and_continues(self):  # SEG-15
        ordered = ["P1", "P2", "P3", "Q1", "P4"]
        self.assertEqual(
            take_within_segment_cap(ordered, 3, segment_of=_segments(*ordered), max_per_segment=2),
            ["P1", "P2", "Q1"],
        )

    def test_take_allows_exactly_the_cap(self):  # SEG-15 (boundary)
        ordered = ["P1", "P2", "Q1"]
        self.assertEqual(
            take_within_segment_cap(ordered, 2, segment_of=_segments(*ordered), max_per_segment=2),
            ["P1", "P2"],
        )

    def test_invalid_segment_cap_inputs_raise(self):  # SEG-25 (one input per guard, L-001)
        ordered = ["P1", "Q1"]
        cases = [
            ("^max_per_segment", {"segment_of": _segments(*ordered), "max_per_segment": 0}),
            ("^max_per_segment", {"segment_of": _segments(*ordered), "max_per_segment": 1.5}),
            ("^max_per_segment", {"segment_of": _segments(*ordered), "max_per_segment": True}),
            ("^segment_of", {"segment_of": None, "max_per_segment": 2}),
            ("^segment_of", {"segment_of": {"P1": "P"}, "max_per_segment": 2}),
        ]
        for pattern, kwargs in cases:
            with self.subTest(pattern=pattern, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    take_within_segment_cap(ordered, 2, **kwargs)

    def test_take_skips_a_ticker_that_would_break_hhi_max(self):  # SEG-37
        ordered = ["P1", "P2", "Q1", "R1", "S1", "T1"]
        # P2 would give 2+1+1+1 at n 5: HHI (4+1+1+1)/25 = 0.28 > 0.25.
        self.assertEqual(
            take_within_segment_cap(
                ordered, 5, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=0.25
            ),
            ["P1", "Q1", "R1", "S1", "T1"],
        )

    def test_take_allows_one_pair_at_n6_under_hhi_max(self):  # SEG-37, A-16
        ordered = ["P1", "P2", "Q1", "Q2", "R1", "S1", "T1"]
        # One pair at n 6: (4+1+1+1+1)/36 = 0.22; a second pair: (4+4+1+1)/36 = 0.28.
        self.assertEqual(
            take_within_segment_cap(
                ordered, 6, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=0.25
            ),
            ["P1", "P2", "Q1", "R1", "S1", "T1"],
        )

    def test_take_accepts_hhi_exactly_at_the_limit(self):  # SEG-37 (boundary)
        ordered = ["P1", "Q1", "R1", "S1"]
        self.assertEqual(
            take_within_segment_cap(
                ordered, 4, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=0.25
            ),
            ["P1", "Q1", "R1", "S1"],
        )

    def test_take_raises_instead_of_returning_a_short_list(self):  # SEG-40
        ordered = ["P1", "P2", "Q1", "Q2", "R1"]
        with self.assertRaisesRegex(SegmentCapInfeasibleError, "hhi_max") as caught:
            take_within_segment_cap(
                ordered, 4, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=0.25
            )
        self.assertIsInstance(caught.exception, ValueError)

    def test_hhi_max_without_cap_raises(self):  # SEG-41 (one input per guard, L-001)
        ordered = ["P1", "Q1"]
        with self.assertRaisesRegex(ValueError, "^hhi_max"):
            take_within_segment_cap(
                ordered, 2, segment_of=_segments(*ordered), max_per_segment=None, hhi_max=0.25
            )
        with self.assertRaisesRegex(ValueError, "^hhi_max"):
            _cap_zone(["P1"], {"P1": 0.9, "Q1": 0.8}, 2, 3, 0.11, max_per_segment=None, hhi_max=0.25)
        for value in (0, 0.0, -0.1, float("nan"), float("inf"), "0.25"):
            with self.subTest(hhi_max=value):
                with self.assertRaisesRegex(ValueError, "^hhi_max"):
                    take_within_segment_cap(
                        ordered, 2, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=value
                    )


def _cap_zone(incumbents, frequencies, n_assets, K, delta, *, max_per_segment=2, scores=None, **extra):
    """apply_protection_zone over letter segments, every candidate eligible, score 1.0, no dedup."""

    return apply_protection_zone(
        incumbents,
        frequencies,
        {t: 1.0 for t in frequencies} if scores is None else scores,
        n_assets,
        K,
        delta,
        eligible=set(frequencies) | set(incumbents),
        dedup_base_company=False,
        segment_of=_segments(*frequencies),
        max_per_segment=max_per_segment,
        **extra,
    )


def _segment_counts(tickers) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ticker in tickers:
        counts[ticker[0]] = counts.get(ticker[0], 0) + 1
    return counts


class ZoneSegmentCapTests(unittest.TestCase):
    def test_third_incumbent_of_a_segment_leaves_with_the_segment_reason(self):  # SEG-19
        frequencies = {"P1": 0.9, "P2": 0.85, "P3": 0.8, "Q1": 0.7, "R1": 0.6, "S1": 0.5}
        selection = _cap_zone(["P1", "P2", "P3"], frequencies, 5, 5, 0.11)
        self.assertEqual(SEGMENT_CAP_DEPARTURE, "saiu: teto do segmento")
        self.assertEqual(selection.kept, ("P1", "P2"))
        self.assertEqual(selection.departures, {"P3": "saiu: teto do segmento"})
        self.assertEqual(selection.exits, 1)
        self.assertEqual(selection.tickers, ("P1", "P2", "Q1", "R1", "S1"))

    def test_two_incumbents_of_a_segment_both_stay(self):  # SEG-19 (boundary: exactly the cap)
        frequencies = {"P1": 0.9, "P2": 0.85, "Q1": 0.7, "R1": 0.6}
        selection = _cap_zone(["P1", "P2"], frequencies, 3, 4, 0.11)
        self.assertEqual(selection.kept, ("P1", "P2"))
        self.assertEqual(selection.reasons, {"P1": "mantido", "P2": "mantido", "Q1": "novo"})
        self.assertEqual(selection.departures, {})

    def test_segment_cut_runs_before_the_capacity_cut(self):  # SEG-20, A-6
        frequencies = {"P1": 0.9, "P2": 0.8, "P3": 0.7, "Q1": 0.6}
        selection = _cap_zone(["P1", "P2", "P3", "Q1"], frequencies, 3, 4, 0.11)
        self.assertEqual(selection.tickers, ("P1", "P2", "Q1"))
        self.assertEqual(selection.kept, ("P1", "P2", "Q1"))
        self.assertEqual(selection.departures, {"P3": "saiu: teto do segmento"})

        frequencies = {"P1": 0.9, "Q1": 0.8, "R1": 0.7, "S1": 0.6}
        selection = _cap_zone(["P1", "Q1", "R1", "S1"], frequencies, 3, 4, 0.11)
        self.assertEqual(selection.tickers, ("P1", "Q1", "R1"))
        self.assertEqual(selection.departures, {"S1": "saiu: vagas reduzidas"})

    def test_fill_skips_a_full_segment(self):  # SEG-21
        frequencies = {"P3": 0.9, "P1": 0.8, "P2": 0.7, "Q2": 0.6}
        selection = _cap_zone(["P1", "P2"], frequencies, 3, 4, 0.5)
        self.assertEqual(selection.tickers, ("P1", "P2", "Q2"))
        self.assertEqual(selection.reasons, {"P1": "mantido", "P2": "mantido", "Q2": "novo"})
        self.assertEqual(selection.entered, ("Q2",))

    def test_challenger_that_would_break_the_cap_is_skipped(self):  # SEG-22
        # Weakest protected Q1; the best outsider P3 sits in full P; R1 is next and leads by >= delta.
        frequencies = {"P3": 0.9, "R1": 0.88, "P1": 0.87, "P2": 0.86, "Q1": 0.5}
        selection = _cap_zone(["P1", "P2", "Q1"], frequencies, 3, 6, 0.1)
        self.assertEqual(selection.replaced, (("Q1", "R1"),))
        self.assertEqual(selection.tickers, ("R1", "P1", "P2"))
        self.assertEqual(selection.departures, {"Q1": "saiu: substituído por R1"})

    def test_challenger_of_the_weakest_segment_fits(self):  # SEG-22 (held minus the weakest)
        frequencies = {"P3": 0.9, "Q1": 0.85, "P1": 0.8, "P2": 0.5}
        selection = _cap_zone(["Q1", "P1", "P2"], frequencies, 3, 6, 0.11)
        self.assertEqual(selection.replaced, (("P2", "P3"),))
        self.assertEqual(selection.tickers, ("P3", "Q1", "P1"))
        self.assertEqual(selection.departures, {"P2": "saiu: substituído por P3"})

    def test_migration_from_13_incumbents_to_5_partitions_every_incumbent(self):  # SEG-19, SEG-20, edge
        frequencies = {
            "P1": 0.95, "Q1": 0.93, "P2": 0.91, "P3": 0.89, "X1": 0.88, "Q2": 0.86, "R1": 0.84,
            "S1": 0.82, "P4": 0.80, "Q3": 0.78, "R2": 0.70, "Q4": 0.60, "R3": 0.50, "P5": 0.30,
            "X2": 0.20,
        }
        incumbents = ["P1", "P2", "P3", "P4", "P5", "Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "S1"]
        selection = _cap_zone(incumbents, frequencies, 5, 8, 0.11)
        self.assertEqual(selection.tickers, ("P1", "Q1", "P2", "Q2", "R1"))
        self.assertEqual(
            selection.departures,
            {
                "P3": "saiu: teto do segmento",
                "S1": "saiu: vagas reduzidas",
                "P4": "saiu: fora do top K", "Q3": "saiu: fora do top K",
                "R2": "saiu: fora do top K", "Q4": "saiu: fora do top K",
                "R3": "saiu: fora do top K", "P5": "saiu: fora do top K",
            },
        )
        self.assertEqual(set(selection.kept) | set(selection.departures), set(incumbents))
        self.assertFalse(set(selection.kept) & set(selection.departures))
        self.assertLessEqual(max(_segment_counts(selection.tickers).values()), 2)
        self.assertEqual(selection.turnover, len(selection.entered))
        self.assertEqual(selection.exits, len(selection.departures))

    def test_cap_at_or_above_n_assets_equals_uncapped(self):  # edge (max_per_segment >= n_assets)
        frequencies = {"A1": 0.9, "B1": 0.85, "A2": 0.8, "A3": 0.75, "C1": 0.7, "B2": 0.5}
        incumbents = ["A2", "A3", "C1", "B2"]
        uncapped = apply_protection_zone(
            incumbents, frequencies, {t: 1.0 for t in frequencies}, 2, 5, 0.11,
            eligible=set(frequencies), dedup_base_company=False,
        )
        # The fixture exercises a swap, a capacity cut and an out-of-K departure.
        self.assertEqual(uncapped.replaced, (("A3", "A1"),))
        self.assertEqual(
            uncapped.departures,
            {"A3": "saiu: substituído por A1", "C1": "saiu: vagas reduzidas", "B2": "saiu: fora do top K"},
        )
        for cap in (2, 3):
            with self.subTest(max_per_segment=cap):
                self.assertEqual(_cap_zone(incumbents, frequencies, 2, 5, 0.11, max_per_segment=cap), uncapped)

    def test_none_cap_equals_the_call_without_keywords(self):  # SEG-24
        frequencies = {"A1": 0.9, "B1": 0.85, "A2": 0.8, "A3": 0.75, "A4": 0.72, "C1": 0.7, "B2": 0.5}
        incumbents = ["A2", "A3", "A4", "C1", "B2"]
        for n_assets in (2, 3, 4):
            with self.subTest(n_assets=n_assets):
                without = apply_protection_zone(
                    incumbents, frequencies, {t: 1.0 for t in frequencies}, n_assets, 5, 0.11,
                    eligible=set(frequencies), dedup_base_company=False,
                )
                self.assertEqual(
                    _cap_zone(incumbents, frequencies, n_assets, 5, 0.11, max_per_segment=None),
                    without,
                )
                # The same fixture capped at 2 differs, so the None path is not the cap path.
                if n_assets >= 3:
                    self.assertNotEqual(_cap_zone(incumbents, frequencies, n_assets, 5, 0.11), without)



class ZoneHhiSkipTests(unittest.TestCase):
    def test_zone_rejects_an_invalid_hhi_max_with_a_cap(self):  # SEG-41 (zone side)
        for value in (0, float("nan")):
            with self.subTest(hhi_max=value):
                with self.assertRaisesRegex(ValueError, "^hhi_max"):
                    _cap_zone(["P1"], {"P1": 0.9, "Q1": 0.8}, 2, 3, 0.11, hhi_max=value)

    def test_fill_skips_a_ticker_that_would_break_hhi_max(self):  # SEG-38
        frequencies = {"P1": 0.9, "P2": 0.85, "Q1": 0.8, "R1": 0.7, "S1": 0.6, "T1": 0.5}
        selection = _cap_zone(["P1"], frequencies, 5, 8, 0.11, hhi_max=0.25)
        self.assertEqual(selection.tickers, ("P1", "Q1", "R1", "S1", "T1"))
        self.assertEqual(selection.kept, ("P1",))

    def test_challenger_that_would_break_hhi_max_is_skipped(self):  # SEG-38
        # Weakest protected Q1. P2 leads it by >= delta but a P pair does not fit at n 5
        # under 0.25; R2 is next and leads by >= delta, so R2 replaces Q1.
        frequencies = {"P1": 0.95, "P2": 0.9, "R2": 0.85, "S1": 0.8, "T1": 0.75, "U1": 0.7, "Q1": 0.5}
        selection = _cap_zone(["P1", "S1", "T1", "U1", "Q1"], frequencies, 5, 8, 0.11, hhi_max=0.25)
        self.assertEqual(selection.replaced, (("Q1", "R2"),))
        self.assertEqual(selection.tickers, ("P1", "R2", "S1", "T1", "U1"))
        self.assertNotIn("P2", selection.tickers)

    def test_challenger_hhi_check_excludes_the_weakest(self):  # SEG-38 (held minus the weakest)
        frequencies = {"Q2": 0.9, "P1": 0.85, "R1": 0.8, "S1": 0.75, "T1": 0.7, "Q1": 0.5}
        selection = _cap_zone(["P1", "R1", "S1", "T1", "Q1"], frequencies, 5, 8, 0.11, hhi_max=0.25)
        self.assertEqual(selection.replaced, (("Q1", "Q2"),))
        self.assertEqual(selection.tickers, ("Q2", "P1", "R1", "S1", "T1"))

    def test_protected_pair_over_hhi_max_keeps_the_higher_ranked_incumbent(self):  # SEG-39, A-5
        frequencies = {"P1": 0.9, "Q1": 0.8, "P2": 0.7, "R1": 0.6, "S1": 0.5, "T1": 0.4}
        selection = _cap_zone(["P1", "Q1", "P2"], frequencies, 5, 8, 0.11, hhi_max=0.25)
        self.assertEqual(selection.kept, ("P1", "Q1"))
        self.assertEqual(selection.departures, {"P2": "saiu: teto do segmento"})
        self.assertEqual(set(selection.tickers), {"P1", "Q1", "R1", "S1", "T1"})

        # Equal frequency: the higher SCORE ranks first and stays (rank tie-break).
        frequencies = {"P1": 0.9, "P2": 0.9, "Q1": 0.8, "R1": 0.6, "S1": 0.5, "T1": 0.4}
        scores = {"P1": 0.5, "P2": 0.8, "Q1": 1.0, "R1": 1.0, "S1": 1.0, "T1": 1.0}
        selection = _cap_zone(["P1", "Q1", "P2"], frequencies, 5, 8, 0.11, hhi_max=0.25, scores=scores)
        self.assertEqual(selection.kept, ("P2", "Q1"))
        self.assertEqual(selection.departures, {"P1": "saiu: teto do segmento"})
        self.assertEqual(set(selection.tickers), {"P2", "Q1", "R1", "S1", "T1"})

    def test_incumbent_past_capacity_is_cut_by_capacity_not_by_hhi(self):  # SEG-20, SEG-39
        frequencies = {"P1": 0.9, "Q1": 0.8, "R1": 0.7, "S1": 0.6, "P2": 0.5}
        selection = _cap_zone(["P1", "Q1", "R1", "S1", "P2"], frequencies, 4, 8, 0.11, hhi_max=0.25)
        self.assertEqual(selection.kept, ("P1", "Q1", "R1", "S1"))
        self.assertEqual(selection.departures, {"P2": "saiu: vagas reduzidas"})

    def test_zone_fill_short_raises_instead_of_a_short_sleeve(self):  # SEG-43
        frequencies = {"P1": 0.9, "P2": 0.8, "Q1": 0.7, "Q2": 0.6, "R1": 0.5}
        with self.assertRaisesRegex(SegmentCapInfeasibleError, "hhi_max"):
            _cap_zone(["P1"], frequencies, 4, 8, 0.11, hhi_max=0.25)



def _segment_hhi(tickers) -> float:
    counts = _segment_counts(tickers)
    return sum((c / len(tickers)) ** 2 for c in counts.values())


def _cap_select(incumbents, frequencies, n_assets, K, delta, *, hhi_of=_segment_hhi, hhi_max=None,
                max_per_segment=2, scores=None):
    """select_with_fallback over letter segments, every candidate eligible, no dedup."""

    return select_with_fallback(
        incumbents,
        frequencies,
        {t: 1.0 for t in frequencies} if scores is None else scores,
        n_assets,
        {"K": K, "delta": delta},
        eligible=set(frequencies) | set(incumbents),
        dedup_base_company=False,
        hhi_of=hhi_of,
        hhi_max=hhi_max,
        segment_of=_segments(*frequencies),
        max_per_segment=max_per_segment,
    )


# Zone protects R1 (rank 5 <= K 5) and fills P1, P2; the capped K = 0 consensus skips P3
# (P full) and takes Q1. Uncapped, the consensus would be P1, P2, P3.
FALLBACK_CAP_FREQUENCIES = {"P1": 0.9, "P2": 0.85, "P3": 0.8, "Q1": 0.7, "R1": 0.6}


class SelectWithFallbackSegmentCapTests(unittest.TestCase):
    def test_fallback_consensus_respects_the_cap(self):  # SEG-23
        hhi_of = lambda tickers: 0.95 if set(tickers) == {"P1", "P2", "R1"} else 0.1  # noqa: E731
        selection, record = _cap_select(
            ["R1"], FALLBACK_CAP_FREQUENCIES, 3, 5, 0.5, hhi_of=hhi_of, hhi_max=0.9
        )
        self.assertEqual(selection.tickers, ("P1", "P2", "Q1"))
        self.assertEqual(selection.departures, {"R1": "saiu: fora do top K"})
        self.assertEqual((record["status"], record["buffer_fallback"]), ("fallback_hhi", "hhi"))

    def test_recorded_consensus_tickers_are_capped(self):  # SEG-23, A-13
        selection, record = _cap_select(
            ["R1"], FALLBACK_CAP_FREQUENCIES, 3, 5, 0.5, hhi_of=lambda tickers: 0.1, hhi_max=0.9
        )
        self.assertEqual(selection.tickers, ("P1", "P2", "R1"))
        self.assertEqual(record["status"], "applied")
        self.assertEqual(record["consensus_tickers"], ["P1", "P2", "Q1"])
        self.assertLessEqual(max(_segment_counts(record["consensus_tickers"]).values()), 2)
        _, uncapped = _cap_select(
            ["R1"], FALLBACK_CAP_FREQUENCIES, 3, 5, 0.5, hhi_of=lambda tickers: 0.1, hhi_max=0.9,
            max_per_segment=None,
        )
        self.assertEqual(uncapped["consensus_tickers"], ["P1", "P2", "P3"])
        self.assertEqual(list(record), list(uncapped))

    def test_protected_same_segment_pair_keeps_the_protection_without_fallback(self):  # SEG-39, SEG-43
        frequencies = {
            "P1": 0.90, "P2": 0.85, "Q1": 0.80, "R1": 0.70, "S1": 0.60, "T1": 0.50, "U1": 0.45,
        }
        selection, record = _cap_select(["P1", "P2", "U1"], frequencies, 5, 8, 0.11, hhi_max=0.25)
        self.assertEqual(record["status"], "applied")
        self.assertIsNone(record["buffer_fallback"])
        self.assertEqual(selection.tickers, ("P1", "Q1", "R1", "S1", "U1"))
        self.assertEqual(selection.kept, ("P1", "U1"))
        self.assertEqual(selection.departures, {"P2": "saiu: teto do segmento"})
        self.assertAlmostEqual(_segment_hhi(selection.tickers), 0.20, places=12)
        # U1's protection survives: T1 leads it by 0.05 < delta, so the zone differs from K = 0.
        self.assertEqual(record["consensus_tickers"], ["P1", "Q1", "R1", "S1", "T1"])

    def test_segment_cap_infeasible_error_propagates_from_both_calls(self):  # SEG-43, AD-029
        original = protection_zone.apply_protection_zone
        frequencies = {"P1": 0.9, "Q1": 0.8, "R1": 0.7, "S1": 0.6, "T1": 0.5}
        for failing_k in ("zone", "consensus"):
            def stub(*args, **kwargs):
                is_zone = args[4] > 0
                if is_zone == (failing_k == "zone"):
                    raise SegmentCapInfeasibleError(f"stub {failing_k} cannot fill")
                return original(*args, **kwargs)

            with self.subTest(failing=failing_k):
                with mock.patch.object(protection_zone, "apply_protection_zone", side_effect=stub):
                    with self.assertRaisesRegex(SegmentCapInfeasibleError, f"stub {failing_k}"):
                        _cap_select(["P1"], frequencies, 4, 7, 0.11, hhi_max=0.25)

    def test_stock_path_never_forwards_hhi_max(self):  # SEG-41, SEG-24
        frequencies = {"ALFA3": 0.9, "BETA3": 0.8, "GAMA3": 0.7, "DELT3": 0.6, "EPSI3": 0.1}
        original = protection_zone.apply_protection_zone
        with mock.patch.object(
            protection_zone, "apply_protection_zone", side_effect=original
        ) as spy:
            selection, record = select_with_fallback(
                ["GAMA3"], frequencies, {t: 1.0 for t in frequencies}, 2, {"K": 3, "delta": 0.11},
                eligible=set(frequencies), dedup_base_company=True,
                hhi_of=lambda tickers: 0.2, hhi_max=0.25,
            )
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            self.assertNotIn("hhi_max", call.kwargs)
            self.assertNotIn("max_per_segment", call.kwargs)
        self.assertEqual(selection.tickers, ("ALFA3", "GAMA3"))
        self.assertEqual(
            record,
            {
                "status": "applied",
                "K": 3,
                "k_offset": 1,
                "delta": 0.11,
                "n_assets": 2,
                "incumbents": ["GAMA3"],
                "kept": ["GAMA3"],
                "replaced": [],
                "entered": ["ALFA3"],
                "departures": {},
                "turnover": 1,
                "exits": 0,
                "buffer_fallback": None,
                "consensus_tickers": ["ALFA3", "BETA3"],
                "consensus_turnover": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
