"""Segment cap and HHI skip in the protection-zone module (issue #56, AD-027..AD-029)."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.protection_zone import (  # noqa: E402
    SegmentCapInfeasibleError,
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
        for value in (0, 0.0, -0.1, float("nan"), float("inf"), "0.25"):
            with self.subTest(hhi_max=value):
                with self.assertRaisesRegex(ValueError, "^hhi_max"):
                    take_within_segment_cap(
                        ordered, 2, segment_of=_segments(*ordered), max_per_segment=2, hhi_max=value
                    )


if __name__ == "__main__":
    unittest.main()
