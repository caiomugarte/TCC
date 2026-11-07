"""Segment cap in the GA fitness (issue #56, SEG-09..SEG-14)."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.optimizer import GeneticAlgorithm  # noqa: E402


def _ga(**overrides) -> GeneticAlgorithm:
    settings = {"n_assets": 3, "lambda_hhi": 0.5, "generations": 5, "pop_size": 4, "random_seed": 0}
    settings.update(overrides)
    return GeneticAlgorithm(**settings)


def _frame(segments, scores=None) -> pd.DataFrame:
    scores = scores if scores is not None else [1.0 - 0.05 * i for i in range(len(segments))]
    return pd.DataFrame({
        "TICKER": [f"T{i:02d}11" for i in range(len(segments))],
        "SETOR": list(segments),
        "SCORE": list(scores),
    })


def _golden_frame() -> pd.DataFrame:
    """Fixed synthetic frame whose GA output was captured at a61164a (SEG-14)."""

    segments = ["Papéis", "Papéis", "Lajes", "Papéis", "Logística", "Lajes", "Híbrido",
                "Shoppings", "Papéis", "Logística", "Híbrido", "Lajes", "Shoppings", "Indefinido"]
    scores = [0.95, 0.91, 0.88, 0.84, 0.80, 0.77, 0.73, 0.70, 0.66, 0.61, 0.57, 0.52, 0.48, 0.44]
    return pd.DataFrame({
        "TICKER": [f"FII{i:02d}11" for i in range(len(scores))],
        "SETOR": segments,
        "SCORE": scores,
    })


# Captured by running the a61164a GeneticAlgorithm(n_assets=5, lambda_hhi=0.3,
# generations=40, pop_size=20) on _golden_frame() before optimizer.py was edited.
PRE_CHANGE_GOLDEN = {
    3: {
        "tickers": ["FII0011", "FII0111", "FII0211", "FII0411", "FII0711"],
        "fitness": 3.8200000000000003,
        "hhi": 0.28,
        "generations_run": 40,
    },
    11: {
        "tickers": ["FII0011", "FII0111", "FII0211", "FII0411", "FII0611"],
        "fitness": 3.8499999999999996,
        "hhi": 0.28,
        "generations_run": 40,
    },
}


class SegmentCapFitnessTests(unittest.TestCase):
    def test_over_cap_candidate_is_infeasible(self):  # SEG-09
        frame = _frame(["Papéis", "Papéis", "Papéis", "Lajes"])
        mask = np.array([1, 1, 1, 0])
        self.assertEqual(_ga(max_per_segment=2).fitness(frame, mask), -np.inf)

    def test_indefinido_is_capped_like_any_segment(self):  # SEG-09, A-4
        frame = _frame(["Indefinido", "Indefinido", "Indefinido", "Lajes"])
        mask = np.array([1, 1, 1, 0])
        self.assertEqual(_ga(max_per_segment=2).fitness(frame, mask), -np.inf)

    def test_candidate_at_cap_scores_like_the_uncapped_ga(self):  # SEG-10
        frame = _frame(["Papéis", "Papéis", "Lajes", "Lajes", "Híbrido"], [0.9, 0.8, 0.7, 0.6, 0.5])
        mask = np.array([1, 1, 1, 1, 0])
        capped = _ga(n_assets=4, max_per_segment=2).fitness(frame, mask)
        uncapped = _ga(n_assets=4).fitness(frame, mask)
        # SCORE sum 3.0, HHI (2/4)^2 + (2/4)^2 = 0.5, penalty 0.5 * 0.5 * 4 = 1.0.
        self.assertEqual(capped, uncapped)
        self.assertAlmostEqual(capped, 3.0 - 0.5 * 0.5 * 4, places=12)

    def test_optimize_never_returns_more_than_cap_per_segment(self):  # SEG-11
        segments = ["Papéis"] * 6 + ["Lajes", "Lajes", "Híbrido", "Shoppings", "Logística", "Indefinido"]
        frame = _frame(segments, [1.0 - 0.04 * i for i in range(len(segments))])
        uncapped_max = []
        for seed in range(10):
            with self.subTest(seed=seed):
                capped = GeneticAlgorithm(
                    n_assets=5, lambda_hhi=0.0, generations=30, pop_size=20,
                    random_seed=seed, max_per_segment=2,
                ).optimize(frame)
                self.assertLessEqual(int(capped["SETOR"].value_counts().max()), 2)
                uncapped = GeneticAlgorithm(
                    n_assets=5, lambda_hhi=0.0, generations=30, pop_size=20, random_seed=seed,
                ).optimize(frame)
                uncapped_max.append(int((uncapped["SETOR"] == "Papéis").sum()))
        # The frame binds: without the cap at least one seed holds 3+ "Papéis".
        self.assertGreaterEqual(max(uncapped_max), 3)

    def test_invalid_cap_raises(self):  # SEG-12
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_per_segment"):
                    _ga(max_per_segment=value)
        self.assertIsNone(_ga(max_per_segment=None).max_per_segment)
        self.assertEqual(_ga(max_per_segment=2).max_per_segment, 2)

    def test_cap_without_setor_column_raises(self):  # SEG-13
        frame = pd.DataFrame({"TICKER": ["A", "B", "C"], "SCORE": [1.0, 0.9, 0.8]})
        with self.assertRaisesRegex(ValueError, "SETOR"):
            _ga(max_per_segment=2).fitness(frame, np.array([1, 1, 1]))


class UncappedGoldenTests(unittest.TestCase):
    def test_uncapped_ga_matches_the_pre_change_golden(self):  # SEG-14
        for seed, golden in PRE_CHANGE_GOLDEN.items():
            for label, extra in (("omitted", {}), ("None", {"max_per_segment": None})):
                with self.subTest(seed=seed, keyword=label):
                    portfolio = GeneticAlgorithm(
                        n_assets=5, lambda_hhi=0.3, generations=40, pop_size=20,
                        random_seed=seed, **extra,
                    ).optimize(_golden_frame())
                    self.assertEqual(list(portfolio["TICKER"]), golden["tickers"])
                    self.assertEqual(float(portfolio.attrs["fitness"]), golden["fitness"])
                    self.assertEqual(portfolio.attrs["hhi"], golden["hhi"])
                    self.assertEqual(portfolio.attrs["generations_run"], golden["generations_run"])
                    self.assertEqual(portfolio.attrs["seed"], seed)


if __name__ == "__main__":
    unittest.main()
