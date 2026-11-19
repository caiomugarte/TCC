"""Fitness do GA avaliado em lote (issue #95).

A avaliação da população inteira com numpy tem de dar o mesmo float, bit a bit, que o
``GeneticAlgorithm.fitness`` cromossomo a cromossomo; e o ``optimize`` tem de devolver a
mesma carteira de antes da mudança.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from core.optimizer import GeneticAlgorithm  # noqa: E402

STOCK_SECTORS = ["Financeiro", "Energia", "Saneamento", "Varejo", "Mineração", "Petróleo", "Saúde", "Telecom"]
FII_SEGMENTS = ["Papéis", "Lajes", "Logística", "Shoppings", "Híbrido", "Indefinido"]


def _frame(m: int, sectors: list, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "TICKER": [f"T{i:03d}" for i in range(m)],
        "SETOR": [sectors[int(k)] for k in rng.integers(0, len(sectors), m)],
        "SCORE": np.sort(rng.normal(0, 1, m))[::-1],
    })
    # O df_ranked de produção carrega o attrs da preparação; o GA não pode depender dele.
    frame.attrs["exclusion_reasons"] = {"X": "y"}
    return frame


def _population(m: int, n_assets: int, size: int, seed: int) -> list:
    """Cromossomos aleatórios, ~10% com número errado de ativos."""

    rng = np.random.default_rng(seed)
    population = []
    for _ in range(size):
        chromosome = np.zeros(m, dtype=int)
        k = n_assets if rng.random() < 0.9 else int(rng.integers(1, m))
        chromosome[rng.choice(m, size=k, replace=False)] = 1
        population.append(chromosome)
    return population


CONSTRAINTS = {
    "sem teto": {},
    "teto de 2 por segmento": {"max_per_segment": 2},
    "hhi_max 0,25": {"hhi_max": 0.25},
}

UNIVERSES = {
    "ações": (lambda: _frame(186, STOCK_SECTORS, 1), 13, 0.5),
    "FIIs": (lambda: _frame(57, FII_SEGMENTS, 2), 5, 0.3),
}


class FitnessBatchEquivalenceTests(unittest.TestCase):
    def _assert_identical(self, ga: GeneticAlgorithm, frame: pd.DataFrame, population: list) -> None:
        expected = np.array([ga.fitness(frame, chromosome) for chromosome in population])
        actual = ga.fitness_batch(frame, population)
        self.assertEqual(actual.dtype, np.float64)
        # assert_array_equal trata NaN == NaN e compara -inf exatamente; floats finitos sem tolerância.
        np.testing.assert_array_equal(actual, expected)

    def test_batch_matches_per_chromosome_fitness_bit_for_bit(self):
        for universe, (make_frame, n_assets, lambda_hhi) in UNIVERSES.items():
            frame = make_frame()
            population = _population(len(frame), n_assets, 3000, seed=7)
            for label, constraint in CONSTRAINTS.items():
                with self.subTest(universe=universe, constraint=label):
                    ga = GeneticAlgorithm(
                        n_assets=n_assets, lambda_hhi=lambda_hhi, generations=1, pop_size=2,
                        random_seed=0, **constraint,
                    )
                    expected = np.array([ga.fitness(frame, c) for c in population])
                    # O caso só prova algo se tiver viáveis e inviáveis.
                    self.assertTrue(np.isfinite(expected).any())
                    self.assertTrue(np.isneginf(expected).any())
                    self._assert_identical(ga, frame, population)

    def test_missing_sector_and_score_follow_the_pandas_rules(self):
        # value_counts ignora SETOR vazio (normaliza pelos preenchidos) e sum ignora SCORE vazio.
        frame = _frame(40, FII_SEGMENTS, 5)
        frame.loc[[1, 4, 9, 20], "SETOR"] = None
        frame.loc[[2, 9, 30], "SCORE"] = np.nan
        population = _population(len(frame), 5, 2000, seed=8)
        all_missing = np.zeros(len(frame), dtype=int)
        all_missing[[1, 4, 9, 20]] = 1
        all_missing[0] = 1
        frame.loc[0, "SETOR"] = None
        population.append(all_missing)
        for label, constraint in CONSTRAINTS.items():
            with self.subTest(constraint=label):
                ga = GeneticAlgorithm(
                    n_assets=5, lambda_hhi=0.3, generations=1, pop_size=2, random_seed=0, **constraint,
                )
                self._assert_identical(ga, frame, population)

    def test_hhi_one_ulp_above_hhi_max_stays_feasible(self):
        # Contagens 2, 2, 1 dão HHI 0.3600000000000001: só a tolerância de 1e-12 o aceita com hhi_max 0,36.
        frame = pd.DataFrame({
            "TICKER": ["A", "B", "C", "D", "E", "F"],
            "SETOR": ["Papéis", "Papéis", "Lajes", "Lajes", "Híbrido", "Híbrido"],
            "SCORE": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        })
        population = [np.array([1, 1, 1, 1, 1, 0]), np.array([1, 1, 1, 0, 1, 1])]
        ga = GeneticAlgorithm(n_assets=5, lambda_hhi=0.3, generations=1, pop_size=2, random_seed=0, hhi_max=0.36)
        self.assertTrue(np.isfinite(ga.fitness_batch(frame, population)).all())
        self._assert_identical(ga, frame, population)

    def test_frame_without_setor_column(self):
        frame = _frame(20, FII_SEGMENTS, 6).drop(columns="SETOR")
        population = _population(len(frame), 5, 200, seed=9)
        for label, constraint in (("sem teto", {}), ("hhi_max 0,25", {"hhi_max": 0.25})):
            with self.subTest(constraint=label):
                ga = GeneticAlgorithm(
                    n_assets=5, lambda_hhi=0.3, generations=1, pop_size=2, random_seed=0, **constraint,
                )
                self._assert_identical(ga, frame, population)
        capped = GeneticAlgorithm(
            n_assets=5, lambda_hhi=0.3, generations=1, pop_size=2, random_seed=0, max_per_segment=2,
        )
        with self.assertRaisesRegex(ValueError, "SETOR"):
            capped.fitness_batch(frame, population)


# Capturado com o GeneticAlgorithm de develop em 2f42049 (fitness cromossomo a cromossomo),
# generations=60, pop_size=30, early_stopping_patience=15, antes da vetorização.
PRE_VECTORIZATION_GOLDEN = {
    ("stocks", 3): (["T000", "T001", "T002", "T003", "T005", "T007", "T010", "T012", "T013", "T014",
                     "T027", "T028", "T035"], 9.411681298102618, 0.18343195266272194, 28),
    ("stocks", 11): (["T000", "T001", "T005", "T006", "T007", "T009", "T010", "T011", "T012", "T013",
                      "T017", "T021", "T038"], 9.233566057954414, 0.2071005917159764, 35),
    ("fiis_cap", 3): (["T000", "T001", "T002", "T003", "T005"], 6.22638522219651, 0.28, 35),
    ("fiis_cap", 11): (["T000", "T001", "T002", "T004", "T007"], 6.080361962208642, 0.28, 24),
    ("fiis_hhimax", 3): (["T000", "T001", "T002", "T006", "T007"], 5.572864942989614, 0.20000000000000004, 24),
    ("fiis_hhimax", 11): (["T000", "T001", "T004", "T006", "T007"], 5.509316828449299, 0.20000000000000004, 19),
}

GOLDEN_CASES = {
    "stocks": (lambda: _frame(60, STOCK_SECTORS, 1), {"n_assets": 13, "lambda_hhi": 0.5}),
    "fiis_cap": (lambda: _frame(30, FII_SEGMENTS, 2), {"n_assets": 5, "lambda_hhi": 0.3, "max_per_segment": 2}),
    "fiis_hhimax": (lambda: _frame(30, FII_SEGMENTS, 3),
                    {"n_assets": 5, "lambda_hhi": 0.3, "hhi_max": 0.25, "max_per_segment": 2}),
}


class OptimizeRegressionTests(unittest.TestCase):
    def test_same_seed_returns_the_pre_vectorization_portfolio(self):
        for (case, seed), (tickers, fitness, hhi, generations) in PRE_VECTORIZATION_GOLDEN.items():
            make_frame, params = GOLDEN_CASES[case]
            with self.subTest(case=case, seed=seed):
                portfolio = GeneticAlgorithm(
                    generations=60, pop_size=30, random_seed=seed, early_stopping_patience=15, **params,
                ).optimize(make_frame())
                self.assertEqual(list(portfolio["TICKER"]), tickers)
                self.assertEqual(float(portfolio.attrs["fitness"]), fitness)
                self.assertEqual(portfolio.attrs["hhi"], hhi)
                self.assertEqual(portfolio.attrs["generations_run"], generations)


if __name__ == "__main__":
    unittest.main()
