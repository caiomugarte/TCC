"""AD-022 equivalence (PZ-13): the zone function reproduces the oracle and the #9 lots.

``scripts/convergence_analysis.zone_portfolio`` is the reference the #9 report was
computed with. Production code never imports ``scripts/``; the test does.
"""

import math
from pathlib import Path
import random
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import convergence_analysis as ca  # noqa: E402
from core.protection_zone import apply_protection_zone, rank_candidates  # noqa: E402

PROTOCOL_PATH = PROJECT_ROOT / "docs" / "validation" / "multirun-convergence-protocol.json"
LOTS_DIR = PROJECT_ROOT / "docs" / "validation" / "multirun-convergence" / "lots"
PROTOCOL, PROTOCOL_SHA256 = ca.load_protocol(PROTOCOL_PATH)

# Recommended delta of #9 (AD-024, Q5); the zone values of A-2.
RECOMMENDED_DELTA = 0.11
DELTA_GRID = list(PROTOCOL["decision"]["zone"]["delta_grid"]) + [RECOMMENDED_DELTA]


def _random_case(rng: random.Random, selector: str):
    """Candidates over distinct companies; stocks may list two classes of one company.

    Half the cases use the previous consensus (the top of the ranking) as incumbents,
    with up to two replaced, because that is where a held incumbent outranks the
    first candidate outside the portfolio. Half use small run counts, so ties are common.
    """

    suffixes = ("3", "4", "5", "6", "11") if selector == "stocks" else ("11",)
    tickers: list[str] = []
    for company in rng.sample(range(100), rng.randint(14, 24)):
        classes = 2 if selector == "stocks" and rng.random() < 0.3 else 1
        tickers.extend(f"C{company:02d}{suffix}" for suffix in rng.sample(suffixes, classes))
    k = rng.randint(3, 8) if rng.random() < 0.5 else rng.randint(9, 150)
    counts = {ticker: rng.randint(1, k) for ticker in tickers}
    scores = {ticker: math.nan if rng.random() < 0.15 else rng.uniform(0.0, 10.0) for ticker in tickers}
    n = rng.randint(1, 13)
    ranking = ca.extended_ranking(counts, scores, selector)
    if rng.random() < 0.5:
        incumbents = ranking[:n]
        for _ in range(rng.randint(0, 2)):
            outside = [ticker for ticker in tickers if ticker not in incumbents]
            incumbents[rng.randrange(n)] = rng.choice(outside)
    else:
        incumbents = rng.sample(tickers, n)
    return k, counts, scores, incumbents, n, ranking


class RandomGridTests(unittest.TestCase):
    def test_portfolio_and_turnover_match_zone_portfolio_on_random_cases(self):  # PZ-13
        rng = random.Random(4650)
        for selector in ("stocks", "fiis"):
            dedup = selector == "stocks"
            for case in range(20):
                k, counts, scores, incumbents, n, ranking = _random_case(rng, selector)
                frequencies = {ticker: count / k for ticker, count in counts.items()}
                with self.subTest(selector=selector, case=case):
                    self.assertEqual(rank_candidates(frequencies, scores, dedup_base_company=dedup), ranking)
                    for K in [0, *range(n, 31)]:
                        for delta in DELTA_GRID:
                            selection = apply_protection_zone(
                                incumbents, frequencies, scores, n, K, delta,
                                eligible=set(counts), dedup_base_company=dedup,
                            )
                            held, turnover = ca.zone_portfolio(
                                incumbents, ranking, counts, k, K, delta, selector
                            )
                            with self.subTest(K=K, delta=delta):
                                self.assertEqual(len(selection.tickers), n)
                                self.assertEqual(set(selection.tickers), held)
                                self.assertEqual(selection.turnover, turnover)


class LotsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lots = ca.load_lots(PROTOCOL, PROTOCOL_SHA256, LOTS_DIR)

    def test_recommended_point_reproduces_the_report_turnover_on_the_lots(self):  # PZ-13
        prefix = PROTOCOL["ga_override"]["run_count"]
        seeds = list(PROTOCOL["seeds"])
        # (mean, max) of turnover over ordered seed pairs, and the recommended k_offset.
        cells = {
            ("arrojado-anchor", "stocks"): (0.40, 1, 6),
            ("premium-2026-11-02", "stocks"): (0.10, 1, 6),
            ("arrojado-anchor", "fiis"): (0.70, 2, 3),
            ("premium-2026-11-02", "fiis"): (0.00, 0, 3),
        }
        for (profile_id, selector), (mean, maximum, k_offset) in cells.items():
            with self.subTest(profile=profile_id, selector=selector):
                series = {
                    seed: ca.SeedSeries(
                        ca.lot_id(profile_id, seed),
                        selector,
                        self.lots[(profile_id, seed)]["selectors"][selector],
                        PROTOCOL,
                    )
                    for seed in seeds
                }
                n = series[seeds[0]].n
                K = n + k_offset
                turnovers = []
                for incumbent_seed in seeds:
                    for new_seed in seeds:
                        if incumbent_seed == new_seed:
                            continue
                        incumbents = series[incumbent_seed].consensus(prefix)
                        new = series[new_seed]
                        counts = new.counts(prefix)
                        frequencies = {ticker: count / prefix for ticker, count in counts.items()}
                        selection = apply_protection_zone(
                            incumbents, frequencies, new.scores, n, K, RECOMMENDED_DELTA,
                            eligible=set(new.candidates) | set(incumbents),
                            dedup_base_company=selector == "stocks",
                        )
                        held, turnover = ca.zone_portfolio(
                            incumbents, new.ranking(prefix), counts, prefix, K, RECOMMENDED_DELTA, selector
                        )
                        self.assertEqual(set(selection.tickers), held)
                        self.assertEqual(selection.turnover, turnover)
                        turnovers.append(selection.turnover)
                self.assertAlmostEqual(sum(turnovers) / len(turnovers), mean, places=12)
                self.assertEqual(max(turnovers), maximum)


if __name__ == "__main__":
    unittest.main()
