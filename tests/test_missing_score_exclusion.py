"""Issue #52: empty DY is 0, stocks still without SCORE leave the universe."""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from config import METRIC_GROUPS, PROFILE_WEIGHTS  # noqa: E402
from core.preprocessing import apply_robustness_filter, preprocess_profile  # noqa: E402
from core.scoring import build_scores  # noqa: E402
from fetch_status_invest import SOURCE_COLUMNS as STOCK_COLUMNS  # noqa: E402
from pipelines.multi_run import (  # noqa: E402
    build_consensus_portfolio,
    run_multi_execution_profile,
)
from pipelines.single_run import run_stock_selection  # noqa: E402


FILTERS = {"cap_min": 100.0, "liq_min": 100.0}
WEIGHTS = dict(PROFILE_WEIGHTS["caio_new"])


def _frame(no_dy=(), no_liquidity=(), tiny=()) -> pd.DataFrame:
    rows = []
    tickers = ("AAA3", "BBB3", "CCC3", "DDD3", "EEE3", "FFF3")
    for index, ticker in enumerate(tickers, start=1):
        row = {column: float(index) + 0.1 * index * index for column in STOCK_COLUMNS}
        size = 1.0 if ticker in tiny else 1000.0
        row.update(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "DY": float("nan") if ticker in no_dy else float(index),
                "VALOR DE MERCADO": size,
                "LIQUIDEZ MEDIA DIARIA": size,
                "SETOR": "Sector A",
            }
        )
        if ticker in no_liquidity:
            for column in METRIC_GROUPS["liquidez"]:
                row[column] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=[*STOCK_COLUMNS, "SETOR"])


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    clean = apply_robustness_filter(preprocess_profile(frame, filters=FILTERS))
    return build_scores(clean, factor_weights=WEIGHTS)


class EmptyDividendYieldTests(unittest.TestCase):
    def test_empty_dy_is_zero_before_standardisation(self):
        clean = preprocess_profile(_frame(no_dy=("CCC3",)), filters=FILTERS)

        self.assertTrue(clean["DY"].notna().all())
        by_ticker = clean.set_index("TICKER")["DY"]
        self.assertEqual(by_ticker.idxmin(), "CCC3")
        self.assertLess(by_ticker["CCC3"], by_ticker.drop("CCC3").min())

    def test_stock_without_dy_gets_finite_score_with_worst_dividend_score(self):
        ranked = _ranked(_frame(no_dy=("CCC3",)))

        self.assertEqual(len(ranked), 6)
        self.assertTrue(ranked["SCORE"].map(math.isfinite).all())
        avg_div = ranked.set_index("TICKER")["AVG_DIV"]
        self.assertEqual(avg_div["CCC3"], avg_div.min())


class MissingScoreExclusionTests(unittest.TestCase):
    def test_stock_without_a_whole_group_leaves_with_reason(self):
        ranked = _ranked(_frame(no_liquidity=("BBB3",), tiny=("FFF3",)))

        self.assertNotIn("BBB3", set(ranked["TICKER"]))
        self.assertTrue(ranked["SCORE"].notna().all())
        reasons = ranked.attrs["exclusion_reasons"]
        self.assertEqual(reasons["BBB3"], ["missing_metric_group:liquidez"])
        self.assertIn("FFF3", reasons)  # eligibility reasons survive scoring
        self.assertEqual(ranked.attrs["eligibility_filters"]["cap_min"], 100.0)
        self.assertEqual(list(ranked["SCORE"]), sorted(ranked["SCORE"], reverse=True))

    def test_named_profile_scoring_also_excludes(self):
        clean = apply_robustness_filter(
            preprocess_profile(_frame(no_liquidity=("BBB3",)), filters=FILTERS)
        )
        ranked = build_scores(clean, "moderado")

        self.assertNotIn("BBB3", set(ranked["TICKER"]))
        self.assertIn("BBB3", ranked.attrs["exclusion_reasons"])

    def _config(self) -> dict:
        return {
            "selection_preset": "fixture",
            "factor_weights": WEIGHTS,
            "liquidity_and_size_filters": FILTERS,
            "system_ga_config": {
                "n_assets": 2,
                "lambda": 0.0,
                "generations": 3,
                "pop_size": 4,
            },
        }

    def test_single_run_records_missing_score_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "stocks.csv"
            _frame(no_liquidity=("BBB3",)).to_csv(input_path, index=False)
            result = run_stock_selection(
                input_path=input_path, config=self._config(), seed=7
            )

        self.assertNotIn("BBB3", result["selected_tickers"])
        self.assertEqual(
            result["exclusion_reasons"]["BBB3"], ["missing_metric_group:liquidez"]
        )

    def test_multi_run_records_missing_score_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "stocks.csv"
            _frame(no_liquidity=("BBB3",)).to_csv(input_path, index=False)
            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                with redirect_stdout(StringIO()):
                    result = run_multi_execution_profile(
                        input_path=input_path,
                        config=self._config(),
                        n_runs=2,
                        parallel=False,
                        use_cache=False,
                        save_interval=1,
                        workspace=root / "workspace",
                        seed=7,
                    )

        self.assertNotIn("BBB3", result["selected_tickers"])
        for run in result["all_runs"]:
            self.assertNotIn("BBB3", run["tickers"])
        self.assertEqual(
            result["exclusion_reasons"]["BBB3"], ["missing_metric_group:liquidez"]
        )


class ConsensusTieBreakTests(unittest.TestCase):
    def test_tie_break_is_frequency_then_score_then_ticker(self):
        df_ranked = pd.DataFrame(
            {
                "TICKER": ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3"],
                "SETOR": ["S1", "S2", "S3", "S4", "S5"],
                "SCORE": [0.5, 0.9, 0.7, 0.7, 2.0],
            }
        )
        # EEEE3 once; the other four twice each, so the top 3 come from a
        # frequency tie: BBBB3 (0.9), then CCCC3 and DDDD3 tie at 0.7 -> ticker.
        results = [
            {"tickers": ["AAAA3", "BBBB3", "CCCC3", "DDDD3"]},
            {"tickers": ["DDDD3", "CCCC3", "BBBB3", "AAAA3"]},
            {"tickers": ["EEEE3"]},
        ]
        config = {"n_assets": 3}
        expected = {"BBBB3", "CCCC3", "DDDD3"}

        rng = random.Random(52)
        for _ in range(20):
            shuffled = [
                {"tickers": rng.sample(r["tickers"], len(r["tickers"]))}
                for r in rng.sample(results, len(results))
            ]
            with redirect_stdout(StringIO()):
                consensus = build_consensus_portfolio(
                    shuffled, df_ranked, selection_config=config
                )
            self.assertEqual(set(consensus["TICKER"]), expected)

        with redirect_stdout(StringIO()):
            two = build_consensus_portfolio(
                results, df_ranked, selection_config={"n_assets": 2}
            )
        self.assertEqual(set(two["TICKER"]), {"BBBB3", "CCCC3"})


SNAPSHOT_REGISTRY = os.environ.get("TCC_SNAPSHOT_REGISTRY")


@unittest.skipUnless(
    SNAPSHOT_REGISTRY, "set TCC_SNAPSHOT_REGISTRY to the registry holding dev-2026-10-29-ibov"
)
class PremiumSnapshotTests(unittest.TestCase):
    def test_premium_0582_ranking_has_no_nan_and_drops_only_no_liquidity(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import benchmark_premium
        import multirun_convergence
        from app.adapters.premium_optimization import _selector_config
        from core.preprocessing import load_raw_data

        protocol = json.loads(
            (PROJECT_ROOT / "docs/validation/multirun-convergence-protocol.json").read_text()
        )
        manifest, snapshot = benchmark_premium.verify_snapshot(
            benchmark_premium.load_declaration(PROJECT_ROOT / protocol["snapshot_declaration"]),
            Path(SNAPSHOT_REGISTRY),
        )
        record = protocol["profiles"]["premium-2026-11-02"]["record"]
        policy = multirun_convergence.resolve_policy(record, manifest, protocol["seeds"][0])
        config = _selector_config(policy["stocks"])
        inputs = benchmark_premium.selector_inputs_from_manifest(manifest, snapshot)

        clean = apply_robustness_filter(
            preprocess_profile(load_raw_data(Path(inputs.stocks.path)), config=config)
        )
        ranked = build_scores(clean, factor_weights=config["factor_weights"])

        self.assertEqual(int(ranked["SCORE"].isna().sum()), 0)
        excluded = set(clean["TICKER"]) - set(ranked["TICKER"])
        self.assertEqual(excluded, {"BAZA3", "BMEB4"})
        for ticker in excluded:
            self.assertEqual(
                ranked.attrs["exclusion_reasons"][ticker],
                ["missing_metric_group:liquidez"],
            )


if __name__ == "__main__":
    unittest.main()
