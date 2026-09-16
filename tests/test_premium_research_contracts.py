from contextlib import redirect_stdout
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from allocation_config import ASSET_CLASSES  # noqa: E402
from core.allocation import DailyReturn  # noqa: E402
from core.optimizer import derive_run_seed, optimize_portfolio  # noqa: E402
from fetch_status_invest import SOURCE_COLUMNS as STOCK_COLUMNS  # noqa: E402
from fetch_status_invest_fii import SOURCE_COLUMNS as FII_COLUMNS  # noqa: E402
from fii_selection import FII_PROFILE_WEIGHTS, run_fii_selection  # noqa: E402
from core.preprocessing import preprocess_profile  # noqa: E402
from core.scoring import build_scores  # noqa: E402
from pipelines.asset_allocation import optimize_window  # noqa: E402
from pipelines.multi_run import run_multi_execution_profile  # noqa: E402
from pipelines.single_run import run_stock_selection  # noqa: E402
from config import PROFILE_WEIGHTS  # noqa: E402


def _stock_frame() -> pd.DataFrame:
    rows = []
    for index, (ticker, sector, market_cap, liquidity) in enumerate((
        ("AAA3", "Sector A", 1000.0, 1000.0),
        ("BBB3", "Sector A", 1000.0, 1000.0),
        ("CCC3", "Sector B", 1000.0, 1000.0),
        ("DDD3", "Sector B", 1000.0, 1000.0),
        ("BAD3", "Sector B", 1.0, 1.0),
    ), start=1):
        row = {column: float(index) for column in STOCK_COLUMNS}
        row.update(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "DY": float(index),
                "VALOR DE MERCADO": market_cap,
                "LIQUIDEZ MEDIA DIARIA": liquidity,
            }
        )
        row["SETOR"] = sector
        rows.append(row)
    return pd.DataFrame(rows, columns=[*STOCK_COLUMNS, "SETOR"])


def _fii_frame() -> pd.DataFrame:
    rows = []
    for ticker, sector, liquidity in (
        ("AAA11", "Segment A", 100000.0),
        ("BBB11", "Segment A", 80000.0),
        ("CCC11", "Segment A", 60000.0),
        ("DDD11", "Segment B", 90000.0),
        ("EEE11", "Segment B", 70000.0),
        ("FFF11", "Segment B", 50000.0),
    ):
        row = {column: 1.0 for column in FII_COLUMNS if column != "SETOR"}
        row.update(
            {
                "TICKER": ticker,
                "PRECO": 100.0,
                "ULTIMO DIVIDENDO": 1.0,
                "DY": 2.0,
                "P/VP": 1.0,
                "LIQUIDEZ MEDIA DIARIA": liquidity,
                "SETOR": sector,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*FII_COLUMNS, "SETOR"])


def _stock_config() -> dict[str, object]:
    return {
        "selection_preset": "fixture",
        "factor_weights": dict(PROFILE_WEIGHTS["caio_new"]),
        "liquidity_and_size_filters": {"cap_min": 100.0, "liq_min": 100.0},
        "system_ga_config": {
            "n_assets": 2,
            "lambda": 0.0,
            "generations": 3,
            "pop_size": 4,
        },
    }


def _fii_config() -> dict[str, object]:
    return {
        "selection_preset": "fixture",
        "factor_weights": dict(FII_PROFILE_WEIGHTS["caio"]),
        "liquidity_and_size_filters": {"liquidity_min": 90000.0},
        "system_ga_config": {
            "n_assets": 2,
            "lambda": 0.0,
            "generations": 3,
            "pop_size": 4,
        },
    }


def _allocation_rows(count: int = 5):
    return tuple(
        DailyReturn(
            date(2020, 1, 1) + timedelta(days=index),
            {name: 0.0 for name in ASSET_CLASSES},
        )
        for index in range(count)
    )


class PremiumResearchContractTests(unittest.TestCase):
    def test_named_caio_stock_weights_remain_usable(self):
        ranked = build_scores(
            preprocess_profile(
                _stock_frame(),
                filters={"cap_min": 100.0, "liq_min": 100.0},
            ),
            "caio",
        )

        self.assertTrue(ranked["SCORE"].notna().all())

    def test_allocation_applies_fixed_income_and_hhi_constraints(self):
        config = {
            "volatility_cap": 1.0,
            "drawdown_cap": 1.0,
            "coarse_step": 0.05,
            "refinement_step": 0.01,
            "refinement_radius": 0.02,
            "hhi_penalties": (0.0,),
            "class_constraints": {
                "minimums": {"fixed_income": 0.40},
                "hhi_max": 0.25,
            },
        }

        window = optimize_window(_allocation_rows(), config=config)

        self.assertTrue(window.frontier)
        for candidate in window.frontier:
            self.assertGreaterEqual(candidate.weights[3], 0.40)
            self.assertLessEqual(candidate.hhi, 0.25 + 1e-12)

    def test_infeasible_class_constraints_return_diagnostic(self):
        config = {
            "volatility_cap": 1.0,
            "drawdown_cap": 1.0,
            "coarse_step": 0.05,
            "refinement_step": 0.01,
            "refinement_radius": 0.02,
            "hhi_penalties": (0.0,),
            "class_constraints": {
                "minimums": {"fixed_income": 0.40},
                "maximums": {"crypto": 0.0},
                "hhi_max": 0.25,
            },
        }

        window = optimize_window(_allocation_rows(), config=config)

        self.assertEqual(window.frontier, ())
        self.assertEqual(window.diagnostic["code"], "infeasible_constraints")

    def test_optimizer_replay_uses_stable_seed(self):
        ranked = pd.DataFrame(
            {
                "TICKER": ["AAA3", "BBB3", "CCC3", "DDD3"],
                "SETOR": ["A", "A", "B", "B"],
                "SCORE": [4.0, 3.0, 2.0, 1.0],
            }
        )
        config = {
            "system_ga_config": {
                "n_assets": 2,
                "lambda": 0.2,
                "generations": 4,
                "pop_size": 4,
            }
        }

        first = optimize_portfolio(ranked, config=config, seed=19)
        second = optimize_portfolio(ranked, config=config, seed=19)

        self.assertEqual(first["TICKER"].tolist(), second["TICKER"].tolist())
        self.assertEqual(first.attrs["seed"], 19)
        self.assertEqual(derive_run_seed(19, 0, "stock"), derive_run_seed(19, 0, "stock"))
        self.assertNotEqual(derive_run_seed(19, 0, "stock"), derive_run_seed(19, 0, "fii"))

    def test_explicit_stock_multi_run_isolated_and_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "stocks.csv"
            _stock_frame().to_csv(input_path, index=False)
            config = _stock_config()
            first_workspace = root / "first"
            second_workspace = root / "second"

            with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
                with redirect_stdout(StringIO()):
                    first = run_multi_execution_profile(
                        input_path=input_path,
                        config=config,
                        n_runs=2,
                        parallel=False,
                        use_cache=False,
                        save_interval=1,
                        workspace=first_workspace,
                        seed=19,
                    )
                    second = run_multi_execution_profile(
                        input_path=input_path,
                        config=config,
                        n_runs=2,
                        parallel=False,
                        use_cache=False,
                        save_interval=1,
                        workspace=second_workspace,
                        seed=19,
                    )

            self.assertEqual(first["selected_tickers"], second["selected_tickers"])
            self.assertEqual(
                [run["seed"] for run in first["all_runs"]],
                [run["seed"] for run in second["all_runs"]],
            )
            for workspace in (first_workspace, second_workspace):
                self.assertTrue((workspace / "run-0" / "portfolio.json").exists())
                self.assertTrue((workspace / "run-1" / "portfolio.json").exists())
                self.assertFalse((workspace / ".checkpoint.json").exists())

    def test_explicit_stock_selection_returns_sleeve_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "stocks.csv"
            workspace = root / "stock-workspace"
            _stock_frame().to_csv(input_path, index=False)

            result = run_stock_selection(
                input_path=input_path,
                config=_stock_config(),
                workspace=workspace,
                seed=19,
            )

            self.assertEqual(len(result["selected_tickers"]), 2)
            self.assertAlmostEqual(sum(result["sleeve_weights"].values()), 1.0)
            self.assertIn("BAD3", result["exclusion_reasons"])
            self.assertTrue((workspace / "stock_selection.json").exists())

    def test_explicit_fii_selection_records_filters_and_isolated_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fiis.csv"
            _fii_frame().to_csv(input_path, index=False)
            workspace = root / "fii-workspace"

            result = run_fii_selection(
                input_path=input_path,
                config=_fii_config(),
                n_runs=2,
                parallel=False,
                workspace=workspace,
                seed=19,
            )

            self.assertEqual(set(result["selected_tickers"]), {"AAA11", "DDD11"})
            self.assertAlmostEqual(sum(result["sleeve_weights"].values()), 1.0)
            self.assertIn("BBB11", result["exclusion_reasons"])
            self.assertTrue((workspace / "run-0" / "portfolio.json").exists())
            self.assertTrue((workspace / "run-1" / "portfolio.json").exists())


if __name__ == "__main__":
    unittest.main()
