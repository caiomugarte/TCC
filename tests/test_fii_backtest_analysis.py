import json
from datetime import date, datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from fii_backtest_analysis import (  # noqa: E402
    build_parser,
    run_period_backtest,
    write_outputs,
)


class FiiBacktestTests(unittest.TestCase):
    def test_backtest_uses_selected_fiis_and_optional_ifix_only_as_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            portfolio_path = Path(directory) / "fiis.json"
            portfolio_path.write_text(
                json.dumps([{"TICKER": "AAA11"}, {"TICKER": "BBB11"}]),
                encoding="utf-8",
            )
            dates = pd.date_range("2020-01-01", periods=3, freq="D")
            prices = pd.DataFrame(
                {
                    "AAA11.SA": [100.0, 110.0, 120.0],
                    "BBB11.SA": [100.0, 105.0, 110.0],
                },
                index=dates,
            )

            with patch("fii_backtest_analysis.fetch_historical_data", return_value=(prices, [])):
                with patch(
                    "fii_backtest_analysis.fetch_ifix_levels",
                    return_value={
                        date(2020, 1, 1): 100.0,
                        date(2020, 1, 2): 102.0,
                        date(2020, 1, 3): 104.0,
                    },
                ):
                    result = run_period_backtest(
                        portfolio_path,
                        1,
                        end_date=datetime(2020, 1, 3),
                        analyze_assets=False,
                    )

            self.assertEqual(result["tickers"], ["AAA11", "BBB11"])
            self.assertEqual(result["available_tickers"], ["AAA11", "BBB11"])
            self.assertAlmostEqual(result["values"].iloc[-1], 11500.0)
            self.assertAlmostEqual(result["metrics"]["retorno_total_pct"], 15.0)
            self.assertEqual(result["benchmark_name"], "IFIX")
            self.assertTrue(result["benchmark_metrics"])

    def test_write_outputs_keeps_fii_prefix_and_portfolio_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            portfolio_path = Path(directory) / "fiis.json"
            portfolio_path.write_text(json.dumps([{"TICKER": "AAA11"}]), encoding="utf-8")
            dates = pd.date_range("2020-01-01", periods=3, freq="D")
            values = pd.Series([10000.0, 10100.0, 10200.0], index=dates)
            result = {
                "period_years": 1,
                "start_date": "2019-12-01",
                "end_date": "2020-01-03",
                "tickers": ["AAA11"],
                "available_tickers": ["AAA11"],
                "missing_tickers": [],
                "values": values,
                "returns": values.pct_change().dropna(),
                "metrics": {"retorno_total_pct": 2.0},
                "asset_analysis": pd.DataFrame(),
                "benchmark_name": "IFIX",
                "benchmark_values": pd.Series(dtype=float),
                "benchmark_metrics": {},
                "relative_metrics": {},
                "benchmark_error": None,
            }

            output_dir = Path(directory) / "outputs"
            paths = write_outputs(
                {"1ano": result},
                "caio",
                portfolio_path,
                output_dir,
            )

            metrics_path = output_dir / "fii_backtest_metrics_caio.json"
            series_path = output_dir / "fii_backtest_series_caio_1ano.csv"
            self.assertIn(metrics_path, paths)
            self.assertTrue(metrics_path.exists())
            self.assertTrue(series_path.exists())
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["portfolio_path"], str(portfolio_path))
            self.assertEqual(payload["periods"]["1ano"]["tickers"], ["AAA11"])

    def test_cli_defaults_to_five_and_ten_year_fii_analysis(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.profile, "caio")
        self.assertEqual(args.years, [5, 10])
        self.assertFalse(args.skip_ifix)


if __name__ == "__main__":
    unittest.main()
