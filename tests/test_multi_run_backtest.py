import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "py"))

from pipelines.multi_run import run_backtest_comparison  # noqa: E402


class MultiRunBacktestTests(unittest.TestCase):
    def test_backtest_rejects_portfolio_with_missing_price_series(self):
        dates = pd.date_range(
            pd.Timestamp.now() - pd.Timedelta(days=395),
            periods=390,
            freq="D",
        )
        prices = pd.DataFrame(
            {("Close", "AAA.SA"): range(100, 490)},
            index=dates,
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)
        portfolio = pd.DataFrame({"TICKER": ["AAA", "BBB"]})

        with patch("yfinance.download", return_value=prices):
            result = run_backtest_comparison(
                portfolio,
                portfolio,
                "test",
                period_years=1,
            )

        self.assertEqual(result, {})
