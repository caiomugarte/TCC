# Context

## User decisions already made

- The target is Caio's whole-portfolio allocation, not another stock-ranking experiment.
- Caio's stock parameters matter as profile calibration, but they are not copied into incompatible classes and the stock analyzer is not reused as the allocation objective.
- The initial international comparison uses only S&P 500 exposure; it must not be described as the global market.
- Taxes and costs are explicitly excluded for now.
- Optimization is allowed to produce zero weight for a class and has no imposed maximum per class. Diversification is evaluated as an outcome through HHI/frontier analysis.
- The preferred decision rule is risk-adjusted total-return thinking with explicit volatility and drawdown caps, while the specified core objective is gross annualized nominal BRL return under those caps.

## Reference artifacts

- Stock portfolio: `outputs/carteira_caio_consensus.json`.
- Stock pipeline: `py/pipelines/multi_run.py` and `py/core/optimizer.py`.
- Existing stock backtest: `py/backtest_analysis.py`, useful for conventions and artifacts but not the new allocation engine.

## Assumptions to keep visible

- IFIX and DI are proxies, not a complete universe of FIIs or fixed-income products.
- Bitcoin represents crypto in the first version because a consistent long history is needed.
- The current target comes from the latest training window; the adaptive backtest is a separate historical policy.
- S&P 500 and BTC conversions need one documented PTAX alignment rule.

