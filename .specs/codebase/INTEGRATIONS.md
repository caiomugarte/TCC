# Integrations

- Existing stock fundamentals are local CSV/JSON/Parquet artifacts.
- Existing price backtests use `yfinance` and may add `.SA` to Brazilian tickers.
- The allocation feature needs external market snapshots for B3 IFIX, B3 DI, BCB PTAX, S&P 500 total return, and BTC/USD. Each snapshot must carry source, retrieval date, cutoff date, currency, and transformation metadata.
- External downloads must be isolated from the pure optimizer and must never silently fill missing prices or replace a missing Caio ticker.

