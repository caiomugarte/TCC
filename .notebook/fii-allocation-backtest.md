# FII Allocation Backtest
> Optimized FII artifact becomes the allocation `fiis` sleeve; IFIX optional

Entry: `py/allocation_data.py:load_snapshot_bundle()`

Current boundary: stock allocation reads `outputs/carteira_caio_consensus.json`
→ `data/allocation/caio_stocks.csv`; FII allocation reads
`outputs/carteira_fii_caio_consensus.json` → `data/allocation/caio_fiis.csv` →
`build_equal_weight_sleeve()` for the `fiis` class. IFIX is optional benchmark
data only.

Fetch boundary: FII artifact → Yahoo `<TICKER>.SA` adjusted levels →
`data/allocation/caio_fiis.csv` → same annual equal-weight sleeve builder →
unchanged five-class allocation/walk-forward.

Standalone analysis: `py/fii_backtest_analysis.py` reads the same fixed FII
artifact and produces 5/10-year FII-only buy-and-hold metrics, per-FII
distribution analysis, time-series/drawdown charts, and optional IFIX
benchmark comparison under `outputs/fii_backtest_*`. It does not rerun FII
selection or five-class allocation.

Decisions: fixed current optimized FII artifact across history, matching
stocks; known look-ahead disclosed. Allocation chooses only class-level `fiis`
weight. IFIX optional benchmark-only and never the `fiis` source when FII
portfolio data exists.

Spec/design/tasks: `.specs/features/fii-allocation-backtest/`

Updated: 2026-08-11
