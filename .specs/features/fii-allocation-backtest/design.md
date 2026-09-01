# FII Allocation Backtest Design

**Spec:** `.specs/features/fii-allocation-backtest/spec.md`
**Context:** `.specs/features/fii-allocation-backtest/context.md`
**Status:** Implemented

## Architecture Overview

The existing allocation pipeline remains the numerical owner. The change is at
the snapshot boundary: fetch and materialize selected FII ticker histories,
then construct the FII class exactly like the stock class.

`FII consensus artifact → snapshot fetch → caio_fiis.csv → snapshot loader →
equal-weight FII sleeve → existing five-class allocation/walk-forward`

IFIX may still be written as an optional benchmark file, but no allocation
return row is built from it when `caio_fiis.csv` is available.

## Reuse Analysis

| Existing component | Reuse |
| --- | --- |
| `fetch_allocation_snapshot.fetch_yahoo_levels()` | FII `.SA` adjusted-close history |
| `fetch_allocation_snapshot._write_levels()` | Multi-ticker FII level CSV |
| `fetch_allocation_snapshot._write_metadata()` | Source, artifact, skips, bias disclosure |
| `allocation_data.load_caio_tickers()` pattern | Generic validated ticker-artifact reader |
| `allocation_data.build_equal_weight_sleeve()` | FII internal annual-rebalanced sleeve |
| `allocation_data.levels_to_returns()` | Common-date simple returns |
| `pipelines.asset_allocation.run_allocation()` | Same current/primary/robustness/walk-forward optimizer |
| `core.allocation` | Unchanged allocation objective and metrics |

## Components

### Snapshot FII history

- **Location:** `py/fetch_allocation_snapshot.py`
- **Purpose:** Fetch selected FII histories beside stock histories.
- **Interface:** Extend `fetch_snapshot()` with an FII portfolio path and an
  optional IFIX toggle.
- **Output:** `data/allocation/caio_fiis.csv`, one column per selected ticker.
- **Failure:** Permanent Yahoo no-data skips and records; transient/API/schema
  failures abort; no replacement ticker is selected.

### Portfolio artifact reader

- **Location:** `py/allocation_data.py`
- **Purpose:** Validate stock and FII consensus JSON ticker identity.
- **Interface:** Generic ticker loader with labels; preserve the current stock
  loader behavior through a compatibility wrapper if needed.
- **Validation:** Existing file, JSON list, non-empty unique `TICKER`s.

### Snapshot bundle construction

- **Location:** `py/allocation_data.py`
- **Purpose:** Build stock and FII sleeves from their selected ticker levels.
- **Interface:** Extend `load_snapshot_bundle()` with the FII portfolio path.
- **Behavior:** Load `caio_fiis.csv`, reject missing selected columns, call
  `build_equal_weight_sleeve()`, and map the result to `ASSET_CLASSES[1]`.
- **IFIX:** Optional file; never used for `ASSET_CLASSES[1]`.

### Allocation CLI wiring

- **Locations:** `py/fetch_allocation_snapshot.py`, `py/run_allocation.py`
- **Purpose:** Make the FII artifact explicit and reproducible from CLI.
- **Defaults:** `outputs/carteira_fii_caio_consensus.json`.
- **Metadata:** Record stock/FII artifact paths, selected tickers, snapshot
  ranges, skips, and fixed-artifact look-ahead disclosure.

## Data Model

```text
data/allocation/
├── caio_stocks.csv       # date,TICKER...
├── caio_fiis.csv         # date,TICKER...
├── ifix.csv              # optional benchmark-only series
└── metadata.json         # artifacts, symbols, skips, provenance, bias note
```

`SnapshotBundle.rows` remains `DailyReturn` keyed by the unchanged five
`ASSET_CLASSES`. No per-security weight enters `DailyReturn` or the allocation
grid.

## Error Handling

| Scenario | Handling |
| --- | --- |
| Missing/invalid FII artifact | Abort before snapshot write |
| Duplicate/blank FII ticker | Abort before network fetch |
| Permanent FII Yahoo no-data | Skip, record reason, continue if any remain |
| Transient FII fetch failure | Abort; do not silently replace |
| Missing selected FII column in snapshot | Loader aborts |
| No IFIX file | Allocation still runs from selected FII sleeve |
| History too short for configured horizon | Existing horizon validation reports failure |

## Tech Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| FII return source | Yahoo adjusted close | Exact stock-sleeve source/treatment reuse |
| Internal FII construction | Equal-weight annual rebalance | Existing stock sleeve contract |
| FII selection timing | Fixed current artifact | User explicitly wants actual optimized portfolio; same as stocks |
| IFIX role | Optional benchmark-only | Preserve comparison without allowing proxy to replace actual portfolio |
| Allocation core | Unchanged | Prevent stock/allocation objective regressions |
