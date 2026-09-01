# FII Allocation Backtest Context

**Gathered:** 2026-08-11
**Spec:** `.specs/features/fii-allocation-backtest/spec.md`
**Status:** Implemented

## Feature Boundary

Use the already optimized FII consensus artifact to build the `fiis` class
inside the existing allocation backtest. The allocation optimizer decides the
class-level FII percentage; the FII selector remains responsible only for
ticker selection.

## Implementation Decisions

### Primary FII source

- Default artifact: `outputs/carteira_fii_caio_consensus.json`.
- Read `TICKER` identity from the artifact; ignore selector-side scores,
  frequency, and any future fields that could be mistaken for allocation
  weights.

### Same treatment as stocks

- Fetch `<TICKER>.SA` history through the existing Yahoo chart path.
- Use adjusted close, with the same distribution treatment as stocks.
- Build an internally equal-weight sleeve with annual rebalancing.
- Use exact common dates and no forward-fill.
- Preserve stock-compatible permanent-failure skipping and metadata.

### Allocation boundary

- Keep the existing five classes and allocation objective unchanged.
- Feed the optimized FII sleeve into `fiis` instead of IFIX.
- Do not add per-FII weights to the selector or allocation class grid.

### IFIX

- IFIX remains optional benchmark-only source data.
- Missing or disabled IFIX must not block allocation.
- IFIX must never populate `fiis` when the optimized FII artifact is present.

### Historical interpretation

- Use one current optimized FII artifact for the whole requested history,
  matching the existing stock allocation flow.
- Disclose look-ahead bias in snapshot metadata and output documentation.
- Historical re-selection per walk-forward window is deferred.

## Agent Discretion

- Exact CLI flag names, default paths, and metadata field layout, provided the
  artifact and selected symbols are explicit and reproducible.
- Whether to retain IFIX by default during snapshot fetch for backwards
  compatibility, provided it is optional and allocation-independent.

## Deferred Ideas

- Historical FII selection/reselection inside each training window.
- Per-FII risk contribution, turnover, or allocation optimization.
- IFIX comparison metrics in allocation output beyond the optional raw
  benchmark artifact.
