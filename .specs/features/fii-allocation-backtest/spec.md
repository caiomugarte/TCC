# FII Allocation Backtest and Portfolio-Backed Allocation

**Status:** Implemented

## Problem Statement

The allocation pipeline currently represents the FII class with IFIX. The FII
selector already produces an optimized portfolio, but that actual ticker set
is not used by allocation or by the allocation backtest.

The allocation pipeline SHALL use the optimized FII portfolio as a fixed,
equal-weight internal sleeve, following the existing stock-sleeve treatment.
The allocation optimizer SHALL continue deciding only the capital percentage
assigned to the FII class.

## Goals

- Fetch reproducible historical levels for the selected FII tickers.
- Build the FII sleeve with the same annual-rebalanced equal-weight approach
  used for stocks.
- Replace IFIX as the `fiis` input to the existing five-class allocation
  optimizer.
- Preserve current allocation metrics, risk caps, frontier, knee, and
  walk-forward reports.
- Keep IFIX available only as optional benchmark data; it SHALL never drive
  the `fiis` class when an optimized FII portfolio is supplied.
- Record the exact FII portfolio artifact and data-quality decisions in
  snapshot metadata and outputs.

## Explicit Boundary

This is an allocation-level backtest of the already optimized FII portfolio.
It is not a historical re-run of FII selection for every walk-forward window.
The selected FII artifact is fixed across the requested historical period,
matching the current stock allocation approach. This creates known look-ahead
bias and SHALL be disclosed in metadata and documentation.

The selector does not assign final portfolio weights. Internal equal weighting
is used only to construct the FII class sleeve; the allocation optimizer sets
the class-level FII percentage.

## Scope

- Add an FII portfolio input to the allocation snapshot fetcher and runner.
- Read tickers from `outputs/carteira_fii_<profile>_consensus.json` (default
  `outputs/carteira_fii_caio_consensus.json`).
- Fetch each FII as `<TICKER>.SA` through the existing Yahoo chart path using
  adjusted close, including distributions through the same treatment as
  stocks.
- Write a reproducible `caio_fiis.csv` level snapshot beside
  `caio_stocks.csv`.
- Keep IFIX fetching/storing optional for benchmark diagnostics, without making
  it required by allocation.
- Build the FII class with the existing `build_equal_weight_sleeve()` and
  annual rebalance behavior.
- Feed that sleeve into the existing allocation optimizer and walk-forward
  backtest under the unchanged `fiis` class name.
- Leave stock selection, stock fetching, and the FII selector unchanged except
  for the new integration contract.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Historical re-selection of FIIs in each training window | Separate strategy-validation feature; current request is the actual optimized portfolio |
| Per-FII allocation weights | Allocation optimizer owns class-level amounts; selector only supplies tickers |
| New allocation objective or risk model | Existing five-class optimizer remains the objective |
| Taxes, costs, cash flows, or turnover modeling | Existing allocation boundary excludes them |
| Automatic FII selector execution from allocation | Allocation consumes an explicit, already-generated portfolio artifact |

## User Stories

### P1: Backtest the optimized FII portfolio ⭐ MVP

**User Story:** As the portfolio researcher, I want the allocation backtest to
use the actual optimized FII ticker set so that the FII class represents the
portfolio I would actually hold rather than IFIX.

**Acceptance Criteria:**

1. WHEN the allocation snapshot is fetched with a valid FII portfolio artifact
   THEN the system SHALL fetch historical data for every selected FII ticker
   using the existing Yahoo source contract.
2. WHEN the snapshot is loaded THEN the system SHALL build the FII sleeve with
   annual-rebalanced equal weights and common-date returns, matching the stock
   sleeve behavior.
3. WHEN allocation runs THEN the `fiis` class SHALL use the selected FII sleeve
   and SHALL NOT use IFIX as its return series.
4. WHEN the current, primary, robustness, and walk-forward analyses run THEN
   they SHALL use the same allocation objective, caps, frontier, knee, and
   reporting logic as before.

**Independent Test:** A fixture FII portfolio with two tickers and fixture
level histories produces a `fiis` return series equal to the annual-rebalanced
equal-weight combination and changes allocation results when its history
differs from IFIX.

### P1: Preserve reproducibility and data quality

**User Story:** As the researcher, I want the source artifact and missing-data
decisions recorded so that a backtest result can be reproduced and audited.

**Acceptance Criteria:**

1. WHEN the snapshot is written THEN metadata SHALL record the FII portfolio
   path, selected tickers, requested date range, source symbols, and skipped
   ticker reasons.
2. WHEN Yahoo returns a permanent no-history failure for a selected FII THEN
   the system SHALL follow the existing stock behavior: skip it, record the
   reason, and continue if at least one selected FII remains.
3. WHEN Yahoo returns a transient or malformed response THEN the snapshot
   fetch SHALL fail rather than silently replace the FII or IFIX data.
4. WHEN no selected FII has usable history THEN the snapshot fetch SHALL fail.
5. WHEN the portfolio artifact is missing, malformed, duplicated, or empty
   THEN the snapshot fetch SHALL fail before writing an allocation snapshot.

**Independent Test:** Mock permanent and transient Yahoo failures and verify
metadata, failure behavior, and atomicity of the resulting snapshot.

### P1: Keep allocation separate from security selection

**User Story:** As the portfolio researcher, I want allocation to choose the
FII class percentage while the FII selector chooses tickers, so that the two
optimization levels do not overwrite each other.

**Acceptance Criteria:**

1. WHEN allocation loads the FII portfolio THEN it SHALL read ticker identity
   only; any selector-side score or frequency fields SHALL not become capital
   weights.
2. WHEN the FII sleeve is constructed THEN its internal equal weighting SHALL
   be applied only inside the `fiis` class return series.
3. WHEN allocation outputs are written THEN class-level FII weights SHALL be
   reported under `fiis`, with no per-FII allocation weights added to the FII
   selector artifact.

**Independent Test:** Run allocation with two different FII ticker artifacts
   and confirm only the FII class return series and resulting class allocation
   change; stock output and selector weights remain untouched.

### P2: Retain optional IFIX reference data

**User Story:** As the researcher, I want to compare the optimized FII sleeve
   with IFIX without allowing IFIX to drive the allocation result.

**Acceptance Criteria:**

1. IF the IFIX source is retained THEN it SHALL be labeled as benchmark-only
   data and SHALL not populate the `fiis` class used by the optimizer.
2. WHEN IFIX collection is disabled THEN the snapshot and allocation SHALL
   still run successfully from the optimized FII portfolio history.

**Independent Test:** Verify that changing or omitting IFIX data cannot change
the optimized `fiis` class when the selected FII history is unchanged.

## Edge Cases

- A selected FII has no Yahoo symbol/history: skip and record it using the
  stock-compatible rule; never replace it with another FII.
- Selected FIIs have different listing dates: use exact common dates, with no
  forward-fill or manufactured observations.
- The FII artifact contains a non-unique ticker: reject it.
- The FII portfolio artifact contains no usable FII after skips: reject the
  snapshot.
- The FII history is shorter than the requested 10-year horizon: preserve the
  common available range and let the existing horizon validation fail if it is
  too short for allocation analysis.
- The selected portfolio changes between runs: record the artifact path and
  selected tickers so outputs are not interpreted as the same strategy.

## Confirmed Decisions

- The optimized FII portfolio is the primary source for the `fiis` class.
- IFIX is optional benchmark-only data and never replaces the optimized FII
  sleeve.
- The current optimized FII artifact is fixed across historical windows,
  matching stocks. This known look-ahead bias is disclosed; historical
  re-selection is deferred.

## Requirement Traceability

| Requirement ID | Story | Status |
| --- | --- | --- |
| FII-ALLOC-01 | P1: Backtest optimized FII portfolio | Verified |
| FII-ALLOC-02 | P1: Backtest optimized FII portfolio | Verified |
| FII-ALLOC-03 | P1: Backtest optimized FII portfolio | Verified |
| FII-ALLOC-04 | P1: Backtest optimized FII portfolio | Verified |
| FII-ALLOC-05 | P1: Preserve reproducibility and data quality | Verified |
| FII-ALLOC-06 | P1: Preserve reproducibility and data quality | Verified |
| FII-ALLOC-07 | P1: Preserve reproducibility and data quality | Verified |
| FII-ALLOC-08 | P1: Keep allocation separate from security selection | Verified |
| FII-ALLOC-09 | P1: Keep allocation separate from security selection | Verified |
| FII-ALLOC-10 | P2: Retain optional IFIX reference data | Verified |

## Success Criteria

- Allocation can run from a selected FII consensus artifact without IFIX
  driving the `fiis` class.
- FII class returns match a fixture equal-weight annual-rebalanced portfolio.
- Existing stock and allocation tests remain green.
- Output metadata identifies the exact FII artifact, symbols, date range, and
  data-quality skips.
- Existing stock flow remains behaviorally unchanged.
