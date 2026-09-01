# Tasks

## T1 — Add allocation configuration and paths

- Depends on: none
- Files: `py/config.py` or a new allocation config module
- Requirements: RQ-04, RQ-07, RQ-09
- Add the five class order, 10/5-year horizons, 3/1-year walk-forward windows, 20% volatility cap, 30% drawdown cap, and snapshot/output paths.
- Verify: import the config and assert the class order and weight-independent constraints are explicit.

## T2 — Implement deterministic allocation metrics and weight validation

- Depends on: T1
- Files: `py/core/allocation.py`
- Requirements: RQ-04, RQ-05
- Implement weight validation, portfolio return calculation, annualized return, volatility, maximum drawdown, Calmar, and class HHI.
- Verify: synthetic unit tests cover valid/invalid weights, zero-weight classes, constant series, and drawdown sign conventions.

## T3 — Implement annual-rebalance simulation and candidate grid

- Depends on: T2
- Files: `py/core/allocation.py`, `tests/test_allocation.py`
- Requirements: RQ-04, RQ-07
- Implement annual rebalancing with next-trading-day execution and deterministic 5%/1% weight grids.
- Verify: a two-class synthetic path shows rebalance dates and the grid contains only non-negative vectors summing to one.

## T4 — Implement feasibility, frontier, and knee selection

- Depends on: T2, T3
- Files: `py/core/allocation.py`, `tests/test_allocation.py`
- Requirements: RQ-05, RQ-06
- Evaluate candidates, filter risk-infeasible results, remove dominated points, and select max-return, lowest-HHI, and knee points with deterministic tie-breaks.
- Verify: tests cover empty feasible sets, a degenerate frontier, endpoint fallback, and a known knee.

## T5 — Add snapshot readers and BRL benchmark transformations

- Depends on: T1
- Files: `py/allocation_data.py`, `data/allocation/README.md`
- Requirements: RQ-01, RQ-02, RQ-03, RQ-09
- Read local snapshots, load the fixed Caio portfolio, build the equal-weight annual-rebalanced stock sleeve, convert USD series using aligned PTAX, and validate metadata/common dates. Keep provider download code out of the numerical core.
- Verify: fixture snapshots load into one aligned daily BRL return table; missing history is reported without forward-filling.

## T6 — Add current-target and walk-forward orchestration

- Depends on: T3, T4, T5
- Files: `py/pipelines/asset_allocation.py`
- Requirements: RQ-06, RQ-07, RQ-08
- Run the latest-window optimization, historical adaptive walk-forward, and fixed baselines. Summarize per-window weights, metrics, feasibility, and allocation stability.
- Verify: an offline fixture run produces deterministic tables and explicitly marks infeasible windows.

## T7 — Add the minimal CLI and allocation-specific artifacts

- Depends on: T6
- Files: `py/run_allocation.py`, `outputs/` through runtime only
- Requirements: RQ-09, RQ-10
- Provide an offline/local-snapshot command with a clear failure message when snapshots are missing. Write target, frontier, walk-forward, baseline, metadata, and summary artifacts under `allocation_caio_*` names.
- Verify: `--help` works without network access; fixture execution writes valid JSON/CSV without touching existing stock outputs.

## T8 — Validate in the repository environment and document limitations

- Depends on: T7
- Files: `.specs/project/STATE.md`, `.notebook/allocation-flow.md`, `.notebook/INDEX.md`
- Requirements: all
- Run focused tests and import checks, record unavailable dependencies or sources, and update persistent project intelligence.
- Verify: report exact commands and outcomes; do not claim a real-market result if required snapshots were unavailable.

## T9 — Define the allocation-profile contract and anchor calibration

**Status:** Complete for v1 policy anchors; empirical reference-sleeve calibration remains a follow-up.

- Depends on: T8
- Files: `py/allocation_profiles.py`, `py/allocation_config.py`, `.specs/features/caio-asset-allocation/spec.md`
- Requirements: RQ-13, RQ-14, RQ-15
- Add allocation-only profile parameters for conservative, moderate, and aggressive anchors; validate scores and risk limits; interpolate a Caio profile from an explicit continuous score. Keep `profiles.py` fundamental weights and `config.py` GA settings out of the allocation contract.
- Verify: synthetic anchor parameters interpolate exactly at 0, 0.5, and 1; invalid scores and caps fail clearly.

## T10 — Connect derived profile parameters to allocation orchestration

**Status:** Complete.

- Depends on: T9
- Files: `py/pipelines/asset_allocation.py`, `py/run_allocation.py`, `py/allocation_config.py`
- Requirements: RQ-04, RQ-14, RQ-16
- Pass derived volatility, drawdown, crypto-risk, and diversification settings into the existing allocation engine and record the profile provenance. Preserve the generic result as an explicit fallback until calibrated Caio inputs are supplied.
- Verify: a fixture run uses the derived caps and its JSON identifies the profile score, source, and parameters.

## T11 — Calibrate and compare the three anchor profiles

**Status:** Pending empirical reference-sleeve calibration.

- Depends on: T9, T10
- Files: profile calibration runner/output and allocation tests
- Requirements: RQ-14, RQ-16
- Measure the three anchor reference sleeves with the same benchmark/date/rebalance rules, derive their allocation risk parameters, and expose conservative/moderate/aggressive comparison results before deriving Caio.
- Verify: anchor outputs use the same metric definitions and clearly distinguish observed metrics from policy caps.

## T12 — Re-run Caio with the derived allocation profile

**Status:** Complete for suitability score `0.831` using the v2 policy anchors.

- Depends on: T11
- Files: allocation outputs and project state notes
- Requirements: RQ-15, RQ-16
- Feed the Caio suitability score into the interpolation, run current and walk-forward allocation, and replace the generic result only after the personalized profile is present and feasible.
- Verify: output provenance shows the score and anchors; no result is labeled personalized when the score or calibration inputs are missing.

## T13 — Add profile-specific risk-adjusted preferences

**Status:** Complete.

- Depends on: T10
- Files: `py/allocation_profiles.py`, `py/allocation_config.py`, `py/core/allocation.py`, `py/pipelines/asset_allocation.py`, `tests/test_allocation.py`, existing allocation spec/state notes
- Requirements: RQ-04, RQ-06, RQ-14, RQ-18
- Add explicit return/volatility/drawdown weights to the three anchors, interpolate them by suitability score, and use them only to rank feasible personalized candidates. Preserve generic fallback ranking, hard caps, crypto-risk cap, HHI penalty, and 5% floor.
- Verify: synthetic candidate test prefers lower risk when profile weights justify the return trade-off; personalized output records interpolated weights and remains deterministic.
