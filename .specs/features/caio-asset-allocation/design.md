# Design

## Boundary

Create a sibling allocation flow rather than extending the stock GA or the existing stock backtest. The flow has four seams:

1. **Snapshot/data seam** — load local benchmark snapshots, validate metadata, align common dates, and produce one daily BRL return table.
2. **Pure allocation seam** — evaluate weight vectors, annual rebalancing, return/volatility/drawdown/Calmar metrics, HHI, feasibility, grid search, non-dominated frontier, and knee selection.
3. **Orchestration seam** — load the fixed Caio sleeve, run current-window and walk-forward analyses, compare baselines, and write allocation-specific artifacts.
4. **Profile calibration seam** — keep fundamental scoring and GA execution inputs separate, calibrate allocation-only risk parameters for the three anchor profiles, and interpolate the Caio allocation profile from a continuous suitability score.

## Proposed repository shape

```text
py/
  core/allocation.py             # deterministic numerical core
  allocation_profiles.py         # allocation-only profile contract and interpolation
  allocation_data.py             # snapshot readers and benchmark transformations
  pipelines/asset_allocation.py  # current target, walk-forward, output orchestration
  run_allocation.py              # small CLI entry point
tests/test_allocation.py         # synthetic-data tests
data/allocation/                 # input snapshots and metadata
outputs/allocation_caio_*        # generated reports and tables
docs/adr/0001-*.md               # architecture decision
```

The exact filenames may be simplified during implementation if an existing module offers a genuinely smaller seam, but stock modules should not be modified just to make the new flow fit.

## Numerical contract

- Inputs to the core are ordered `DailyReturn` rows (date plus one return per class) and a class-name order. The snapshot seam may use pandas or CSV readers, but the optimizer remains dependency-light and does not require pandas.
- Weights are a named vector in `[0, 1]` with sum 1 within a small numeric tolerance.
- The configured wallet floor is 5% per sleeve; candidate grids violating that lower bound are excluded before feasibility and frontier selection.
- Annual rebalancing applies the target on the first common trading day after each 12-month anniversary; the return from each day is calculated from the holdings in force that day.
- Annualized return uses the realized portfolio value over the actual number of elapsed years.
- Volatility uses the standard deviation of daily portfolio returns times `sqrt(252)`.
- Maximum drawdown is the minimum percentage below the running peak.
- HHI is `sum(weights ** 2)` over classes.
- Risk contribution is the signed target-weight covariance contribution, `weight_i * Cov(return_i, return_portfolio) / Var(return_portfolio)`, over the selected window. Contributions approximately sum to 1; negative values indicate variance reduction. It is reported as a diagnostic and does not alter selection.
- The optional risk-budget scenario marks a candidate infeasible when any signed contribution is above the configured cap. The first comparison uses a 25% cap and leaves the unrestricted candidate set and default target unchanged.
- Crypto sensitivity scenarios reuse the current knee's other weights, set crypto to 10%, 15%, or 20%, and add the difference to fixed income; they are comparisons, not a second optimizer.
- Grid enumeration starts at 5% increments and refines candidate neighborhoods at 1% increments. No random search or GA is used.
- A candidate is feasible only when both training risk caps pass. An empty feasible set is a first-class result.
- Allocation profile parameters are separate from stock `PROFILE_WEIGHTS` and `GA_CONFIG`; the allocation core receives already-derived caps and preferences rather than importing the stock optimizer.
- Profile interpolation is piecewise linear over score anchors 0, 0.5, and 1.0. A missing or uncalibrated score is explicit and may use the documented generic fallback only for comparison, never as a personalized Caio claim.
- Personalized selection uses anchor/interpolated weights for return, volatility, and drawdown. Each metric is min-max normalized across feasible candidates in the current training window; return and drawdown are maximized, volatility is minimized. The weighted quality score is reduced by the existing HHI penalty. These weights are policy inputs, not historical guarantees.

## Frontier and knee

For each feasible candidate, retain return, volatility, drawdown, HHI, and weights. Remove candidates dominated on return, volatility, drawdown, and concentration according to the report's comparison rule. For the decision points, reduce that feasible frontier to the return-versus-HHI trade-off: a candidate with lower return and no lower concentration is not a knee candidate even if it has a safer drawdown. Identify the highest-return and lowest-HHI endpoints, normalize return and HHI between those endpoints, and select the point with maximum perpendicular distance from their connecting line. Keep endpoint fallbacks for a one-point or degenerate frontier.

## Data contract

Each snapshot has a stable class identifier, date, value/return, source URL or provider, retrieval timestamp, currency, total-return/distribution treatment, and coverage metadata. The pipeline should fail loudly for a required missing class or insufficient history. It should not mutate source files during analysis.

Profile calibration inputs must identify the anchor profile, source portfolio artifact, measurement window, and risk metrics used to derive its allocation parameters. The calibration report must distinguish observed historical metrics from policy limits.

## Failure handling

- Invalid weights: raise a clear validation error.
- Missing dates/columns: return a reportable data-quality error, not a filled series.
- No feasible training window: mark the window infeasible and continue other windows.
- Missing Caio ticker history: mark the window incomplete; do not substitute another ticker.
- Network unavailable: use existing snapshots or stop with instructions to acquire them.
- Missing profile calibration input or an inconsistent anchor order: fail with a profile-specific message; do not substitute stock fundamental weights or GA settings as allocation risk limits.
