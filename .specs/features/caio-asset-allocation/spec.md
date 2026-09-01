# Caio asset-class allocation

## Problem

The stock optimizer answers which individual Brazilian stocks fit Caio's fundamental profile. It does not answer how Caio's whole capital should be divided among asset classes. This feature adds that analysis without re-running stock selection for every class.

## Requirements

### RQ-01 — Asset universe

Evaluate exactly five sleeves: Brazilian stocks, FIIs, international equity exposure, fixed income, and crypto. BDRs and ETFs are access instruments inside a sleeve, not separate sleeves.

### RQ-02 — Caio-specific stock sleeve

Use Caio's existing parameters and result artifacts as calibration evidence. Use `outputs/carteira_caio_consensus.json` as the reference stock portfolio, keep its tickers fixed, and rebalance those tickers equally once per year. Do not invoke the stock GA inside the allocation optimizer.

### RQ-03 — Comparable benchmark series

Build daily total-return series in BRL over the common available dates:

| Sleeve | Initial benchmark | Transformation |
| --- | --- | --- |
| Brazilian stocks | Fixed Caio reference portfolio | Equal-weight annual rebalance |
| FIIs | B3 IFIX | Use total return, including distributions |
| International equity | S&P 500 Total Return | Convert USD to BRL with BCB PTAX |
| Fixed income | BCB SGS 12 daily CDI factor, cross-checked against B3 DI | Proxy for post-fixed CDB/Tesouro Selic |
| Crypto | BTC/USD from one documented provider | Convert to BRL with BCB PTAX |

Record source, retrieval date, cutoff, currency, and transformations for each snapshot. Use common dates; do not forward-fill prices or silently replace missing history.

### RQ-04 — Objective and constraints

Find non-negative weights summing to 100% that maximize gross nominal annualized BRL return, subject to annualized volatility <= 20% and maximum drawdown >= -30% in the training window. Personalized profiles may additionally rank feasible candidates with their explicit return/volatility/drawdown preferences. Taxes, fees, spreads, and cash flows are out of scope. There is no class-specific maximum.

### RQ-05 — Diversification frontier

Measure class concentration with HHI on the target weights. Sweep the HHI penalty with a deterministic coarse 5 percentage-point grid and 1 percentage-point refinement. Retain non-dominated feasible allocations and expose the frontier instead of returning only one opaque optimum.

### RQ-06 — Knee target

Select the default current target from the feasible return-versus-HHI trade-off as the point with the greatest normalized distance from the line joining the highest-return and lowest-HHI endpoints. For the generic fallback, volatility and drawdown remain hard caps, not extra reasons for a lower-return point with the same HHI to become the knee. Personalized profiles may rank feasible candidates with explicit risk-adjusted preferences separately from the diagnostic knee. Report the max-return, most-diversified, knee, and personalized profile-winner allocations. The knee is a decision aid, not a universal truth.

### RQ-07 — Historical validation

Use a 10-year primary horizon and a 5-year robustness horizon. In each walk-forward step, train on the previous three years, choose weights, execute them on the next one year starting on the next trading day, then re-optimize annually. Apply constraints only to training; report test violations and windows with no feasible solution.

### RQ-08 — Baselines

Compare the adaptive and current-target results with 100% Caio stocks, equal 20% allocation across all five sleeves, and 100% DI. Mark a baseline as infeasible when it violates a risk cap rather than changing the cap.

### RQ-09 — Outputs

Write reproducible snapshots, the latest target, frontier, walk-forward windows, baselines, metrics, feasibility status, and stability summaries under an allocation-specific output prefix. Include the exact input artifact and cutoff dates.

### RQ-10 — Risk budget diagnostic

For the current target, selected walk-forward allocations, and baselines, report each class's signed contribution to target-weight portfolio variance. Keep this diagnostic class-agnostic and informational; do not change the unrestricted optimizer.

### RQ-11 — Risk budget comparison

Run a separate comparison scenario with a 25% maximum positive contribution to target-weight portfolio variance for every class. Apply the same return, volatility, drawdown, grid, rebalancing, and walk-forward rules; retain the unrestricted result as the default benchmark and do not impose a class-weight maximum.

### RQ-12 — Crypto sensitivity comparisons

Starting from the unrestricted knee, evaluate fixed 10%, 15%, and 20% crypto scenarios with the removed crypto weight transferred to fixed income. Report metrics and risk contributions for the current training, primary horizon, and robustness horizon without replacing the unrestricted target.

### RQ-13 — Profile source boundaries

Keep the stock-profile inputs separate by meaning: `profiles.py` fundamental-group weights describe how individual Brazilian stocks are scored, while `config.py` GA settings describe how the stock-selection algorithm executes. Neither dictionary is an asset-class allocation policy, and the allocation optimizer shall not reinterpret either one as class percentages.

### RQ-14 — Allocation profile calibration

Define allocation-only parameters for the conservative, moderate, and aggressive anchor profiles. Each anchor shall contain a volatility cap, drawdown cap, optional positive crypto variance-contribution cap, a diversification preference, and normalized return/volatility/drawdown objective weights summing to one. Calibrate the numeric anchors from consistently measured profile reference sleeves and the suitability risk inputs; policy assumptions must be labeled and must not be presented as observed facts.

### RQ-15 — Caio allocation profile

Accept Caio's continuous suitability score as the allocation-profile input and interpolate allocation-only parameters piecewise between the conservative (0), moderate (0.5), and aggressive (1) anchors, using the same interpolation convention as the stock-profile questionnaire. Do not infer one score by mixing fundamental weights with GA settings.

### RQ-16 — Profile provenance and outputs

Every calibrated allocation result shall record the profile name, score, anchor parameters, derived parameters, calibration inputs, and whether the result used a generic fallback or a personalized profile. The selected allocation and all feasibility checks shall use the derived parameters shown in that record.

### RQ-17 — Minimum sleeve allocation

Every allocation candidate shall assign at least 5% to each of the five sleeves. This is a lower bound requested for wallet diversification; it does not add a class-specific maximum.

### RQ-18 — Profile-specific risk-adjusted ranking

For a personalized profile, normalize annualized return, annualized volatility, and maximum drawdown across feasible candidates in the same training window. Rank candidates with the interpolated profile weights, then apply the existing HHI penalty as a concentration penalty. Keep volatility, drawdown, crypto-risk, and minimum-sleeve rules as hard feasibility checks. The generic fallback retains the existing return-minus-HHI ranking.

## Acceptance criteria

- The allocation command can run from local snapshots without network access.
- The same inputs and configuration produce the same weights and metrics.
- No stock-selection GA code is imported by the allocation optimizer.
- A synthetic-data test suite verifies the numerical core and its risk/weight invariants.
- Missing data and infeasible windows are visible in the result rather than silently repaired.
- The report clearly labels results as gross, historical, proxy-based analysis—not tax-adjusted personal advice.
