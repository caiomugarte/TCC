# Project state

## Status

Core implementation and first real-market execution are complete for the
documented 2016-07-21 through 2026-07-21 snapshot.

## Confirmed decisions

- Optimize allocation percentages, not individual securities.
- Preserve Caio's current stock-selection parameters as calibration evidence and use `outputs/carteira_caio_consensus.json` as the fixed stock sleeve.
- Use five classes: Brazilian stocks, FIIs, international exposure, fixed income, and crypto.
- Optimize gross nominal BRL total return with annualized volatility <= 20% and maximum drawdown >= -30%.
- Do not impose a class-specific maximum; weights are non-negative and sum to 100%.
- Show the return/diversification frontier and choose the mathematically defined knee as the default current target.
- Use a 10-year primary horizon and a 5-year robustness horizon, with 3-year training and 1-year test walk-forward windows.
- Exclude taxes, costs, cash flows, and Sharpe from the core objective.
- Report class-agnostic target-weight variance contributions as a risk-budget diagnostic, while preserving the unrestricted allocation as the benchmark until a threshold is chosen.
- Run a separate 25% maximum positive variance-contribution scenario for comparison; it is not a class-weight maximum and cannot replace the unrestricted default without a later decision.
- Compare fixed 10%, 15%, and 20% crypto versions of the knee by moving the difference to fixed income; keep the unrestricted 30% target as the benchmark.
- Treat `py/profiles.py` fundamental-group weights and `py/config.py` stock-GA execution settings as separate stock-selection inputs; neither is an asset-allocation risk profile.
- Add a separate allocation-profile calibration layer with conservative, moderate, and aggressive anchors, then interpolate Caio's allocation parameters from an explicit suitability score.
- V2 allocation anchors remain explicit questionnaire-policy assumptions: conservative 10%/15%/25% volatility/drawdown/crypto-risk caps with HHI penalty 0.50 and return/volatility/drawdown preferences 30%/35%/35%; moderate 15%/25%/40% with 0.25 and 50%/25%/25%; aggressive 20%/35%/50% with 0.10 and 70%/15%/15%. Empirical reference-sleeve calibration remains separate and pending.
- The optimized FII consensus artifact is the primary fixed `fiis` allocation sleeve, built with the same Yahoo adjusted-close and annual equal-weight treatment as stocks; IFIX is optional benchmark-only data and never replaces the optimized FII portfolio. The fixed artifact's historical look-ahead is explicit and documented; historical FII re-selection is deferred.

## Known repository risks

- The stock GA and the existing backtest have different responsibilities and should remain untouched by the new core.
- The existing backtest is buy-and-hold and uses a fixed 10% risk-free proxy for Sharpe.
- Current Caio artifacts and current CLI defaults do not necessarily describe the same number of GA runs.
- No standard test suite is configured; new pure logic needs focused, dependency-light tests.

## Implemented surface

- `py/core/allocation.py`: deterministic metrics, annual rebalancing, simplex grid, penalty sweep, frontier, and knee selection.
- `py/allocation_data.py`: offline snapshot contract, fixed-ticker stock/FII sleeves, PTAX conversions, and common-date alignment.
- `py/fetch_allocation_snapshot.py`: reproducible source collector for Yahoo FII/stock levels, optional B3 IFIX, BCB PTAX, and BCB SGS CDI series 12.
- `py/pipelines/asset_allocation.py`: current target, baselines, walk-forward, stability, and output writers.
- `py/run_allocation.py`: local-snapshot CLI.
- `tests/test_allocation.py`: dependency-light allocation, snapshot, FII integration, and CLI tests.

## Latest execution

- Snapshot: 2,416 common daily rows from 2016-07-21 through 2026-07-21.
- Current return/HHI knee: 20% Brazilian stocks, 10% FIIs, 25% S&P 500 total return, 15% fixed income, and 30% crypto.
- Current knee metrics on the latest three-year training window: 27.01% annualized return, 18.95% annualized volatility, and -16.49% maximum drawdown.
- Risk-budget diagnostic for the current knee: crypto is 30% of capital but approximately 80.94% of target-weight portfolio variance; primary walk-forward knee risk shares for crypto were 38.54%, 34.94%, and 84.57%.
- The 25% maximum variance-contribution comparison produced no feasible current or walk-forward allocation under the existing return, volatility, and drawdown rules; the closest current frontier point had a 30.27% maximum class contribution.
- The crypto sensitivity comparison is reported in `outputs/allocation_caio.json` for the current training, primary, and robustness horizons.
- On the current training window, the 10%, 15%, and 20% crypto sensitivities produced 19.69%/9.31%/-8.33%, 21.74%/11.70%/-10.56%, and 23.64%/14.14%/-12.65% for return/volatility/drawdown; their crypto variance shares were 44.20%, 59.77%, and 69.80%.
- In the 10-year primary horizon, the 10% scenario remained within caps, while 15% and 20% violated the volatility and drawdown caps.
- Max-return endpoint: 15% Brazilian stocks, 0% FIIs, 40% S&P 500 total return, 15% fixed income, and 30% crypto; 28.59% return, 19.97% volatility, -19.43% drawdown.
- The first primary walk-forward test window exceeded the volatility cap (30.55%) and drawdown cap (-34.24%), so the knee is not a guarantee and the stability report must be read with the out-of-sample flags.
- Personalized v2 execution: `python3 py/run_allocation.py --suitability-score 0.831 --profile-name caio_new` produces `outputs/allocation_caio_new_personalized.json`; derived caps are 18.31% volatility, 31.62% drawdown, and 46.62% crypto variance contribution, with interpolated return/volatility/drawdown preferences 63.24%/18.38%/18.38%. The current profile winner is 50% Brazilian stocks, 5% FIIs, 20% S&P 500, 5% fixed income, and 20% crypto on the available 2021-07-29 through 2026-07-21 snapshot; the new ranking leaves the winner unchanged.
- Allocation floor update: every optimized sleeve now has a 5% minimum. Re-running score 0.831 yields a current profile winner of 50% Brazilian stocks, 5% FIIs, 20% S&P 500, 5% fixed income, and 20% crypto; current-window metrics are 28.20% annualized return, 17.04% volatility, and -11.85% maximum drawdown.

## Next action

FII allocation integration is implemented and verified with fixture snapshots.
Next: optionally refresh the real snapshot with the selected FII artifact, then
measure the three anchor reference sleeves empirically and compare the result
with the v2 policy-anchor personalized run. Results remain historical, gross,
and proxy-based; they are not tax-adjusted personal advice.
