# Caio Allocation Flow
> New benchmark-sleeve allocation path; separate from stock selection

Entry: `py/run_allocation.py` (offline/local snapshot CLI)

Flow: documented benchmark snapshots → common daily BRL returns → fixed profile stock sleeve + four benchmark sleeves → allocation-profile calibration → deterministic weight grid → profile risk-cap feasibility → return/HHI frontier → knee/current target → risk-budget diagnostics/scenarios → walk-forward and baseline outputs.

Implementation: `py/core/allocation.py` is dependency-light and owns the
metrics, annual-rebalance simulator, grid, HHI penalty sweep, frontier, knee,
and target-weight variance contributions. `py/allocation_data.py` owns
CSV/metadata validation, fixed-ticker stock/FII sleeve construction, and PTAX
multiplication. The orchestration and writers live in
`py/pipelines/asset_allocation.py`. `py/allocation_profiles.py` owns the
allocation-only profile contract and conservative/moderate/aggressive score
interpolation.

Decisions: five classes; BDRs/ETFs are instruments inside international exposure; S&P 500 Total Return is the first international proxy; the optimized FII artifact is the fixed `fiis` sleeve with IFIX optional benchmark-only; BCB SGS 12 supplies the daily CDI/DI factor cross-checked against B3; BTC is the first crypto sleeve; USD series use BCB PTAX; no taxes/costs/cash flows; 5% minimum per class and no class-weight maximum; 20% volatility and 30% maximum drawdown caps; 10-year primary and 5-year robustness horizons; 3-year train/1-year test walk-forward. The knee is selected from the feasible return-vs-HHI trade-off after risk caps. The first alternative scenario caps each positive variance contribution at 25% while retaining the unrestricted result.

Do not reuse: `py/core/optimizer.py` or `py/backtest_analysis.py` as the allocation objective. The former selects binary stock masks and penalizes sector HHI; the latter is a buy-and-hold stock backtest with a fixed 10% Sharpe proxy.

Data risk: `outputs/carteira_caio_consensus.json` is the fixed reference, but its run history may not match current defaults. Record the exact artifact and benchmark snapshot metadata in every allocation output.

Verification: `python3 -m unittest discover -s tests -v` passes 45 tests; the
CLI help, a complete synthetic output run, the ten-year snapshot fetch, and
the real CLI run pass. The generated snapshot aligns 2,416 common daily rows;
the current target is recorded in `outputs/allocation_caio.json`. The current
knee's 30% crypto target contributes approximately 80.94% of target-weight
portfolio variance; this is diagnostic only and does not cap or re-rank the
optimizer. The first 25% variance-contribution scenario is recorded in the
JSON output but has no feasible current or walk-forward selection under the
existing return and risk caps. Fixed crypto sensitivity scenarios at 10%, 15%,
and 20% move the difference to fixed income and are reported alongside the
unrestricted knee. On the current training window their crypto variance
contributions are 44.20%, 59.77%, and 69.80%; the 10% version is the only one
of the three that remains within caps in the 10-year primary horizon.

Profile boundary: `py/profiles.py` contains fundamental-indicator group
weights and `py/config.py` contains stock-GA execution settings. Neither is
reused as an allocation policy. The allocation pipeline accepts an explicit
`AllocationProfile`; without one it labels the result as a generic fallback.
The CLI also has one explicit alias: `--profile-name caio_last` defaults to
suitability score `0.0` (conservative anchor) when no score is supplied; an
explicit score overrides the alias. This remains a declared policy mapping,
not inference from stock-GA weights.
`py/run_allocation.py --suitability-score SCORE` now interpolates explicit v2
allocation anchors and selects a named `profile_winner` by weighted
return/volatility/drawdown quality minus the derived HHI penalty, after hard
risk-cap feasibility. V2 anchor preferences are policy assumptions; empirical
reference-sleeve calibration remains pending. Current `caio_new` run: score
0.831, interpolated preferences 63.24%/18.38%/18.38%, derived volatility cap
18.31%, drawdown cap 31.62%, crypto variance-contribution cap 46.62%, current
profile winner 50/5/20/5/20 across stocks/FIIs/S&P500/DI/crypto.

Updated: 2026-08-11
