# Suitability Profile Flow
> Continuous questionnaire output mapped to the repository's profile dictionaries

Entry: `data_preprocessing.py:main()`

Flow: `data_preprocessing.py:FILTERS` → `preprocess()` → `data/processed/fundamentals_clean_{perfil}.csv` → `profiles.py:build_scores()` → `ga.py:run_ga()`

Weights: `profiles.py:PROFILE_WEIGHTS` uses `liquidez`, `rent`, `value`, `growth`, `div`; `build_scores()` averages the matching columns in `GROUPS` and sums the weighted group averages.

GA: `ga.py:PERFIL_CONFIG` supplies `n_assets`, `lambda`, `generations`, `pop_size`; `ga.py:fitness()` applies `lambda` as an HHI sector-concentration penalty, not direct return volatility.

Filters: `data_preprocessing.py:FILTERS` controls `VALOR DE MERCADO` and `LIQUIDEZ MEDIA DIARIA`; the current base anchors are conservador, moderado and arrojado. `caio` is an additional bespoke profile.

Current modular Caio config: `py/config.py` — cap >= 3,000,000,000; liquidity >= 1,050,000; weights liquidez/rent/value/growth/div = 0.29/0.14/0.26/0.04/0.26; GA = 10 assets, lambda 0.37, 600 generations, population 400; crossover/mutation = 0.80/0.02.

Consensus: `py/pipelines/multi_run.py:run_multi_execution_profile()` uses seed `42 + run_id`; `build_consensus_portfolio()` selects the 10 most frequent tickers, ties by SCORE then ticker, and removes same-company share classes. `py/core/optimizer.py` stops after 50 generations without improvement (min delta 1e-6). Python `random` is not seeded, so exact replay is not guaranteed from the NumPy seed alone.

Artifact gotcha: `outputs/carteira_caio_consensus.json` and `outputs/metrics_stability_caio.csv` indicate 150 runs (run IDs 0–149, seeds 42–191), while current `N_RUNS`/CLI default is 30. `outputs/multiple_runs_summary.json` reports a different 30-run consensus; verify matching artifacts before citing it.

HTML: `suitability.html` interpolates the three base anchors, limits short horizon/near-term liquidity, and exports a `personalizado` Python dict. Custom profile entries must be merged into `py/config.py` before modular execution.

`caio_new`: `py/config.py` — cap >= 470,000,000; liquidity >= 202,000; weights liquidez/rent/value/growth/div = 0.1338/0.2169/0.2169/0.3324/0.10; GA = 14 assets, lambda 0.151, 470 generations, population 280. `run_optimized.py --production --profile caio_new` writes profile-specific consensus, best-individual, comparison and stability artifacts. Requires `data/processed/fundamentals_clean_caio_new.csv` when cache is enabled.

Profile registry gotcha: modular execution reads `py/config.py` (`PROFILES`, `FILTERS`, `GA_CONFIG`, `PROFILE_WEIGHTS`); legacy execution still reads `py/ga.py:PERFIL_CONFIG` and `py/profiles.py:PROFILE_WEIGHTS`. New profiles need matching entries in both paths when legacy scripts remain in use. Cached optimization requires `data/processed/fundamentals_clean_{perfil}.csv`; generate it with `py/main.py --preprocess`.

Backtest gotcha: `run_multi_execution_profile()` saves stock-selection artifacts even when Yahoo Finance cannot resolve; comparison backtest fields then remain empty. Python 3.9 cannot evaluate existing PEP 604 annotations (`int | float`); use Python 3.10+ for current modular runner despite `pyproject.toml` declaring >=3.9.

Updated: 2026-07-24
