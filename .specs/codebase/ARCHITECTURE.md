# Architecture

The repository is a script-oriented pipeline with three broad layers:

1. Data preparation in `py/data_preprocessing.py` and `py/core/preprocessing.py`.
2. Fundamental scoring and stock selection in `py/profiles.py`, `py/core/scoring.py`, `py/ga.py`, and `py/core/optimizer.py`.
3. Portfolio comparison/backtesting in `py/backtest_analysis.py` and output-oriented pipeline modules under `py/pipelines/`.

The allocation feature should be a sibling flow with a small pure numerical core and an orchestration/data-snapshot boundary. It must not call the stock GA or modify the semantics of the existing stock backtest.

