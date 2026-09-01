# Structure

```text
py/
  config.py                  # paths and stock-profile configuration
  core/                      # preprocessing, scoring, stock optimizer, metrics
  pipelines/                 # single- and multi-run stock workflows
  utils/                     # cache helper
  backtest_analysis.py       # existing stock/benchmark backtest script
  requirements.txt           # runtime dependencies
outputs/                     # generated stock portfolios, metrics, and plots
data/                        # raw and processed stock data
.cache/                      # preprocessing cache
```

The allocation implementation should add only the smallest set of sibling modules and an explicit output namespace so it can coexist with the current stock artifacts.

