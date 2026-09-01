# Concerns

- `py/profiles.py` and modular `py/config.py` contain different Caio weight definitions; the allocation feature must use the canonical saved consensus portfolio rather than infer a new stock portfolio from whichever dictionary is imported.
- `py/pipelines/multi_run.py` can produce consensus artifacts from a run count that differs from the current default. The analysis should record the exact input artifact and its metadata.
- `py/backtest_analysis.py` is large, duplicated in places, uses buy-and-hold, forward-fills short price gaps, and treats a fixed 10% proxy as a risk-free rate. Reusing it would violate the allocation decisions.
- Data availability differs across classes. The common date range and incomplete-window status must be explicit in reports.
- The local `.venv` has no scientific dependencies at the moment; full execution may require installing `py/requirements.txt` outside the current sandbox.

