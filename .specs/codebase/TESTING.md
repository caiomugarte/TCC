# Testing

There is no maintained pytest/unittest suite. `py/test.py`, `py/test_imports.py`, and `test_comparison.py` are executable smoke scripts rather than isolated tests. A pytest bytecode artifact exists locally, but pytest is not declared in the project configuration.

The allocation core should therefore expose deterministic functions that can be tested with small synthetic pandas Series/DataFrames. Tests should cover weight validity, annualized metrics, drawdown, annual rebalancing, grid enumeration, feasibility, frontier dominance, and knee selection without network access.

