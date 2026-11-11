"""Point-in-time rebuild of the Status Invest stock indicators from CVM and B3 (issue #59).

The package rebuilds the 19 fields of the stock selector for each requested date
from primary sources and compares them with the real Status Invest values. Each
module holds one rule; the CLI only reads and writes files.

Exit codes of the CLI:

    0  report written
    1  the report files could not be written
    2  invalid arguments, or an output or cache directory inside data/, snapshots/ or the baselines tree
    3  a given path does not exist
    4  integrity violation: cached file hash mismatch, PIT leak, Status Invest CSV
       missing a required column or repeating a ticker
    5  a required download failed, or a required file is absent from the cache in --offline mode
"""
