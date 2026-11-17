"""Entry point of the PIT rebuild: python3 py/pit_rebuild <fetch|report|collect> ... (run from the repository root).

fetch and report are the measurement tool (cli.py); collect builds the stock CSV of a snapshot (collector.py, #78).
"""

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this package directory. Put py/ first so that
# "pit_rebuild" imports the package and no module of this directory shadows a stdlib one.
sys.path[0] = str(Path(__file__).resolve().parents[1])

if __name__ == "__main__":
    if sys.argv[1:2] == ["collect"]:
        from pit_rebuild.collector import main  # noqa: E402

        sys.exit(main(sys.argv[2:]))
    from pit_rebuild.cli import main  # noqa: E402

    sys.exit(main())
