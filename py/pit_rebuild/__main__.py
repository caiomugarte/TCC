"""Entry point of the PIT rebuild: python3 py/pit_rebuild <fetch|report|fii-report|collect> ... (run from the repository root).

fetch and report are the measurement tool (cli.py); fii-* are the FII commands (fii_cli.py);
collect builds the stock CSV of a snapshot (collector.py, #78).
"""

import sys
from pathlib import Path

# Running the directory puts it on sys.path[0]; swap it for py/ so that
# "pit_rebuild" imports the package and no module of this directory shadows a stdlib one.
sys.path[0] = str(Path(__file__).resolve().parents[1])

from pit_rebuild import cli, fii_cli  # noqa: E402

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "collect":
        from pit_rebuild.collector import main as collect_main  # noqa: E402

        sys.exit(collect_main(sys.argv[2:]))
    sys.exit(fii_cli.main() if command.startswith("fii-") else cli.main())
