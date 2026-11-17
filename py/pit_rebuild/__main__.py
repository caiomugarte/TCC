"""Entry point of the PIT rebuild: python3 py/pit_rebuild <fetch|report|fii-report> ... (run from the repository root)."""

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this package directory. Put py/ first so that
# "pit_rebuild" imports the package and no module of this directory shadows a stdlib one.
sys.path[0] = str(Path(__file__).resolve().parents[1])

from pit_rebuild import cli, fii_cli  # noqa: E402

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(fii_cli.main() if command.startswith("fii-") else cli.main())
