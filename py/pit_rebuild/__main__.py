"""Entry point of the PIT rebuild: python3 py/pit_rebuild <fetch|report> ... (run from the repository root)."""

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this package directory. Put py/ first so that
# "pit_rebuild" imports the package and no module of this directory shadows a stdlib one.
sys.path[0] = str(Path(__file__).resolve().parents[1])

from pit_rebuild.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
