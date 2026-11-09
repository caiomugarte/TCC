"""Write guard for the research tool and atomic report writes (SAFE-01, RPT-01, RPT-05).

The tool never writes into the frozen or production trees. The output and cache
directories are resolved through symlinks before the check, so a link into
data/ is refused as well.
"""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

from .errors import ArgumentError

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("data", "snapshots", "baselines")


def assert_writable_dir(path: Path, label: str) -> Path:
    """Return the resolved directory; raise ArgumentError when it is inside a forbidden tree."""
    resolved = Path(path).resolve()
    for name in FORBIDDEN:
        root = (REPO_ROOT / name).resolve()
        if resolved == root or root in resolved.parents:
            raise ArgumentError(label, f"{resolved} is inside {name}/, which this tool never writes")
    return resolved


def write_files_atomically(files: Mapping[Path, bytes]) -> None:
    """Replace every target or none of them.

    Each file is written to a temp sibling first. The targets are then replaced
    one by one. When a replace fails, the targets already replaced get their
    previous bytes back (or are removed when they did not exist), every temp is
    removed, and the error is raised again.
    """
    temps: dict[Path, Path] = {}
    previous: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    try:
        for target, data in files.items():
            temps[target] = _write_temp(target, data)
        for target in files:
            previous[target] = target.read_bytes() if target.exists() else None
        for target in files:
            os.replace(temps[target], target)
            replaced.append(target)
    except BaseException:
        for target in replaced:
            _restore(target, previous[target])
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        raise


def _write_temp(target: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _restore(target: Path, data: bytes | None) -> None:
    if data is None:
        target.unlink(missing_ok=True)
        return
    os.replace(_write_temp(target, data), target)
