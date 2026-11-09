"""Read one Status Invest CSV read-only and select the eligible universe (SAFE-02, A-1).

The file is read once as bytes. The sha256 recorded in the report is the hash of
the same bytes the rows come from. A blank or non-numeric cell is None.
"""

from dataclasses import dataclass
import csv
import hashlib
import io
import math
from pathlib import Path

from .errors import IntegrityError, MissingPathError
from .fields import ALL_FIELDS

# A-1: production applies these thresholds (py/core/preprocessing.py:103-106).
# The arrojado anchor of api/app/services/premium_policy.py holds the same values.
ELIGIBILITY = {"cap_min": 200_000_000, "liq_min": 50_000}
REQUIRED_COLUMNS = ("TICKER", "SETOR", *ALL_FIELDS)


@dataclass(frozen=True)
class SICsv:
    path: Path
    sha256: str
    rows: dict[str, dict[str, float | None]]
    setor: dict[str, str]


def load(path: Path) -> SICsv:
    path = Path(path)
    if not path.is_file():
        raise MissingPathError(str(path), "Status Invest CSV does not exist")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IntegrityError(str(path), f"Status Invest CSV is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    for column in REQUIRED_COLUMNS:
        if column not in header:
            raise IntegrityError(column, f"required column missing from {path}")
    rows: dict[str, dict[str, float | None]] = {}
    setor: dict[str, str] = {}
    for record in reader:
        ticker = record["TICKER"].strip()
        if ticker in rows:
            raise IntegrityError(ticker, f"ticker repeated in {path}")
        rows[ticker] = {field: _number(record[field]) for field in ALL_FIELDS}
        setor[ticker] = (record["SETOR"] or "").strip()
    return SICsv(path=path, sha256=hashlib.sha256(raw).hexdigest(), rows=rows, setor=setor)


def eligible(si: SICsv) -> list[str]:
    """Tickers with VALOR DE MERCADO >= cap_min and LIQUIDEZ MEDIA DIARIA >= liq_min, sorted."""
    return sorted(
        ticker
        for ticker, values in si.rows.items()
        if _at_least(values["VALOR DE MERCADO"], ELIGIBILITY["cap_min"])
        and _at_least(values["LIQUIDEZ MEDIA DIARIA"], ELIGIBILITY["liq_min"])
    )


def _number(cell: str | None) -> float | None:
    text = (cell or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _at_least(value: float | None, minimum: float) -> bool:
    return value is not None and value >= minimum
