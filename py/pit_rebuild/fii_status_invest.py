"""The Status Invest FII CSV, read only, and the universe the FII selector takes as eligible (issue #60).

Eligibility mirrors the selector: the core rule of py/fii_selection.py (CORE_POSITIVE_COLUMNS > 0
and DY >= 0, apply_fii_eligibility) with the widest Premium floor, the arrojado one of
api/app/services/premium_policy.py (FII_RULES: size_min 200 M, liq_min 50 k).
"""

import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path

from .errors import IntegrityError, MissingPathError

NUMERIC_FIELDS = (
    "LIQUIDEZ MEDIA DIARIA",
    "N COTISTAS",
    "PATRIMONIO",
    "N COTAS",
    "PERCENTUAL EM CAIXA",
    "P/VP",
    "CAGR DIVIDENDOS 3 ANOS",
    "CAGR VALOR COTA 3 ANOS",
    "DY",
    "ULTIMO DIVIDENDO",
    "PRECO",
)
CATEGORY_FIELDS = ("GESTAO", "SETOR")
FIELDS = NUMERIC_FIELDS + CATEGORY_FIELDS  # the 13 indicators of the FII selector
CORE_POSITIVE = ("PRECO", "P/VP", "LIQUIDEZ MEDIA DIARIA", "PATRIMONIO", "N COTISTAS", "N COTAS")
FLOOR = {"size_min": 200_000_000, "liq_min": 50_000}
ELIGIBILITY_SOURCE = [
    "py/fii_selection.py: CORE_POSITIVE_COLUMNS, apply_fii_eligibility",
    "api/app/services/premium_policy.py: FII_RULES['arrojado']['filters']",
]
REQUIRED_COLUMNS = ("TICKER", *FIELDS)


@dataclass(frozen=True)
class SIFiiCsv:
    path: Path
    sha256: str
    rows: dict[str, dict[str, float | None]]
    text: dict[str, dict[str, str]]


def load(path: Path) -> SIFiiCsv:
    path = Path(path)
    if not path.is_file():
        raise MissingPathError(str(path), "Status Invest FII CSV does not exist")
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IntegrityError(str(path), f"Status Invest FII CSV is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(content))
    header = reader.fieldnames or []
    for column in REQUIRED_COLUMNS:
        if column not in header:
            raise IntegrityError(column, f"required column missing from {path}")
    rows: dict[str, dict[str, float | None]] = {}
    text: dict[str, dict[str, str]] = {}
    for record in reader:
        ticker = record["TICKER"].strip().upper()
        if ticker in rows:
            raise IntegrityError(ticker, f"ticker repeated in {path}")
        rows[ticker] = {field: _number(record[field]) for field in NUMERIC_FIELDS}
        text[ticker] = {field: (record[field] or "").strip() for field in CATEGORY_FIELDS}
    return SIFiiCsv(path=path, sha256=hashlib.sha256(raw).hexdigest(), rows=rows, text=text)


def eligible(si: SIFiiCsv) -> list[str]:
    """Tickers the FII selector takes as eligible at the date of the CSV, sorted."""
    return sorted(ticker for ticker, values in si.rows.items() if _eligible(values))


def _eligible(values: dict[str, float | None]) -> bool:
    if any(values[column] is None or values[column] <= 0 for column in CORE_POSITIVE):
        return False
    if values["DY"] is None or values["DY"] < 0:
        return False
    return values["PATRIMONIO"] >= FLOOR["size_min"] and values["LIQUIDEZ MEDIA DIARIA"] >= FLOOR["liq_min"]


def _number(cell: str | None) -> float | None:
    text = (cell or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
