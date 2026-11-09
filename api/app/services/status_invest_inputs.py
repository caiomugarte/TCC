from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from snapshot_manifest import file_sha256


STATUS_INVEST_TIMEOUT_SECONDS = 30


class StatusInvestInputError(RuntimeError):
    code = "status_invest_input_failed"


class SelectorSnapshotChangedError(RuntimeError):
    """The selector CSV bytes no longer match the SHA-256 recorded for the run."""


class SelectorPriceError(ValueError):
    """A selected ticker has no usable PRECO in the run's selector CSV."""


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    provider: str
    retrieved_at: str
    sha256: str

    def provenance(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class StatusInvestInputs:
    stocks: SourceSnapshot
    fiis: SourceSnapshot

    def provenance(self) -> dict[str, dict[str, str]]:
        return {
            "stocks": self.stocks.provenance(),
            "fiis": self.fiis.provenance(),
        }


def _validate_csv(path: Path, validator: Callable[[list[Mapping[str, str]]], None]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("dataset is missing or empty")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("dataset has no header")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values())
        or not str(row.get("TICKER", "")).strip()
        for row in rows
    ):
        raise ValueError("dataset has malformed or empty rows")
    validator(rows)


def _source_snapshot(path: Path) -> SourceSnapshot:
    return SourceSnapshot(
        path=path.resolve(),
        provider="statusinvest",
        retrieved_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        sha256=file_sha256(path),
    )


def refresh_inputs(
    workspace: Path,
    *,
    stock_collector: Callable[..., Any] | None = None,
    fii_collector: Callable[..., Any] | None = None,
    timeout: int = STATUS_INVEST_TIMEOUT_SECONDS,
) -> StatusInvestInputs:
    """Refresh and validate immutable Status Invest files for one run workspace."""

    from fetch_status_invest import refresh as refresh_stocks, validate_rows as validate_stocks
    from fetch_status_invest_fii import refresh as refresh_fiis, validate_rows as validate_fiis

    collectors = (
        (
            "stocks",
            stock_collector or refresh_stocks,
            validate_stocks,
            Path(workspace) / "status-invest" / "stocks.csv",
        ),
        (
            "fiis",
            fii_collector or refresh_fiis,
            validate_fiis,
            Path(workspace) / "status-invest" / "fiis.csv",
        ),
    )
    snapshots: dict[str, SourceSnapshot] = {}
    failures: list[tuple[str, Exception]] = []
    for role, collector, validator, path in collectors:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            collector(output_path=path, timeout=timeout)
            _validate_csv(path, validator)
            snapshots[role] = _source_snapshot(path)
        except Exception as exc:
            failures.append((role, exc))
    if failures:
        role, error = failures[0]
        raise StatusInvestInputError(f"Status Invest {role} input failed validation") from error
    return StatusInvestInputs(stocks=snapshots["stocks"], fiis=snapshots["fiis"])


def _price_cents(ticker: str, raw: str | None) -> tuple[Decimal, int]:
    try:
        value = Decimal((raw or "").strip())
    except InvalidOperation:
        raise SelectorPriceError(f"{ticker}: PRECO {raw!r} is not a number") from None
    if not value.is_finite():
        raise SelectorPriceError(f"{ticker}: PRECO {raw!r} is not finite")
    cents = int(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)
    if cents <= 0:
        raise SelectorPriceError(f"{ticker}: PRECO {raw!r} is not positive")
    return value, cents


def load_selector_prices(snapshot: SourceSnapshot, tickers: Sequence[str]) -> dict[str, int]:
    """Return the PRECO of each ticker, in cents, from the run's hashed selector CSV.

    The bytes are read once; the same bytes are hashed against the recorded
    SHA-256 and then parsed, so every price comes from the hashed file.
    """

    try:
        data = Path(snapshot.path).read_bytes()
    except OSError as exc:
        raise SelectorPriceError(f"selector CSV {snapshot.path} could not be read") from exc
    if hashlib.sha256(data).hexdigest() != snapshot.sha256:
        raise SelectorSnapshotChangedError(f"selector CSV {snapshot.path} changed after the run recorded it")
    wanted = [str(ticker).strip().upper() for ticker in tickers]
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    has_price_column = "PRECO" in (reader.fieldnames or [])
    found: dict[str, tuple[Decimal, int]] = {}
    for row in reader:
        ticker = str(row.get("TICKER") or "").strip().upper()
        if ticker not in wanted:
            continue
        if not has_price_column:
            raise SelectorPriceError(f"{ticker}: selector CSV has no PRECO column")
        value, cents = _price_cents(ticker, row.get("PRECO"))
        if ticker in found and found[ticker][0] != value:
            raise SelectorPriceError(f"{ticker}: conflicting PRECO values in duplicate rows")
        found.setdefault(ticker, (value, cents))
    missing = [ticker for ticker in wanted if ticker not in found]
    if missing:
        raise SelectorPriceError(f"{', '.join(missing)}: no row in the selector CSV")
    return {ticker: found[ticker][1] for ticker in wanted}
