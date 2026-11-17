from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from snapshot_manifest import SnapshotChangedError, SnapshotUnavailableError


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


def _check_recorded_hashes(manifest: Any, recorded: Mapping[str, str]) -> None:
    for key, source in manifest.sources.items():
        expected = recorded.get(key)
        if expected is not None and str(expected).lower() != source.sha256:
            raise SnapshotChangedError(
                f"source {key} hash differs from the one recorded for the run"
            )


def load_snapshot_inputs(
    manifest: Any,
    workspace: Path,
    recorded_hashes: Mapping[str, str] | None = None,
) -> StatusInvestInputs:
    """Selector inputs read only from the run's snapshot, never collected live.

    Each stock/FII CSV named by the manifest is read once; its bytes are checked
    against the manifest SHA-256 and against the hashes recorded for the run,
    validated, and copied into the run workspace so the engines read exactly
    the verified bytes. A missing file raises SnapshotUnavailableError; a hash
    divergence or a malformed CSV raises SnapshotChangedError.
    """

    from fetch_status_invest import validate_rows as validate_stocks
    from fetch_status_invest_fii import validate_rows as validate_fiis

    _check_recorded_hashes(manifest, recorded_hashes or {})
    retrieved_at = str(manifest.created_at)
    snapshots: dict[str, SourceSnapshot] = {}
    for role, key, validator in (
        ("stocks", "stock", validate_stocks),
        ("fiis", "fii", validate_fiis),
    ):
        source = manifest.selector_source_for(key)
        source_path = source.resolved_path()
        try:
            data = source_path.read_bytes()
        except OSError as exc:
            raise SnapshotUnavailableError(f"source {key} not found: {source_path}") from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != source.sha256:
            raise SnapshotChangedError(f"SHA-256 mismatch for source {key}: {source_path}")
        path = Path(workspace) / "snapshot" / f"{role}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            _validate_csv(path, validator)
        except Exception as exc:
            raise SnapshotChangedError(f"source {key} is not a valid selector CSV") from exc
        snapshots[role] = SourceSnapshot(
            path=path.resolve(),
            provider=source.provider,
            retrieved_at=str(source.metadata.get("retrieved_at") or retrieved_at),
            sha256=digest,
        )
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
