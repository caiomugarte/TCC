"""Local snapshot readers and benchmark transformations for allocation.

The numerical optimizer consumes dated daily returns. This module is the
boundary that turns documented level snapshots into that representation and
keeps currency conversions and data-quality checks visible.
"""

import csv
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from allocation_config import ASSET_CLASSES
from core.allocation import DailyReturn, simulate_portfolio


class SnapshotError(ValueError):
    """Raised when a local allocation snapshot is incomplete or malformed."""


LevelSeries = Dict[date, float]


@dataclass(frozen=True)
class SnapshotBundle:
    """Aligned BRL daily returns plus provenance for one analysis run."""

    rows: Tuple[DailyReturn, ...]
    metadata: Mapping[str, object]
    start_date: date
    end_date: date


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError as exc:
        raise SnapshotError(f"invalid ISO date: {raw!r}") from exc


def read_levels_csv(path: Path) -> Dict[str, LevelSeries]:
    """Read `date,<series>...` level data without filling missing values."""

    if not path.exists():
        raise SnapshotError(f"snapshot file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise SnapshotError(f"{path} must contain a date column")
        series_names = [name for name in reader.fieldnames if name != "date"]
        if not series_names:
            raise SnapshotError(f"{path} must contain at least one level column")

        result = {name: {} for name in series_names}
        for line_number, row in enumerate(reader, start=2):
            current_date = _parse_date(row["date"] or "")
            for name in series_names:
                raw_value = (row.get(name) or "").strip()
                if not raw_value:
                    raise SnapshotError(
                        f"missing value for {name} on line {line_number} in {path}"
                    )
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise SnapshotError(
                        f"invalid value for {name} on line {line_number} in {path}"
                    ) from exc
                if value <= 0:
                    raise SnapshotError(
                        f"level for {name} on {current_date} must be positive"
                    )
                if current_date in result[name]:
                    raise SnapshotError(f"duplicate date {current_date} in {path}")
                result[name][current_date] = value
    return result


def read_metadata(path: Path) -> Mapping[str, object]:
    """Read and minimally validate snapshot provenance metadata."""

    if not path.exists():
        raise SnapshotError(f"metadata file not found: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid metadata JSON: {path}") from exc
    required = ("source", "retrieved_at", "cutoff_date")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise SnapshotError(f"metadata missing required fields: {missing}")
    return metadata


def _single_series(series: Mapping[str, LevelSeries], name: str = "value") -> LevelSeries:
    if name in series:
        return dict(series[name])
    if len(series) == 1:
        return dict(next(iter(series.values())))
    raise SnapshotError(f"expected series {name!r}; found {sorted(series)}")


def multiply_levels(left: LevelSeries, right: LevelSeries) -> LevelSeries:
    """Multiply two level series on exact common dates only."""

    common_dates = sorted(set(left).intersection(right))
    if not common_dates:
        raise SnapshotError("series have no common dates")
    return {current_date: left[current_date] * right[current_date] for current_date in common_dates}


def levels_to_returns(
    level_series: Mapping[str, LevelSeries],
    class_names: Sequence[str],
) -> Tuple[DailyReturn, ...]:
    """Convert aligned level series to dated simple returns.

    The first common date is retained with zero return so a simulator can use
    it as the initial observation. Every later return uses the previous
    available observation; absent dates are never manufactured.
    """

    classes = tuple(class_names)
    if set(level_series) != set(classes):
        raise SnapshotError(
            f"level series must contain exactly {list(classes)}; found {sorted(level_series)}"
        )
    common_dates = set(next(iter(level_series.values())))
    for series in level_series.values():
        common_dates.intersection_update(series)
    dates = sorted(common_dates)
    if not dates:
        raise SnapshotError("benchmark series have no common dates")

    rows = []
    previous_date = dates[0]
    rows.append(DailyReturn(previous_date, {name: 0.0 for name in classes}))
    for current_date in dates[1:]:
        current_returns = {
            name: level_series[name][current_date] / level_series[name][previous_date] - 1.0
            for name in classes
        }
        rows.append(DailyReturn(current_date, current_returns))
        previous_date = current_date
    return tuple(rows)


def build_equal_weight_sleeve(
    ticker_levels: Mapping[str, LevelSeries],
    rebalance_years: int = 1,
) -> LevelSeries:
    """Build a fixed-ticker, equal-weight annual-rebalanced sleeve."""

    tickers = tuple(sorted(ticker_levels))
    if not tickers:
        raise SnapshotError("at least one Caio ticker is required")
    ticker_rows = levels_to_returns(ticker_levels, tickers)
    target = {ticker: 1.0 / len(tickers) for ticker in tickers}
    path = simulate_portfolio(
        ticker_rows,
        target,
        tickers,
        annual_rebalance=True,
        rebalance_years=rebalance_years,
    )
    return dict(zip(path.dates, path.values))


def load_portfolio_tickers(
    portfolio_path: Path,
    portfolio_label: str,
) -> Tuple[str, ...]:
    """Load and validate ticker identity from a consensus artifact."""

    if not portfolio_path.exists():
        raise SnapshotError(f"{portfolio_label} portfolio not found: {portfolio_path}")
    try:
        payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid {portfolio_label} portfolio JSON: {portfolio_path}") from exc
    if not isinstance(payload, list) or not payload:
        raise SnapshotError(f"{portfolio_label} portfolio must be a non-empty JSON list")

    tickers = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SnapshotError(
                f"{portfolio_label} portfolio item {index} must be an object with TICKER"
            )
        raw_ticker = item.get("TICKER")
        if not isinstance(raw_ticker, str) or not raw_ticker.strip():
            raise SnapshotError(
                f"{portfolio_label} portfolio item {index} must contain a non-empty TICKER"
            )
        tickers.append(raw_ticker.strip().upper())

    if len(set(tickers)) != len(tickers):
        raise SnapshotError(f"{portfolio_label} portfolio must contain unique TICKER values")
    return tuple(tickers)


def load_caio_tickers(portfolio_path: Path) -> Tuple[str, ...]:
    """Load the fixed ticker list from the canonical Caio consensus artifact."""

    return load_portfolio_tickers(portfolio_path, "Caio")


def _metadata_tickers(metadata: Mapping[str, object], key: str) -> set[str]:
    raw_tickers = metadata.get(key, ())
    if not isinstance(raw_tickers, (list, tuple, set)):
        return set()
    return {
        str(ticker).strip().upper()
        for ticker in raw_tickers
        if str(ticker).strip()
    }


def load_snapshot_bundle(
    snapshot_dir: Path,
    portfolio_path: Path,
    fii_portfolio_path: Optional[Path] = None,
    rebalance_years: int = 1,
) -> SnapshotBundle:
    """Load the documented offline snapshot layout into BRL daily returns."""

    metadata = read_metadata(snapshot_dir / "metadata.json")
    skipped_tickers = _metadata_tickers(metadata, "skipped_tickers")
    tickers = tuple(
        ticker
        for ticker in load_caio_tickers(portfolio_path)
        if ticker.upper() not in skipped_tickers
    )
    if not tickers:
        raise SnapshotError("all portfolio tickers were skipped")
    ticker_data = read_levels_csv(snapshot_dir / "caio_stocks.csv")
    missing_tickers = [ticker for ticker in tickers if ticker not in ticker_data]
    if missing_tickers:
        raise SnapshotError(
            "Caio ticker history is incomplete; no replacements are allowed: "
            f"{missing_tickers}"
        )
    stock_sleeve = build_equal_weight_sleeve(
        {ticker: ticker_data[ticker] for ticker in tickers},
        rebalance_years=rebalance_years,
    )

    if fii_portfolio_path is None:
        default_fii_path = portfolio_path.parent / "carteira_fii_caio_consensus.json"
        if default_fii_path.exists() and (snapshot_dir / "caio_fiis.csv").exists():
            fii_portfolio_path = default_fii_path

    if fii_portfolio_path is None:
        fiis = _single_series(read_levels_csv(snapshot_dir / "ifix.csv"))
    else:
        fii_tickers = tuple(
            ticker
            for ticker in load_portfolio_tickers(fii_portfolio_path, "FII")
            if ticker not in _metadata_tickers(metadata, "fii_skipped_tickers")
        )
        if not fii_tickers:
            raise SnapshotError("all FII portfolio tickers were skipped")
        fii_data = read_levels_csv(snapshot_dir / "caio_fiis.csv")
        missing_fii_tickers = [ticker for ticker in fii_tickers if ticker not in fii_data]
        if missing_fii_tickers:
            raise SnapshotError(
                "FII ticker history is incomplete; no replacements are allowed: "
                f"{missing_fii_tickers}"
            )
        fiis = build_equal_weight_sleeve(
            {ticker: fii_data[ticker] for ticker in fii_tickers},
            rebalance_years=rebalance_years,
        )
    sp500_usd = _single_series(read_levels_csv(snapshot_dir / "sp500_total_return_usd.csv"))
    fixed_income = _single_series(read_levels_csv(snapshot_dir / "di.csv"))
    btc_usd = _single_series(read_levels_csv(snapshot_dir / "btc_usd.csv"))
    ptax = _single_series(read_levels_csv(snapshot_dir / "ptax.csv"))
    international_brl = multiply_levels(sp500_usd, ptax)
    crypto_brl = multiply_levels(btc_usd, ptax)

    levels = {
        ASSET_CLASSES[0]: stock_sleeve,
        ASSET_CLASSES[1]: fiis,
        ASSET_CLASSES[2]: international_brl,
        ASSET_CLASSES[3]: fixed_income,
        ASSET_CLASSES[4]: crypto_brl,
    }
    rows = levels_to_returns(levels, ASSET_CLASSES)
    return SnapshotBundle(
        rows=rows,
        metadata=metadata,
        start_date=rows[0].date,
        end_date=rows[-1].date,
    )
