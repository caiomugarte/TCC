#!/usr/bin/env python3
"""Fetch the documented benchmark snapshot used by ``run_allocation``.

This module is deliberately separate from the optimizer.  It downloads raw
observations, applies only the source-specific transformations documented in
the snapshot metadata, and writes a reproducible local snapshot.
"""

import argparse
import base64
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import ALLOCATION_DATA_DIR, ALLOCATION_OUTPUTS_DIR  # noqa: E402
from allocation_data import load_caio_tickers, load_portfolio_tickers  # noqa: E402


LevelSeries = Dict[date, float]

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
B3_IFIX_URL = (
    "https://sistemaswebb3-listados.b3.com.br/"
    "indexStatisticsProxy/IndexCall/GetPortfolioDay/{token}"
)
BCB_PTAX_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,"
    "dataFinalCotacao=@dataFinalCotacao)"
)
BCB_SGS_CDI_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
USER_AGENT = "caio-allocation-snapshot/1.0"


class SnapshotFetchError(RuntimeError):
    """Raised when a required source cannot produce a valid series."""


_PERMANENT_TICKER_FAILURE_MARKERS = (
    "http error 404",
    "symbol may be delisted",
    "yahoo returned no chart",
    "yahoo returned no quote series",
    "yahoo returned no usable observations",
)


def _is_permanent_ticker_failure(error: SnapshotFetchError) -> bool:
    """Return whether Yahoo definitively has no usable series for a ticker."""

    message = str(error).casefold()
    return any(marker in message for marker in _PERMANENT_TICKER_FAILURE_MARKERS)


def _parse_iso_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {raw}") from exc


def _fetch_json_payload(url: str, timeout: int = 45) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SnapshotFetchError(f"could not fetch {url}: {exc}") from exc
    return payload


def _fetch_json(url: str, timeout: int = 45) -> Mapping[str, object]:
    payload = _fetch_json_payload(url, timeout)
    if not isinstance(payload, dict):
        raise SnapshotFetchError(f"source returned a non-object JSON payload: {url}")
    return payload


def _parse_decimal(raw: object) -> float:
    if raw is None:
        raise ValueError("missing decimal")
    text = str(raw).strip()
    if "," in text and "." in text:
        # B3's en-us endpoint uses comma thousands and dot decimals, while
        # some B3 exports use the inverse convention.
        if text.rfind(",") < text.rfind("."):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    value = float(text)
    if value <= 0:
        raise ValueError(f"level must be positive: {raw!r}")
    return value


def _write_levels(path: Path, series: Mapping[str, LevelSeries]) -> Tuple[date, date, int]:
    names = tuple(series)
    if not names or any(not values for values in series.values()):
        raise SnapshotFetchError(f"cannot write an empty snapshot file: {path}")

    common_dates = set(next(iter(series.values())))
    for values in series.values():
        common_dates.intersection_update(values)
    dates = sorted(common_dates)
    if not dates:
        raise SnapshotFetchError(f"series have no common dates for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date",) + names)
        writer.writeheader()
        for current_date in dates:
            writer.writerow(
                {
                    "date": current_date.isoformat(),
                    **{name: f"{series[name][current_date]:.12g}" for name in names},
                }
            )
    return dates[0], dates[-1], len(dates)


def fetch_yahoo_levels(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    adjusted: bool,
    timeout: int = 45,
) -> LevelSeries:
    """Fetch one daily Yahoo chart series without filling missing observations."""

    start_timestamp = int(
        datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    )
    end_timestamp = int(
        datetime.combine(
            end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        ).timestamp()
    )
    query = urlencode(
        {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    payload = _fetch_json(YAHOO_CHART_URL.format(symbol=symbol) + "?" + query, timeout)
    chart = payload.get("chart")
    if not isinstance(chart, dict) or not chart.get("result"):
        error = chart.get("error") if isinstance(chart, dict) else None
        raise SnapshotFetchError(f"Yahoo returned no chart for {symbol}: {error}")
    results = chart["result"]
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise SnapshotFetchError(f"malformed Yahoo chart response for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise SnapshotFetchError(f"malformed Yahoo chart response for {symbol}")
    quote_rows = indicators.get("quote")
    if not isinstance(quote_rows, list):
        raise SnapshotFetchError(f"malformed Yahoo chart response for {symbol}")
    if not quote_rows or not isinstance(quote_rows[0], dict):
        raise SnapshotFetchError(f"Yahoo returned no quote series for {symbol}")
    values = quote_rows[0].get("close")
    if values is None:
        raise SnapshotFetchError(f"Yahoo returned no quote series for {symbol}")
    if not isinstance(values, list):
        raise SnapshotFetchError(f"malformed Yahoo chart response for {symbol}")
    if adjusted:
        adjusted_rows = indicators.get("adjclose") or []
        if adjusted_rows and not isinstance(adjusted_rows[0], dict):
            raise SnapshotFetchError(f"malformed Yahoo adjusted-close response for {symbol}")
        adjusted_values = adjusted_rows[0].get("adjclose") if adjusted_rows else None
        if adjusted_values is not None and not isinstance(adjusted_values, list):
            raise SnapshotFetchError(f"malformed Yahoo adjusted-close response for {symbol}")
        if adjusted_values:
            values = adjusted_values

    levels: LevelSeries = {}
    for timestamp, raw_value in zip(timestamps, values):
        if raw_value is None:
            continue
        current_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
        if not start_date <= current_date <= end_date:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise SnapshotFetchError(f"malformed Yahoo observation for {symbol}") from exc
        if value > 0:
            levels[current_date] = value
    if not levels:
        raise SnapshotFetchError(f"Yahoo returned no usable observations for {symbol}")
    return levels


def _b3_filter_token(index_name: str, year: int) -> str:
    payload = json.dumps(
        {"index": index_name, "language": "en-us", "year": str(year)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def parse_b3_ifix_payload(
    payload: Mapping[str, object],
    year: int,
    start_date: date,
    end_date: date,
) -> LevelSeries:
    """Turn B3's month-column daily response into dated IFIX levels."""

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise SnapshotFetchError(f"B3 IFIX response has no results for {year}")
    levels: LevelSeries = {}
    for raw_row in raw_results:
        if not isinstance(raw_row, dict):
            continue
        try:
            day = int(raw_row.get("day", 0))
        except (TypeError, ValueError):
            continue
        if day < 1 or day > 31:
            continue
        for month in range(1, 13):
            raw_value = raw_row.get(f"rateValue{month}")
            if raw_value in (None, ""):
                continue
            try:
                current_date = date(year, month, day)
                value = _parse_decimal(raw_value)
            except (ValueError, TypeError):
                continue
            if start_date <= current_date <= end_date:
                levels[current_date] = value
    return levels


def fetch_ifix_levels(
    start_date: date,
    end_date: date,
    *,
    timeout: int = 45,
) -> LevelSeries:
    levels: LevelSeries = {}
    for year in range(start_date.year, end_date.year + 1):
        url = B3_IFIX_URL.format(token=_b3_filter_token("IFIX", year))
        payload = _fetch_json(url, timeout)
        levels.update(parse_b3_ifix_payload(payload, year, start_date, end_date))
    if not levels:
        raise SnapshotFetchError("B3 IFIX returned no observations")
    return levels


def fetch_ptax_levels(
    start_date: date,
    end_date: date,
    *,
    timeout: int = 45,
) -> LevelSeries:
    """Fetch closing PTAX midpoint (BRL per USD) for each available date."""

    query = urlencode(
        {
            "@moeda": "'USD'",
            "@dataInicial": f"'{start_date:%m-%d-%Y}'",
            "@dataFinalCotacao": f"'{end_date:%m-%d-%Y}'",
            "$format": "json",
        }
    )
    payload = _fetch_json(BCB_PTAX_URL + "?" + query, timeout)
    raw_values = payload.get("value")
    if not isinstance(raw_values, list):
        raise SnapshotFetchError("BCB PTAX response has no value list")

    levels: LevelSeries = {}
    for raw_value in raw_values:
        if not isinstance(raw_value, dict) or raw_value.get("tipoBoletim") != "Fechamento":
            continue
        timestamp = str(raw_value.get("dataHoraCotacao", ""))
        try:
            current_date = date.fromisoformat(timestamp[:10])
            buy = float(raw_value["cotacaoCompra"])
            sell = float(raw_value["cotacaoVenda"])
        except (KeyError, TypeError, ValueError):
            continue
        if start_date <= current_date <= end_date:
            levels[current_date] = (buy + sell) / 2.0
    if not levels:
        raise SnapshotFetchError("BCB PTAX returned no closing observations")
    return levels


def fetch_bcb_cdi_levels(
    start_date: date,
    end_date: date,
    *,
    timeout: int = 45,
) -> LevelSeries:
    """Fetch BCB SGS 12 daily CDI factors and compound them into a level.

    SGS 12 is the daily CDI rate in percent per day. Its factors match the
    annual DI values from B3's FTP files after applying B3's 252-business-day
    convention, while the BCB API is practical for a ten-year download.
    """

    rates: Dict[date, float] = {}
    current_start = start_date
    while current_start <= end_date:
        next_year = date(current_start.year + 1, 1, 1)
        current_end = min(end_date, next_year - timedelta(days=1))
        query = urlencode(
            {
                "formato": "json",
                "dataInicial": f"{current_start:%d/%m/%Y}",
                "dataFinal": f"{current_end:%d/%m/%Y}",
            }
        )
        payload = _fetch_json_payload(BCB_SGS_CDI_URL + "?" + query, timeout)
        if not isinstance(payload, list):
            raise SnapshotFetchError("BCB SGS CDI response is not a list")
        for raw_value in payload:
            if not isinstance(raw_value, dict):
                continue
            try:
                current_date = datetime.strptime(
                    str(raw_value["data"]), "%d/%m/%Y"
                ).date()
                daily_percent = float(str(raw_value["valor"]).replace(",", "."))
            except (KeyError, TypeError, ValueError):
                continue
            if start_date <= current_date <= end_date:
                rates[current_date] = daily_percent / 100.0
        current_start = next_year

    if not rates:
        raise SnapshotFetchError("BCB SGS CDI returned no observations")

    levels: LevelSeries = {}
    value = 100.0
    for current_date in sorted(rates):
        if levels:
            value *= 1.0 + rates[current_date]
        levels[current_date] = value
    return levels


def _write_metadata(
    path: Path,
    *,
    start_date: date,
    end_date: date,
    sources: Mapping[str, object],
    skipped_tickers: Sequence[str] = (),
    skipped_ticker_reasons: Optional[Mapping[str, str]] = None,
    fii_skipped_tickers: Sequence[str] = (),
    fii_skipped_ticker_reasons: Optional[Mapping[str, str]] = None,
    portfolio_artifacts: Optional[Mapping[str, str]] = None,
) -> None:
    retrieved_at = datetime.now(timezone(timedelta(hours=-3))).isoformat()
    metadata = {
        "source": "B3, Banco Central do Brasil, and Yahoo Finance chart API",
        "retrieved_at": retrieved_at,
        "cutoff_date": end_date.isoformat(),
        "requested_start_date": start_date.isoformat(),
        "notes": (
            "Daily common-date intersection; no forward-fill. Caio stocks use "
            "Yahoo adjusted close and equal-weight annual rebalancing. IFIX and "
            "S&P 500 TR are total-return levels. USD levels are converted with "
            "closing PTAX midpoint. Fixed income uses BCB SGS 12 daily CDI "
            "factors (cross-checked against B3 DI FTP "
            "using the 252-business-day convention). BTC uses Yahoo BTC-USD close. "
            "Ibovespa is a price-index benchmark and does not include dividends. "
            "Permanent Yahoo no-data ticker failures are skipped and recorded; "
            "transient fetch failures stop the snapshot."
        ),
        "sources": sources,
        "portfolio_artifacts": dict(portfolio_artifacts or {}),
        "skipped_tickers": list(skipped_tickers),
        "skipped_ticker_reasons": dict(skipped_ticker_reasons or {}),
        "fii_skipped_tickers": list(fii_skipped_tickers),
        "fii_skipped_ticker_reasons": dict(fii_skipped_ticker_reasons or {}),
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_portfolio_levels(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    label: str,
    timeout: int,
    skip_tickers: Sequence[str] = (),
) -> Tuple[Dict[str, LevelSeries], Tuple[str, ...], list[str], Dict[str, str]]:
    skipped = {str(ticker).strip().upper() for ticker in skip_tickers if str(ticker).strip()}
    ticker_set = {ticker.upper() for ticker in tickers}
    unknown = skipped - ticker_set
    if unknown:
        raise SnapshotFetchError(
            f"skip tickers not found in {label} portfolio: {sorted(unknown)}"
        )
    skipped_reasons = {
        ticker: "explicitly skipped for this allocation snapshot"
        for ticker in skipped
    }
    levels: Dict[str, LevelSeries] = {}
    for index, ticker in enumerate(tickers, start=1):
        if ticker.upper() in skipped:
            continue
        print(f"Fetching {label} ticker {index}/{len(tickers)}: {ticker}.SA", flush=True)
        try:
            levels[ticker] = fetch_yahoo_levels(
                f"{ticker}.SA", start_date, end_date, adjusted=True, timeout=timeout
            )
        except SnapshotFetchError as exc:
            if not _is_permanent_ticker_failure(exc):
                raise
            ticker_key = ticker.upper()
            skipped.add(ticker_key)
            skipped_reasons[ticker_key] = str(exc)
            print(
                f"Skipping {label} {ticker}: Yahoo has no usable history ({exc})",
                flush=True,
            )

    valid_tickers = tuple(ticker for ticker in tickers if ticker in levels)
    if not valid_tickers:
        raise SnapshotFetchError(f"no {label} tickers have usable Yahoo history")
    return levels, valid_tickers, sorted(skipped), skipped_reasons


def fetch_snapshot(
    snapshot_dir: Path,
    portfolio_path: Path,
    start_date: date,
    end_date: date,
    *,
    fii_portfolio_path: Optional[Path] = None,
    include_ifix: bool = True,
    timeout: int = 45,
    skip_tickers: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    if start_date >= end_date:
        raise SnapshotFetchError("start date must be before end date")
    if not include_ifix and fii_portfolio_path is None:
        raise SnapshotFetchError("FII portfolio is required when IFIX is disabled")

    portfolio_tickers = load_caio_tickers(portfolio_path)
    fii_selected_tickers: Tuple[str, ...] = ()
    if fii_portfolio_path is not None:
        fii_selected_tickers = load_portfolio_tickers(fii_portfolio_path, "FII")

    stock_levels, valid_tickers, skipped, skipped_reasons = _fetch_portfolio_levels(
        portfolio_tickers,
        start_date,
        end_date,
        label="Caio",
        timeout=timeout,
        skip_tickers=skip_tickers or (),
    )

    fii_levels: Dict[str, LevelSeries] = {}
    fii_valid_tickers: Tuple[str, ...] = ()
    fii_skipped: list[str] = []
    fii_skipped_reasons: Dict[str, str] = {}
    if fii_portfolio_path is not None:
        (
            fii_levels,
            fii_valid_tickers,
            fii_skipped,
            fii_skipped_reasons,
        ) = _fetch_portfolio_levels(
            fii_selected_tickers,
            start_date,
            end_date,
            label="FII",
            timeout=timeout,
        )

    print("Fetching S&P 500 total return", flush=True)
    sp500 = fetch_yahoo_levels(
        "^SP500TR", start_date, end_date, adjusted=False, timeout=timeout
    )
    print("Fetching BTC/USD", flush=True)
    btc = fetch_yahoo_levels(
        "BTC-USD", start_date, end_date, adjusted=False, timeout=timeout
    )
    print("Fetching Ibovespa", flush=True)
    ibovespa = fetch_yahoo_levels(
        "^BVSP", start_date, end_date, adjusted=False, timeout=timeout
    )
    ifix = None
    if include_ifix:
        print("Fetching B3 IFIX", flush=True)
        ifix = fetch_ifix_levels(start_date, end_date, timeout=timeout)
    print("Fetching BCB CDI/DI daily factor", flush=True)
    di = fetch_bcb_cdi_levels(start_date, end_date, timeout=timeout)
    print("Fetching BCB PTAX", flush=True)
    ptax = fetch_ptax_levels(start_date, end_date, timeout=timeout)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ranges = {}
    ranges["caio_stocks.csv"] = _write_levels(snapshot_dir / "caio_stocks.csv", stock_levels)
    if fii_portfolio_path is not None:
        ranges["caio_fiis.csv"] = _write_levels(snapshot_dir / "caio_fiis.csv", fii_levels)
    if include_ifix:
        ranges["ifix.csv"] = _write_levels(snapshot_dir / "ifix.csv", {"value": ifix})
    ranges["sp500_total_return_usd.csv"] = _write_levels(
        snapshot_dir / "sp500_total_return_usd.csv", {"value": sp500}
    )
    ranges["di.csv"] = _write_levels(snapshot_dir / "di.csv", {"value": di})
    ranges["btc_usd.csv"] = _write_levels(snapshot_dir / "btc_usd.csv", {"value": btc})
    ranges["ibovespa.csv"] = _write_levels(
        snapshot_dir / "ibovespa.csv", {"value": ibovespa}
    )
    ranges["ptax.csv"] = _write_levels(snapshot_dir / "ptax.csv", {"value": ptax})
    sources = {
        "caio_stocks": {
            "provider": "Yahoo Finance chart API",
            "portfolio_path": str(portfolio_path),
            "selected_tickers": list(portfolio_tickers),
            "symbols": [f"{ticker}.SA" for ticker in valid_tickers],
        },
        "sp500_total_return_usd": {"provider": "Yahoo Finance chart API", "symbol": "^SP500TR"},
        "btc_usd": {"provider": "Yahoo Finance chart API", "symbol": "BTC-USD"},
        "ibovespa": {
            "provider": "Yahoo Finance chart API",
            "symbol": "^BVSP",
            "treatment": "price index; dividends not included",
        },
        "di": {
            "provider": "Banco Central do Brasil SGS API",
            "series": 12,
            "unit": "% per business day",
            "endpoint": BCB_SGS_CDI_URL,
            "cross_check": "B3 DI FTP annual rate converted with 252-business-day factor",
        },
        "ptax": {"provider": "Banco Central do Brasil PTAX OData", "currency": "USD", "quote": "closing midpoint"},
    }
    if fii_portfolio_path is not None:
        sources["caio_fiis"] = {
            "provider": "Yahoo Finance chart API",
            "portfolio_path": str(fii_portfolio_path),
            "selected_tickers": list(fii_selected_tickers),
            "symbols": [f"{ticker}.SA" for ticker in fii_valid_tickers],
            "treatment": "adjusted close; equal-weight annual rebalancing in allocation loader",
            "lookahead_bias": "fixed current optimized FII artifact across historical windows",
            "skipped_tickers": fii_skipped,
            "skipped_ticker_reasons": fii_skipped_reasons,
        }
    if include_ifix:
        sources["ifix"] = {
            "provider": "B3 Index Statistics API",
            "index": "IFIX",
            "role": "benchmark-only",
        }
    _write_metadata(
        snapshot_dir / "metadata.json",
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        skipped_tickers=sorted(skipped),
        skipped_ticker_reasons=skipped_reasons,
        fii_skipped_tickers=fii_skipped,
        fii_skipped_ticker_reasons=fii_skipped_reasons,
        portfolio_artifacts={
            "caio_stocks": str(portfolio_path),
            **(
                {"caio_fiis": str(fii_portfolio_path)}
                if fii_portfolio_path is not None
                else {}
            ),
        },
    )
    return {
        "ranges": ranges,
        "tickers": valid_tickers,
        "skipped_tickers": sorted(skipped),
        "skipped_ticker_reasons": skipped_reasons,
        "fii_tickers": fii_valid_tickers,
        "fii_skipped_tickers": fii_skipped,
        "fii_skipped_ticker_reasons": fii_skipped_reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    today = date.today()
    default_start = today.replace(year=today.year - 10)
    parser = argparse.ArgumentParser(
        description="Fetch the reproducible benchmark snapshot for Caio allocation."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=ALLOCATION_DATA_DIR)
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=ALLOCATION_OUTPUTS_DIR / "carteira_caio_consensus.json",
    )
    parser.add_argument(
        "--fii-portfolio",
        type=Path,
        default=ALLOCATION_OUTPUTS_DIR / "carteira_fii_caio_consensus.json",
        help="fixed FII portfolio JSON",
    )
    parser.add_argument("--start-date", type=_parse_iso_date, default=default_start)
    parser.add_argument("--end-date", type=_parse_iso_date, default=today)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument(
        "--skip-ticker",
        action="append",
        default=[],
        help="Skip ticker only in allocation snapshot; repeat for multiple tickers",
    )
    parser.add_argument(
        "--skip-ifix",
        "--no-ifix",
        action="store_true",
        help="Do not fetch optional IFIX benchmark data",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = fetch_snapshot(
            args.snapshot_dir,
            args.portfolio,
            args.start_date,
            args.end_date,
            fii_portfolio_path=args.fii_portfolio,
            include_ifix=not args.skip_ifix,
            timeout=args.timeout,
            skip_tickers=args.skip_ticker,
        )
    except (SnapshotFetchError, ValueError) as exc:
        print(f"Snapshot could not be fetched: {exc}", file=sys.stderr)
        return 2
    print(f"Snapshot written to {args.snapshot_dir}")
    print(f"  tickers: {len(result['tickers'])}")
    if result["skipped_tickers"]:
        print(f"  skipped: {', '.join(result['skipped_tickers'])}")
        for ticker in result["skipped_tickers"]:
            print(f"    {ticker}: {result['skipped_ticker_reasons'][ticker]}")
    for name, bounds in result["ranges"].items():
        print(f"  {name}: {bounds[0]} to {bounds[1]} ({bounds[2]} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
