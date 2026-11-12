"""Recent B3 trading filter of the Status Invest universe (issue #63).

Status Invest keeps listing tickers that no longer trade on B3. After each
collection, every stock and FII with no cash-market session in the
MAX_QUOTE_AGE_DAYS calendar days before the collection date is dropped from
the run's CSV, and the removal is recorded in the run workspace.

Trading comes from the B3 yearly COTAHIST file: TPMERC 010, CODBDI 02, 07 and
08 for stocks and CODBDI 12 for FIIs. In January the previous year's file is
read too. The yearly file is about 40 MB, so it is downloaded at most once per
day into a cache directory outside the run workspace and reused across runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Callable, Mapping
import urllib.request
import zipfile

from pit_rebuild import cotahist
from pit_rebuild.sources import cotahist_url

COTAHIST_CACHE_ENV = "PRUMO_COTAHIST_CACHE_DIR"
DOWNLOAD_TIMEOUT_SECONDS = 180
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
REASON_NO_RECENT_TRADE = "no_recent_trade"
NO_TRADE = "nenhum"
FILTER_RECORD_NAME = "universe-filter.json"
# Brazil has had no daylight saving time since 2019: B3 session dates are UTC-3 dates.
B3_TIMEZONE = timezone(timedelta(hours=-3))
ROLE_CODBDI: Mapping[str, tuple[str, ...]] = {
    "stocks": cotahist.QUOTE_CODBDI,
    "fiis": cotahist.FII_CODBDI,
}
_DAY_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LINE = re.compile(r"[^\n]*\n|[^\n]+")


class B3TradingUnavailableError(RuntimeError):
    """The COTAHIST file could not be downloaded or read; the run must not continue unfiltered."""

    code = "b3_trading_unavailable"


def b3_today() -> date:
    return datetime.now(B3_TIMEZONE).date()


def default_cache_root() -> Path:
    configured = os.environ.get(COTAHIST_CACHE_ENV)
    if configured:
        return Path(configured)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "prumo" / "cotahist"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read()


@dataclass(frozen=True)
class CotahistFile:
    url: str
    path: Path
    sha256: str
    downloaded_on: date

    def record(self) -> dict[str, str]:
        return {"url": self.url, "sha256": self.sha256, "downloaded_on": self.downloaded_on.isoformat()}


class CotahistCache:
    """Yearly COTAHIST zips kept under <root>/<download date>/; one download per file per day."""

    _lock = threading.Lock()

    def __init__(self, root: Path | None = None, downloader: Callable[[str], bytes] = download):
        self.root = Path(root) if root is not None else default_cache_root()
        self.downloader = downloader

    def get(self, year: int, today: date) -> CotahistFile:
        url = cotahist_url(year)
        directory = self.root / today.isoformat()
        path = directory / url.rsplit("/", 1)[1]
        with self._lock:
            if not path.is_file():
                data = self.downloader(url)
                if not data:
                    raise ValueError(f"empty COTAHIST download from {url}")
                directory.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".part")
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                    os.replace(temporary, path)
                except BaseException:
                    Path(temporary).unlink(missing_ok=True)
                    raise
                self._prune(today)
        return CotahistFile(url=url, path=path, sha256=_sha256(path), downloaded_on=today)

    def _prune(self, today: date) -> None:
        """Drop day directories older than yesterday; yesterday stays for runs that cross midnight."""
        keep_from = (today - timedelta(days=1)).isoformat()
        for child in self.root.iterdir():
            if child.is_dir() and _DAY_DIRECTORY.match(child.name) and child.name < keep_from:
                shutil.rmtree(child, ignore_errors=True)


def cotahist_years(collected_on: date) -> list[int]:
    """The yearly files that cover the window: the previous year too in January."""
    if collected_on.month == 1:
        return [collected_on.year - 1, collected_on.year]
    return [collected_on.year]


def filter_recent_trades(
    csv_paths: Mapping[str, Path],
    workspace: Path,
    *,
    collected_on: date,
    cache: CotahistCache,
) -> dict[str, object]:
    """Drop the tickers with no session in the window from each CSV and write the record.

    csv_paths maps "stocks" and "fiis" to the run's collected CSVs. Raises
    B3TradingUnavailableError when COTAHIST cannot be downloaded or read.
    """

    tickers = {role: _tickers(path) for role, path in csv_paths.items()}
    try:
        files = [cache.get(year, collected_on) for year in cotahist_years(collected_on)]
        quotes = cotahist.load(
            [file.path for file in files],
            set().union(*tickers.values()),
            codbdi=sorted({code for role in csv_paths for code in ROLE_CODBDI[role]}),
        )
    except Exception as exc:
        raise B3TradingUnavailableError("B3 COTAHIST could not be downloaded or read") from exc
    window_start = collected_on - timedelta(days=cotahist.MAX_QUOTE_AGE_DAYS)
    if not any(window_start <= day <= collected_on for day in quotes.sessions):
        raise B3TradingUnavailableError(
            f"B3 COTAHIST has no session between {window_start.isoformat()} and {collected_on.isoformat()}"
        )

    removed: dict[str, list[dict[str, str]]] = {}
    for role, path in csv_paths.items():
        dropped: dict[str, str] = {}
        for ticker in tickers[role]:
            last = _last_session(quotes, ticker, ROLE_CODBDI[role], collected_on)
            if last is None or (collected_on - last).days > cotahist.MAX_QUOTE_AGE_DAYS:
                dropped[ticker] = last.isoformat() if last is not None else NO_TRADE
        if dropped:
            _drop_rows(path, set(dropped))
        removed[role] = [
            {"ticker": ticker, "reason": REASON_NO_RECENT_TRADE, "last_trade": last}
            for ticker, last in sorted(dropped.items())
        ]

    record = {
        "collected_on": collected_on.isoformat(),
        "window_days": cotahist.MAX_QUOTE_AGE_DAYS,
        "cotahist": [file.record() for file in files],
        "csv_sha256": {role: _sha256(path) for role, path in csv_paths.items()},
        "removed": removed,
    }
    record_path = Path(workspace) / "status-invest" / FILTER_RECORD_NAME
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def _last_session(quotes: cotahist.Quotes, ticker: str, codbdi: tuple[str, ...], on: date) -> date | None:
    days = [row.day for row in quotes.rows.get(ticker, []) if row.codbdi in codbdi and row.day <= on]
    return max(days) if days else None


def _tickers(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted({str(row.get("TICKER") or "").strip().upper() for row in csv.DictReader(handle)} - {""})


def _drop_rows(path: Path, tickers: set[str]) -> None:
    """Rewrite path without the rows of tickers, keeping every other byte (header, BOM, line ends, order)."""
    data = path.read_bytes()
    text = data.decode("utf-8")
    lines = _LINE.findall(text)
    header = next(csv.reader([lines[0]]))
    column = [name.lstrip("﻿") for name in header].index("TICKER")
    kept = [lines[0]]
    for line in lines[1:]:
        fields = next(csv.reader([line]), [])
        if not fields:
            kept.append(line)
            continue
        if len(fields) != len(header):
            raise ValueError(f"{path.name}: a row spans several lines; it cannot be filtered line by line")
        if fields[column].strip().upper() not in tickers:
            kept.append(line)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write("".join(kept).encode("utf-8"))
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
