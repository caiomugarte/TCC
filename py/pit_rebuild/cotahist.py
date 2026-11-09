"""B3 COTAHIST daily files: quotes, liquidity, ISIN, share class and sessions as of d (PIT-01, IND-01, A-7, A-15, A-16, A-17).

COTAHIST is fixed width. Columns are 1-based and inclusive: TIPREG 1-2, DATA 3-10,
CODBDI 11-12, CODNEG 13-24, TPMERC 25-27, ESPECI 40-49, PREULT 109-121 (two implied
decimals), VOLTOT 171-188 (two implied decimals), FATCOT 211-217, CODISI 231-242.

Only standard-lot records of the cash market are kept (TIPREG 01, TPMERC 010, CODBDI 02)
for the requested tickers. The market calendar is every session date of every TIPREG 01
record, whatever the ticker, so next_session holds for the whole market (A-16).
"""

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
import io
from pathlib import Path
import zipfile

from .errors import IntegrityError

LINE_LENGTH = 245
MAX_QUOTE_AGE_DAYS = 30  # A-7: no session within 30 calendar days before d means no_quote
LIQUIDITY_SESSIONS = 30  # A-7: LIQUIDEZ MEDIA DIARIA averages the last 30 sessions


@dataclass(frozen=True)
class Session:
    """One trading record of a ticker: close (PREULT / FATCOT), VOLTOT, ISIN and the first ESPECI token."""

    day: date
    close: float
    voltot: float
    isin: str
    especi: str


@dataclass
class Quotes:
    """Records of the requested tickers, and the session calendar of every ticker in the files."""

    sessions: list[date] = field(default_factory=list)
    rows: dict[str, list[Session]] = field(default_factory=dict)
    dates: dict[str, list[date]] = field(default_factory=dict)


def load(zip_paths: Iterable[Path], tickers: set[str]) -> Quotes:
    """Read the COTAHIST zips, keeping the lot records of the tickers and every session date."""
    days: set[date] = set()
    by_ticker: dict[str, dict[date, Session]] = {}
    for path in zip_paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.upper().endswith(".TXT"):
                    continue
                with archive.open(info) as raw:
                    for text in io.TextIOWrapper(raw, encoding="latin-1"):
                        if not text.startswith("01"):
                            continue
                        record = _parse(text.rstrip("\r\n"), info.filename)
                        days.add(record.day)
                        if record.tpmerc != "010" or record.codbdi != "02" or record.ticker not in tickers:
                            continue
                        close = record.price / record.fatcot if record.fatcot > 1 else record.price
                        session = Session(day=record.day, close=close, voltot=record.voltot, isin=record.isin, especi=record.especi)
                        by_ticker.setdefault(record.ticker, {})[record.day] = session
    quotes = Quotes(sessions=sorted(days))
    for ticker, by_day in by_ticker.items():
        ordered = [by_day[day] for day in sorted(by_day)]
        quotes.rows[ticker] = ordered
        quotes.dates[ticker] = [row.day for row in ordered]
    return quotes


def quote_at(q: Quotes, ticker: str, d: date, max_age_days: int = MAX_QUOTE_AGE_DAYS) -> tuple[date, float] | None:
    """(session date, close) of the last session on or before d; None when it is more than max_age_days old (A-7)."""
    index = _last_index(q, ticker, d)
    if index < 0:
        return None
    row = q.rows[ticker][index]
    if (d - row.day).days > max_age_days:
        return None
    return row.day, row.close


def avg_volume(q: Quotes, ticker: str, d: date, sessions: int = LIQUIDITY_SESSIONS) -> float | None:
    """Mean VOLTOT of the ticker's last `sessions` sessions on or before d (fewer when fewer exist)."""
    index = _last_index(q, ticker, d)
    if index < 0:
        return None
    window = q.rows[ticker][max(0, index + 1 - sessions) : index + 1]
    return sum(row.voltot for row in window) / len(window)


def isin(q: Quotes, ticker: str) -> str | None:
    rows = q.rows.get(ticker)
    return rows[-1].isin if rows else None


def share_class(q: Quotes, ticker: str) -> str | None:
    """First token of the ticker's ESPECI, e.g. ON, PN, PNA, PNB or UNT (A-17)."""
    rows = q.rows.get(ticker)
    return rows[-1].especi if rows else None


def next_session(q: Quotes, day: date) -> date | None:
    """First market session strictly after day. None before the first loaded session or after the last (A-16)."""
    if not q.sessions or day < q.sessions[0]:
        return None
    index = bisect_right(q.sessions, day)
    return q.sessions[index] if index < len(q.sessions) else None


def class_prices(q: Quotes, tickers: Iterable[str], d: date) -> dict[str, tuple[str, float] | None]:
    """Price of each share class of a company: ON from the ON ticker, PREF from the most liquid preferred one (A-17).

    Units (UNT) are skipped. A ticker counts only when quote_at finds a session in time. Ties in
    average volume go to the smaller ticker.
    """
    best: dict[str, tuple[float, str, float] | None] = {"ON": None, "PREF": None}
    for ticker in sorted(set(tickers)):
        cls = _price_class(q, ticker)
        if cls is None:
            continue
        quote = quote_at(q, ticker, d)
        volume = avg_volume(q, ticker, d)
        if quote is None or volume is None:
            continue
        candidate = (-volume, ticker, quote[1])
        if best[cls] is None or candidate < best[cls]:
            best[cls] = candidate
    return {cls: None if item is None else (item[1], item[2]) for cls, item in best.items()}


@dataclass(frozen=True)
class _Line:
    day: date
    codbdi: str
    ticker: str
    tpmerc: str
    especi: str
    price: float
    voltot: float
    fatcot: int
    isin: str


def _parse(text: str, member: str) -> _Line:
    if len(text) < LINE_LENGTH:
        raise IntegrityError(member, f"COTAHIST line has {len(text)} columns, expected {LINE_LENGTH}")
    day_text = text[2:10]
    try:
        day = date(int(day_text[0:4]), int(day_text[4:6]), int(day_text[6:8]))
    except ValueError as exc:
        raise IntegrityError(member, f"COTAHIST DATA is not a date: {day_text!r}") from exc
    return _Line(
        day=day,
        codbdi=text[10:12],
        ticker=text[12:24].strip(),
        tpmerc=text[24:27],
        especi=_first_token(text[39:49]),
        price=_scaled(text[108:121], member, "PREULT", 100),
        voltot=_scaled(text[170:188], member, "VOLTOT", 100),
        fatcot=_integer(text[210:217], member, "FATCOT"),
        isin=text[230:242].strip(),
    )


def _scaled(cell: str, member: str, name: str, divisor: int) -> float:
    return _integer(cell, member, name) / divisor


def _integer(cell: str, member: str, name: str) -> int:
    try:
        return int(cell)
    except ValueError as exc:
        raise IntegrityError(member, f"COTAHIST {name} is not a number: {cell!r}") from exc


def _first_token(cell: str) -> str:
    parts = cell.split()
    return parts[0] if parts else ""


def _price_class(q: Quotes, ticker: str) -> str | None:
    especi = share_class(q, ticker)
    if especi == "ON":
        return "ON"
    if especi is not None and especi.startswith("PN"):
        return "PREF"
    return None


def _last_index(q: Quotes, ticker: str, d: date) -> int:
    dates = q.dates.get(ticker)
    if not dates:
        return -1
    return bisect_right(dates, d) - 1
