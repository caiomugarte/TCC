"""B3 COTAHIST daily files: quotes, liquidity, ISIN, share class and sessions as of d (PIT-01, IND-01, A-7, A-15, A-16, A-17).

COTAHIST is fixed width. Columns are 1-based and inclusive: TIPREG 1-2, DATA 3-10,
CODBDI 11-12, CODNEG 13-24, TPMERC 25-27, ESPECI 40-49, PREULT 109-121 (two implied
decimals), VOLTOT 171-188 (two implied decimals), FATCOT 211-217, CODISI 231-242.

Quote records of the cash market are kept (TIPREG 01, TPMERC 010, CODBDI 02, 07 and 08; A-7)
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
LIQUIDITY_SESSIONS = 21  # A-7 (F5): LIQUIDEZ MEDIA DIARIA averages the last 21 sessions
QUOTE_CODBDI = ("02", "07", "08")  # A-7 (F1): lote padrão, recuperação extrajudicial, recuperação judicial
FII_CODBDI = ("12",)  # fundos imobiliários (issue #63)
LOT_CODBDI = "02"  # on the same session a lot line replaces a recovery line, whatever the line order


@dataclass(frozen=True)
class Session:
    """One trading record of a ticker: close (PREULT / FATCOT), VOLTOT, ISIN, the first ESPECI token and CODBDI."""

    day: date
    close: float
    voltot: float
    isin: str
    especi: str
    codbdi: str


@dataclass
class Quotes:
    """Records of the requested tickers, and the session calendar of every ticker in the files."""

    sessions: list[date] = field(default_factory=list)
    rows: dict[str, list[Session]] = field(default_factory=dict)
    dates: dict[str, list[date]] = field(default_factory=dict)


def load(zip_paths: Iterable[Path], tickers: set[str], codbdi: Iterable[str] = QUOTE_CODBDI) -> Quotes:
    """Read the COTAHIST zips, keeping the lot records of the tickers and every session date.

    codbdi names the cash-market CODBDI codes kept; the default is the stock lot and recovery codes (A-7).
    """
    kept_codbdi = frozenset(codbdi)
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
                        if record.tpmerc != "010" or record.codbdi not in kept_codbdi or record.ticker not in tickers:
                            continue
                        close = record.price / record.fatcot if record.fatcot > 1 else record.price
                        session = Session(
                            day=record.day, close=close, voltot=record.voltot, isin=record.isin,
                            especi=record.especi, codbdi=record.codbdi,
                        )
                        by_session = by_ticker.setdefault(record.ticker, {})
                        kept = by_session.get(record.day)
                        if kept is not None and kept.codbdi == LOT_CODBDI and session.codbdi != LOT_CODBDI:
                            continue  # a recovery line never replaces the lot line of its session (A-7)
                        by_session[record.day] = session
    quotes = Quotes(sessions=sorted(days))
    for ticker, by_day in by_ticker.items():
        ordered = [by_day[day] for day in sorted(by_day)]
        quotes.rows[ticker] = ordered
        quotes.dates[ticker] = [row.day for row in ordered]
    return quotes


def quote_at(q: Quotes, ticker: str, d: date, max_age_days: int | None = MAX_QUOTE_AGE_DAYS) -> tuple[date, float] | None:
    """(session date, close) of the last session on or before d; None when it is more than max_age_days old (A-7).

    max_age_days=None sets no age limit.
    """
    index = _last_index(q, ticker, d)
    if index < 0:
        return None
    row = q.rows[ticker][index]
    if max_age_days is not None and (d - row.day).days > max_age_days:
        return None
    return row.day, row.close


def last_trade(q: Quotes, ticker: str, d: date) -> tuple[date, float] | None:
    """(session date, close) of the ticker's last session on or before d, at any age (F14)."""
    return quote_at(q, ticker, d, max_age_days=None)


def reference_stale(q: Quotes, ticker: str, d: date) -> bool:
    """True when the ticker has no session in the MAX_QUOTE_AGE_DAYS before d, d included (D-A)."""
    return quote_at(q, ticker, d) is None


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


def class_prices(q: Quotes, tickers: Iterable[str], d: date) -> dict[str, tuple[str, float, date] | None]:
    """(ticker, close, session day) of each share class of a company: ON from the ON ticker, PREF from a preferred one (A-17).

    Units (UNT) are skipped. A class is priced from the ticker with the highest average volume among those with a
    session within MAX_QUOTE_AGE_DAYS before d (ties: the smaller ticker). When none of its tickers has one, the
    ticker with the latest session on or before d prices it at any age (ties: higher average volume, then the
    smaller ticker); the caller reports that such a price is stale (F14).
    """
    fresh: dict[str, tuple[tuple, str, float, date]] = {}
    latest: dict[str, tuple[tuple, str, float, date]] = {}
    for ticker in sorted(set(tickers)):
        cls = _price_class(q, ticker)
        if cls is None:
            continue
        last = last_trade(q, ticker, d)
        volume = avg_volume(q, ticker, d)
        if last is None or volume is None:
            continue
        session_day, close = last
        by_volume = ((-volume, ticker), ticker, close, session_day)
        if quote_at(q, ticker, d) is not None and (cls not in fresh or by_volume[0] < fresh[cls][0]):
            fresh[cls] = by_volume
        by_recency = ((-session_day.toordinal(), -volume, ticker), ticker, close, session_day)
        if cls not in latest or by_recency[0] < latest[cls][0]:
            latest[cls] = by_recency
    result: dict[str, tuple[str, float, date] | None] = {}
    for cls in ("ON", "PREF"):
        chosen = fresh.get(cls) or latest.get(cls)
        result[cls] = None if chosen is None else (chosen[1], chosen[2], chosen[3])
    return result


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
