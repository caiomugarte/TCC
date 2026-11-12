"""Dividends per share over the 12 months to d, from B3 or Yahoo Finance (D-4, A-15, A-16, SRC-05, PIT-02).

B3 is the first source for a company: its listed cash-dividend history, every page
fetched, complete, non-empty and reaching back to d - 365 days. Otherwise the ticker's
Yahoo dividends are used, with the reason recorded. One window rule holds for both
sources: d - 365 days < ex_date <= d. A B3 record's ex-date is the first market session
after its lastDatePriorEx (A-16). B3 amounts are divided by the split, bonus and grouping multipliers
of the company's stock events that fall after the dividend's ex-date and by d (A-20).

The parsers take the bytes of the cached responses. A response that is not valid JSON
raises ValueError; the caller reports that source as failed.
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json

from .errors import PITLeakError

WINDOW_DAYS = 365  # A-16: d - 365 days < ex_date <= d
SHORT_HISTORY_REASON = "b3_short_history"
SPLIT_LABELS = ("DESDOBRAMENTO", "BONIFICACAO", "GRUPAMENTO")  # A-20: the stockDividends labels that change share counts
REST_CAPITAL_RETURN = "REST CAP DIN"  # A-15 (F16): a capital return, not a dividend


@dataclass(frozen=True)
class StockEvent:
    """A split, bonus or grouping of one class ISIN, from the company record's stockDividends (A-20)."""

    isin: str
    label: str
    multiplier: float  # DESDOBRAMENTO and BONIFICACAO: 1 + factor / 100; GRUPAMENTO: factor
    last_date_prior: date


@dataclass(frozen=True)
class B3Company:
    trading_name: str
    code_cvm: str | None
    total_shares: int | None  # totalNumberShares without the "." grouping; unit only (A-18)
    stock_events: tuple[StockEvent, ...] = ()
    retrieved_on: date | None = None  # R of B-3: the day the record was retrieved; set by the CLI, never by parse_b3_company


@dataclass(frozen=True)
class B3Dividend:
    type_stock: str  # ON, PN, PNA, PNB, UNT ...
    value: float  # valueCash with "," as decimal separator
    last_date_prior_ex: date


@dataclass(frozen=True)
class B3History:
    records: list[B3Dividend]
    total_records: int  # page 1 totalRecords; 0 when page 1 is missing
    complete: bool  # every record of every page 1..totalPages was fetched (A-15)
    raw_records: int = 0  # every record received, before the F16 filters
    duplicates: int = 0  # records identical in every field to one already received (A-15)
    capital_returns: int = 0  # REST CAP DIN records, dropped (A-15)


@dataclass(frozen=True)
class Dividend:
    amount: float
    ex_date: date | None  # None: no ex-date in the loaded data, never counts (A-16)


def parse_b3_company(payload: bytes) -> B3Company | None:
    """The first record of GetListedSupplementCompany; None for an empty list."""
    items = json.loads(payload)
    if not items:
        return None
    item = items[0]
    total = item.get("totalNumberShares")
    return B3Company(
        trading_name=item["tradingName"].strip(),
        code_cvm=(item.get("codeCVM") or None),
        total_shares=int(total.replace(".", "").strip()) if total else None,
        stock_events=_stock_events(item.get("stockDividends")),
    )


def _stock_events(entries) -> tuple[StockEvent, ...]:
    """The split, bonus and grouping entries of stockDividends; entries identical in every field count once (A-20)."""
    events: list[StockEvent] = []
    seen: set[str] = set()
    for entry in entries or []:
        key = json.dumps(entry, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        label = entry["label"].strip()
        if label not in SPLIT_LABELS:
            continue
        factor = _decimal(entry["factor"])
        multiplier = factor if label == "GRUPAMENTO" else 1 + factor / 100
        events.append(
            StockEvent(
                isin=(entry.get("isinCode") or "").strip(),
                label=label,
                multiplier=multiplier,
                last_date_prior=_br_date(entry["lastDatePrior"]),
            )
        )
    return tuple(events)


def parse_b3_cash(pages: Sequence[bytes]) -> B3History:
    """The records of GetListedCashDividends over its pages (SRC-05, A-15, F16).

    Every record received counts in raw_records, and complete compares that raw count with page 1's
    totalRecords (with pages 1..totalPages all present). A record identical in every field to one already
    received is a duplicate and is dropped; a REST CAP DIN record is a capital return and is dropped.
    """
    records: list[B3Dividend] = []
    seen_items: set[str] = set()
    seen_pages: set[int] = set()
    total_records = 0
    total_pages = 0
    page_one = False
    raw_records = duplicates = capital_returns = 0
    for payload in pages:
        body = json.loads(payload)
        page = body["page"]
        number = int(page["pageNumber"])
        seen_pages.add(number)
        if number == 1:
            page_one = True
            total_records = int(page["totalRecords"])
            total_pages = int(page["totalPages"])
        for item in body["results"]:
            raw_records += 1
            key = json.dumps(item, sort_keys=True)
            if key in seen_items:
                duplicates += 1
                continue
            seen_items.add(key)
            if (item.get("corporateAction") or "").strip() == REST_CAPITAL_RETURN:
                capital_returns += 1
                continue
            records.append(
                B3Dividend(
                    type_stock=item["typeStock"].strip(),
                    value=_decimal(item["valueCash"]),
                    last_date_prior_ex=_br_date(item["lastDatePriorEx"]),
                )
            )
    complete = (
        page_one
        and raw_records == total_records
        and all(number in seen_pages for number in range(1, total_pages + 1))
    )
    return B3History(
        records=records,
        total_records=total_records,
        complete=complete,
        raw_records=raw_records,
        duplicates=duplicates,
        capital_returns=capital_returns,
    )


def parse_yahoo(payload: bytes) -> list[Dividend]:
    """Dividend events of a Yahoo chart response (events=div); ex-date is the UTC date of the epoch."""
    body = json.loads(payload)
    results = body["chart"]["result"]
    if not results:
        raise ValueError("Yahoo chart response has no result")
    events = (results[0].get("events") or {}).get("dividends") or {}
    dividends = [
        Dividend(amount=float(event["amount"]), ex_date=datetime.fromtimestamp(int(event["date"]), tz=timezone.utc).date())
        for event in events.values()
    ]
    return sorted(dividends, key=lambda item: item.ex_date)


def choose_source(history: B3History | None, d: date) -> tuple[str, str | None]:
    """("b3", None) or ("yahoo", reason) for one company at d (A-15)."""
    if history is None:
        return "yahoo", "b3_failed"
    if history.complete and history.total_records == 0:
        return "yahoo", "b3_empty"
    if not history.complete:
        return "yahoo", "b3_incomplete"
    earliest = min((record.last_date_prior_ex for record in history.records), default=None)
    if earliest is None or earliest > d - timedelta(days=WINDOW_DAYS):  # no record left after the F16 filters: none reaches d
        return "yahoo", SHORT_HISTORY_REASON
    return "b3", None


def resolve(
    choice: tuple[str, str | None],
    b3_dividends: list[Dividend] | None,
    yahoo: list[Dividend] | None,
) -> tuple[str, str | None, list[Dividend] | None]:
    """The source, reason and dividends a ticker uses at d after the Yahoo fallback (A-15, SRC-05).

    A complete empty B3 history with Yahoo failed is a confirmed non-payer: 0 dividends from B3.
    """
    source, reason = choice
    if source == "b3":
        return "b3", None, b3_dividends
    if yahoo is not None:
        return "yahoo", reason, yahoo
    if reason == "b3_empty":
        return "b3", "b3_empty_yahoo_failed", []
    return "none", "no_dividend_source", None


def class_records(records: Iterable[B3Dividend], share_class: str) -> list[B3Dividend]:
    """The records whose typeStock is the ticker's share class (ON, PN, PNA, UNT ...)."""
    return [record for record in records if record.type_stock == share_class]


def b3_ex_dates(
    records: Iterable[B3Dividend],
    next_session: Callable[[date], date | None],
    *,
    events: Sequence[StockEvent] = (),
    isin: str | None = None,
    d: date | None = None,
) -> list[Dividend]:
    """B3 records with ex_date = the first market session after lastDatePriorEx (A-16), each amount divided by split_divisor (A-20)."""
    dividends: list[Dividend] = []
    for record in records:
        ex_date = next_session(record.last_date_prior_ex)
        divisor = split_divisor(events, isin, ex_date, d, next_session)
        dividends.append(Dividend(amount=record.value / divisor, ex_date=ex_date))
    return dividends


def unit_dividends(
    records: Iterable[B3Dividend],
    composition: dict[str, int],
    next_session: Callable[[date], date | None],
    *,
    events: Sequence[StockEvent] = (),
    isins: Mapping[str, str] | None = None,
    d: date | None = None,
) -> list[Dividend]:
    """Dividends of a unit: its UNT records when present, else each class record times its count in the unit.

    isins holds the ISIN of "UNT" and of each class; each record is adjusted by the events of its own ISIN (A-20).
    """
    records = list(records)
    isins = isins or {}
    units = class_records(records, "UNT")
    if units:
        return b3_ex_dates(units, next_session, events=events, isin=isins.get("UNT"), d=d)
    dividends: list[Dividend] = []
    for share_class, count in composition.items():
        for record in class_records(records, share_class):
            ex_date = next_session(record.last_date_prior_ex)
            divisor = split_divisor(events, isins.get(share_class), ex_date, d, next_session)
            dividends.append(Dividend(amount=record.value * count / divisor, ex_date=ex_date))
    return dividends


def split_divisor(
    events: Iterable[StockEvent],
    isin: str | None,
    ex_date: date | None,
    d: date | None,
    next_session: Callable[[date], date | None],
) -> float:
    """The product of the multipliers of the ISIN's events with ex-date after ex_date and by d (A-20).

    An event's ex-date is the first market session after its last_date_prior; an event with none does not apply.
    The divisor is 1.0 when isin, ex_date or d is None, or when no event applies.
    """
    if isin is None or ex_date is None or d is None:
        return 1.0
    divisor = 1.0
    for event in events:
        if event.isin != isin:
            continue
        event_ex = next_session(event.last_date_prior)
        if event_ex is not None and ex_date < event_ex <= d:
            divisor *= event.multiplier
    return divisor


def event_ex_date(event: StockEvent, next_session: Callable[[date], date | None]) -> date:
    """The ex-date of a stock event: the first market session after its last_date_prior (B-1, A-20).

    When the calendar holds no such session (before its first or after its last session), it is the calendar day after last_date_prior.
    """
    session = next_session(event.last_date_prior)
    return session if session is not None else event.last_date_prior + timedelta(days=1)


def class_events(
    events: Iterable[StockEvent],
    isins: Collection[str],
    after: date,
    until: date | None,
    next_session: Callable[[date], date | None],
) -> list[tuple[StockEvent, date]]:
    """The events of the ISINs in isins with after < ex-date <= until, in ex-date order; until None has no upper bound (SHR-01, SHR-02, B-2).

    An event that B3 lists under two of the ISINs counts once, under the first of them in events: the key is its label,
    multiplier and lastDatePrior.
    """
    wanted = set(isins)
    seen: set[tuple[str, float, date]] = set()
    found: list[tuple[StockEvent, date]] = []
    for event in events:
        key = (event.label, event.multiplier, event.last_date_prior)
        if event.isin not in wanted or key in seen:
            continue
        seen.add(key)
        ex = event_ex_date(event, next_session)
        if ex <= after or (until is not None and ex > until):
            continue
        found.append((event, ex))
    return sorted(found, key=lambda item: item[1])


def b3_earliest_raw(records: Iterable[B3Dividend]) -> date | None:
    """The earliest raw date B3 gives: the day after the earliest lastDatePriorEx (F3, COV-02); None without records."""
    dates = [record.last_date_prior_ex for record in records]
    return min(dates) + timedelta(days=1) if dates else None


def in_window(ex_date: date | None, d: date) -> bool:
    """The only window rule of the tool: d - 365 days < ex_date <= d. No ex-date never counts (A-16)."""
    if ex_date is None:
        return False
    return d - timedelta(days=WINDOW_DAYS) < ex_date <= d


def sum_dividends(dividends: Iterable[Dividend], d: date, ticker: str = "") -> float:
    """Sum the amounts given; a dividend with an ex-date after d is a PIT leak (PIT-02, exit 4)."""
    total = 0.0
    for dividend in dividends:
        if dividend.ex_date is None:
            continue
        if dividend.ex_date > d:
            raise PITLeakError(ticker, d, dividend.ex_date)
        total += dividend.amount
    return total


def dps_12m(dividends: Iterable[Dividend], d: date, ticker: str = "") -> float:
    """Dividends per share over the 12 months to d: the window filter, then the summed records (DY)."""
    return sum_dividends([item for item in dividends if in_window(item.ex_date, d)], d, ticker)


def _decimal(text: str) -> float:
    text = text.strip()
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def _br_date(text: str) -> date:
    return datetime.strptime(text.strip(), "%d/%m/%Y").date()
