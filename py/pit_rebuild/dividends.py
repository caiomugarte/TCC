"""Dividends per share over the 12 months to d, from B3 or Yahoo Finance (D-4, A-15, A-16, SRC-05, PIT-02).

B3 is the first source for a company: its listed cash-dividend history, every page
fetched, complete, non-empty and reaching back to d - 365 days. Otherwise the ticker's
Yahoo dividends are used, with the reason recorded. One window rule holds for both
sources: d - 365 days < ex_date <= d. A B3 record's ex-date is the first market session
after its lastDatePriorEx (A-16).

The parsers take the bytes of the cached responses. A response that is not valid JSON
raises ValueError; the caller reports that source as failed.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json

from .errors import PITLeakError

WINDOW_DAYS = 365  # A-16: d - 365 days < ex_date <= d
SHORT_HISTORY_REASON = "b3_short_history"


@dataclass(frozen=True)
class B3Company:
    trading_name: str
    code_cvm: str | None
    total_shares: int | None  # totalNumberShares without the "." grouping; unit only (A-18)


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
    )


def parse_b3_cash(pages: Sequence[bytes]) -> B3History:
    """The records of GetListedCashDividends over its pages (SRC-05).

    complete holds when the records equal page 1's totalRecords and pages 1..totalPages are all present.
    """
    records: list[B3Dividend] = []
    seen_pages: set[int] = set()
    total_records = 0
    total_pages = 0
    page_one = False
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
            records.append(
                B3Dividend(
                    type_stock=item["typeStock"].strip(),
                    value=_decimal(item["valueCash"]),
                    last_date_prior_ex=_br_date(item["lastDatePriorEx"]),
                )
            )
    complete = (
        page_one
        and len(records) == total_records
        and all(number in seen_pages for number in range(1, total_pages + 1))
    )
    return B3History(records=records, total_records=total_records, complete=complete)


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
    earliest = min(record.last_date_prior_ex for record in history.records)
    if earliest > d - timedelta(days=WINDOW_DAYS):
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


def b3_ex_dates(records: Iterable[B3Dividend], next_session: Callable[[date], date | None]) -> list[Dividend]:
    """B3 records with ex_date = the first market session after lastDatePriorEx (A-16)."""
    return [Dividend(amount=record.value, ex_date=next_session(record.last_date_prior_ex)) for record in records]


def unit_dividends(
    records: Iterable[B3Dividend],
    composition: dict[str, int],
    next_session: Callable[[date], date | None],
) -> list[Dividend]:
    """Dividends of a unit: its UNT records when present, else each class record times its count in the unit."""
    records = list(records)
    units = class_records(records, "UNT")
    if units:
        return b3_ex_dates(units, next_session)
    dividends: list[Dividend] = []
    for share_class, count in composition.items():
        for record in class_records(records, share_class):
            ex_date = next_session(record.last_date_prior_ex)
            dividends.append(Dividend(amount=record.value * count, ex_date=ex_date))
    return dividends


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
