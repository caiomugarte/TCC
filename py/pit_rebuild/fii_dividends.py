"""FII distributions: the B3 fund page first, Yahoo only as a marked fallback (issue #60).

The B3 page (fundsProxy GetListedSupplementFunds) lists the cash distributions of the fund. A
record counts when its label is RENDIMENTO and its ISIN is the quota ISIN of the ticker at d;
the ex-date is the session after lastDatePrior. The page keeps a short history, so a window it
does not reach goes to Yahoo, and the source and reason are reported. The CVM monthly report has
no per-quota distribution: its Percentual_Dividend_Yield_Mes is not used (see the report).
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json

from .dividends import WINDOW_DAYS, Dividend

INCOME_LABEL = "RENDIMENTO"
CAGR_YEARS = 3


@dataclass(frozen=True)
class B3Payment:
    isin: str
    last_date_prior: date
    rate: float
    label: str


@dataclass(frozen=True)
class B3Fund:
    name: str
    quantity: int | None
    payments: tuple[B3Payment, ...]
    labels: tuple[str, ...]


def parse_b3_fund(payload: bytes) -> B3Fund:
    body = json.loads(payload)
    payments = []
    for entry in body.get("cashDividends") or []:
        try:
            payments.append(B3Payment(
                isin=(entry.get("isinCode") or "").strip(),
                last_date_prior=datetime.strptime(entry["lastDatePrior"].strip(), "%d/%m/%Y").date(),
                rate=_decimal(entry["rate"]),
                label=(entry.get("label") or "").strip().upper(),
            ))
        except (KeyError, ValueError, AttributeError):
            continue  # a record without a date or a rate is not a distribution the window can place
    quantity_text = (body.get("quantity") or "").replace(".", "").strip()
    return B3Fund(
        name=(body.get("fund") or "").strip(),
        quantity=int(quantity_text) if quantity_text.isdigit() else None,
        payments=tuple(payments),
        labels=tuple(sorted({item.label for item in payments})),
    )


def b3_dividends(fund: B3Fund, isin: str, next_session: Callable[[date], date | None]) -> list[Dividend]:
    """The RENDIMENTO records of the quota ISIN, ex-date on the session after lastDatePrior, sorted."""
    found = []
    for payment in fund.payments:
        if payment.label != INCOME_LABEL or payment.isin != isin:
            continue
        ex_date = next_session(payment.last_date_prior) or payment.last_date_prior + timedelta(days=1)
        found.append(Dividend(amount=payment.rate, ex_date=ex_date))
    return sorted(found, key=lambda item: item.ex_date)


def choose(
    b3: Sequence[Dividend] | None, yahoo: Sequence[Dividend] | None, start: date
) -> tuple[str, str | None, Sequence[Dividend] | None]:
    """(source, reason, dividends) for a window that opens after `start`.

    B3 is the source when its earliest ex-date is on or before start. Otherwise Yahoo, with the
    reason B3 was not used; with Yahoo failed too, no source.
    """
    if b3 is None:
        reason = "b3_failed"
    elif not b3:
        reason = "b3_empty"
    elif min(item.ex_date for item in b3) <= start:
        return "b3", None, b3
    else:
        reason = "b3_short_history"
    if yahoo is None:
        return "none", f"{reason}_yahoo_failed", None
    return "yahoo", reason, yahoo


def window_start(d: date, years_back: int = 0) -> date:
    """The exclusive start of the 12 months that end `years_back` years before d."""
    return _years_before(d, years_back) - timedelta(days=WINDOW_DAYS)


def sum_12m(dividends: Iterable[Dividend], d: date) -> float:
    """Sum of the amounts with d - 365 days < ex-date <= d."""
    start = d - timedelta(days=WINDOW_DAYS)
    return sum(item.amount for item in dividends if item.ex_date is not None and start < item.ex_date <= d)


def last_dividend(dividends: Iterable[Dividend], d: date) -> float | None:
    """The amount of the newest record with ex-date on or before d."""
    known = [item for item in dividends if item.ex_date is not None and item.ex_date <= d]
    return max(known, key=lambda item: item.ex_date).amount if known else None


def cagr_3y(dividends: Sequence[Dividend], d: date) -> float | None:
    """((12 months to d) / (12 months to d - 3 years)) ** (1/3) - 1, in percent.

    None when the history does not reach the start of the old window or the old sum is zero.
    """
    dated = [item for item in dividends if item.ex_date is not None]
    if not dated or min(item.ex_date for item in dated) > window_start(d, CAGR_YEARS):
        return None
    old = sum_12m(dated, _years_before(d, CAGR_YEARS))
    if old <= 0:
        return None
    return ((sum_12m(dated, d) / old) ** (1 / CAGR_YEARS) - 1) * 100


def _years_before(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # 29 February
        return d.replace(year=d.year - years, day=28)


def _decimal(text: str) -> float:
    text = text.strip()
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)
