"""Share counts at d: the CVM count moved by the B3 events of its classes (AD-033, SHR-01..SHR-08).

A class count at d is the filed count times the A-18 factor f times the events of the class ISINs with ex-date in
(DT_REFER, d] (M_d, B-2). f is chosen on the filed count times the events of every ISIN the class had, with ex-date
in (DT_REFER, R], R the B3 retrieval date (M_all, B-3). Pure: the caller reads the filing, the ISIN sets, the events
and the B3 total, and gets back the numbers and the lists that the report prints.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math

from . import cvm
from .dividends import StockEvent, class_events

EVENT_EPSILON = 1e-9  # SHR-03: a class whose event product is within this distance of 1 is not listed as adjusted


@dataclass(frozen=True)
class Adjusted:
    """The counts of one company at d, the A-18 factor and the events that moved them (SHR-01..SHR-04, SHR-08)."""

    n_on: int
    n_pref: int
    factor: float  # f: the A-18 power of ten
    verified: bool  # the B3 total confirmed f
    residual: float | None  # the event-adjusted total times f over the B3 total; None without a B3 total
    scaled_total: float  # the filed total times f, events left out
    applied: list[dict]  # the classes whose events changed the count: {class, isins, factor, events}
    no_isin: list[str]  # the classes with shares and no ISIN at d


def adjust(
    filed: cvm.Shares,
    isins_d: Mapping[str, Collection[str]],
    isins_all: Mapping[str, Collection[str]],
    events: Sequence[StockEvent],
    b3_total: int | None,
    retrieved_on: date | None,
    d: date,
    next_session: Callable[[date], date | None],
) -> Adjusted:
    """Adjust one company's filed counts to d.

    isins_d[cls] are the ISINs of the class at d (I_d, B-2); isins_all[cls] also holds every ISIN the class had after d
    (I_all, B-3). retrieved_on None leaves the window of the scale without an upper bound.
    """
    counts = {"ON": filed.on, "PREF": filed.pref}
    m_all = {
        cls: _product(class_events(events, isins_all.get(cls, ()), filed.dt_refer, retrieved_on, next_session))
        for cls in counts
    }
    scale_total = round(sum(count * m_all[cls] for cls, count in counts.items()))
    factor, verified = cvm.share_scale(scale_total, b3_total) if scale_total > 0 else (1.0, False)
    residual = None if b3_total is None or b3_total <= 0 else scale_total * factor / b3_total

    applied: list[dict] = []
    no_isin: list[str] = []
    m_d: dict[str, float] = {}
    for cls, count in counts.items():
        isins = tuple(isins_d.get(cls, ()))
        if count > 0 and not isins:
            no_isin.append(cls)
            m_d[cls] = 1.0
            continue
        pairs = class_events(events, isins, filed.dt_refer, d, next_session)
        m_d[cls] = _product(pairs)
        if abs(m_d[cls] - 1.0) > EVENT_EPSILON:
            applied.append({
                "class": cls,
                "isins": sorted(isins),
                "factor": m_d[cls],
                "events": [
                    {"isin": event.isin, "label": event.label, "ex_date": ex, "multiplier": event.multiplier}
                    for event, ex in pairs
                ],
            })
    return Adjusted(
        n_on=round(filed.on * factor * m_d["ON"]),
        n_pref=round(filed.pref * factor * m_d["PREF"]),
        factor=factor,
        verified=verified,
        residual=residual,
        scaled_total=(filed.on + filed.pref) * factor,
        applied=applied,
        no_isin=no_isin,
    )


def _product(pairs: Sequence[tuple[StockEvent, date]]) -> float:
    return math.prod((event.multiplier for event, _ in pairs), start=1.0)
