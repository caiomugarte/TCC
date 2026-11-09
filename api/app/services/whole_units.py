"""Whole-unit buy list for one asset class (whole-units-v1).

Pure integer-cents rule: floor each ticker's target by its price, then spend
the class's remaining cash one unit at a time on the ticker furthest below its
own target among those whose unit still fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

FII_MINIMUM_HOLDINGS = 3
REASON_DID_NOT_FIT = "não coube no capital"
REASON_FII_MINIMUM = "mínimo de 3 FIIs"


@dataclass(frozen=True)
class UnitLine:
    ticker: str
    price_cents: int
    target_cents: int
    quantity: int
    forced: bool


@dataclass(frozen=True)
class WholeUnitAllocation:
    lines: tuple[UnitLine, ...]
    spent_cents: int
    cash_residual_cents: int
    over_target_cents: int


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(
    tickers: Sequence[str],
    target_cents: Mapping[str, int],
    price_cents: Mapping[str, int],
    class_target_cents: int,
    minimum_count: int,
) -> None:
    if not tickers:
        raise ValueError("tickers must not be empty")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique")
    for ticker in tickers:
        if ticker not in target_cents:
            raise ValueError(f"{ticker}: no target")
        if ticker not in price_cents:
            raise ValueError(f"{ticker}: no price")
        price = price_cents[ticker]
        if not _is_int(price) or price <= 0:
            raise ValueError(f"{ticker}: price must be a positive int of cents")
        target = target_cents[ticker]
        if not _is_int(target) or target < 0:
            raise ValueError(f"{ticker}: target must be a non-negative int of cents")
    if not _is_int(class_target_cents) or class_target_cents < 0:
        raise ValueError("class target must be a non-negative int of cents")
    if sum(target_cents[ticker] for ticker in tickers) != class_target_cents:
        raise ValueError("ticker targets must add up to the class target")
    if not _is_int(minimum_count) or minimum_count < 0:
        raise ValueError("minimum_count must be a non-negative int")


def allocate_whole_units(
    tickers: Sequence[str],
    target_cents: Mapping[str, int],
    price_cents: Mapping[str, int],
    class_target_cents: int,
    *,
    minimum_count: int = 0,
) -> WholeUnitAllocation:
    _validate(tickers, target_cents, price_cents, class_target_cents, minimum_count)
    order = list(tickers)
    quantity = {ticker: target_cents[ticker] // price_cents[ticker] for ticker in order}
    remaining = class_target_cents - sum(quantity[t] * price_cents[t] for t in order)
    while True:
        fits = [t for t in order if price_cents[t] <= remaining]
        if not fits:
            break
        chosen = max(
            fits,
            key=lambda t: (target_cents[t] - quantity[t] * price_cents[t], -order.index(t)),
        )
        quantity[chosen] += 1
        remaining -= price_cents[chosen]

    # The FII minimum buys one unit of the best-placed tickers left at zero,
    # even above the class target; no remainder loop runs after it.
    forced: set[str] = set()
    missing = min(minimum_count, len(order)) - sum(1 for t in order if quantity[t] > 0)
    for ticker in order:
        if missing <= 0:
            break
        if quantity[ticker] == 0:
            quantity[ticker] = 1
            forced.add(ticker)
            missing -= 1

    lines = tuple(
        UnitLine(t, price_cents[t], target_cents[t], quantity[t], t in forced) for t in order
    )
    spent = sum(line.quantity * line.price_cents for line in lines)
    return WholeUnitAllocation(
        lines=lines,
        spent_cents=spent,
        cash_residual_cents=max(0, class_target_cents - spent),
        over_target_cents=max(0, spent - class_target_cents),
    )
