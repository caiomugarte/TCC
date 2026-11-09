"""Tolerance, differences and per-indicator statistics of the rebuild against Status Invest (CMP-01, CMP-02, A-2, A-3, A-4, A-12).

Tolerance (D-1, A-2): a relative indicator is within when |calc - ref| / |ref| <= 5 %; a percent
indicator (PERCENT_FIELDS) is within when |calc - ref| <= 0.5 p.p. The rebuilt value is rounded to
2 decimals before the comparison. With reference 0 a relative indicator is within only when the
rounded rebuilt value is 0.00 (A-3).

Denominator (A-4): an eligible ticker whose Status Invest value is blank is out of the denominator
and counted in reference_blank, except DY: a blank DY is the reference 0.00, stays in the
denominator and is counted in reference_blank_as_zero. A ticker in the denominator with no rebuilt
value is outside tolerance and is listed with its reason.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field as dataclass_field
import math
import statistics

from .fields import tolerance_kind

RELATIVE_TOLERANCE_PCT = 5.0  # D-1: relative difference <= 5 %
PP_TOLERANCE = 0.5  # A-2: percentage points for the percent indicators
ROUNDING_DECIMALS = 2  # A-3: the CSV precision
WORST_CASES = 10  # CMP-02: the 10 worst cases per indicator and date
BLANK_DY_REFERENCE = 0.0  # A-4: a blank DY in Status Invest is 0.00 (production fills it, preprocessing.py:280-285)


@dataclass
class FieldStats:
    """The CMP-02 statistics of one indicator over one group of tickers."""

    field: str
    diff_unit: str  # "relative_pct" or "pp"
    denominator: int = 0
    within: int = 0
    pct_within: float | None = None
    reference_blank: int = 0
    reference_blank_with_value: int = 0
    reference_blank_as_zero: int = 0
    median_diff: float | None = None
    p90_diff: float | None = None
    worst: list[dict] = dataclass_field(default_factory=list)
    missing: list[dict] = dataclass_field(default_factory=list)


def reference(field: str, raw: float | None) -> tuple[float | None, bool]:
    """The reference value of a Status Invest cell and whether it is a blank DY read as zero (A-4)."""
    if raw is not None:
        return raw, False
    if field == "DY":
        return BLANK_DY_REFERENCE, True
    return None, False


def within(field: str, ref: float | None, calc: float | None) -> bool:
    """Whether the rebuilt value is within the tolerance of the reference (CMP-01, D-1, A-2, A-3)."""
    if ref is None or calc is None:
        return False
    rebuilt = round(calc, ROUNDING_DECIMALS)
    if tolerance_kind(field) == "pp":
        return round(abs(rebuilt - ref), 9) <= PP_TOLERANCE
    if ref == 0:
        return rebuilt == 0.0
    return round(abs(rebuilt - ref) / abs(ref) * 100, 9) <= RELATIVE_TOLERANCE_PCT


def difference(field: str, ref: float | None, calc: float | None) -> float | None:
    """|calc - ref| in percent of |ref| for relative indicators, in points for percent ones; None for a zero reference (A-12)."""
    if ref is None or calc is None:
        return None
    rebuilt = round(calc, ROUNDING_DECIMALS)
    if tolerance_kind(field) == "pp":
        return abs(rebuilt - ref)
    if ref == 0:
        return None
    return abs(rebuilt - ref) / abs(ref) * 100


def p90(values: Iterable[float]) -> float | None:
    """The nearest-rank 90th percentile: the ceil(0.9 n)-th smallest value (A-12)."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (9 * len(ordered) + 9) // 10  # ceil(0.9 n) in integers, 1-based
    return ordered[rank - 1]


def summarize(field: str, pairs: Iterable[tuple[str, float | None, float | None, str | None]]) -> FieldStats:
    """CMP-02 statistics of one indicator. pairs hold (ticker, raw Status Invest value, rebuilt value, reason of a missing rebuilt value)."""
    stats = FieldStats(field=field, diff_unit="pp" if tolerance_kind(field) == "pp" else "relative_pct")
    diffs: list[tuple[float, str, float, float]] = []
    for ticker, raw, calc, reason in pairs:
        ref, is_blank_dy = reference(field, raw)
        if raw is None and not is_blank_dy:
            stats.reference_blank += 1
            if calc is not None:
                stats.reference_blank_with_value += 1
            continue
        stats.denominator += 1
        if is_blank_dy:
            stats.reference_blank_as_zero += 1
        if calc is None:
            stats.missing.append({"ticker": ticker, "reason": reason or "no_value"})
            continue
        if within(field, ref, calc):
            stats.within += 1
        delta = difference(field, ref, calc)
        if delta is not None:
            diffs.append((delta, ticker, ref, round(calc, ROUNDING_DECIMALS)))
    if stats.denominator:
        stats.pct_within = 100 * stats.within / stats.denominator
    values = [item[0] for item in diffs]
    if values:
        stats.median_diff = statistics.median(values)
        stats.p90_diff = p90(values)
    worst = sorted(diffs, key=lambda item: (-item[0], item[1]))[:WORST_CASES]
    stats.worst = [{"ticker": ticker, "reference": ref, "rebuilt": calc, "diff": delta} for delta, ticker, ref, calc in worst]
    return stats
