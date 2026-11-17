"""Tolerance, statistics and category mapping of the FII rebuild against Status Invest (issue #60).

The rules are those of the stock comparison (compare.py): within when the relative difference is
at most 5 %, or at most 0.5 percentage point for the indicators in percent; the rebuilt value is
rounded to 2 decimals first; a reference of 0 is matched only by a rebuilt 0.00. The denominator is
the eligible tickers with a reference value; a ticker in it without a rebuilt value is outside.

GESTAO and SETOR are categories: the CVM value is translated by a fixed table written before the
measurement, and the ticker is within when the translation contains the Status Invest value.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field as dataclass_field
import statistics

from .compare import PP_TOLERANCE, RELATIVE_TOLERANCE_PCT, ROUNDING_DECIMALS, WORST_CASES, p90

PERCENT_FIELDS = frozenset({"DY", "PERCENTUAL EM CAIXA", "CAGR DIVIDENDOS 3 ANOS", "CAGR VALOR COTA 3 ANOS"})

# CVM Segmento_Atuacao -> the Status Invest segments it stands for. A CVM value with no entry, or
# with an empty set, has no counterpart and never matches.
SEGMENT_MAP: dict[str, frozenset[str]] = {
    "Shoppings": frozenset({"Shoppings"}),
    "Logística": frozenset({"Imóveis Industriais e Logísticos", "Logística"}),
    "Escritórios": frozenset({"Lajes Corporativas"}),
    "Lajes Corporativas": frozenset({"Lajes Corporativas"}),
    "Residencial": frozenset({"Imóveis Residenciais"}),
    "Hospital": frozenset({"Hospitalar"}),
    "Hotel": frozenset({"Hotéis"}),
    "Varejo": frozenset({"Varejo"}),
    "Educacional": frozenset({"Educacional"}),
    "Híbrido": frozenset({"Misto"}),
    "Multicategoria": frozenset({"Misto"}),
    "Outros": frozenset(),
}
# CVM Tipo_Gestao -> Status Invest GESTAO.
MANAGEMENT_MAP: dict[str, frozenset[str]] = {
    "Ativa": frozenset({"Ativa"}),
    "Passiva": frozenset({"Passiva"}),
    "Definida": frozenset({"Passiva"}),
}
CATEGORY_MAPS = {"SETOR": SEGMENT_MAP, "GESTAO": MANAGEMENT_MAP}


@dataclass
class FieldStats:
    field: str
    diff_unit: str
    denominator: int = 0
    within: int = 0
    pct_within: float | None = None
    reference_blank: int = 0
    median_diff: float | None = None
    p90_diff: float | None = None
    worst: list[dict] = dataclass_field(default_factory=list)
    missing: list[dict] = dataclass_field(default_factory=list)


@dataclass
class CategoryStats:
    field: str
    denominator: int = 0
    within: int = 0
    pct_within: float | None = None
    reference_blank: int = 0
    crosstab: Counter = dataclass_field(default_factory=Counter)  # (CVM value, Status Invest value) -> tickers
    mismatches: list[dict] = dataclass_field(default_factory=list)
    missing: list[dict] = dataclass_field(default_factory=list)


def is_percent(field: str) -> bool:
    return field in PERCENT_FIELDS


def within(field: str, ref: float | None, calc: float | None) -> bool:
    if ref is None or calc is None:
        return False
    rebuilt = round(calc, ROUNDING_DECIMALS)
    if is_percent(field):
        return round(abs(rebuilt - ref), 9) <= PP_TOLERANCE
    if ref == 0:
        return rebuilt == 0.0
    return round(abs(rebuilt - ref) / abs(ref) * 100, 9) <= RELATIVE_TOLERANCE_PCT


def difference(field: str, ref: float | None, calc: float | None) -> float | None:
    if ref is None or calc is None:
        return None
    rebuilt = round(calc, ROUNDING_DECIMALS)
    if is_percent(field):
        return abs(rebuilt - ref)
    if ref == 0:
        return None
    return abs(rebuilt - ref) / abs(ref) * 100


def summarize(field: str, pairs: Iterable[tuple[str, float | None, float | None, str | None]]) -> FieldStats:
    """pairs hold (ticker, Status Invest value, rebuilt value, reason of a missing rebuilt value).

    worst holds the WORST_CASES cases outside the tolerance: the missing rebuilt values first (by ticker),
    then the largest differences. A case within the tolerance is never a worst case.
    """
    stats = FieldStats(field=field, diff_unit="pp" if is_percent(field) else "relative_pct")
    diffs: list[tuple[float, str, float, float]] = []
    pairs_seen = list(pairs)
    for ticker, ref, calc, reason in pairs_seen:
        if ref is None:
            stats.reference_blank += 1
            continue
        stats.denominator += 1
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
    outside = sorted((item for item in diffs if not within(field, item[2], item[3])), key=lambda item: (-item[0], item[1]))
    missing = sorted(stats.missing, key=lambda item: item["ticker"])
    refs = {ticker: ref for ticker, ref, _, _ in pairs_seen}
    worst = [{"ticker": item["ticker"], "reference": refs[item["ticker"]], "rebuilt": None, "diff": None, "reason": item["reason"]} for item in missing]
    worst += [{"ticker": t, "reference": ref, "rebuilt": calc, "diff": delta} for delta, t, ref, calc in outside]
    stats.worst = worst[:WORST_CASES]
    return stats


def mapped(field: str, cvm_value: str | None) -> frozenset[str]:
    return CATEGORY_MAPS[field].get(cvm_value or "", frozenset())


def summarize_categories(field: str, triples: Iterable[tuple[str, str, str | None, str | None]]) -> CategoryStats:
    """triples hold (ticker, Status Invest value, CVM value, reason of a missing CVM value)."""
    stats = CategoryStats(field=field)
    for ticker, ref, cvm_value, reason in triples:
        if not ref:
            stats.reference_blank += 1
            continue
        stats.denominator += 1
        if cvm_value is None:
            stats.missing.append({"ticker": ticker, "reason": reason or "no_value"})
            continue
        stats.crosstab[(cvm_value, ref)] += 1
        if ref in mapped(field, cvm_value):
            stats.within += 1
        else:
            stats.mismatches.append({"ticker": ticker, "reference": ref, "rebuilt": cvm_value})
    if stats.denominator:
        stats.pct_within = 100 * stats.within / stats.denominator
    stats.mismatches.sort(key=lambda item: item["ticker"])
    return stats
