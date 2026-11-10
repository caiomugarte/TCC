"""Earliest years per source and field, DY coverage and the effort of the full rebuild (COV-01, COV-02, COV-03, A-13).

The earliest year of a field is the latest earliest year among its sources plus its lookback
(COV-02): 5 fiscal years for the two CAGR fields, 2 for PEG RATIO (two trailing-12-month incomes a
year apart), 1 for the other trailing-12-month flows, 0 for balance-sheet-only and price-only
fields. DY takes its earliest year from the dividend source that was actually used.

The effort (A-13) projects the load by bytes: the seconds per byte measured on the run, times the
full-rebuild bytes of each yearly-file kind. The compute is measured per decision date and
multiplied by the 180 monthly dates of 2012-01 to 2026-12. The load is never multiplied by 180.
"""

from collections.abc import Callable, Mapping
from datetime import date
import re
import statistics

from .sources import FCA_FIRST_YEAR

PROBE_FIRST_YEAR = 2005  # COV-01: COTAHIST probe runs from 2005 to 2012
PROBE_LAST_YEAR = 2012
FIRST_DECISION_YEAR = 2012  # the full rebuild covers the decision months 2012-01 .. 2026-12
LAST_DECISION_YEAR = 2026
DFP_YEARS_BACK = 7  # SRC-01: DFP from year(d) - 7
ITR_YEARS_BACK = 2  # SRC-01: ITR from year(d) - 2
COTAHIST_YEARS_BACK = 1  # SRC-01: COTAHIST from year(d) - 1
DECISION_DATES = (LAST_DECISION_YEAR - FIRST_DECISION_YEAR + 1) * 12  # 180
LOOKBACK = {"CAGR": 5, "PEG": 2, "TTM": 1, "BALANCE_PRICE": 0}
RETENTION = {
    "b3": "histórico completo da empresa (GetListedCashDividends, todas as páginas)",
    "yahoo": "a partir de 2000-01-01 (period1 da requisição)",
}
# Field -> (sources whose earliest year it needs, lookback kind). DY is read from the dividend source.
FIELD_RULES: dict[str, tuple[tuple[str, ...], str]] = {
    "PRECO": (("COTAHIST",), "BALANCE_PRICE"),
    "LIQUIDEZ MEDIA DIARIA": (("COTAHIST",), "BALANCE_PRICE"),
    "VALOR DE MERCADO": (("COTAHIST", "DFP", "ITR"), "BALANCE_PRICE"),
    "P/L": (("COTAHIST", "DFP", "ITR"), "TTM"),
    "P/VP": (("COTAHIST", "DFP", "ITR"), "BALANCE_PRICE"),
    "PSR": (("COTAHIST", "DFP", "ITR"), "TTM"),
    "EV/EBIT": (("COTAHIST", "DFP", "ITR"), "TTM"),
    "DY": (("DIVIDENDS",), "BALANCE_PRICE"),
    "ROE": (("DFP", "ITR"), "TTM"),
    "ROA": (("DFP", "ITR"), "TTM"),
    "ROIC": (("DFP", "ITR"), "TTM"),
    "MARG. LIQUIDA": (("DFP", "ITR"), "TTM"),
    "MARGEM EBIT": (("DFP", "ITR"), "TTM"),
    "LIQ. CORRENTE": (("DFP", "ITR"), "BALANCE_PRICE"),
    "DIVIDA LIQUIDA / EBIT": (("DFP", "ITR"), "TTM"),
    "DIV. LIQ. / PATRI.": (("DFP", "ITR"), "BALANCE_PRICE"),
    "CAGR RECEITAS 5 ANOS": (("DFP",), "CAGR"),
    "CAGR LUCROS 5 ANOS": (("DFP",), "CAGR"),
    "PEG RATIO": (("COTAHIST", "DFP", "ITR"), "PEG"),
}
_LISTING = re.compile(r'href="(dfp|itr|fca)_cia_aberta_(\d{4})\.zip"', re.IGNORECASE)


def earliest_years(
    listings: Mapping[str, bytes],
    cotahist_probe: Mapping[int, bool],
) -> dict[str, dict[str, int | bool | None]]:
    """The earliest year of each CVM source from its directory listing, and of COTAHIST from the probe (COV-01).

    A listing holds the yearly zips of one source (nginx autoindex). The probe year present at 2005 is
    the probe floor: COTAHIST is then reported as "<= 2005" and never with an earlier year.
    """
    result: dict[str, dict[str, int | bool | None]] = {}
    for source, html in listings.items():
        years = [
            int(match.group(2))
            for match in _LISTING.finditer(html.decode("utf-8", "replace"))
            if match.group(1).upper() == source
        ]
        result[source] = {"year": min(years) if years else None, "at_probe_floor": False}
    present = [year for year in sorted(cotahist_probe) if cotahist_probe[year] and PROBE_FIRST_YEAR <= year <= PROBE_LAST_YEAR]
    if present:
        result["COTAHIST"] = {"year": present[0], "at_probe_floor": present[0] == PROBE_FIRST_YEAR}
    else:
        result["COTAHIST"] = {"year": None, "at_probe_floor": False}
    return result


def field_earliest(field: str, earliest: Mapping[str, int | None]) -> int | None:
    """The earliest year a field can be rebuilt: the latest source year plus its lookback (COV-02). None if a source year is unknown."""
    sources, kind = FIELD_RULES[field]
    years = [earliest.get(source) for source in sources]
    if any(year is None for year in years):
        return None
    return max(years) + LOOKBACK[kind]


def dy_coverage(by_source: Mapping[str, Mapping[str, date | None]]) -> dict[str, object]:
    """DY's coverage per dividend source from each ticker's earliest raw date (COV-02, F3).

    by_source maps a source to {ticker: earliest_raw_date}; a ticker counts whether or not it has a date.
    The dates are reported as given: no session calendar and no d cut them. The median is over the tickers'
    years. The DY year comes from B3 when any ticker used B3, else from Yahoo.
    """
    sources: dict[str, dict[str, object]] = {}
    for source in ("b3", "yahoo"):
        dates = dict(by_source.get(source, {}))
        known = sorted(day for day in dates.values() if day is not None)
        earliest = known[0] if known else None
        years = [day.year for day in known]
        sources[source] = {
            "tickers": len(dates),
            "earliest_raw_date": earliest.isoformat() if earliest else None,
            "earliest_year": earliest.year if earliest else None,
            "median_ticker_year": statistics.median(years) if years else None,
            "retention": RETENTION[source],
        }
    used = next((source for source in ("b3", "yahoo") if sources[source]["tickers"] > 0), None)
    return {
        "by_source": sources,
        "earliest_year_from": used,
        "earliest_year": None if used is None else sources[used]["earliest_year"],
    }


def full_rebuild_years(earliest: Mapping[str, int | None]) -> dict[str, tuple[int, int]]:
    """The years each source is read for the full rebuild: SRC-01's rule, clipped to each source's earliest year (COV-03)."""
    first = {
        "DFP": FIRST_DECISION_YEAR - DFP_YEARS_BACK,
        "ITR": FIRST_DECISION_YEAR - ITR_YEARS_BACK,
        "FCA": FCA_FIRST_YEAR,
        "COTAHIST": FIRST_DECISION_YEAR - COTAHIST_YEARS_BACK,
    }
    return {
        source: (max(start, earliest.get(source) or start), LAST_DECISION_YEAR)
        for source, start in first.items()
    }


def effort(
    load: Mapping[str, tuple[float, int]],
    compute_seconds: float,
    n_dates: int,
    sizes: Mapping[str, int],
    kind_of_url: Callable[[str], str],
) -> dict[str, object]:
    """The projected time of the full rebuild (A-13).

    load holds (seconds, bytes) measured per yearly-file kind. sizes holds the Content-Length of the
    full-rebuild URLs. A kind measured with 0 bytes has no projection, and the total is then None.
    """
    if n_dates < 1:
        raise ValueError("effort needs at least one measured decision date")
    full_bytes: dict[str, int] = {kind: 0 for kind in load}
    for url, size in sizes.items():
        kind = kind_of_url(url)
        full_bytes[kind] = full_bytes.get(kind, 0) + size
    per_kind: dict[str, dict[str, object]] = {}
    load_projected: float | None = 0.0
    for kind, (seconds, measured) in load.items():
        projected: float | None = None
        if measured > 0:
            projected = seconds / measured * full_bytes.get(kind, 0)
            if load_projected is not None:
                load_projected += projected
        else:
            load_projected = None
        per_kind[kind] = {
            "seconds": seconds,
            "bytes": measured,
            "full_bytes": full_bytes.get(kind, 0),
            "projected_seconds": projected,
        }
    compute_per_date = compute_seconds / n_dates
    compute_projected = compute_per_date * DECISION_DATES
    total = None if load_projected is None else load_projected + compute_projected
    return {
        "decision_dates": DECISION_DATES,
        "dates_measured": n_dates,
        "load": per_kind,
        "load_projected_seconds": load_projected,
        "compute_seconds_per_date": compute_per_date,
        "compute_projected_seconds": compute_projected,
        "total_projected_seconds": total,
        "full_bytes": sum(full_bytes.values()),
        "sizes": [{"url": url, "bytes": size} for url, size in sorted(sizes.items())],
        "assumption": "compute por data com o universo de hoje",
    }
