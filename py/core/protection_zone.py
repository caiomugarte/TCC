"""Protection zone for the Premium consensus (AD-022, issue #46).

Pure functions over plain mappings: the rule the Premium selectors apply after
they count the GA runs. The stock and FII selectors call this module, and the
API only reads the incumbents and records the provenance. No pandas here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

# Same suffixes as the base-company rule of build_consensus_portfolio
# (py/pipelines/multi_run.py) and scripts/convergence_analysis.py (AD-022).
BASE_COMPANY_SUFFIXES = ("11", "3", "4", "5", "6")

# Same tolerance as the FII HHI check (py/fii_selection.py).
TOLERANCE = 1e-12


def base_company(ticker: str) -> str:
    """The company behind a ticker: the first class suffix is removed (TAEE11 -> TAEE)."""

    for suffix in BASE_COMPANY_SUFFIXES:
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def rank_candidates(
    frequencies: Mapping[str, float],
    scores: Mapping[str, float],
    *,
    dedup_base_company: bool,
) -> list[str]:
    """Every candidate by frequency desc, score desc, ticker asc (PZ-02).

    The sort runs over the mapping's own order. A NaN score compares neither
    below nor above, so a tie that contains a NaN keeps first-appearance order,
    as the production sort does (AD-025). With ``dedup_base_company``, a ticker
    whose company is already ranked is dropped.
    """

    ordered = sorted(frequencies, key=lambda t: (-frequencies[t], -float(scores[t]), t))
    if not dedup_base_company:
        return ordered
    ranking: list[str] = []
    companies: set[str] = set()
    for ticker in ordered:
        company = base_company(ticker)
        if company not in companies:
            companies.add(company)
            ranking.append(ticker)
    return ranking


@dataclass(frozen=True)
class ZoneSelection:
    """One class's zone portfolio. ``tickers`` is in rank order."""

    tickers: tuple[str, ...]
    reasons: Mapping[str, str]
    departures: Mapping[str, str]
    kept: tuple[str, ...]
    replaced: tuple[tuple[str, str], ...]
    entered: tuple[str, ...]
    turnover: int
    exits: int


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def validate_zone_inputs(
    n_assets: int,
    K: int,
    delta: float,
    frequencies: Mapping[str, float],
) -> None:
    """Raise ValueError naming the field that breaks the zone contract (PZ-11)."""

    if not _is_integer(n_assets) or n_assets < 1:
        raise ValueError("n_assets must be an integer >= 1")
    if not _is_integer(K) or K < 0:
        raise ValueError("K must be an integer >= 0")
    if not isinstance(delta, numbers.Real) or not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be a finite number >= 0")
    for ticker, frequency in frequencies.items():
        if not 0.0 <= frequency <= 1.0:
            raise ValueError(f"frequency of {ticker} must be in [0, 1]")
