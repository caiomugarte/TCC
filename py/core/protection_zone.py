"""Protection zone for the Premium consensus (AD-022, issue #46).

Pure functions over plain mappings: the rule the Premium selectors apply after
they count the GA runs. The stock and FII selectors call this module, and the
API only reads the incumbents and records the provenance. No pandas here.
"""

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
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


SEGMENT_CAP_DEPARTURE = "saiu: teto do segmento"


class SegmentCapInfeasibleError(ValueError):
    """A greedy walk under the segment cap and ``hhi_max`` cannot reach ``n_assets`` (SEG-40, SEG-43)."""


def validate_segment_cap(
    max_per_segment: int | None,
    segment_of: Mapping[str, str] | None,
    tickers: Iterable[str],
    hhi_max: float | None = None,
) -> None:
    """Raise ValueError naming the field that breaks the segment-cap contract (SEG-25, SEG-41).

    ``hhi_max`` is accepted only together with a cap: without one, the HHI skip
    never runs and the caller keeps the pre-change path.
    """

    if max_per_segment is None:
        if hhi_max is not None:
            raise ValueError("hhi_max is only accepted together with max_per_segment")
        return
    if not _is_integer(max_per_segment) or max_per_segment < 1:
        raise ValueError("max_per_segment must be an integer >= 1")
    if not isinstance(segment_of, Mapping):
        raise ValueError("segment_of is required when max_per_segment is set")
    missing = sorted(ticker for ticker in tickers if ticker not in segment_of)
    if missing:
        raise ValueError(f"segment_of has no segment for {missing}")
    if hhi_max is not None and (
        not isinstance(hhi_max, numbers.Real)
        or isinstance(hhi_max, bool)
        or not math.isfinite(hhi_max)
        or hhi_max <= 0
    ):
        raise ValueError("hhi_max must be a finite number > 0")


def _fits(
    ticker: str,
    held_counts: Counter,
    size: int,
    *,
    segment_of: Mapping[str, str],
    max_per_segment: int,
    hhi_max: float | None,
    n_assets: int,
) -> bool:
    """Whether ``ticker`` may join a held set of ``size`` tickers (spec A-16).

    The segment must hold fewer than ``max_per_segment`` tickers and, with
    ``hhi_max``, the lowest HHI any completion to ``n_assets`` can reach (every
    empty slot in a new segment) must stay within it.
    """

    count = held_counts[segment_of[ticker]]
    if count >= max_per_segment:
        return False
    if hhi_max is None:
        return True
    squares = sum(c * c for c in held_counts.values())
    projected = squares + 2 * count + 1 + (n_assets - size - 1)
    return projected / (n_assets * n_assets) <= hhi_max + TOLERANCE


def take_within_segment_cap(
    ordered: Sequence[str],
    n_assets: int,
    *,
    segment_of: Mapping[str, str] | None,
    max_per_segment: int | None,
    hhi_max: float | None = None,
) -> list[str]:
    """The first ``n_assets`` of ``ordered`` that fit the segment cap and ``hhi_max`` (SEG-15, SEG-37).

    Without a cap it is the plain prefix. A ticker that does not fit is skipped
    and the walk goes on; a walk that ends short raises (SEG-40).
    """

    validate_segment_cap(max_per_segment, segment_of, ordered, hhi_max)
    if max_per_segment is None:
        return list(ordered[:n_assets])
    picked: list[str] = []
    counts: Counter = Counter()
    for ticker in ordered:
        if len(picked) >= n_assets:
            break
        if not _fits(
            ticker, counts, len(picked), segment_of=segment_of,
            max_per_segment=max_per_segment, hhi_max=hhi_max, n_assets=n_assets,
        ):
            continue
        picked.append(ticker)
        counts[segment_of[ticker]] += 1
    if len(picked) < n_assets:
        raise SegmentCapInfeasibleError(
            f"cannot fill n_assets {n_assets} within max_per_segment {max_per_segment} "
            f"and hhi_max {hhi_max}: {len(picked)} fit"
        )
    return picked


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


def apply_protection_zone(
    incumbents: Iterable[str],
    frequencies: Mapping[str, float],
    scores: Mapping[str, float],
    n_assets: int,
    K: int,
    delta: float,
    *,
    eligible: Collection[str],
    dedup_base_company: bool,
    segment_of: Mapping[str, str] | None = None,
    max_per_segment: int | None = None,
    hhi_max: float | None = None,
) -> ZoneSelection:
    """Apply the AD-022 zone to one class and give a reason for every ticker (PZ-01).

    Only eligible tickers are ranked, so an ineligible ticker is never selected.
    With ``dedup_base_company`` the ranking holds one ticker per company, so the
    portfolio never holds two tickers of one company (PZ-09). ``K = 0`` returns
    the pure consensus.

    With ``max_per_segment`` (issue #56, AD-027) no segment of ``segment_of``
    holds more than the cap: a qualifying incumbent of a full segment leaves
    with ``saiu: teto do segmento`` before the ``n_assets`` cut, and the fill
    and the challengers skip a ticker that would break the cap. With ``hhi_max``
    as well (accepted only with the cap), the protection walk, the fill and the
    challengers also skip a ticker that would push the sleeve above the
    projected HHI bound (spec A-16, AD-029); an incumbent the walk rejects
    leaves with the same reason, and a fill that cannot reach ``n_assets``
    raises SegmentCapInfeasibleError. Without a cap the rule is the AD-022
    one, unchanged.
    """

    validate_zone_inputs(n_assets, K, delta, frequencies)
    incumbent_set = {ticker.strip().upper() for ticker in incumbents}
    eligible_set = set(eligible)
    # SPEC_DEVIATION: design step 2 ranks every picked ticker; this ranks eligible picks only.
    # Reason: PZ-06 says an ineligible incumbent is never selected, and the fill draws from this
    # ranking. Production picks are always eligible (they come from the ranked frame), so no output changes.
    candidates = {ticker: f for ticker, f in frequencies.items() if ticker in eligible_set}
    validate_segment_cap(max_per_segment, segment_of, candidates, hhi_max)
    ranking = rank_candidates(candidates, scores, dedup_base_company=dedup_base_company)
    rank = {ticker: position for position, ticker in enumerate(ranking, start=1)}

    # Protection: eligible incumbents ranked within K, the n_assets best only (PZ-03, PZ-08).
    qualifying = sorted(
        (t for t in incumbent_set if t in rank and rank[t] <= K), key=rank.__getitem__
    )
    segment_capped: set[str] = set()
    if max_per_segment is None:
        protected = qualifying[:n_assets]
        capped = set(qualifying[n_assets:])
    else:
        # Segment cut before the capacity cut (A-5): walk in rank order and keep
        # an incumbent only while it fits (SEG-19, SEG-39). Once n_assets are
        # accepted the rest are cut by capacity anyway, so only the cap decides
        # whether they leave as "teto do segmento" or "vagas reduzidas".
        accepted: list[str] = []
        accepted_counts: Counter = Counter()
        for ticker in qualifying:
            if _fits(
                ticker, accepted_counts, len(accepted), segment_of=segment_of,
                max_per_segment=max_per_segment,
                hhi_max=hhi_max if len(accepted) < n_assets else None,
                n_assets=n_assets,
            ):
                accepted.append(ticker)
                accepted_counts[segment_of[ticker]] += 1
            else:
                segment_capped.add(ticker)
        protected = accepted[:n_assets]
        capped = set(accepted[n_assets:])

    # Fill the vacancies in rank order (PZ-04), skipping a ticker that does not fit (SEG-21, SEG-38).
    held = set(protected)
    if max_per_segment is None:
        for ticker in ranking:
            if len(held) >= n_assets:
                break
            held.add(ticker)
    else:
        held_counts = Counter(segment_of[t] for t in held)
        for ticker in ranking:
            if len(held) >= n_assets:
                break
            if ticker in held or not _fits(
                ticker, held_counts, len(held), segment_of=segment_of,
                max_per_segment=max_per_segment, hhi_max=hhi_max, n_assets=n_assets,
            ):
                continue
            held.add(ticker)
            held_counts[segment_of[ticker]] += 1
        if len(held) < n_assets:
            raise SegmentCapInfeasibleError(
                f"protection zone cannot fill n_assets {n_assets} within max_per_segment "
                f"{max_per_segment} and hhi_max {hhi_max}: {len(held)} fit"
            )

    def fits_without(ticker: str, weakest: str) -> bool:
        """A challenger must fit the held set once the challenged incumbent leaves (SEG-22, SEG-38)."""

        if max_per_segment is None:
            return True
        remaining = held - {weakest}
        return _fits(
            ticker, Counter(segment_of[t] for t in remaining), len(remaining),
            segment_of=segment_of, max_per_segment=max_per_segment, hhi_max=hhi_max,
            n_assets=n_assets,
        )

    # Swaps (PZ-05): the weakest protected ticker still held gives way to the best
    # outside candidate when it ranks above it and leads by delta. A replaced
    # ticker never returns. Every ranked ticker shares no company with another
    # one, so a challenger never shares a company with the portfolio.
    protected_held = set(protected)
    replaced: set[str] = set()
    swaps: list[tuple[str, str]] = []
    while protected_held:
        weakest = max(protected_held, key=rank.__getitem__)
        challenger = next(
            (
                t for t in ranking
                if t not in held and t not in replaced and fits_without(t, weakest)
            ),
            None,
        )
        if challenger is None:
            break
        if not (
            rank[challenger] < rank[weakest]
            and candidates[challenger] >= candidates[weakest] + delta - TOLERANCE
        ):
            break
        held.remove(weakest)
        held.add(challenger)
        protected_held.remove(weakest)
        replaced.add(weakest)
        swaps.append((weakest, challenger))

    tickers = tuple(sorted(held, key=rank.__getitem__))
    kept = tuple(sorted(t for t in held if t in incumbent_set))
    entered = tuple(sorted(t for t in held if t not in incumbent_set))
    swapped_in = {challenger: weakest for weakest, challenger in swaps}
    swapped_out = {weakest: challenger for weakest, challenger in swaps}

    reasons: dict[str, str] = {}
    for ticker in tickers:
        if ticker in swapped_in:
            reasons[ticker] = f"substituiu {swapped_in[ticker]}"
        elif ticker in incumbent_set:
            reasons[ticker] = "mantido"
        else:
            reasons[ticker] = "novo"

    departures: dict[str, str] = {}
    for ticker in sorted(incumbent_set - held):
        if ticker not in eligible_set:
            departures[ticker] = "saiu: inelegível"
        elif ticker in swapped_out:
            departures[ticker] = f"saiu: substituído por {swapped_out[ticker]}"
        elif ticker in segment_capped:
            departures[ticker] = SEGMENT_CAP_DEPARTURE
        elif ticker in capped:
            departures[ticker] = "saiu: vagas reduzidas"
        else:
            departures[ticker] = "saiu: fora do top K"

    return ZoneSelection(
        tickers=tickers,
        reasons=reasons,
        departures=departures,
        kept=kept,
        replaced=tuple(swaps),
        entered=entered,
        turnover=len(entered),
        exits=len(departures),
    )


def select_with_fallback(
    incumbents: Iterable[str],
    frequencies: Mapping[str, float],
    scores: Mapping[str, float],
    n_assets: int,
    zone: Mapping[str, object],
    *,
    eligible: Collection[str],
    dedup_base_company: bool,
    hhi_of: Callable[[Sequence[str]], float],
    hhi_max: float | None,
) -> tuple[ZoneSelection, dict[str, object]]:
    """Run the zone for one class and return the portfolio used with its class record.

    The zone falls back to the pure consensus (K = 0) only when ``hhi_max`` is set
    and the zone portfolio's HHI exceeds it by more than the tolerance (PZ-14).
    The record has the fields of the design's class record; the adapter writes
    the reduced record for a class it does not send to a selector.
    """

    incumbent_list = sorted({ticker.strip().upper() for ticker in incumbents})
    K = zone["K"]
    delta = zone["delta"]
    zone_selection = apply_protection_zone(
        incumbent_list, frequencies, scores, n_assets, K, delta,
        eligible=eligible, dedup_base_company=dedup_base_company,
    )
    consensus = apply_protection_zone(
        incumbent_list, frequencies, scores, n_assets, 0, delta,
        eligible=eligible, dedup_base_company=dedup_base_company,
    )

    selection, status, buffer_fallback = zone_selection, "applied", None
    if not incumbent_list:
        status = "no_incumbents"
    elif hhi_max is not None and hhi_of(zone_selection.tickers) > hhi_max + TOLERANCE:
        selection, status, buffer_fallback = consensus, "fallback_hhi", "hhi"

    record: dict[str, object] = {
        "status": status,
        "K": int(K),
        "k_offset": int(K) - int(n_assets),
        "delta": float(delta),
        "n_assets": int(n_assets),
        "incumbents": incumbent_list,
        "kept": list(selection.kept),
        "replaced": [{"out": out, "in": into} for out, into in selection.replaced],
        "entered": list(selection.entered),
        "departures": dict(sorted(selection.departures.items())),
        "turnover": selection.turnover,
        "exits": selection.exits,
        "buffer_fallback": buffer_fallback,
        "consensus_tickers": sorted(consensus.tickers),
        "consensus_turnover": consensus.turnover,
    }
    return selection, record
