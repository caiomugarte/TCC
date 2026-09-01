"""Deterministic numerical core for asset-class allocation.

This module deliberately has no dependency on the stock-selection GA. It
accepts dated daily return rows, evaluates named weight vectors, and exposes
the risk-constrained return/concentration frontier.
"""

from dataclasses import dataclass
from datetime import date
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class AllocationError(ValueError):
    """Raised when allocation inputs cannot be evaluated safely."""


@dataclass(frozen=True)
class DailyReturn:
    """One dated row of total returns keyed by asset class."""

    date: date
    returns: Mapping[str, float]


@dataclass(frozen=True)
class PortfolioPath:
    """Portfolio value path produced by a deterministic simulation."""

    dates: Tuple[date, ...]
    values: Tuple[float, ...]
    daily_returns: Tuple[float, ...]
    rebalance_dates: Tuple[date, ...]


@dataclass(frozen=True)
class Metrics:
    """Gross portfolio metrics used by the allocation objective."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    calmar: float


@dataclass(frozen=True)
class Candidate:
    """Evaluated allocation candidate."""

    weights: Tuple[float, ...]
    metrics: Metrics
    hhi: float
    feasible: bool


def _as_class_tuple(class_names: Sequence[str]) -> Tuple[str, ...]:
    classes = tuple(class_names)
    if not classes or len(set(classes)) != len(classes):
        raise AllocationError("class_names must contain unique classes")
    return classes


def validate_weights(
    weights: Mapping[str, float],
    class_names: Sequence[str],
    tolerance: float = 1e-9,
) -> Tuple[float, ...]:
    """Return weights in class order after enforcing simplex invariants."""

    classes = _as_class_tuple(class_names)
    if set(weights) != set(classes):
        missing = sorted(set(classes) - set(weights))
        extra = sorted(set(weights) - set(classes))
        raise AllocationError(
            "weights must contain exactly the requested classes "
            f"(missing={missing}, extra={extra})"
        )

    ordered = tuple(float(weights[name]) for name in classes)
    if any(not math.isfinite(weight) for weight in ordered):
        raise AllocationError("weights must be finite")
    if any(weight < -tolerance or weight > 1 + tolerance for weight in ordered):
        raise AllocationError("weights must be between 0 and 1")
    if abs(sum(ordered) - 1.0) > tolerance:
        raise AllocationError("weights must sum to 1")

    return tuple(0.0 if abs(weight) <= tolerance else weight for weight in ordered)


def _normalise_rows(
    rows: Iterable[DailyReturn],
    class_names: Sequence[str],
) -> Tuple[DailyReturn, ...]:
    classes = _as_class_tuple(class_names)
    normalised = sorted(rows, key=lambda row: row.date)
    if not normalised:
        raise AllocationError("at least one daily return row is required")

    dates = [row.date for row in normalised]
    if len(set(dates)) != len(dates):
        raise AllocationError("daily return rows must not contain duplicate dates")

    checked = []
    for row in normalised:
        if set(row.returns) != set(classes):
            raise AllocationError(
                f"return row on {row.date} must contain exactly {list(classes)}"
            )
        values = {}
        for name in classes:
            value = float(row.returns[name])
            if not math.isfinite(value) or value < -1.0:
                raise AllocationError(
                    f"return for {name} on {row.date} must be finite and >= -100%"
                )
            values[name] = value
        checked.append(DailyReturn(row.date, values))
    return tuple(checked)


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # February 29 has no anniversary in non-leap years.
        return value.replace(year=value.year + years, day=28)


def class_hhi(weights: Mapping[str, float], class_names: Sequence[str]) -> float:
    """Return HHI for class weights; 1 is concentrated and lower is broader."""

    ordered = validate_weights(weights, class_names)
    return sum(weight * weight for weight in ordered)


def _covariance_matrix(
    return_vectors: Sequence[Tuple[float, ...]],
) -> Tuple[Tuple[float, ...], ...]:
    if len(return_vectors) < 2:
        raise AllocationError("at least two daily return rows are required")
    means = tuple(
        sum(values[index] for values in return_vectors) / len(return_vectors)
        for index in range(len(return_vectors[0]))
    )
    return tuple(
        tuple(
            sum(
                (values[i] - means[i]) * (values[j] - means[j])
                for values in return_vectors
            )
            / (len(return_vectors) - 1)
            for j in range(len(means))
        )
        for i in range(len(means))
    )


def _risk_contributions_from_covariance(
    covariance: Sequence[Sequence[float]],
    weights: Tuple[float, ...],
) -> Tuple[float, ...]:
    portfolio_variance = sum(
        weights[i] * covariance[i][j] * weights[j]
        for i in range(len(weights))
        for j in range(len(weights))
    )
    if portfolio_variance <= 0:
        return tuple(0.0 for _ in weights)
    return tuple(
        weights[i]
        * sum(covariance[i][j] * weights[j] for j in range(len(weights)))
        / portfolio_variance
        for i in range(len(weights))
    )


def risk_contributions(
    rows: Iterable[DailyReturn],
    weights: Mapping[str, float],
    class_names: Sequence[str],
) -> Dict[str, float]:
    """Return each class's signed contribution to target-weight variance.

    Contributions sum to approximately 1. Negative values indicate a class
    reduced portfolio variance over the sampled period. This is a diagnostic
    based on static target weights; it does not change candidate selection.
    """

    classes = _as_class_tuple(class_names)
    target = validate_weights(weights, classes)
    checked_rows = _normalise_rows(rows, classes)
    return dict(
        zip(
            classes,
            _risk_contributions_from_covariance(
                _covariance_matrix(
                    tuple(
                        tuple(row.returns[name] for name in classes)
                        for row in checked_rows
                    )
                ),
                target,
            ),
        )
    )


def _simulate_checked_rows(
    checked_rows: Sequence[DailyReturn],
    target: Tuple[float, ...],
    classes: Tuple[str, ...],
    annual_rebalance: bool,
    rebalance_years: int,
) -> PortfolioPath:
    if rebalance_years < 1:
        raise AllocationError("rebalance_years must be positive")

    holdings = list(target)
    portfolio_value = 1.0
    next_anniversary = _add_years(checked_rows[0].date, rebalance_years)
    values: List[float] = []
    daily_returns: List[float] = []
    rebalance_dates: List[date] = []

    for row in checked_rows:
        if annual_rebalance and row.date > next_anniversary:
            total_value = sum(holdings)
            holdings = [total_value * weight for weight in target]
            rebalance_dates.append(row.date)
            while row.date > next_anniversary:
                next_anniversary = _add_years(next_anniversary, rebalance_years)

        start_value = sum(holdings)
        if start_value <= 0:
            raise AllocationError("portfolio value became non-positive")
        day_return = sum(
            (holding / start_value) * row.returns[name]
            for holding, name in zip(holdings, classes)
        )
        portfolio_value *= 1.0 + day_return
        holdings = [
            holding * (1.0 + row.returns[name])
            for holding, name in zip(holdings, classes)
        ]
        values.append(portfolio_value)
        daily_returns.append(day_return)

    return PortfolioPath(
        dates=tuple(row.date for row in checked_rows),
        values=tuple(values),
        daily_returns=tuple(daily_returns),
        rebalance_dates=tuple(rebalance_dates),
    )


def _metrics_checked_vectors(
    checked_rows: Sequence[DailyReturn],
    return_vectors: Sequence[Tuple[float, ...]],
    target: Tuple[float, ...],
    classes: Tuple[str, ...],
    annual_rebalance: bool,
    rebalance_years: int,
) -> Metrics:
    """Calculate metrics without allocating a path object per candidate."""

    if rebalance_years < 1:
        raise AllocationError("rebalance_years must be positive")
    holdings = list(target)
    portfolio_value = 1.0
    next_anniversary = _add_years(checked_rows[0].date, rebalance_years)
    peak = 0.0
    max_drawdown = 0.0
    count = 0
    mean = 0.0
    sum_squared_differences = 0.0

    for row, return_vector in zip(checked_rows, return_vectors):
        if annual_rebalance and row.date > next_anniversary:
            total_value = sum(holdings)
            holdings = [total_value * weight for weight in target]
            while row.date > next_anniversary:
                next_anniversary = _add_years(next_anniversary, rebalance_years)

        start_value = sum(holdings)
        if start_value <= 0:
            raise AllocationError("portfolio value became non-positive")
        day_return = sum(
            (holding / start_value) * value
            for holding, value in zip(holdings, return_vector)
        )
        portfolio_value *= 1.0 + day_return
        peak = max(peak, portfolio_value)
        if peak > 0:
            max_drawdown = min(max_drawdown, portfolio_value / peak - 1.0)

        count += 1
        difference = day_return - mean
        mean += difference / count
        sum_squared_differences += difference * (day_return - mean)
        holdings = [
            holding * (1.0 + value)
            for holding, value in zip(holdings, return_vector)
        ]

    total_return = portfolio_value - 1.0
    elapsed_days = (checked_rows[-1].date - checked_rows[0].date).days
    elapsed_years = elapsed_days / 365.25
    if elapsed_years <= 0:
        elapsed_years = max(count / 252.0, 1.0 / 252.0)
    annualized_return = (
        portfolio_value ** (1.0 / elapsed_years) - 1.0
        if portfolio_value > 0
        else -1.0
    )
    variance = sum_squared_differences / (count - 1) if count > 1 else 0.0
    annualized_volatility = math.sqrt(max(variance, 0.0)) * math.sqrt(252.0)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        calmar=calmar,
    )


def simulate_portfolio(
    rows: Iterable[DailyReturn],
    weights: Mapping[str, float],
    class_names: Sequence[str],
    annual_rebalance: bool = True,
    rebalance_years: int = 1,
) -> PortfolioPath:
    """Simulate a single-investment portfolio from daily total returns.

    Rebalancing happens at the start of the first trading row strictly after
    each anniversary. This models a decision made at the anniversary close
    and executed on the next available trading day.
    """

    classes = _as_class_tuple(class_names)
    target = validate_weights(weights, classes)
    checked_rows = _normalise_rows(rows, classes)
    return _simulate_checked_rows(
        checked_rows, target, classes, annual_rebalance, rebalance_years
    )


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def portfolio_metrics(path: PortfolioPath) -> Metrics:
    """Calculate annualized return, volatility, drawdown, and Calmar."""

    if not path.values:
        raise AllocationError("portfolio path is empty")
    total_return = path.values[-1] - 1.0
    if path.values[-1] <= 0:
        annualized_return = -1.0
    else:
        elapsed_days = (path.dates[-1] - path.dates[0]).days
        elapsed_years = elapsed_days / 365.25
        if elapsed_years <= 0:
            elapsed_years = max(len(path.values) / 252.0, 1.0 / 252.0)
        annualized_return = path.values[-1] ** (1.0 / elapsed_years) - 1.0

    volatility = _sample_standard_deviation(path.daily_returns) * math.sqrt(252.0)
    peak = 0.0
    max_drawdown = 0.0
    for value in path.values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = min(max_drawdown, value / peak - 1.0)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return Metrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=volatility,
        max_drawdown=max_drawdown,
        calmar=calmar,
    )


def is_feasible(
    metrics: Metrics,
    volatility_cap: float,
    drawdown_cap: float,
    tolerance: float = 1e-9,
) -> bool:
    """Check the training risk caps without relaxing them."""

    if volatility_cap < 0 or drawdown_cap < 0:
        raise AllocationError("risk caps must be non-negative")
    return (
        metrics.annualized_volatility <= volatility_cap + tolerance
        and metrics.max_drawdown >= -drawdown_cap - tolerance
    )


def _compositions(total: int, parts: int) -> Iterable[Tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def simplex_grid(
    class_names: Sequence[str],
    step: float,
) -> Tuple[Tuple[float, ...], ...]:
    """Enumerate a deterministic simplex grid at the requested step."""

    classes = _as_class_tuple(class_names)
    if not 0 < step <= 1:
        raise AllocationError("grid step must be in (0, 1]")
    units = round(1.0 / step)
    if abs(units * step - 1.0) > 1e-9:
        raise AllocationError("grid step must divide 1 exactly")
    return tuple(
        tuple(unit / units for unit in composition)
        for composition in _compositions(units, len(classes))
    )


def bounded_simplex_grid(
    centers: Iterable[Tuple[float, ...]],
    class_names: Sequence[str],
    step: float,
    radius: float,
) -> Tuple[Tuple[float, ...], ...]:
    """Refine neighborhoods around coarse candidates at a finer step."""

    classes = _as_class_tuple(class_names)
    if radius < 0:
        raise AllocationError("refinement radius must be non-negative")
    units = round(1.0 / step)
    radius_units = round(radius * units)
    results = set()
    for center in centers:
        if len(center) != len(classes):
            raise AllocationError("center has the wrong number of weights")
        center_units = [round(weight * units) for weight in center]
        if sum(center_units) != units:
            raise AllocationError("center must lie on the requested fine grid")
        lower = [max(0, value - radius_units) for value in center_units]
        upper = [min(units, value + radius_units) for value in center_units]
        for composition in _bounded_compositions(units, lower, upper, 0):
            results.add(tuple(value / units for value in composition))
    return tuple(sorted(results))


def _bounded_compositions(
    remaining: int,
    lower: Sequence[int],
    upper: Sequence[int],
    index: int,
) -> Iterable[Tuple[int, ...]]:
    if index == len(lower) - 1:
        if lower[index] <= remaining <= upper[index]:
            yield (remaining,)
        return
    remaining_lower = sum(lower[index + 1:])
    remaining_upper = sum(upper[index + 1:])
    start = max(lower[index], remaining - remaining_upper)
    stop = min(upper[index], remaining - remaining_lower)
    for value in range(start, stop + 1):
        for rest in _bounded_compositions(
            remaining - value, lower, upper, index + 1
        ):
            yield (value,) + rest


def evaluate_candidates(
    rows: Iterable[DailyReturn],
    class_names: Sequence[str],
    candidates: Iterable[Tuple[float, ...]],
    volatility_cap: float,
    drawdown_cap: float,
    annual_rebalance: bool = True,
    risk_contribution_cap: Optional[float] = None,
    risk_contribution_caps: Optional[Mapping[str, float]] = None,
) -> Tuple[Candidate, ...]:
    """Evaluate candidates in stable order and mark feasibility."""

    classes = _as_class_tuple(class_names)
    checked_rows = _normalise_rows(rows, classes)
    return_vectors = tuple(
        tuple(row.returns[name] for name in classes) for row in checked_rows
    )
    if risk_contribution_cap is not None:
        risk_contribution_cap = float(risk_contribution_cap)
        if not math.isfinite(risk_contribution_cap) or risk_contribution_cap < 0:
            raise AllocationError(
                "risk contribution cap must be finite and non-negative"
            )
    if risk_contribution_caps is not None:
        unknown = set(risk_contribution_caps) - set(classes)
        if unknown:
            raise AllocationError(
                f"risk contribution caps contain unknown classes: {sorted(unknown)}"
            )
        risk_contribution_caps = {
            name: float(cap) for name, cap in risk_contribution_caps.items()
        }
        if any(
            not math.isfinite(cap) or cap < 0
            for cap in risk_contribution_caps.values()
        ):
            raise AllocationError(
                "risk contribution caps must be finite and non-negative"
            )
    covariance = (
        _covariance_matrix(return_vectors)
        if risk_contribution_cap is not None or risk_contribution_caps is not None
        else None
    )
    evaluated = []
    for raw_weights in candidates:
        if len(raw_weights) != len(classes):
            raise AllocationError("candidate has the wrong number of weights")
        weights = validate_weights(
            {name: raw_weights[index] for index, name in enumerate(classes)},
            classes,
        )
        metrics = _metrics_checked_vectors(
            checked_rows,
            return_vectors,
            weights,
            classes,
            annual_rebalance,
            1,
        )
        feasible = is_feasible(metrics, volatility_cap, drawdown_cap)
        if feasible and covariance is not None:
            contributions = _risk_contributions_from_covariance(covariance, weights)
            if risk_contribution_cap is not None:
                feasible = all(
                    contribution <= risk_contribution_cap + 1e-9
                    for contribution in contributions
                )
            if feasible and risk_contribution_caps is not None:
                feasible = all(
                    contributions[classes.index(name)] <= cap + 1e-9
                    for name, cap in risk_contribution_caps.items()
                )
        evaluated.append(
            Candidate(
                weights=weights,
                metrics=metrics,
                hhi=sum(weight * weight for weight in weights),
                feasible=feasible,
            )
        )
    return tuple(evaluated)


def _dominates(left: Candidate, right: Candidate, tolerance: float = 1e-12) -> bool:
    left_values = (
        left.metrics.annualized_return,
        -left.metrics.annualized_volatility,
        left.metrics.max_drawdown,
        -left.hhi,
    )
    right_values = (
        right.metrics.annualized_return,
        -right.metrics.annualized_volatility,
        right.metrics.max_drawdown,
        -right.hhi,
    )
    no_worse = all(a >= b - tolerance for a, b in zip(left_values, right_values))
    strictly_better = any(a > b + tolerance for a, b in zip(left_values, right_values))
    return no_worse and strictly_better


def non_dominated_frontier(candidates: Iterable[Candidate]) -> Tuple[Candidate, ...]:
    """Keep feasible candidates not dominated on return, risk, drawdown, HHI."""

    feasible = sorted(
        (candidate for candidate in candidates if candidate.feasible),
        key=lambda candidate: (
            -candidate.metrics.annualized_return,
            candidate.metrics.annualized_volatility,
            -candidate.metrics.max_drawdown,
            candidate.hhi,
            candidate.weights,
        ),
    )
    # Return is processed from high to low. The maintained list is the
    # skyline of the processed prefix, so each candidate is compared only
    # with current frontier points instead of every pair in the grid.
    frontier: List[Candidate] = []
    for candidate in feasible:
        if any(_dominates(other, candidate) for other in frontier):
            continue
        frontier = [
            other for other in frontier if not _dominates(candidate, other)
        ]
        frontier.append(candidate)
    return tuple(
        sorted(
            frontier,
            key=lambda candidate: (
                candidate.hhi,
                -candidate.metrics.annualized_return,
                candidate.weights,
            ),
        )
    )


def penalty_sweep(
    candidates: Iterable[Candidate],
    penalties: Iterable[float],
    risk_adjusted_weights: Optional[Mapping[str, float]] = None,
) -> Dict[float, Candidate]:
    """Select feasible winners with optional profile risk preferences."""

    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible:
        return {}
    if risk_adjusted_weights is not None:
        required = {"return", "volatility", "drawdown"}
        if set(risk_adjusted_weights) != required:
            raise AllocationError(
                "risk-adjusted weights must contain return, volatility, and drawdown"
            )
        weights = {name: float(value) for name, value in risk_adjusted_weights.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise AllocationError("risk-adjusted weights must be finite and non-negative")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise AllocationError("risk-adjusted weights must sum to 1")
        returns = [candidate.metrics.annualized_return for candidate in feasible]
        volatilities = [candidate.metrics.annualized_volatility for candidate in feasible]
        drawdowns = [candidate.metrics.max_drawdown for candidate in feasible]

        def quality(value: float, low: float, high: float, maximize: bool) -> float:
            if high == low:
                return 1.0
            return (
                (value - low) / (high - low)
                if maximize
                else (high - value) / (high - low)
            )

        return_low, return_high = min(returns), max(returns)
        volatility_low, volatility_high = min(volatilities), max(volatilities)
        drawdown_low, drawdown_high = min(drawdowns), max(drawdowns)

        def risk_adjusted_score(candidate: Candidate) -> float:
            return (
                weights["return"]
                * quality(
                    candidate.metrics.annualized_return,
                    return_low,
                    return_high,
                    True,
                )
                + weights["volatility"]
                * quality(
                    candidate.metrics.annualized_volatility,
                    volatility_low,
                    volatility_high,
                    False,
                )
                + weights["drawdown"]
                * quality(
                    candidate.metrics.max_drawdown,
                    drawdown_low,
                    drawdown_high,
                    True,
                )
            )

        risk_adjusted_scores = {
            candidate.weights: risk_adjusted_score(candidate)
            for candidate in feasible
        }

    winners: Dict[float, Candidate] = {}
    for raw_penalty in penalties:
        penalty = float(raw_penalty)
        if not math.isfinite(penalty) or penalty < 0:
            raise AllocationError("HHI penalties must be finite and non-negative")
        if risk_adjusted_weights is None:
            key = lambda candidate: (
                candidate.metrics.annualized_return - penalty * candidate.hhi,
                candidate.metrics.annualized_return,
                -candidate.hhi,
                tuple(-weight for weight in candidate.weights),
            )
        else:
            key = lambda candidate: (
                risk_adjusted_scores[candidate.weights] - penalty * candidate.hhi,
                candidate.metrics.annualized_return,
                -candidate.hhi,
                tuple(-weight for weight in candidate.weights),
            )
        winners[penalty] = max(feasible, key=key)
    return winners


def _max_return(candidates: Sequence[Candidate]) -> Candidate:
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.metrics.annualized_return,
            candidate.hhi,
            candidate.metrics.annualized_volatility,
            candidate.weights,
        ),
    )


def _min_hhi(candidates: Sequence[Candidate]) -> Candidate:
    return min(
        candidates,
        key=lambda candidate: (
            candidate.hhi,
            -candidate.metrics.annualized_return,
            candidate.metrics.annualized_volatility,
            candidate.weights,
        ),
    )


def select_frontier_points(
    frontier: Sequence[Candidate],
) -> Dict[str, Candidate]:
    """Return the return/concentration endpoints and a geometric knee.

    Volatility and drawdown have already acted as hard feasibility caps. The
    default choice therefore uses the return-versus-HHI trade-off among the
    feasible frontier points, rather than allowing a lower-return, safer
    point with the same concentration to pull the knee toward cash.
    """

    if not frontier:
        raise AllocationError("cannot select a point from an empty frontier")
    tradeoff: List[Candidate] = []
    lowest_hhi_seen = math.inf
    for candidate in sorted(
        frontier,
        key=lambda item: (
            -item.metrics.annualized_return,
            item.hhi,
            item.weights,
        ),
    ):
        if candidate.hhi < lowest_hhi_seen - 1e-12:
            tradeoff.append(candidate)
            lowest_hhi_seen = candidate.hhi

    max_return = _max_return(tradeoff)
    min_hhi = _min_hhi(tradeoff)
    if max_return.weights == min_hhi.weights or len(tradeoff) == 1:
        knee = max_return
    else:
        return_low = min(
            max_return.metrics.annualized_return,
            min_hhi.metrics.annualized_return,
        )
        return_range = abs(
            max_return.metrics.annualized_return
            - min_hhi.metrics.annualized_return
        )
        hhi_low = min_hhi.hhi
        hhi_range = abs(max_return.hhi - min_hhi.hhi)

        def point(candidate: Candidate) -> Tuple[float, float]:
            x = (candidate.hhi - hhi_low) / hhi_range if hhi_range else 0.0
            y = (
                (candidate.metrics.annualized_return - return_low) / return_range
                if return_range
                else 0.0
            )
            return x, y

        start = point(min_hhi)
        end = point(max_return)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)

        def distance(candidate: Candidate) -> float:
            x, y = point(candidate)
            if length == 0:
                return 0.0
            return abs(dy * x - dx * y + end[0] * start[1] - end[1] * start[0]) / length

        # The frontier also contains points that trade substantially more
        # concentration for a safer drawdown. They are valid frontier points,
        # but they are outside the max-return ↔ minimum-HHI trade-off segment
        # and must not pull the default knee toward a 100% cash-like sleeve.
        segment = tuple(
            candidate
            for candidate in tradeoff
            if hhi_low - 1e-12 <= candidate.hhi <= hhi_low + hhi_range + 1e-12
            and return_low - 1e-12
            <= candidate.metrics.annualized_return
            <= return_low + return_range + 1e-12
        )
        knee = max(
            segment or (min_hhi, max_return),
            key=lambda candidate: (
                distance(candidate),
                candidate.metrics.annualized_return,
                -candidate.hhi,
                tuple(-weight for weight in candidate.weights),
            ),
        )

    return {
        "max_return": max_return,
        "most_diversified": min_hhi,
        "knee": knee,
    }
