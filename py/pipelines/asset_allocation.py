"""Orchestration for Caio's asset-class allocation analysis."""

import csv
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import sys
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PY_ROOT = Path(__file__).resolve().parent.parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import ALLOCATION_CONFIG, ASSET_CLASSES  # noqa: E402
from allocation_profiles import AllocationProfile  # noqa: E402
from core.allocation import (  # noqa: E402
    Candidate,
    DailyReturn,
    bounded_simplex_grid,
    evaluate_candidates,
    non_dominated_frontier,
    penalty_sweep,
    portfolio_metrics,
    risk_contributions,
    select_frontier_points,
    simulate_portfolio,
    simplex_grid,
)


@dataclass(frozen=True)
class WindowEvaluation:
    """All allocation choices evaluated for one training window."""

    start_date: date
    end_date: date
    candidates: Tuple[Candidate, ...]
    frontier: Tuple[Candidate, ...]
    selected: Mapping[str, Candidate]
    penalty_winners: Mapping[float, Candidate]


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _sorted_rows(rows: Iterable[DailyReturn]) -> Tuple[DailyReturn, ...]:
    result = tuple(sorted(rows, key=lambda row: row.date))
    if not result:
        raise ValueError("at least one daily return row is required")
    return result


def _latest_window(rows: Sequence[DailyReturn], years: int) -> Tuple[DailyReturn, ...]:
    end_date = rows[-1].date
    start_date = _add_years(end_date, -years)
    selected = tuple(row for row in rows if row.date >= start_date)
    if len(selected) < 2:
        raise ValueError(f"not enough rows for a {years}-year window")
    return selected


def _candidate_key(candidate: Candidate) -> Tuple[float, float, float, Tuple[float, ...]]:
    return (
        -candidate.metrics.annualized_return,
        candidate.hhi,
        candidate.metrics.annualized_volatility,
        candidate.weights,
    )


def _risk_violation(
    candidate: Candidate,
    volatility_cap: float,
    drawdown_cap: float,
) -> Tuple[float, float, float, float, Tuple[float, ...]]:
    volatility_excess = max(0.0, candidate.metrics.annualized_volatility - volatility_cap)
    drawdown_excess = max(0.0, -drawdown_cap - candidate.metrics.max_drawdown)
    return (
        volatility_excess + drawdown_excess,
        volatility_excess,
        drawdown_excess,
        -candidate.metrics.annualized_return,
        candidate.weights,
    )


def optimize_window(
    rows: Iterable[DailyReturn],
    class_names: Sequence[str] = ASSET_CLASSES,
    config: Optional[Mapping[str, float]] = None,
) -> WindowEvaluation:
    """Search one training window with coarse and local fine grids."""

    settings = dict(config or ALLOCATION_CONFIG["caio"])
    checked_rows = _sorted_rows(rows)
    minimum_weight = float(settings.get("minimum_class_weight", 0.0))
    if minimum_weight < 0.0 or minimum_weight * len(class_names) > 1.0 + 1e-9:
        raise ValueError("minimum class weight must leave a valid simplex")

    def eligible(weights: Tuple[float, ...]) -> bool:
        return all(weight >= minimum_weight - 1e-12 for weight in weights)

    risk_contribution_cap = settings.get("risk_contribution_cap")
    risk_contribution_caps = settings.get("risk_contribution_caps")
    coarse_weights = tuple(
        weights
        for weights in simplex_grid(class_names, float(settings["coarse_step"]))
        if eligible(weights)
    )
    if not coarse_weights:
        raise ValueError("minimum class weight leaves no coarse-grid candidates")
    coarse_candidates = evaluate_candidates(
        checked_rows,
        class_names,
        coarse_weights,
        float(settings["volatility_cap"]),
        float(settings["drawdown_cap"]),
        risk_contribution_cap=risk_contribution_cap,
        risk_contribution_caps=risk_contribution_caps,
    )
    coarse_frontier = non_dominated_frontier(coarse_candidates)

    if coarse_frontier:
        frontier_centers = sorted(
            coarse_frontier,
            key=lambda candidate: (candidate.hhi, _candidate_key(candidate)),
        )[:30]
        centers = [candidate.weights for candidate in frontier_centers]
        centers.extend(
            candidate.weights
            for candidate in sorted(coarse_candidates, key=_candidate_key)[:10]
        )
        centers.extend(
            candidate.weights
            for candidate in sorted(
                coarse_candidates,
                key=lambda candidate: (candidate.hhi, _candidate_key(candidate)),
            )[:10]
        )
    else:
        # A fine-grid candidate may be feasible even when no 5% point is. The
        # search remains honest: the selected candidates are still marked by
        # the same caps and an empty final frontier remains a valid outcome.
        centers = [
            candidate.weights
            for candidate in sorted(
                coarse_candidates,
                key=lambda candidate: _risk_violation(
                    candidate,
                    float(settings["volatility_cap"]),
                    float(settings["drawdown_cap"]),
                ),
            )[:20]
        ]

    fine_weights = tuple(
        weights
        for weights in bounded_simplex_grid(
            centers,
            class_names,
            float(settings["refinement_step"]),
            float(settings["refinement_radius"]),
        )
        if eligible(weights)
    )
    if not fine_weights:
        raise ValueError("minimum class weight leaves no fine-grid candidates")
    fine_candidates = evaluate_candidates(
        checked_rows,
        class_names,
        fine_weights,
        float(settings["volatility_cap"]),
        float(settings["drawdown_cap"]),
        risk_contribution_cap=risk_contribution_cap,
        risk_contribution_caps=risk_contribution_caps,
    )

    by_weights = {candidate.weights: candidate for candidate in coarse_candidates}
    by_weights.update({candidate.weights: candidate for candidate in fine_candidates})
    candidates = tuple(by_weights[key] for key in sorted(by_weights))
    frontier = non_dominated_frontier(candidates)
    selected = select_frontier_points(frontier) if frontier else {}
    penalty_winners = penalty_sweep(
        candidates,
        settings.get("hhi_penalties", (0.0,)),
        risk_adjusted_weights=settings.get("risk_adjusted_weights"),
    )
    return WindowEvaluation(
        start_date=checked_rows[0].date,
        end_date=checked_rows[-1].date,
        candidates=candidates,
        frontier=frontier,
        selected=selected,
        penalty_winners=penalty_winners,
    )


def _metrics_dict(candidate: Candidate) -> Dict[str, float]:
    return {
        "total_return": candidate.metrics.total_return,
        "annualized_return": candidate.metrics.annualized_return,
        "annualized_volatility": candidate.metrics.annualized_volatility,
        "max_drawdown": candidate.metrics.max_drawdown,
        "calmar": candidate.metrics.calmar,
        "hhi": candidate.hhi,
        "feasible": candidate.feasible,
    }


def candidate_record(
    candidate: Candidate,
    class_names: Sequence[str],
    risk_rows: Optional[Sequence[DailyReturn]] = None,
) -> Dict[str, object]:
    weights = {
        name: candidate.weights[index]
        for index, name in enumerate(class_names)
    }
    record: Dict[str, object] = {
        "weights": weights,
        "metrics": _metrics_dict(candidate),
    }
    if risk_rows is not None:
        record["risk_contribution"] = risk_contributions(
            risk_rows,
            weights,
            class_names,
        )
    return record


def window_record(
    window: WindowEvaluation,
    class_names: Sequence[str],
    risk_rows: Optional[Sequence[DailyReturn]] = None,
    include_frontier: bool = True,
    profile_penalty: Optional[float] = None,
) -> Dict[str, object]:
    selected = {
        name: candidate_record(candidate, class_names, risk_rows)
        for name, candidate in window.selected.items()
    }
    if profile_penalty is not None:
        profile_winner = window.penalty_winners.get(profile_penalty)
        if profile_winner is not None:
            selected["profile_winner"] = candidate_record(
                profile_winner,
                class_names,
                risk_rows,
            )
    record: Dict[str, object] = {
        "start_date": window.start_date.isoformat(),
        "end_date": window.end_date.isoformat(),
        "candidate_count": len(window.candidates),
        "selected": selected,
        "hhi_penalty_sweep": [
            {
                "penalty": penalty,
                "winner": candidate_record(candidate, class_names, risk_rows),
            }
            for penalty, candidate in sorted(window.penalty_winners.items())
        ],
    }
    if include_frontier:
        record["frontier"] = [
            candidate_record(candidate, class_names)
            for candidate in window.frontier
        ]
    return record


def _baseline_weights(class_names: Sequence[str]) -> Dict[str, Tuple[float, ...]]:
    classes = tuple(class_names)
    equal = tuple(1.0 / len(classes) for _ in classes)
    return {
        "caio_stocks_100": tuple(1.0 if index == 0 else 0.0 for index in range(len(classes))),
        "equal_20": equal,
        "di_100": tuple(1.0 if name == "fixed_income" else 0.0 for name in classes),
    }


def evaluate_baselines(
    rows: Iterable[DailyReturn],
    class_names: Sequence[str],
    config: Mapping[str, float],
) -> Dict[str, Dict[str, object]]:
    """Evaluate fixed comparison allocations under the same risk caps."""

    checked_rows = _sorted_rows(rows)
    results = {}
    candidates = evaluate_candidates(
        checked_rows,
        class_names,
        _baseline_weights(class_names).values(),
        float(config["volatility_cap"]),
        float(config["drawdown_cap"]),
    )
    for name, candidate in zip(_baseline_weights(class_names), candidates):
        results[name] = candidate_record(candidate, class_names, checked_rows)
    return results


def evaluate_crypto_weight_scenarios(
    current_rows: Sequence[DailyReturn],
    primary_rows: Sequence[DailyReturn],
    robustness_rows: Sequence[DailyReturn],
    class_names: Sequence[str],
    base_weights: Mapping[str, float],
    crypto_weights: Iterable[float],
    config: Mapping[str, object],
) -> Dict[str, object]:
    """Compare lower crypto weights by moving the difference to fixed income."""

    classes = tuple(class_names)
    volatility_cap = float(config["volatility_cap"])
    drawdown_cap = float(config["drawdown_cap"])
    scenarios = {}
    for raw_crypto_weight in crypto_weights:
        crypto_weight = float(raw_crypto_weight)
        target = dict(base_weights)
        difference = target["crypto"] - crypto_weight
        if difference < 0:
            continue
        target["crypto"] = crypto_weight
        target["fixed_income"] += difference

        def evaluate(rows: Sequence[DailyReturn]) -> Dict[str, object]:
            checked_rows = _sorted_rows(rows)
            candidate = evaluate_candidates(
                checked_rows,
                classes,
                [tuple(target[name] for name in classes)],
                volatility_cap,
                drawdown_cap,
            )[0]
            return candidate_record(candidate, classes, checked_rows)

        scenario_name = f"crypto_{int(crypto_weight * 100)}pct_to_fixed_income"
        scenarios[scenario_name] = {
            "crypto_weight": crypto_weight,
            "redistributed_to": "fixed_income",
            "current_training": evaluate(current_rows),
            "primary_horizon": evaluate(primary_rows),
            "robustness_horizon": evaluate(robustness_rows),
        }
    return scenarios


def run_walk_forward(
    rows: Iterable[DailyReturn],
    class_names: Sequence[str],
    config: Mapping[str, float],
    include_frontier: bool = True,
    profile_penalty: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Run rolling three-year training/one-year test windows."""

    checked_rows = _sorted_rows(rows)
    training_years = int(config["training_years"])
    test_years = int(config["test_years"])
    windows = []
    train_start = checked_rows[0].date

    while True:
        train_end = _add_years(train_start, training_years)
        test_end = _add_years(train_end, test_years)
        train_rows = tuple(
            row for row in checked_rows
            if train_start <= row.date <= train_end
        )
        test_rows = tuple(
            row for row in checked_rows
            if train_end < row.date <= test_end
        )
        if len(train_rows) < 2 or len(test_rows) < 2:
            break

        window = optimize_window(train_rows, class_names, config)
        record: Dict[str, object] = {
            "train_start": train_rows[0].date.isoformat(),
            "train_end": train_rows[-1].date.isoformat(),
            "test_start": test_rows[0].date.isoformat(),
            "test_end": test_rows[-1].date.isoformat(),
            "training": window_record(
                window,
                class_names,
                train_rows,
                include_frontier=include_frontier,
                profile_penalty=profile_penalty,
            ),
            "test": None,
        }
        selection_key = "profile_winner" if profile_penalty is not None else "knee"
        selected_candidate = (
            window.penalty_winners.get(profile_penalty)
            if profile_penalty is not None
            else window.selected.get("knee")
        )
        record["selection"] = selection_key
        if selected_candidate is not None:
            test_path = simulate_portfolio(
                test_rows,
                {
                    name: selected_candidate.weights[index]
                    for index, name in enumerate(class_names)
                },
                class_names,
                annual_rebalance=False,
            )
            test_metrics = portfolio_metrics(test_path)
            record["test"] = {
                "metrics": {
                    "total_return": test_metrics.total_return,
                    "annualized_return": test_metrics.annualized_return,
                    "annualized_volatility": test_metrics.annualized_volatility,
                    "max_drawdown": test_metrics.max_drawdown,
                    "calmar": test_metrics.calmar,
                },
                "volatility_cap_violated": test_metrics.annualized_volatility > float(config["volatility_cap"]),
                "drawdown_cap_violated": test_metrics.max_drawdown < -float(config["drawdown_cap"]),
                "weights": {
                    name: selected_candidate.weights[index]
                    for index, name in enumerate(class_names)
                },
            }
        windows.append(record)
        next_start = test_rows[0].date
        if next_start <= train_start:
            break
        train_start = next_start
    return windows


def allocation_stability(
    windows: Sequence[Mapping[str, object]],
    class_names: Sequence[str],
    selection_key: str = "knee",
) -> Dict[str, object]:
    """Summarize selected training weights across walk-forward windows."""

    observations = []
    for window in windows:
        selected = window["training"]["selected"].get(selection_key)
        if selected:
            observations.append(selected["weights"])
    summary: Dict[str, object] = {"n_windows": len(observations), "classes": {}}
    for name in class_names:
        values = [float(observation[name]) for observation in observations]
        if not values:
            continue
        average = sum(values) / len(values)
        dispersion = math.sqrt(
            sum((value - average) ** 2 for value in values) / len(values)
        )
        summary["classes"][name] = {
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "dispersion": dispersion,
        }
    return summary


def run_risk_budget_scenario(
    current_rows: Sequence[DailyReturn],
    primary_rows: Sequence[DailyReturn],
    robustness_rows: Sequence[DailyReturn],
    class_names: Sequence[str],
    config: Mapping[str, object],
    risk_contribution_cap: float,
) -> Dict[str, object]:
    """Run one risk-budget alternative without changing the default target."""

    scenario_config = dict(config)
    scenario_config["risk_contribution_cap"] = risk_contribution_cap
    current = optimize_window(current_rows, class_names, scenario_config)
    primary = run_walk_forward(
        primary_rows,
        class_names,
        scenario_config,
        include_frontier=False,
    )
    robustness = run_walk_forward(
        robustness_rows,
        class_names,
        scenario_config,
        include_frontier=False,
    )
    return {
        "risk_contribution_cap": risk_contribution_cap,
        "current_target": window_record(
            current,
            class_names,
            current_rows,
            include_frontier=False,
        ),
        "walk_forward": {
            "primary": primary,
            "robustness": robustness,
        },
        "stability": {
            "primary": allocation_stability(primary, class_names),
            "robustness": allocation_stability(robustness, class_names),
        },
    }


def run_allocation(
    rows: Iterable[DailyReturn],
    metadata: Optional[Mapping[str, object]] = None,
    profile: str = "caio",
    allocation_profile: Optional[AllocationProfile] = None,
) -> Dict[str, object]:
    """Run the current target, horizons, baselines, and walk-forward reports."""

    if profile not in ALLOCATION_CONFIG:
        raise ValueError(f"unknown allocation profile: {profile}")
    config = dict(ALLOCATION_CONFIG[profile])
    profile_record: Dict[str, object] = {
        "status": "generic_fallback",
        "reason": "no calibrated allocation profile supplied",
    }
    profile_penalty = None
    if allocation_profile is not None:
        config["volatility_cap"] = allocation_profile.volatility_cap
        config["drawdown_cap"] = allocation_profile.drawdown_cap
        if allocation_profile.crypto_risk_contribution_cap is not None:
            config["risk_contribution_caps"] = {
                "crypto": allocation_profile.crypto_risk_contribution_cap,
            }
        if allocation_profile.risk_adjusted_weights:
            config["risk_adjusted_weights"] = dict(
                allocation_profile.risk_adjusted_weights
            )
        profile_penalty = allocation_profile.hhi_penalty
        config["hhi_penalties"] = tuple(
            sorted(set(config.get("hhi_penalties", ())) | {profile_penalty})
        )
        profile_record = {
            "status": "personalized",
            "parameters": allocation_profile.as_dict(),
            "selection": {
                "method": (
                    "risk-adjusted return/volatility/drawdown score minus HHI penalty"
                    if allocation_profile.risk_adjusted_weights
                    else "return minus HHI penalty"
                ),
                "hhi_penalty": profile_penalty,
            },
        }
    checked_rows = _sorted_rows(rows)
    primary_rows = _latest_window(
        checked_rows, int(config["primary_horizon_years"])
    )
    robustness_rows = _latest_window(
        checked_rows, int(config["robustness_horizon_years"])
    )
    current_rows = _latest_window(checked_rows, int(config["training_years"]))
    current = optimize_window(current_rows, ASSET_CLASSES, config)
    current_target = (
        current.penalty_winners.get(profile_penalty)
        if profile_penalty is not None
        else current.selected.get("knee")
    )
    crypto_weight_scenarios = {}
    if current_target is not None:
        base_weights = {
            name: current_target.weights[index]
            for index, name in enumerate(ASSET_CLASSES)
        }
        crypto_weight_scenarios = evaluate_crypto_weight_scenarios(
            current_rows,
            primary_rows,
            robustness_rows,
            ASSET_CLASSES,
            base_weights,
            config.get("crypto_weight_scenarios", ()),
            config,
        )

    primary_walk_forward = run_walk_forward(
        primary_rows,
        ASSET_CLASSES,
        config,
        profile_penalty=profile_penalty,
    )
    robustness_walk_forward = run_walk_forward(
        robustness_rows,
        ASSET_CLASSES,
        config,
        profile_penalty=profile_penalty,
    )
    risk_budget_scenarios = {
        name: run_risk_budget_scenario(
            current_rows,
            primary_rows,
            robustness_rows,
            ASSET_CLASSES,
            config,
            float(cap),
        )
        for name, cap in config.get("risk_budget_scenarios", {}).items()
    }
    return {
        "profile": profile,
        "allocation_profile": profile_record,
        "classes": list(ASSET_CLASSES),
        "config": config,
        "data": {
            "start_date": checked_rows[0].date.isoformat(),
            "end_date": checked_rows[-1].date.isoformat(),
            "rows": len(checked_rows),
            "metadata": dict(metadata or {}),
        },
        "current_target": window_record(
            current,
            ASSET_CLASSES,
            current_rows,
            profile_penalty=profile_penalty,
        ),
        "baselines": {
            "primary": evaluate_baselines(primary_rows, ASSET_CLASSES, config),
            "robustness": evaluate_baselines(robustness_rows, ASSET_CLASSES, config),
        },
        "walk_forward": {
            "primary": primary_walk_forward,
            "robustness": robustness_walk_forward,
        },
        "stability": {
            "primary": allocation_stability(
                primary_walk_forward,
                ASSET_CLASSES,
                selection_key=("profile_winner" if profile_penalty is not None else "knee"),
            ),
            "robustness": allocation_stability(
                robustness_walk_forward,
                ASSET_CLASSES,
                selection_key=("profile_winner" if profile_penalty is not None else "knee"),
            ),
        },
        "risk_budget_scenarios": risk_budget_scenarios,
        "crypto_weight_scenarios": crypto_weight_scenarios,
    }


def write_outputs(
    result: Mapping[str, object],
    output_dir: Path,
    prefix: str = "allocation_caio",
) -> Tuple[Path, Path, Path]:
    """Write JSON, frontier CSV, and walk-forward CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    frontier_path = output_dir / f"{prefix}_frontier.csv"
    walk_forward_path = output_dir / f"{prefix}_walk_forward.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    classes = list(result["classes"])
    frontier = result["current_target"]["frontier"]
    with frontier_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["point"] + classes + [
            "total_return", "annualized_return", "annualized_volatility",
            "max_drawdown", "calmar", "hhi", "feasible"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, point in enumerate(frontier):
            row = {"point": index + 1}
            row.update(point["weights"])
            row.update(point["metrics"])
            writer.writerow(row)

    with walk_forward_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["horizon", "train_start", "train_end", "test_start", "test_end", "feasible"] + classes
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for horizon, windows in result["walk_forward"].items():
            for window in windows:
                selection_key = window.get("selection", "knee")
                selected = window["training"]["selected"].get(selection_key)
                row = {
                    "horizon": horizon,
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "feasible": bool(selected),
                }
                row.update((selected or {}).get("weights", {}))
                writer.writerow(row)
    return json_path, frontier_path, walk_forward_path
