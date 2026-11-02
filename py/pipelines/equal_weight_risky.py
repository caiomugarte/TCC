"""Premium class allocation: 1/N across allowed risky classes plus fixed income.

ADR 0003 replaces the class grid search with this method. The risky share is
the largest whole percentage whose volatility and maximum drawdown over the
last ``window_years`` of history stay within the profile caps; the rest goes
to fixed income.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import PurePath
from typing import Any, Dict, Mapping, Optional, Sequence

from allocation_config import ASSET_CLASSES
from snapshot_manifest import ALLOCATION_FILES
from core.allocation import (
    AllocationError,
    DailyReturn,
    Metrics,
    _add_years,
    is_feasible,
    portfolio_metrics,
    simulate_portfolio,
)

METHOD_ID = "equal-weight-risky-v1"
RISKY_CLASSES = ("brazilian_stocks", "fiis", "international_equity", "crypto")
FIXED_INCOME = "fixed_income"
WINDOW_YEARS = 10
FX_CONVERTED_CLASSES = ("international_equity", "crypto")


def _weights_for(risky_steps: int, allowed: Sequence[str]) -> Dict[str, float]:
    weights = {name: 0.0 for name in ASSET_CLASSES}
    if allowed and risky_steps:
        share = risky_steps / 100.0
        for name in allowed:
            weights[name] = share / len(allowed)
        weights[FIXED_INCOME] = 1.0 - share
    else:
        weights[FIXED_INCOME] = 1.0
    return weights


def _window(rows: Sequence[DailyReturn], years: int) -> tuple[tuple[DailyReturn, ...], bool]:
    """Return the rows of the last ``years`` and whether history was shorter."""

    ordered = sorted(rows, key=lambda row: row.date)
    if not ordered:
        raise AllocationError("at least one daily return row is required")
    start = _add_years(ordered[-1].date, -years)
    window = tuple(row for row in ordered if row.date >= start)
    return window, ordered[0].date > start


def _source_entry(sources: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Describe the series behind one class without exposing local paths.

    Manifest-backed metadata names either a per-class source or one
    ``allocation`` directory source (with optional ``metadata.files``);
    legacy snapshot metadata keys sources by file stem.
    """

    source = sources.get(key)
    if isinstance(source, Mapping):
        return {
            "source": key,
            "file": PurePath(str(source.get("path", ""))).name or None,
            "provider": source.get("provider"),
            "sha256": source.get("sha256"),
        }
    filename = ALLOCATION_FILES[key]
    allocation = sources.get("allocation")
    if isinstance(allocation, Mapping):
        files = (allocation.get("metadata") or {}).get("files") or {}
        if isinstance(files, Mapping) and files.get(key):
            filename = str(files[key])
        return {
            "source": "allocation",
            "file": filename,
            "provider": allocation.get("provider"),
            "sha256": allocation.get("sha256"),
        }
    entry: Dict[str, Any] = {"source": None, "file": filename}
    legacy = sources.get(PurePath(filename).stem)
    if isinstance(legacy, Mapping):
        entry["details"] = dict(legacy)
    return entry


def class_series(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map each allocation class to the snapshot series it was built from."""

    raw = metadata.get("sources") if isinstance(metadata, Mapping) else None
    sources = raw if isinstance(raw, Mapping) else {}
    result = {}
    for name in ASSET_CLASSES:
        entry = _source_entry(sources, name)
        if name in FX_CONVERTED_CLASSES:
            entry["fx"] = _source_entry(sources, "ptax")
        result[name] = entry
    return result


def _cap(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if value is None:
        raise AllocationError(f"allocation config is missing {key}")
    return float(value)


def run_equal_weight_risky(
    rows: Sequence[DailyReturn],
    *,
    allocation_config: Mapping[str, Any],
    class_constraints: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    window_years: int = WINDOW_YEARS,
) -> Dict[str, Any]:
    """Return 1/N risky weights scaled with fixed income to the profile caps.

    ``allocation_config`` must carry ``volatility_cap`` and ``drawdown_cap``.
    From ``class_constraints`` only ``maximum_weights[cls] == 0`` (class
    excluded) and ``minimum_weights.fixed_income`` (risky share ceiling) are
    used; class HHI and the crypto risk-contribution cap are not applied.
    """

    volatility_cap = _cap(allocation_config, "volatility_cap")
    drawdown_cap = _cap(allocation_config, "drawdown_cap")
    constraints = class_constraints or {}
    maximums = constraints.get("maximum_weights") or {}
    minimums = constraints.get("minimum_weights") or {}
    excluded = [name for name in RISKY_CLASSES if name in maximums and float(maximums[name]) <= 0]
    allowed = [name for name in RISKY_CLASSES if name not in excluded]
    fixed_income_minimum = float(minimums.get(FIXED_INCOME, 0.0))
    max_steps = int((1.0 - fixed_income_minimum) * 100 + 1e-9) if allowed else 0
    max_steps = max(0, min(100, max_steps))

    window, short_history = _window(rows, window_years)

    def measure(steps: int) -> Metrics:
        weights = _weights_for(steps, allowed)
        return portfolio_metrics(
            simulate_portfolio(window, weights, ASSET_CLASSES, annual_rebalance=True)
        )

    # Full scan: drawdown is not guaranteed monotonic in the risky share.
    chosen_steps, chosen_metrics, within_caps = 0, measure(0), False
    for steps in range(max_steps + 1):
        metrics = chosen_metrics if steps == 0 else measure(steps)
        if is_feasible(metrics, volatility_cap, drawdown_cap):
            chosen_steps, chosen_metrics, within_caps = steps, metrics, True

    weights = _weights_for(chosen_steps, allowed)
    first, last = window[0].date, window[-1].date
    provenance = {
        "method": METHOD_ID,
        "risky_share": chosen_steps / 100.0,
        "risky_share_step": 0.01,
        "risky_share_ceiling": max_steps / 100.0,
        "risky_classes": allowed,
        "excluded_classes": excluded,
        "fixed_income_minimum": fixed_income_minimum,
        "window": {
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "requested_years": window_years,
            "years_used": round((last - first).days / 365.25, 4),
            "full_history": short_history,
            "observations": len(window),
        },
        "risk": {
            "annualized_volatility": chosen_metrics.annualized_volatility,
            "volatility_cap": volatility_cap,
            "max_drawdown": abs(chosen_metrics.max_drawdown),
            "drawdown_cap": drawdown_cap,
            "within_caps": within_caps,
        },
        "class_series": class_series(metadata),
    }
    return {
        "status": "ok",
        "method": METHOD_ID,
        "weights": weights,
        "metrics": asdict(chosen_metrics),
        "provenance": provenance,
    }
