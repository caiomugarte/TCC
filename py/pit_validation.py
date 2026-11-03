"""Point-in-time audit and walk-forward validation of the class allocation.

Two operator commands, run from the repository root:

    python3 py/pit_validation.py audit --protocol PATH --out PATH
    python3 py/pit_validation.py report --protocol PATH --audit PATH --out-dir DIR

``audit`` classifies every series the class allocation reads by the PIT rule
(AD-018). ``report`` runs the walk-forward protocol (AD-019) with the
production decision function only when the audit qualifies every series, and
writes ``class-allocation-oos.json`` and ``class-allocation-oos.md``. Every
rule that changes an outcome is a pure function here; the CLI only reads and
writes files. Exit codes: 0 written, 2 invalid protocol or arguments, 3 a path
does not exist, 4 an integrity violation.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import ALLOCATION_PROFILE_ANCHORS, ASSET_CLASSES  # noqa: E402
from allocation_data import (  # noqa: E402
    SnapshotError,
    load_snapshot_bundle,
    read_levels_csv,
    read_metadata,
)
from allocation_profiles import build_anchor_profiles, interpolate_profile  # noqa: E402
from core.allocation import DailyReturn, PortfolioPath, portfolio_metrics  # noqa: E402
from pipelines.equal_weight_risky import run_equal_weight_risky  # noqa: E402
from snapshot_manifest import ALLOCATION_FILES, file_sha256  # noqa: E402

QUALIFYING_KINDS = ("exchange_index_close", "official_fixing", "official_rate", "spot_close")
REVISABLE_KINDS = ("vendor_adjusted_close", "vendor_composite_close")
REQUIRED_BENCHMARKS = ("cdi", "ibov", "naive_equal_weight")
REQUIRED_SERIES = ("ibovespa", "ifix", "sp500_total_return_usd", "di", "btc_usd", "ptax")
# The file each series is read from by load_snapshot_bundle; the audit must hash the same one.
LOADER_FILES = {Path(filename).stem: filename for filename in ALLOCATION_FILES.values()}
IDENTIFIER_KEYS = ("symbol", "index", "series", "currency")
LAYERS = (
    "class_allocation",
    "stock_selection",
    "fii_selection",
    "basic_recommendation",
    "premium_recommendation",
)
SELECTION_LAYERS = ("stock_selection", "fii_selection")
STATISTICS = ("min", "p25", "median", "p75", "max")
METRICS = ("annualized_return", "annualized_volatility", "max_drawdown")
SERIES_ORDER = ("strategy",) + REQUIRED_BENCHMARKS

MIN_DAYS_APART = 30
MIN_SHARED_DATES = 250
MAX_REL_DIFF = 1e-9
MIN_PATHS = 2

BENCHMARK_WEIGHTS = {
    "cdi": {name: (1.0 if name == "fixed_income" else 0.0) for name in ASSET_CLASSES},
    "ibov": {name: (1.0 if name == "brazilian_stocks" else 0.0) for name in ASSET_CLASSES},
    "naive_equal_weight": {name: 1.0 / len(ASSET_CLASSES) for name in ASSET_CLASSES},
}

AUDIT_FILE = "pit-audit.json"
REPORT_JSON = "class-allocation-oos.json"
REPORT_MD = "class-allocation-oos.md"


class PITValidationError(Exception):
    """A failure the CLI reports as one stderr line and an exit code."""

    exit_code = 1

    def __init__(self, field_name: str, reason: str):
        super().__init__(f"{field_name}: {reason}")
        self.field_name = field_name
        self.reason = reason


class ProtocolError(PITValidationError):
    exit_code = 2


class MissingPathError(PITValidationError):
    exit_code = 3


class IntegrityError(PITValidationError):
    exit_code = 4


class PITLeakError(IntegrityError):
    """A decision received a row dated after its decision date."""

    def __init__(self, decision_date: date, offending_date: date):
        super().__init__(
            "decision",
            f"decision {decision_date.isoformat()} received a row dated "
            f"{offending_date.isoformat()}",
        )
        self.decision_date = decision_date
        self.offending_date = offending_date


# --------------------------------------------------------------------------- protocol


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _non_negative_finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and value >= 0


def validate_protocol(protocol: Any) -> Mapping[str, Any]:
    """Raise ``ProtocolError`` naming the first field that is not declared."""

    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol", "must be a JSON object")
    for key in ("protocol_id", "snapshot_dir"):
        if not isinstance(protocol.get(key), str) or not protocol[key]:
            raise ProtocolError(key, "must be a non-empty string")
    retrievals = protocol.get("retrievals")
    if (
        not isinstance(retrievals, list)
        or not retrievals
        or not all(isinstance(item, str) and item for item in retrievals)
    ):
        raise ProtocolError("retrievals", "must be a non-empty list of directories")
    if protocol["snapshot_dir"] not in retrievals:
        raise ProtocolError("retrievals", "must include snapshot_dir")

    series = protocol.get("series")
    if not isinstance(series, Mapping) or set(series) != set(REQUIRED_SERIES):
        raise ProtocolError("series", f"must name exactly {list(REQUIRED_SERIES)}")
    for name, filename in series.items():
        if not isinstance(filename, str) or not filename:
            raise ProtocolError(f"series.{name}", "must be a file name")
        if filename != LOADER_FILES.get(name):
            raise ProtocolError(
                f"series.{name}",
                f"must be {LOADER_FILES.get(name)!r}, the file the report loads",
            )
    kinds = protocol.get("series_kind")
    if not isinstance(kinds, Mapping):
        raise ProtocolError("series_kind", "must be an object")
    for name in REQUIRED_SERIES:
        kind = kinds.get(name)
        if kind not in QUALIFYING_KINDS + REVISABLE_KINDS:
            raise ProtocolError(f"series_kind.{name}", f"unknown kind {kind!r}")

    benchmarks = protocol.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ProtocolError("benchmarks", "must be a list")
    for name in REQUIRED_BENCHMARKS:
        if name not in benchmarks:
            raise ProtocolError("benchmarks", f"missing {name}")
    for name in benchmarks:
        if name not in REQUIRED_BENCHMARKS:
            raise ProtocolError("benchmarks", f"unknown benchmark {name!r}")

    cost = protocol.get("cost")
    cost = cost if isinstance(cost, Mapping) else {}
    if not _non_negative_finite(cost.get("cost_bps")):
        raise ProtocolError("cost.cost_bps", "must be a finite non-negative number")
    for key in ("basis", "taxes"):
        value = cost.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"cost.{key}", "must be a non-empty string")
    if "sensitivity_bps" in cost:
        values = cost["sensitivity_bps"]
        if (
            not isinstance(values, list)
            or not values
            or not all(_non_negative_finite(value) for value in values)
        ):
            raise ProtocolError(
                "cost.sensitivity_bps",
                "must be a non-empty list of finite non-negative numbers",
            )

    decision = protocol.get("decision")
    if not isinstance(decision, Mapping):
        raise ProtocolError("decision", "must be an object")
    if decision.get("execution_lag_common_dates") != 1:
        raise ProtocolError("decision.execution_lag_common_dates", "must be 1")
    for key in ("min_lookback_years", "max_lookback_years", "test_months"):
        value = decision.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ProtocolError(f"decision.{key}", "must be a positive integer")
    offsets = decision.get("offset_months")
    if (
        not isinstance(offsets, list)
        or not offsets
        or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ProtocolError("decision.offset_months", "must be distinct non-negative integers")

    profiles = protocol.get("profiles")
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(name in ALLOCATION_PROFILE_ANCHORS for name in profiles)
    ):
        raise ProtocolError("profiles", f"must be anchors from {sorted(ALLOCATION_PROFILE_ANCHORS)}")

    method = protocol.get("method")
    if not isinstance(method, Mapping):
        raise ProtocolError("method", "must be an object")
    try:
        date.fromisoformat(str(method.get("adopted_on")))
    except ValueError as exc:
        raise ProtocolError("method.adopted_on", "must be an ISO date") from exc

    layers = protocol.get("layers")
    if not isinstance(layers, Mapping):
        raise ProtocolError("layers", "must be an object")
    for name in SELECTION_LAYERS:
        layer = layers.get(name)
        if not isinstance(layer, Mapping) or not isinstance(layer.get("missing"), list):
            raise ProtocolError(f"layers.{name}", "must list its missing data")
    return protocol


def load_json(path: Path, label: str) -> Tuple[Any, bytes]:
    if not path.is_file():
        raise MissingPathError(label, f"path does not exist: {path}")
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        error = ProtocolError if label == "--protocol" else IntegrityError
        raise error(label, f"invalid JSON in {path}") from exc


def require_directories(protocol: Mapping[str, Any]) -> None:
    for retrieval in [protocol["snapshot_dir"], *protocol["retrievals"]]:
        if not Path(retrieval).is_dir():
            raise MissingPathError("retrievals", f"path does not exist: {retrieval}")


# --------------------------------------------------------------------------- audit


@dataclass(frozen=True)
class RetrievalFile:
    """One retrieval's copy of one series; ``levels`` is None when absent."""

    retrieval: str
    sha256: Optional[str]
    provider: Optional[str]
    identifier: Mapping[str, Any]
    retrieved_at: str
    levels: Optional[Mapping[date, float]]


def _retrieved_date(raw: str) -> date:
    return date.fromisoformat(str(raw)[:10])


def compare_levels(
    snapshot: Mapping[date, float],
    other: Mapping[date, float],
) -> Dict[str, Any]:
    """Shared dates, completeness and the worst daily-factor disagreement."""

    shared = sorted(set(snapshot).intersection(other))
    result: Dict[str, Any] = {
        "shared_dates": len(shared),
        "first_shared_date": shared[0].isoformat() if shared else None,
        "last_shared_date": shared[-1].isoformat() if shared else None,
        "overlap_complete": False,
        "max_rel_diff_daily_factor": None,
    }
    if not shared:
        return result
    inside = [d for d in snapshot if shared[0] <= d <= shared[-1]]
    result["overlap_complete"] = len(inside) == len(shared)
    worst = 0.0
    finite = True
    for previous, current in zip(shared, shared[1:]):
        f_snap = snapshot[current] / snapshot[previous]
        f_other = other[current] / other[previous]
        if not all(math.isfinite(f) and f != 0.0 for f in (f_snap, f_other)):
            finite = False
            break
        diff = abs(f_snap / f_other - 1.0)
        if not math.isfinite(diff):
            finite = False
            break
        worst = max(worst, diff)
    # A zero or non-finite factor cannot show agreement: None reads as `values_revised`.
    result["max_rel_diff_daily_factor"] = worst if len(shared) > 1 and finite else None
    return result


def audit_series(
    name: str,
    kind: str,
    snapshot_dir: str,
    files: Sequence[RetrievalFile],
) -> Dict[str, Any]:
    """Classify one series by the door-3 rule (AD-018)."""

    present = [item for item in files if item.levels is not None]
    groups: Dict[str, List[RetrievalFile]] = {}
    for item in present:
        groups.setdefault(str(item.sha256), []).append(item)

    retrieval_records = []
    for item in files:
        if item.levels is None:
            retrieval_records.append(
                {"retrieval": item.retrieval, "present": False, "reasons": ["series_absent"]}
            )
            continue
        dates = sorted(item.levels)
        retrieval_records.append(
            {
                "retrieval": item.retrieval,
                "present": True,
                "provider": item.provider,
                "identifier": dict(item.identifier),
                "retrieved_at": item.retrieved_at,
                "sha256": item.sha256,
                "same_bytes_as": [
                    other.retrieval
                    for other in groups[str(item.sha256)]
                    if other.retrieval != item.retrieval
                ],
                "first_date": dates[0].isoformat() if dates else None,
                "last_date": dates[-1].isoformat() if dates else None,
                "rows": len(dates),
            }
        )

    def group_record(members: Sequence[RetrievalFile]) -> Dict[str, Any]:
        latest = max(members, key=lambda item: _retrieved_date(item.retrieved_at))
        return {
            "members": [item.retrieval for item in members],
            "sha256": members[0].sha256,
            "retrieved_at": latest.retrieved_at,
        }

    record: Dict[str, Any] = {
        "series": name,
        "series_kind": kind,
        "retrievals": retrieval_records,
        "groups": [group_record(members) for members in groups.values()],
        "pairs": [],
        "status": "PIT_MISSING",
        "reasons": [],
        "evidence": None,
        "days_apart": None,
        "verified_range": None,
        "unverified_dates": None,
    }

    snapshot_item = next((item for item in present if item.retrieval == snapshot_dir), None)
    reasons: List[str] = []
    if kind not in QUALIFYING_KINDS:
        reasons.append("kind_revisable")
    if snapshot_item is None:
        reasons.append("series_absent")
        record["reasons"] = sorted(set(reasons))
        return record

    snapshot_members = groups[str(snapshot_item.sha256)]
    snapshot_group = group_record(snapshot_members)
    snapshot_date = _retrieved_date(snapshot_group["retrieved_at"])
    qualifying = []
    for sha, members in groups.items():
        if sha == str(snapshot_item.sha256):
            continue
        other_group = group_record(members)
        other = members[0]
        days_apart = abs((_retrieved_date(other_group["retrieved_at"]) - snapshot_date).days)
        comparison = compare_levels(snapshot_item.levels, other.levels)
        pair_reasons = []
        if other.provider != snapshot_item.provider or dict(other.identifier) != dict(
            snapshot_item.identifier
        ):
            pair_reasons.append("provider_mismatch")
        if days_apart < MIN_DAYS_APART:
            pair_reasons.append("no_retrieval_pair_30d")
        if comparison["shared_dates"] < MIN_SHARED_DATES:
            pair_reasons.append("overlap_below_250")
        if not comparison["overlap_complete"]:
            pair_reasons.append("overlap_incomplete")
        worst = comparison["max_rel_diff_daily_factor"]
        if worst is None or worst > MAX_REL_DIFF:
            pair_reasons.append("values_revised")
        pair = {
            "snapshot_group": snapshot_group["members"],
            "snapshot_retrieved_at": snapshot_group["retrieved_at"],
            "other_group": other_group["members"],
            "other_retrieved_at": other_group["retrieved_at"],
            "days_apart": days_apart,
            **comparison,
            "qualifies": not pair_reasons,
            "reasons": pair_reasons,
        }
        record["pairs"].append(pair)
        if pair_reasons:
            reasons.extend(pair_reasons)
        else:
            qualifying.append(pair)
    if not record["pairs"]:
        reasons.append("no_retrieval_pair_30d")
    if any(item.levels is None for item in files):
        absent_reason = ["series_absent"]
    else:
        absent_reason = []

    if qualifying:
        best = max(qualifying, key=lambda pair: (pair["shared_dates"], pair["days_apart"]))
        start = date.fromisoformat(best["first_shared_date"])
        end = date.fromisoformat(best["last_shared_date"])
        outside = sorted(d for d in snapshot_item.levels if d < start or d > end)
        record["days_apart"] = best["days_apart"]
        record["verified_range"] = {"start": start.isoformat(), "end": end.isoformat()}
        record["unverified_dates"] = {
            "count": len(outside),
            "first": outside[0].isoformat() if outside else None,
            "last": outside[-1].isoformat() if outside else None,
        }
    if qualifying and kind in QUALIFYING_KINDS:
        record["status"] = "PIT_QUALIFIED"
        record["evidence"] = "non_revision"
        record["reasons"] = []
    else:
        record["reasons"] = sorted(set(reasons + absent_reason))
    return record


def _source_identity(metadata: Mapping[str, Any], name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    sources = metadata.get("sources") if isinstance(metadata, Mapping) else None
    source = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(source, Mapping):
        return None, {}
    identifier = {key: source[key] for key in IDENTIFIER_KEYS if key in source}
    return source.get("provider"), identifier


def read_retrieval_file(retrieval: str, name: str, filename: str) -> RetrievalFile:
    directory = Path(retrieval)
    try:
        metadata = read_metadata(directory / "metadata.json")
    except SnapshotError as exc:
        raise IntegrityError(f"retrievals.{retrieval}", str(exc)) from exc
    provider, identifier = _source_identity(metadata, name)
    path = directory / filename
    if not path.is_file():
        return RetrievalFile(retrieval, None, provider, identifier, str(metadata["retrieved_at"]), None)
    try:
        series = read_levels_csv(path)
    except SnapshotError as exc:
        raise IntegrityError(f"series.{name}", str(exc)) from exc
    if len(series) != 1:
        raise IntegrityError(f"series.{name}", f"{path} must hold one level column")
    return RetrievalFile(
        retrieval,
        file_sha256(path),
        provider,
        identifier,
        str(metadata["retrieved_at"]),
        next(iter(series.values())),
    )


def build_audit(protocol: Mapping[str, Any], protocol_bytes: bytes) -> Dict[str, Any]:
    series_records = {}
    for name in REQUIRED_SERIES:
        filename = protocol["series"][name]
        files = [read_retrieval_file(retrieval, name, filename) for retrieval in protocol["retrievals"]]
        record = audit_series(name, protocol["series_kind"][name], protocol["snapshot_dir"], files)
        record["file"] = filename
        series_records[name] = record
    layers = {}
    for name in SELECTION_LAYERS:
        layer = protocol["layers"][name]
        layers[name] = {
            "status": "PIT_MISSING",
            "missing": list(layer["missing"]),
            "excluded_lookahead": dict(layer.get("excluded_lookahead") or {}),
        }
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "snapshot_dir": protocol["snapshot_dir"],
        "thresholds": {
            "min_days_apart": MIN_DAYS_APART,
            "min_shared_dates": MIN_SHARED_DATES,
            "max_rel_diff_daily_factor": MAX_REL_DIFF,
        },
        "series": series_records,
        "layers": layers,
    }


# --------------------------------------------------------------------------- run period


def run_period_from_audit(audit: Mapping[str, Any]) -> Optional[Tuple[date, date]]:
    """Intersection of the verified ranges; None when a series has none."""

    starts, ends = [], []
    for name in REQUIRED_SERIES:
        verified = audit["series"][name].get("verified_range")
        if not verified:
            return None
        starts.append(date.fromisoformat(verified["start"]))
        ends.append(date.fromisoformat(verified["end"]))
    start, end = max(starts), min(ends)
    return (start, end) if start <= end else None


def trim_rows(
    rows: Sequence[DailyReturn],
    start: date,
    end: date,
) -> Tuple[Tuple[DailyReturn, ...], int]:
    """Keep common rows in ``[start, end]``; the first one earns nothing."""

    kept = [row for row in rows if start <= row.date <= end]
    dropped = len(rows) - len(kept)
    if not kept:
        return (), dropped
    first = DailyReturn(kept[0].date, {name: 0.0 for name in kept[0].returns})
    return (first, *kept[1:]), dropped


# --------------------------------------------------------------------------- schedule


def add_months(value: date, months: int) -> date:
    index = value.month - 1 + months
    year, month = value.year + index // 12, index % 12 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Schedule:
    offset: int
    decisions: Tuple[date, ...]
    executions: Tuple[date, ...]
    windows: Tuple[Tuple[date, date], ...]

    @property
    def period_start(self) -> Optional[date]:
        return self.executions[0] if self.executions else None

    @property
    def period_end(self) -> Optional[date]:
        return self.executions[-1] if self.executions else None


def anchor_executions(
    dates: Sequence[date],
    run_start: date,
    run_end: date,
    min_lookback_years: int,
    offset: int,
    step_months: int,
) -> List[Tuple[date, date]]:
    """(decision_j, exec_j) for every anchor <= run_end that has an execution."""

    pairs = []
    j = 0
    while True:
        anchor = add_months(run_start, 12 * min_lookback_years + offset + step_months * j)
        if anchor > run_end:
            return pairs
        decision = max((d for d in dates if d <= anchor), default=None)
        execution = min((d for d in dates if decision is not None and d > decision), default=None)
        if decision is None or execution is None or execution > run_end:
            return pairs
        pairs.append((decision, execution))
        j += 1


def build_schedule(
    dates: Sequence[date],
    run_start: date,
    run_end: date,
    min_lookback_years: int,
    offset: int,
    step_months: int,
) -> Schedule:
    """Door 4: windows ``(exec_j, exec_{j+1}]`` with exec_{j+1} <= run_end."""

    pairs = anchor_executions(dates, run_start, run_end, min_lookback_years, offset, step_months)
    if len(pairs) < 2:
        executions = (pairs[0][1],) if pairs else ()
        return Schedule(offset, (), executions, ())
    decisions = tuple(decision for decision, _ in pairs[:-1])
    executions = tuple(execution for _, execution in pairs)
    windows = tuple(zip(executions[:-1], executions[1:]))
    return Schedule(offset, decisions, executions, windows)


def check_schedule(
    schedule: Schedule,
    dates: Sequence[date],
    run_start: date,
    run_end: date,
    min_lookback_years: int,
    step_months: int,
) -> None:
    """Raise ``IntegrityError`` on an overlap, a gap or a dropped window."""

    label = f"schedule.offset_{schedule.offset}"
    pairs = anchor_executions(
        dates, run_start, run_end, min_lookback_years, schedule.offset, step_months
    )
    expected_end = pairs[-1][1] if len(pairs) >= 2 else None
    if not schedule.windows:
        if expected_end is not None:
            raise IntegrityError(label, f"no window, expected a path ending {expected_end}")
        return
    if len(schedule.decisions) != len(schedule.windows) or len(schedule.executions) != len(
        schedule.windows
    ) + 1:
        raise IntegrityError(label, "decisions, executions and windows do not line up")
    start, end = schedule.period_start, schedule.period_end
    if end != expected_end:
        raise IntegrityError(
            label, f"period_end {end} differs from the last execution {expected_end} (dropped window)"
        )
    seen: Dict[date, int] = {}
    for j, (lower, upper) in enumerate(schedule.windows):
        if lower >= upper:
            raise IntegrityError(label, f"window {j} ({lower}, {upper}] is empty")
        if j and lower != schedule.windows[j - 1][1]:
            kind = "overlap" if lower < schedule.windows[j - 1][1] else "gap"
            raise IntegrityError(
                label, f"{kind} between window {j - 1} and window {j} at {lower}"
            )
        for d in dates:
            if lower < d <= upper:
                seen[d] = seen.get(d, 0) + 1
    if schedule.windows[0][0] != start:
        raise IntegrityError(label, f"first window starts on {schedule.windows[0][0]}, not {start}")
    expected = [d for d in dates if start < d <= end]
    twice = [d for d, count in seen.items() if count > 1]
    if twice:
        raise IntegrityError(label, f"overlap: {min(twice)} falls in two windows")
    missing = [d for d in expected if d not in seen]
    if missing or len(seen) != len(expected):
        raise IntegrityError(label, f"gap: {min(missing) if missing else start} is in no window")


# --------------------------------------------------------------------------- decisions

Strategy = Callable[..., Mapping[str, Any]]


def default_strategy(rows: Sequence[DailyReturn], *, allocation_config: Mapping[str, Any]):
    return run_equal_weight_risky(rows, allocation_config=allocation_config)


def rows_for_decision(rows: Sequence[DailyReturn], decision: date) -> Tuple[DailyReturn, ...]:
    return tuple(row for row in rows if row.date <= decision)


def decide(
    rows: Sequence[DailyReturn],
    decision: date,
    strategy: Strategy,
    allocation_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Call the strategy on rows dated <= ``decision`` and nothing else."""

    visible = rows_for_decision(rows, decision)
    for row in visible:
        if row.date > decision:
            raise PITLeakError(decision, row.date)
    if not visible or max(row.date for row in visible) != decision:
        last = max((row.date for row in visible), default=None)
        raise IntegrityError(
            "decision", f"decision {decision.isoformat()} must see its own row; last row {last}"
        )
    result = strategy(visible, allocation_config=allocation_config)
    window = (result.get("provenance") or {}).get("window") or {}
    return {
        "weights": {name: float(result["weights"][name]) for name in ASSET_CLASSES},
        "years_used": window.get("years_used"),
        "full_history": window.get("full_history"),
    }


def profile_config(profile: str) -> Dict[str, Any]:
    anchors = build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS)
    return interpolate_profile(
        float(ALLOCATION_PROFILE_ANCHORS[profile]["score"]),
        anchors,
        name=profile,
        calibration_source="Basic generic allocation anchor",
        calibration_inputs={"generic_profile": profile},
    ).as_dict()


# --------------------------------------------------------------------------- simulation


def execution_factor(
    cost_bps: float,
    held: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    turnover = sum(abs(target[name] - held.get(name, 0.0)) for name in ASSET_CLASSES)
    return 1.0 - cost_bps / 10000.0 * turnover


def simulate_path(
    rows: Sequence[DailyReturn],
    schedule: Schedule,
    weights: Sequence[Mapping[str, float]],
    cost_bps: float,
) -> PortfolioPath:
    """Value from 1.0 before exec_0; each exec_j row is earned by the old weights."""

    if len(weights) != len(schedule.windows):
        raise IntegrityError(f"schedule.offset_{schedule.offset}", "one weight vector per window")
    start, end = schedule.period_start, schedule.period_end
    executions = {execution: j for j, execution in enumerate(schedule.executions)}
    path_rows = [row for row in rows if start <= row.date <= end]
    value = execution_factor(cost_bps, {}, weights[0])
    holdings = [value * weights[0][name] for name in ASSET_CLASSES]
    dates, values, daily = [path_rows[0].date], [value], [value - 1.0]
    for row in path_rows[1:]:
        previous = value
        holdings = [h * (1.0 + row.returns[name]) for h, name in zip(holdings, ASSET_CLASSES)]
        value = sum(holdings)
        j = executions.get(row.date)
        if j is not None and j < len(weights):
            held = {name: h / value for h, name in zip(holdings, ASSET_CLASSES)}
            value *= execution_factor(cost_bps, held, weights[j])
            holdings = [value * weights[j][name] for name in ASSET_CLASSES]
        dates.append(row.date)
        values.append(value)
        daily.append(value / previous - 1.0)
    return PortfolioPath(tuple(dates), tuple(values), tuple(daily), tuple(schedule.executions))


def path_metrics(path: PortfolioPath) -> Dict[str, float]:
    metrics = portfolio_metrics(path)
    return {
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "max_drawdown": metrics.max_drawdown,
        "total_return": metrics.total_return,
    }


def series_paths_metrics(
    rows: Sequence[DailyReturn],
    schedule: Schedule,
    strategy_weights: Sequence[Mapping[str, float]],
    cost_bps: float,
) -> Dict[str, Dict[str, float]]:
    result = {"strategy": path_metrics(simulate_path(rows, schedule, strategy_weights, cost_bps))}
    for name in REQUIRED_BENCHMARKS:
        weights = [BENCHMARK_WEIGHTS[name]] * len(schedule.windows)
        result[name] = path_metrics(simulate_path(rows, schedule, weights, cost_bps))
    return result


# --------------------------------------------------------------------------- distribution


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    p25, median, p75 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "paths": len(ordered),
        "min": ordered[0],
        "p25": p25,
        "median": median,
        "p75": p75,
        "max": ordered[-1],
    }


def summarize_excess(values: Sequence[float]) -> Dict[str, Any]:
    result = summarize(values)
    result["share_positive"] = sum(1 for value in values if value > 0) / len(values)
    return result


def distribution(path_records: Sequence[Mapping[str, Any]], key: str = "metrics") -> Dict[str, Any]:
    per_series = {
        series: {
            metric: summarize([path[key][series][metric] for path in path_records])
            for metric in METRICS
        }
        for series in SERIES_ORDER
    }
    excess = {
        benchmark: summarize_excess(
            [
                path[key]["strategy"]["annualized_return"]
                - path[key][benchmark]["annualized_return"]
                for path in path_records
            ]
        )
        for benchmark in REQUIRED_BENCHMARKS
    }
    return {"series": per_series, "excess_annualized_return": excess}


def sensitivity_row(cost_bps: float, metrics_per_path: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "cost_bps": cost_bps,
        "annualized_return": {
            series: summarize([path[series]["annualized_return"] for path in metrics_per_path])
            for series in SERIES_ORDER
        },
        "excess_annualized_return": {
            benchmark: summarize_excess(
                [
                    path["strategy"]["annualized_return"] - path[benchmark]["annualized_return"]
                    for path in metrics_per_path
                ]
            )
            for benchmark in REQUIRED_BENCHMARKS
        },
    }


# --------------------------------------------------------------------------- verdict


def class_verdict(
    audit: Mapping[str, Any],
    paths_per_profile: Mapping[str, int],
) -> Tuple[str, List[str]]:
    missing = []
    for name in REQUIRED_SERIES:
        record = audit["series"][name]
        if record["status"] != "PIT_QUALIFIED":
            missing.extend(f"{name}: {reason}" for reason in record["reasons"] or ["unqualified"])
    if not paths_per_profile or any(count < MIN_PATHS for count in paths_per_profile.values()):
        missing.append("insufficient_history")
    return ("UNVERIFIABLE" if missing else "VALIDATED"), missing


def selection_overlap(
    run_end: Optional[date],
    adopted_on: date,
    first_executions: Sequence[date],
) -> str:
    if run_end is not None and run_end < adopted_on:
        return "full"
    if not first_executions or min(first_executions) >= adopted_on:
        return "none"
    return "partial"


def verdicts_for(class_status: str) -> Dict[str, str]:
    return {
        "class_allocation": class_status,
        "stock_selection": "UNVERIFIABLE",
        "fii_selection": "UNVERIFIABLE",
        "basic_recommendation": class_status,
        "premium_recommendation": "UNVERIFIABLE",
    }


# --------------------------------------------------------------------------- report


def check_audit_protocol(protocol: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    """Exit 4 when the audit was computed under other audit inputs than this protocol's."""

    def mismatch(field: str, recorded: Any, current: Any) -> None:
        if recorded != current:
            raise IntegrityError(
                f"audit.{field}",
                f"audit recorded {json.dumps(recorded, sort_keys=True)}, "
                f"protocol has {json.dumps(current, sort_keys=True)}",
            )

    mismatch("snapshot_dir", audit.get("snapshot_dir"), protocol["snapshot_dir"])
    for name in REQUIRED_SERIES:
        record = audit["series"].get(name)
        if not isinstance(record, Mapping):
            raise IntegrityError(f"audit.{name}", "has no series record")
        mismatch(f"series.{name}", record.get("file"), protocol["series"][name])
        mismatch(f"series_kind.{name}", record.get("series_kind"), protocol["series_kind"][name])
        recorded = [entry.get("retrieval") for entry in record.get("retrievals") or []]
        mismatch(f"retrievals.{name}", recorded, list(protocol["retrievals"]))
    layers = audit.get("layers") if isinstance(audit.get("layers"), Mapping) else {}
    for name in SELECTION_LAYERS:
        layer = protocol["layers"][name]
        recorded = layers.get(name) if isinstance(layers.get(name), Mapping) else {}
        mismatch(f"layers.{name}.missing", recorded.get("missing"), list(layer["missing"]))
        mismatch(
            f"layers.{name}.excluded_lookahead",
            recorded.get("excluded_lookahead"),
            dict(layer.get("excluded_lookahead") or {}),
        )


def check_audit_matches(protocol: Mapping[str, Any], audit: Mapping[str, Any]) -> Dict[str, str]:
    """Return the snapshot file hashes; exit 4 when the audit is about other files."""

    files = {}
    series = audit.get("series") if isinstance(audit, Mapping) else None
    if not isinstance(series, Mapping):
        raise IntegrityError("audit", "has no series records")
    check_audit_protocol(protocol, audit)
    for name in REQUIRED_SERIES:
        filename = protocol["series"][name]
        path = Path(protocol["snapshot_dir"]) / filename
        if not path.is_file():
            raise MissingPathError(f"series.{name}", f"path does not exist: {path}")
        current = file_sha256(path)
        files[filename] = current
        record = series.get(name)
        recorded = None
        if isinstance(record, Mapping):
            for entry in record.get("retrievals") or []:
                if entry.get("retrieval") == protocol["snapshot_dir"]:
                    recorded = entry.get("sha256")
        if recorded != current:
            raise IntegrityError(
                f"audit.{name}",
                f"audit sha256 {recorded} differs from {path} ({current})",
            )
    return files


def check_audit_current(
    protocol: Mapping[str, Any], protocol_bytes: bytes, audit: Mapping[str, Any]
) -> None:
    """Exit 4 when a fresh audit of the retrievals differs from the audit document.

    ``protocol_sha256`` is left out: a protocol edit that leaves the audit inputs
    unchanged needs no re-audit (``check_audit_protocol`` names the inputs).
    """

    fresh = json.loads(dump_json(build_audit(protocol, protocol_bytes)))
    recorded = audit.get("series") if isinstance(audit.get("series"), Mapping) else {}
    for name in REQUIRED_SERIES:
        if fresh["series"][name] != recorded.get(name):
            raise IntegrityError(
                f"audit.{name}",
                "differs from a fresh audit of the current files; re-run audit",
            )
    for key in sorted(set(fresh) | set(audit)):
        if key != "protocol_sha256" and fresh.get(key) != audit.get(key):
            raise IntegrityError(
                f"audit.{key}",
                "differs from a fresh audit of the current files; re-run audit",
            )


def run_report(
    protocol: Mapping[str, Any],
    protocol_bytes: bytes,
    audit: Mapping[str, Any],
    audit_bytes: bytes,
    strategy: Optional[Strategy] = None,
) -> Dict[str, Any]:
    """Compute the report document; raises before anything is written."""

    strategy = strategy or default_strategy
    snapshot_dir = Path(protocol["snapshot_dir"])
    files = check_audit_matches(protocol, audit)
    check_audit_current(protocol, protocol_bytes, audit)
    try:
        bundle = load_snapshot_bundle(snapshot_dir)
    except SnapshotError as exc:
        raise IntegrityError("snapshot_dir", str(exc)) from exc
    cutoff_raw = str(bundle.metadata.get("cutoff_date"))
    last_common = bundle.rows[-1].date
    if cutoff_raw != last_common.isoformat():
        raise IntegrityError(
            "snapshot.cutoff_date",
            f"cutoff_date {cutoff_raw} differs from the last common date {last_common.isoformat()}",
        )

    decision = protocol["decision"]
    lookback = decision["min_lookback_years"]
    step = decision["test_months"]
    adopted_on = date.fromisoformat(protocol["method"]["adopted_on"])
    period = run_period_from_audit(audit)
    rows: Tuple[DailyReturn, ...] = ()
    unverified = len(bundle.rows)
    schedules: List[Schedule] = []
    if period is not None:
        rows, unverified = trim_rows(bundle.rows, *period)
    if rows:
        run_start, run_end = rows[0].date, rows[-1].date
        dates = [row.date for row in rows]
        for offset in decision["offset_months"]:
            schedule = build_schedule(dates, run_start, run_end, lookback, offset, step)
            check_schedule(schedule, dates, run_start, run_end, lookback, step)
            schedules.append(schedule)
    else:
        run_start = run_end = None

    with_windows = [schedule for schedule in schedules if schedule.windows]
    paths_per_profile = {profile: len(with_windows) for profile in protocol["profiles"]}
    class_status, class_missing = class_verdict(audit, paths_per_profile)
    overlap = selection_overlap(
        run_end, adopted_on, [schedule.executions[0] for schedule in schedules if schedule.executions]
    )
    missing = {
        "class_allocation": class_missing,
        "basic_recommendation": list(class_missing),
        "stock_selection": list(audit["layers"]["stock_selection"]["missing"]),
        "fii_selection": list(audit["layers"]["fii_selection"]["missing"]),
        "premium_recommendation": [
            "stock_selection: UNVERIFIABLE",
            "fii_selection: UNVERIFIABLE",
        ]
        + [f"class_allocation: {item}" for item in class_missing],
    }
    report: Dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "environment": {"python": platform.python_version()},
        "snapshot": {
            "dir": protocol["snapshot_dir"],
            "cutoff_date": cutoff_raw,
            "files": files,
        },
        "run_period": {
            "start": run_start.isoformat() if run_start else None,
            "end": run_end.isoformat() if run_end else None,
            "unverified_common_rows": unverified,
        },
        "evidence": {"class_allocation": "non_revision"},
        "selection_overlap": {"class_allocation": overlap},
        "verdicts": verdicts_for(class_status),
        "missing": missing,
    }
    if class_status != "VALIDATED":
        return report

    cost = protocol["cost"]
    headline_bps = cost["cost_bps"]
    sensitivity = cost.get("sensitivity_bps")
    profiles: Dict[str, Any] = {}
    for profile in protocol["profiles"]:
        config = profile_config(profile)
        path_records = []
        sensitivity_metrics: Dict[int, List[Dict[str, Any]]] = {}
        for schedule in with_windows:
            decided = [decide(rows, t, strategy, config) for t in schedule.decisions]
            weights = [item["weights"] for item in decided]
            path_records.append(
                {
                    "offset": schedule.offset,
                    "period_start": schedule.period_start.isoformat(),
                    "period_end": schedule.period_end.isoformat(),
                    "decision_dates": [d.isoformat() for d in schedule.decisions],
                    "execution_dates": [d.isoformat() for d in schedule.executions],
                    "windows": [
                        {
                            "decision_date": schedule.decisions[j].isoformat(),
                            "execution_date": lower.isoformat(),
                            "end_date": upper.isoformat(),
                            "weights": decided[j]["weights"],
                            "years_used": decided[j]["years_used"],
                            "full_history": decided[j]["full_history"],
                        }
                        for j, (lower, upper) in enumerate(schedule.windows)
                    ],
                    "metrics": series_paths_metrics(rows, schedule, weights, headline_bps),
                }
            )
            for index, bps in enumerate(sensitivity or []):
                sensitivity_metrics.setdefault(index, []).append(
                    series_paths_metrics(rows, schedule, weights, bps)
                )
        profile_report: Dict[str, Any] = {
            "paths": path_records,
            "distribution": distribution(path_records),
        }
        if sensitivity is not None:
            profile_report["cost_sensitivity"] = [
                sensitivity_row(bps, sensitivity_metrics[index])
                for index, bps in enumerate(sensitivity)
            ]
        profiles[profile] = profile_report
    report["profiles"] = profiles
    return report


# --------------------------------------------------------------------------- markdown


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%".replace(".", ",")


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(f"`{item}`" for item in value) or "-"
    return str(value)


def regeneration_commands() -> List[str]:
    return [
        "python3 py/pit_validation.py audit --protocol docs/validation/protocol.json "
        "--out docs/validation/pit-audit.json",
        "python3 py/pit_validation.py report --protocol docs/validation/protocol.json "
        "--audit docs/validation/pit-audit.json --out-dir docs/validation",
    ]


def render_markdown(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    verdicts = report["verdicts"]
    overlap = report["selection_overlap"]["class_allocation"]
    run_period = report["run_period"]
    cost = protocol["cost"]
    lines = [
        "# Validação point-in-time fora da amostra: alocação entre classes",
        "",
        f"- Protocolo: `{report['protocol_id']}`",
        f"- `protocol_sha256`: `{report['protocol_sha256']}`",
        f"- `audit_sha256`: `{report['audit_sha256']}`",
        f"- `run_period`: {_cell(run_period['start'])} a {_cell(run_period['end'])} "
        f"(`unverified_common_rows` {run_period['unverified_common_rows']})",
        "",
        "## Veredito por camada",
        "",
        "| Camada | Veredito | Evidência | Sobreposição com a escolha do método |",
        "| --- | --- | --- | --- |",
    ]
    for layer in LAYERS:
        evidence = report["evidence"]["class_allocation"] if layer in (
            "class_allocation",
            "basic_recommendation",
        ) else "-"
        layer_overlap = overlap if layer in ("class_allocation", "basic_recommendation") else "-"
        lines.append(f"| `{layer}` | `{verdicts[layer]}` | `{evidence}` | `{layer_overlap}` |")
    lines += ["", "## O que falta", ""]
    for layer in LAYERS:
        items = report["missing"][layer]
        lines.append(f"- `{layer}`: " + ("; ".join(items) if items else "nada"))

    lines += [
        "",
        "## Auditoria point-in-time por série",
        "",
        "Regra (AD-018): par entre o grupo do arquivo lido e outra captação com "
        f"`days_apart` de pelo menos {MIN_DAYS_APART} dias, pelo menos {MIN_SHARED_DATES} datas "
        f"compartilhadas, nenhuma data faltando no intervalo compartilhado e fatores diários iguais "
        f"dentro de `1e-9`.",
        "",
        "| Série | Tipo | Status | Motivos | `days_apart` | Datas compartilhadas | "
        "Datas não verificadas | Primeira data | Última data | sha256 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in REQUIRED_SERIES:
        record = audit["series"][name]
        best = None
        for pair in record["pairs"]:
            if best is None or (pair["shared_dates"], pair["days_apart"]) > (
                best["shared_dates"],
                best["days_apart"],
            ):
                best = pair
        if record["status"] == "PIT_QUALIFIED":
            best = max(
                (pair for pair in record["pairs"] if pair["qualifies"]),
                key=lambda pair: (pair["shared_dates"], pair["days_apart"]),
            )
        verified = record.get("verified_range") or {}
        unverified = record.get("unverified_dates") or {}
        sha = report["snapshot"]["files"].get(record.get("file"), "-")
        lines.append(
            f"| `{name}` | `{record['series_kind']}` | `{record['status']}` | "
            f"{_cell(record['reasons'])} | {_cell(record.get('days_apart'))} | "
            f"{_cell(best['shared_dates'] if best else None)} | {_cell(unverified.get('count'))} | "
            f"{_cell(verified.get('start'))} | {_cell(verified.get('end'))} | `{sha}` |"
        )

    lines += ["", "## Leitura obrigatória", ""]
    lines.append(
        "- O método (ADR 0003), seus tetos (`ALLOCATION_PROFILE_ANCHORS`) e sua janela de 10 anos "
        f"(`WINDOW_YEARS`) foram escolhidos em {protocol['method']['adopted_on']} depois de ver o "
        "período 2019-2026."
    )
    lines.append(f"- `selection_overlap`: `{overlap}`.")
    if overlap == "full":
        lines.append(
            "- Toda janela avaliada precede essa data: o período avaliado é o mesmo em que o método "
            "foi escolhido, então o veredito diz que os números eram alcançáveis com os dados da "
            "época, não que a escolha do método foi testada fora da amostra."
        )
    lines.append(
        "- Os 12 deslocamentos se sobrepõem no calendário: mostram a sensibilidade à data de "
        "início e não são amostras independentes."
    )
    decisions = [
        window
        for profile in (report.get("profiles") or {}).values()
        for path in profile["paths"]
        for window in path["windows"]
    ]
    false_count = sum(1 for window in decisions if window["full_history"] is not True)
    if decisions:
        history = (
            "`full_history` verdadeiro em todas as decisões"
            if not false_count
            else f"`full_history` falso em {false_count} decisões"
        )
    else:
        history = "nenhuma decisão foi avaliada nesta execução"
    lines.append(
        "- Nenhuma decisão vê a janela de 10 anos da produção: o lookback vai de "
        f"{protocol['decision']['min_lookback_years']} anos até o histórico verificado ({history}). "
        "`full_history` verdadeiro significa que o histórico disponível na decisão era menor que a "
        "janela de 10 anos, então a estratégia usou todo ele; e a ADR 0003 rejeita janelas de 3 anos "
        "por não conterem nenhuma crise."
    )
    days = ", ".join(
        f"`{name}` {_cell(audit['series'][name].get('days_apart'))}" for name in REQUIRED_SERIES
    )
    lines.append(
        "- Nível de evidência: não revisão observada entre duas captações (`days_apart` dias); "
        f"nenhum vintage as-of. `days_apart` por série: {days}."
    )
    lines.append(
        "- A série IBOV (Yahoo `^BVSP`) é rotulada \"retorno total (metodologia B3)\". As duas "
        "declarações conflitam: `data/allocation/metadata.json` diz \"price index; dividends not "
        "included\"; a ADR 0003 trata o Ibovespa como retorno total."
    )
    lines.append(
        f"- Custos: `cost_bps` {cost['cost_bps']:g} por lado (one-way) sobre o valor negociado em "
        "todas as classes e benchmarks; impostos excluídos."
    )

    profiles = report.get("profiles")
    if profiles:
        lines += _distribution_markdown(profiles, cost)

    lines += ["", "## Como regenerar", "", "```sh", *regeneration_commands(), "```", ""]
    return "\n".join(lines)


SERIES_LABELS = {
    "strategy": "estratégia",
    "cdi": "CDI",
    "ibov": "IBOV",
    "naive_equal_weight": "1/5 ingênuo",
}
METRIC_LABELS = {
    "annualized_return": "Retorno anualizado",
    "annualized_volatility": "Volatilidade anualizada",
    "max_drawdown": "Drawdown máximo",
}
TABLE_HEADER = [
    "| Série | min | p25 | mediana | p75 | max | caminhos |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]


def _stats_row(label: str, stats: Mapping[str, Any]) -> str:
    cells = " | ".join(_pct(stats[key]) for key in STATISTICS)
    return f"| {label} | {cells} | {stats['paths']} |"


def _distribution_markdown(profiles: Mapping[str, Any], cost: Mapping[str, Any]) -> List[str]:
    lines = ["", "## Distribuição por perfil", ""]
    for profile, data in profiles.items():
        lines += [f"### {profile}", ""]
        dist = data["distribution"]
        for metric in METRICS:
            lines += [
                f"{METRIC_LABELS[metric]} (custo `cost_bps` {cost['cost_bps']:g}):",
                "",
                *TABLE_HEADER,
            ]
            for series in SERIES_ORDER:
                lines.append(_stats_row(SERIES_LABELS[series], dist["series"][series][metric]))
            lines.append("")
        lines += ["Excesso de retorno anualizado da estratégia sobre cada benchmark:", ""]
        lines += [
            "| Benchmark | min | p25 | mediana | p75 | max | caminhos | positivos |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for benchmark in REQUIRED_BENCHMARKS:
            stats = dist["excess_annualized_return"][benchmark]
            lines.append(
                _stats_row(SERIES_LABELS[benchmark], stats) + f" {_pct(stats['share_positive'])} |"
            )
        lines.append("")
        if "cost_sensitivity" in data:
            lines += [
                "Sensibilidade ao custo (retorno anualizado, mesmos caminhos e benchmarks):",
                "",
                "| `cost_bps` | Série | min | p25 | mediana | p75 | max | caminhos |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in data["cost_sensitivity"]:
                for series in SERIES_ORDER:
                    lines.append(
                        f"| {row['cost_bps']:g} "
                        + _stats_row(SERIES_LABELS[series], row["annualized_return"][series])
                    )
            lines.append("")
    return lines


# --------------------------------------------------------------------------- output


def dump_json(document: Any) -> str:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_atomically(outputs: Sequence[Tuple[Path, str]]) -> None:
    """Write every output to a temporary file first; rename only when all exist."""

    staged: List[Tuple[str, Path]] = []
    try:
        for target, content in outputs:
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            )
            staged.append((handle.name, target))
            with handle:
                handle.write(content)
            os.chmod(handle.name, 0o644)
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


# --------------------------------------------------------------------------- CLI


def _load_protocol(path: str) -> Tuple[Mapping[str, Any], bytes]:
    protocol, raw = load_json(Path(path), "--protocol")
    validate_protocol(protocol)
    require_directories(protocol)
    return protocol, raw


def command_audit(args: argparse.Namespace) -> None:
    protocol, raw = _load_protocol(args.protocol)
    out = Path(args.out)
    if not out.parent.is_dir():
        raise MissingPathError("--out", f"path does not exist: {out.parent}")
    audit = build_audit(protocol, raw)
    write_atomically([(out, dump_json(audit))])


def command_report(args: argparse.Namespace) -> None:
    protocol, raw = _load_protocol(args.protocol)
    audit, audit_raw = load_json(Path(args.audit), "--audit")
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise MissingPathError("--out-dir", f"path does not exist: {out_dir}")
    report = run_report(protocol, raw, audit, audit_raw)
    markdown = render_markdown(report, protocol, audit)
    write_atomically([(out_dir / REPORT_JSON, dump_json(report)), (out_dir / REPORT_MD, markdown)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pit_validation", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="classify every series by the PIT rule")
    audit.add_argument("--protocol", required=True)
    audit.add_argument("--out", required=True)
    audit.set_defaults(handler=command_audit)
    report = commands.add_parser("report", help="run the walk-forward and write the report")
    report.add_argument("--protocol", required=True)
    report.add_argument("--audit", required=True)
    report.add_argument("--out-dir", required=True)
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        args.handler(args)
    except PITValidationError as exc:
        print(f"pit_validation: {exc.field_name}: {exc.reason}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"pit_validation: output: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
