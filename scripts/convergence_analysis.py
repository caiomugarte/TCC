"""Prefix analysis of the multi-run convergence lots (issue #9).

Reads the pre-registered protocol (``docs/validation/multirun-convergence-protocol.json``)
and the lots ``scripts/multirun_convergence.py run`` wrote, and builds the report:
fitness CV and pairwise Jaccard per prefix, ticker frequencies, the consensus per
prefix (recomputed with the production ``build_consensus_portfolio`` and
``fii_selection._consensus``), turnover between seeds and within a seed, the gap at
the edge of the consensus, the simulated production stop, the protection-zone
simulation (AD-022) and the three recommendations of the protocol's decision block.

Every number the report emits is computed in stdlib arithmetic (``math.fsum``,
``math.sqrt``, ``fractions.Fraction``) and cross-checked against the production
functions within ``TOLERANCE``, so the report's bytes do not depend on the numpy
build. The functions below are pure; ``run_report`` is the only one touching files.
"""

from __future__ import annotations

from collections import Counter
import contextlib
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "api", PROJECT_ROOT / "py", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SELECTORS = ("stocks", "fiis")
LOT_SCHEMA = "multirun-lot-v1"
REPORT_SCHEMA = "multirun-convergence-report-v1"
COST_SCHEMA = "multirun-convergence-cost-v1"
TOLERANCE = 1e-12
# Production's base-company rule (build_consensus_portfolio.get_base_ticker).
STOCK_SUFFIXES = ("11", "3", "4", "5", "6")
DATA_INSUFFICIENT = "data_insufficient"
REPORT_FILES = ("multirun-convergence.json", "multirun-convergence-cost.json", "multirun-convergence.md")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INTEGRITY = 4

TOP_FIELDS = (
    "protocol_id",
    "issue",
    "snapshot_declaration",
    "capital_brl",
    "seeds",
    "profiles",
    "production_run",
    "ga_override",
    "prefixes",
    "curve_step",
    "rank_window_max",
    "production_stop_rule",
    "decision",
)


class ProtocolError(ValueError):
    """The protocol is invalid; ``field`` names the offending field."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class IntegrityError(ValueError):
    """A lot (or the set of lots) fails an integrity check; ``where`` names it."""

    def __init__(self, where: str, reason: str) -> None:
        super().__init__(f"{where}: {reason}")
        self.where = where
        self.reason = reason


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def dumps(value: object) -> str:
    """The one serialization of every generated file and lot (door 1, door 2)."""

    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_files_atomically(files: Mapping[Path, str]) -> None:
    """Write every file to a temporary sibling first, then rename them all."""

    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in files.items():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            tmp.write_text(text, encoding="utf-8")
            staged.append((tmp, path))
        for tmp, path in staged:
            os.replace(tmp, path)
    finally:
        for tmp, _path in staged:
            if tmp.exists():
                tmp.unlink()


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) and math.isfinite(value)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(field, "must be an object")
    return value


def _require_fields(value: Mapping[str, Any], prefix: str, fields: Iterable[str]) -> None:
    for name in fields:
        if name not in value:
            raise ProtocolError(f"{prefix}{name}", "missing")


def _strictly_increasing_positive(values: object, field: str) -> list[int]:
    if not isinstance(values, list) or not values or not all(_is_int(v) and v > 0 for v in values):
        raise ProtocolError(field, "must be a non-empty list of positive integers")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ProtocolError(field, "must be strictly increasing")
    return list(values)


def validate_protocol(protocol: object) -> dict[str, Any]:
    """Raise ``ProtocolError`` naming the first invalid field; return the protocol."""

    protocol = _require_mapping(protocol, "protocol")
    _require_fields(protocol, "", TOP_FIELDS)
    if not isinstance(protocol["protocol_id"], str) or not protocol["protocol_id"]:
        raise ProtocolError("protocol_id", "must be a non-empty string")
    if not isinstance(protocol["snapshot_declaration"], str) or not protocol["snapshot_declaration"]:
        raise ProtocolError("snapshot_declaration", "must be a path")
    if not _is_number(protocol["capital_brl"]) or protocol["capital_brl"] <= 0:
        raise ProtocolError("capital_brl", "must be a positive number")

    ga = _require_mapping(protocol["ga_override"], "ga_override")
    _require_fields(ga, "ga_override.", ("adaptive_mode", "run_count"))
    if ga["adaptive_mode"] is not False:
        raise ProtocolError("ga_override.adaptive_mode", "must be false")
    if not _is_int(ga["run_count"]) or ga["run_count"] <= 0:
        raise ProtocolError("ga_override.run_count", "must be a positive integer")
    run_count = ga["run_count"]

    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2 or not all(_is_int(s) for s in seeds):
        raise ProtocolError("seeds", "must be at least two integers")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds", "must be distinct")
    rule = protocol["decision"].get("run_count") if isinstance(protocol["decision"], Mapping) else None
    leave_one_out = rule.get("leave_one_seed_out") if isinstance(rule, Mapping) else None
    if leave_one_out is not None and len(seeds) < 3:
        raise ProtocolError("seeds", "leave_one_seed_out needs at least three seeds")

    prefixes = _strictly_increasing_positive(protocol["prefixes"], "prefixes")
    if prefixes[-1] != run_count:
        raise ProtocolError("prefixes", f"must end at ga_override.run_count ({run_count})")
    step = protocol["curve_step"]
    if not _is_int(step) or step <= 0:
        raise ProtocolError("curve_step", "must be a positive integer")
    if any(k % step for k in prefixes):
        raise ProtocolError("curve_step", f"{step} does not divide every prefix {prefixes}")
    window = protocol["rank_window_max"]
    if not _is_int(window) or window <= 0:
        raise ProtocolError("rank_window_max", "must be a positive integer")

    profiles = _require_mapping(protocol["profiles"], "profiles")
    if not profiles:
        raise ProtocolError("profiles", "must declare at least one profile")
    for profile_id, declared in profiles.items():
        field = f"profiles.{profile_id}"
        declared = _require_mapping(declared, field)
        _require_fields(declared, f"{field}.", ("record", "expected_n_assets"))
        record = _require_mapping(declared["record"], f"{field}.record")
        if "anchor" in record:
            if not isinstance(record["anchor"], str):
                raise ProtocolError(f"{field}.record.anchor", "must be a profile name")
        else:
            _require_fields(
                record,
                f"{field}.record.",
                ("suitability_score", "raw_score", "dimensions", "generic_profile", "restrictions"),
            )
        expected = _require_mapping(declared["expected_n_assets"], f"{field}.expected_n_assets")
        for selector in SELECTORS:
            value = expected.get(selector)
            if not _is_int(value) or value <= 0:
                raise ProtocolError(f"{field}.expected_n_assets.{selector}", "must be a positive integer")
            if value > window:
                raise ProtocolError("rank_window_max", f"is below {field} n_assets {value}")

    production = _require_mapping(protocol["production_run"], "production_run")
    _require_fields(production, "production_run.", ("id", "profile", "seed", "stocks", "fiis"))
    if production["profile"] not in profiles:
        raise ProtocolError("production_run.profile", "is not a declared profile")
    if production["seed"] not in seeds:
        raise ProtocolError("production_run.seed", "is not a declared seed")
    for selector in SELECTORS:
        tickers = production[selector]
        if not isinstance(tickers, list) or not all(isinstance(t, str) and t for t in tickers):
            raise ProtocolError(f"production_run.{selector}", "must be a list of tickers")

    stop = _require_mapping(protocol["production_stop_rule"], "production_stop_rule")
    _require_fields(stop, "production_stop_rule.", ("target_cv", "target_jaccard", "min_runs", "check_every"))
    for name in ("target_cv", "target_jaccard"):
        if not _is_number(stop[name]):
            raise ProtocolError(f"production_stop_rule.{name}", "must be a number")
    if not _is_int(stop["min_runs"]) or stop["min_runs"] <= 0:
        raise ProtocolError("production_stop_rule.min_runs", "must be a positive integer")
    every = _require_mapping(stop["check_every"], "production_stop_rule.check_every")
    for selector in SELECTORS:
        if not _is_int(every.get(selector)) or every[selector] <= 0:
            raise ProtocolError(f"production_stop_rule.check_every.{selector}", "must be a positive integer")

    decision = _require_mapping(protocol["decision"], "decision")
    _require_fields(decision, "decision.", ("run_count", "target", "zone"))
    rule = _require_mapping(decision["run_count"], "decision.run_count")
    _require_fields(
        rule, "decision.run_count.", ("candidates", "epsilon_swaps", "statistics", "leave_one_seed_out")
    )
    candidates = _strictly_increasing_positive(rule["candidates"], "decision.run_count.candidates")
    if not set(candidates) <= set(prefixes):
        raise ProtocolError("decision.run_count.candidates", f"must be a subset of prefixes {prefixes}")
    if candidates[-1] != run_count:
        raise ProtocolError("decision.run_count.candidates", f"must end at ga_override.run_count ({run_count})")
    if not _is_number(rule["epsilon_swaps"]) or rule["epsilon_swaps"] < 0:
        raise ProtocolError("decision.run_count.epsilon_swaps", "must be a non-negative number")
    statistics = rule["statistics"]
    if (
        not isinstance(statistics, list)
        or not statistics
        or len(set(statistics)) != len(statistics)
        or not set(statistics) <= {"mean", "median"}
    ):
        raise ProtocolError("decision.run_count.statistics", "must be distinct values of mean, median")
    if rule["leave_one_seed_out"] != "max":
        raise ProtocolError("decision.run_count.leave_one_seed_out", "must be \"max\"")

    target = _require_mapping(decision["target"], "decision.target")
    _require_fields(target, "decision.target.", ("prefix", "targets"))
    if target["prefix"] not in prefixes or not _is_int(target["prefix"]):
        raise ProtocolError("decision.target.prefix", f"must be one of the prefixes {prefixes}")
    targets = target["targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or not set(targets) <= {"target_jaccard", "target_cv"}
    ):
        raise ProtocolError("decision.target.targets", "must be distinct values of target_jaccard, target_cv")

    zone = _require_mapping(decision["zone"], "decision.zone")
    _require_fields(
        zone,
        "decision.zone.",
        ("quantile", "quantile_method", "delta_round_up", "delta_arithmetic", "delta_grid"),
    )
    if not _is_number(zone["quantile"]) or not 0 < zone["quantile"] <= 1:
        raise ProtocolError("decision.zone.quantile", "must be in (0, 1]")
    if zone["quantile_method"] != "inverted_cdf":
        raise ProtocolError("decision.zone.quantile_method", "must be \"inverted_cdf\"")
    if not _is_number(zone["delta_round_up"]) or zone["delta_round_up"] <= 0:
        raise ProtocolError("decision.zone.delta_round_up", "must be a positive number")
    if zone["delta_arithmetic"] != "fraction_from_counts":
        raise ProtocolError("decision.zone.delta_arithmetic", "must be \"fraction_from_counts\"")
    grid = zone["delta_grid"]
    if not isinstance(grid, list) or not grid or not all(_is_number(d) and d >= 0 for d in grid):
        raise ProtocolError("decision.zone.delta_grid", "must be a non-empty list of non-negative numbers")
    return dict(protocol)


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    """Read and validate the protocol; return it with the sha256 of its bytes."""

    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ProtocolError("protocol", f"cannot read {path}: {exc.strerror}") from exc
    try:
        protocol = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("protocol", f"not valid JSON: {exc}") from exc
    return validate_protocol(protocol), sha256_bytes(data)


def lot_path(lots_dir: Path, profile_id: str, seed: int) -> Path:
    return Path(lots_dir) / profile_id / f"seed-{seed}.json"


def cost_path(lots_dir: Path, profile_id: str, seed: int) -> Path:
    return Path(lots_dir) / profile_id / f"seed-{seed}.cost.json"


def lot_id(profile_id: str, seed: int) -> str:
    return f"{profile_id}:{seed}"


# --------------------------------------------------------------------------
# Metrics (stdlib arithmetic)
# --------------------------------------------------------------------------


def fitness_cv(values: Sequence[float]) -> float:
    """Population standard deviation over the mean; 0.0 when the mean is 0."""

    k = len(values)
    mean = math.fsum(values) / k
    if mean == 0:
        return 0.0
    deviation = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / k)
    return deviation / mean


def jaccard(a: set | frozenset, b: set | frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pair_jaccards(ticker_sets: Sequence[frozenset]) -> list[float]:
    """Jaccard of every unordered pair, ordered so the first k(k-1)/2 cover the first k runs."""

    values: list[float] = []
    for j in range(len(ticker_sets)):
        for i in range(j):
            values.append(jaccard(ticker_sets[i], ticker_sets[j]))
    return values


def jaccard_mean(ticker_sets: Sequence[set | frozenset]) -> float:
    values = pair_jaccards([frozenset(s) for s in ticker_sets])
    return math.fsum(values) / len(values) if values else 0.0


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def summary(values: Sequence[float], keys: Sequence[str] = ("mean", "median", "min", "max")) -> dict[str, float]:
    functions = {"mean": mean, "median": median, "min": min, "max": max}
    return {key: functions[key](values) for key in keys}


def inverted_cdf(values: Sequence[Any], quantile: float) -> Any:
    """Smallest value whose empirical CDF reaches ``quantile`` (exact index)."""

    ordered = sorted(values)
    index = math.ceil(Fraction(repr(quantile)) * len(ordered)) - 1
    return ordered[max(index, 0)]


def round_up(value: Fraction, step: float) -> float:
    """``ceil(value / step) * step`` in exact fractions, emitted as a float."""

    unit = Fraction(repr(step))
    m = math.ceil(Fraction(value) / unit)
    return float(m * unit)


# --------------------------------------------------------------------------
# Ranking and consensus
# --------------------------------------------------------------------------


def base_company(ticker: str) -> str:
    for suffix in STOCK_SUFFIXES:
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def count_picks(runs: Sequence[Mapping[str, Any]]) -> Counter:
    return Counter(ticker for run in runs for ticker in run["tickers"])


def score_value(info: Mapping[str, Any]) -> float:
    """A candidate's SCORE; a lot stores production's NaN SCORE as null."""

    return math.nan if info.get("score") is None else float(info["score"])


def extended_ranking(counts: Mapping[str, int], scores: Mapping[str, float], selector: str) -> list[str]:
    """Every picked ticker by count desc, score desc, ticker asc (door 3's ``R``).

    For stocks a ticker whose base company is already ranked is left out, as
    ``build_consensus_portfolio`` does. ``counts`` must be built in production's
    order (runs in order, each run's sorted tickers): with a NaN score the sort
    key compares neither below nor above, so production's order then depends on
    first appearance, and the same ``sorted`` call over the same sequence
    reproduces it (door 5).
    """

    ordered = sorted(counts, key=lambda t: (-counts[t], -float(scores[t]), t))
    if selector != "stocks":
        return ordered
    ranking: list[str] = []
    seen: set[str] = set()
    for ticker in ordered:
        base = base_company(ticker)
        if base not in seen:
            ranking.append(ticker)
            seen.add(base)
    return ranking


def production_consensus(
    runs: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    selector: str,
    n_assets: int,
) -> list[str]:
    """The consensus the production function returns on these runs, sorted."""

    import pandas as pd

    tickers = sorted(candidates)
    ranked = pd.DataFrame(
        {
            "TICKER": tickers,
            "SCORE": [score_value(candidates[t]) for t in tickers],
            "SETOR": [candidates[t].get("sector") for t in tickers],
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if selector == "stocks":
            from pipelines.multi_run import build_consensus_portfolio

            portfolio = build_consensus_portfolio(
                [{"tickers": list(run["tickers"])} for run in runs],
                ranked,
                selection_config={"n_assets": int(n_assets)},
            )
        else:
            from fii_selection import _consensus

            portfolio = _consensus(
                [pd.DataFrame({"TICKER": list(run["tickers"])}) for run in runs],
                ranked,
                int(n_assets),
            )
    return sorted(str(t) for t in portfolio["TICKER"].tolist())


# --------------------------------------------------------------------------
# Production stop (simulated)
# --------------------------------------------------------------------------


def adaptive_stop_at(
    metrics: Mapping[int, tuple[float, float]],
    run_count: int,
    rule: Mapping[str, Any],
    selector: str,
) -> int | None:
    """First k >= min_runs where production's check would stop, or None.

    ``metrics[k]`` is ``(fitness_cv, jaccard_mean)`` over the first k runs.
    Stocks check when k is a multiple of ``check_every.stocks`` or k is
    ``run_count`` (``py/pipelines/multi_run.py``); FIIs every
    ``check_every.fiis`` runs (``py/fii_selection.py``).
    """

    every = int(rule["check_every"][selector])
    for k in range(int(rule["min_runs"]), run_count + 1):
        checked = k % every == 0 or (selector == "stocks" and k == run_count)
        if not checked:
            continue
        cv, jm = metrics[k]
        if cv <= rule["target_cv"] and jm >= rule["target_jaccard"]:
            return k
    return None


# --------------------------------------------------------------------------
# Protection zone (door 3, AD-022)
# --------------------------------------------------------------------------


def zone_portfolio(
    incumbents: Iterable[str],
    ranking: Sequence[str],
    counts: Mapping[str, int],
    k: int,
    K: int,
    delta: float,
    selector: str,
) -> tuple[set[str], int]:
    """Apply the door-3 zone rule; return the portfolio and its turnover."""

    incumbents = list(incumbents)
    n = len(incumbents)
    rank = {ticker: index + 1 for index, ticker in enumerate(ranking)}
    stocks = selector == "stocks"

    def freq(ticker: str) -> float:
        return counts.get(ticker, 0) / k

    def bases(portfolio: Sequence[str], excluding: str | None = None) -> set[str]:
        return {base_company(t) for t in portfolio if t != excluding}

    protected = {t for t in incumbents if t in rank and rank[t] <= K}
    portfolio = [t for t in incumbents if t in protected]
    for ticker in ranking:
        if len(portfolio) >= n:
            break
        if ticker in portfolio or (stocks and base_company(ticker) in bases(portfolio)):
            continue
        portfolio.append(ticker)

    replaced: set[str] = set()
    while True:
        held = [t for t in portfolio if t in protected]
        if not held:
            break
        weakest = max(held, key=lambda t: rank[t])
        challenger = None
        for ticker in ranking:
            if ticker in portfolio or ticker in replaced:
                continue
            if stocks and base_company(ticker) in bases(portfolio, excluding=weakest):
                continue
            challenger = ticker
            break
        if challenger is None:
            break
        if rank[challenger] < rank[weakest] and freq(challenger) >= freq(weakest) + delta - TOLERANCE:
            portfolio[portfolio.index(weakest)] = challenger
            replaced.add(weakest)
            protected.discard(weakest)
        else:
            break
    held_set = set(portfolio)
    return held_set, n - len(held_set & set(incumbents))


def zone_k_values(n: int, window: int) -> list[int]:
    return [0, *range(n, window + 1)]


# --------------------------------------------------------------------------
# Lots
# --------------------------------------------------------------------------


def check_lot(
    lot: object,
    where: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    profile_id: str,
    seed: int,
) -> dict[str, Any]:
    """Structural integrity of one lot (S3 checks that need no other lot)."""

    if not isinstance(lot, Mapping):
        raise IntegrityError(where, "lot is not an object")
    if lot.get("schema") != LOT_SCHEMA:
        raise IntegrityError(where, f"schema is {lot.get('schema')!r}, expected {LOT_SCHEMA}")
    if lot.get("protocol_sha256") != protocol_sha256:
        raise IntegrityError(where, "protocol_sha256 differs from the protocol file")
    if lot.get("protocol_id") != protocol["protocol_id"]:
        raise IntegrityError(where, "protocol_id differs from the protocol")
    if lot.get("profile_id") != profile_id or lot.get("seed") != seed:
        raise IntegrityError(where, "profile_id or seed differs from its path")
    if lot.get("mode") not in ("measured", "smoke"):
        raise IntegrityError(where, f"mode is {lot.get('mode')!r}")
    for key in ("snapshot", "code"):
        if not isinstance(lot.get(key), Mapping):
            raise IntegrityError(where, f"{key} is missing")
    run_count = protocol["ga_override"]["run_count"]
    selectors = lot.get("selectors")
    if not isinstance(selectors, Mapping):
        raise IntegrityError(where, "selectors is missing")
    for selector in SELECTORS:
        data = selectors.get(selector)
        label = f"{where} {selector}"
        if not isinstance(data, Mapping):
            raise IntegrityError(label, "selector is missing")
        expected_n = protocol["profiles"][profile_id]["expected_n_assets"][selector]
        if data.get("n_assets") != expected_n:
            raise IntegrityError(label, f"n_assets {data.get('n_assets')!r} differs from {expected_n}")
        if data.get("run_count") != run_count:
            raise IntegrityError(label, f"run_count {data.get('run_count')!r} differs from {run_count}")
        runs = data.get("runs")
        if not isinstance(runs, list):
            raise IntegrityError(label, "runs is missing")
        ids = [run.get("run_id") if isinstance(run, Mapping) else None for run in runs]
        if ids != list(range(run_count)):
            raise IntegrityError(
                label, f"runs are missing, duplicated or out of run_id order (expected 0..{run_count - 1})"
            )
        for run in runs:
            tickers = run.get("tickers")
            if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                raise IntegrityError(label, f"run {run['run_id']} has no ticker list")
            if not _is_number(run.get("fitness")):
                raise IntegrityError(label, f"run {run['run_id']} has no finite fitness")
        candidates = data.get("candidates")
        if not isinstance(candidates, Mapping):
            raise IntegrityError(label, "candidates is missing")
        picked = {t for run in runs for t in run["tickers"]}
        if set(candidates) != picked:
            raise IntegrityError(label, "inconsistent candidate scores: candidates differ from the picked tickers")
        for ticker, info in candidates.items():
            if not isinstance(info, Mapping) or "score" not in info:
                raise IntegrityError(label, f"inconsistent candidate scores: {ticker} has no score")
            if info["score"] is not None and not _is_number(info["score"]):
                raise IntegrityError(label, f"inconsistent candidate scores: {ticker} score is not a number or null")
        selected = data.get("selected_tickers")
        if not isinstance(selected, list) or len(selected) != expected_n:
            raise IntegrityError(label, "selected_tickers does not hold n_assets tickers")
    return dict(lot)


def _read_json(path: Path, where: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntegrityError(where, f"missing file {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(where, f"cannot read {path}: {exc}") from exc


def load_lots(
    protocol: Mapping[str, Any], protocol_sha256: str, lots_dir: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    """Every declared lot, checked one by one and against each other."""

    lots: dict[tuple[str, int], dict[str, Any]] = {}
    for profile_id in protocol["profiles"]:
        for seed in protocol["seeds"]:
            where = lot_id(profile_id, seed)
            path = lot_path(lots_dir, profile_id, seed)
            if not path.is_file():
                raise IntegrityError(where, f"no lot at {path}")
            lot = _read_json(path, where)
            lots[(profile_id, seed)] = check_lot(lot, where, protocol, protocol_sha256, profile_id, seed)
    first_key = next(iter(lots))
    first = lots[first_key]
    for key, lot in lots.items():
        for label, getter in (
            ("protocol_sha256", lambda l: l.get("protocol_sha256")),
            ("protocol_commit", lambda l: l.get("protocol_commit")),
            ("snapshot", lambda l: l.get("snapshot")),
            ("code_fingerprint", lambda l: l["code"].get("code_fingerprint")),
            ("experiment_fingerprint", lambda l: l["code"].get("experiment_fingerprint")),
            ("mode", lambda l: l.get("mode")),
        ):
            if getter(lot) != getter(first):
                raise IntegrityError(
                    lot_id(*key), f"lots disagree on {label} (this lot vs {lot_id(*first_key)})"
                )
    return lots


# --------------------------------------------------------------------------
# Per-cell analysis
# --------------------------------------------------------------------------


def _production_cv(values: Sequence[float]) -> float:
    from core.metrics import coefficient_of_variation

    return float(coefficient_of_variation(list(values)))


def _production_jaccard(a: set, b: set) -> float:
    from core.metrics import jaccard_similarity

    return float(jaccard_similarity(a, b))


def _production_stability(selector: str, runs: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """(fitness_cv, jaccard_mean) the production selector computes on these runs."""

    if selector == "stocks":
        from pipelines.multi_run import analyze_stability

        stability = analyze_stability([dict(run) for run in runs])
        return float(stability["fitness"]["cv"]), float(stability["portfolio_similarity"]["jaccard_mean"])
    from fii_selection import _fii_stability

    stability = _fii_stability([dict(run) for run in runs])
    return float(stability["fitness_cv"]), float(stability["jaccard_mean"])


class SeedSeries:
    """One seed's selector lot with its per-prefix metrics."""

    def __init__(self, where: str, selector: str, data: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
        self.where = where
        self.selector = selector
        self.n = int(data["n_assets"])
        self.runs = list(data["runs"])
        self.candidates = dict(data["candidates"])
        self.scores = {t: score_value(info) for t, info in self.candidates.items()}
        self.selected = list(data["selected_tickers"])
        self.fitness = [float(run["fitness"]) for run in self.runs]
        self.sets = [frozenset(run["tickers"]) for run in self.runs]
        self.pairs = pair_jaccards(self.sets)
        self.run_count = len(self.runs)
        for (i, j), value in zip(
            ((i, j) for j in range(self.run_count) for i in range(j)), self.pairs
        ):
            if abs(value - _production_jaccard(set(self.sets[i]), set(self.sets[j]))) > TOLERANCE:
                raise IntegrityError(where, f"jaccard of runs {i} and {j} differs from core.metrics.jaccard_similarity")
        self._consensus: dict[int, list[str]] = {}
        self._ranking: dict[int, list[str]] = {}
        self._counts: dict[int, Counter] = {}

    def cv(self, k: int) -> float:
        return fitness_cv(self.fitness[:k])

    def jm(self, k: int) -> float:
        count = k * (k - 1) // 2
        return math.fsum(self.pairs[:count]) / count if count else 0.0

    def counts(self, k: int) -> Counter:
        if k not in self._counts:
            self._counts[k] = count_picks(self.runs[:k])
        return self._counts[k]

    def ranking(self, k: int) -> list[str]:
        if k not in self._ranking:
            self._ranking[k] = extended_ranking(self.counts(k), self.scores, self.selector)
        return self._ranking[k]

    def consensus(self, k: int) -> list[str]:
        if k not in self._consensus:
            consensus = production_consensus(self.runs[:k], self.candidates, self.selector, self.n)
            if sorted(self.ranking(k)[: self.n]) != consensus:
                raise IntegrityError(
                    self.where, f"extended ranking at k={k} disagrees with the production consensus"
                )
            self._consensus[k] = consensus
        return self._consensus[k]


def _pairs(seeds: Sequence[int]) -> list[tuple[int, int]]:
    return [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]


def _ordered_pairs(seeds: Sequence[int]) -> list[tuple[int, int]]:
    return [(a, b) for a in seeds for b in seeds if a != b]


def swapped(n: int, a: Sequence[str], b: Sequence[str]) -> int:
    return n - len(set(a) & set(b))


def seed_turnover(series: Mapping[int, SeedSeries], seeds: Sequence[int], k: int, n: int) -> dict[str, Any]:
    pairs = []
    for a, b in _pairs(seeds):
        ca, cb = series[a].consensus(k), series[b].consensus(k)
        pairs.append({"seeds": [a, b], "jaccard": jaccard(set(ca), set(cb)), "swapped": swapped(n, ca, cb)})
    return {
        "k": k,
        "pairs": pairs,
        "swapped": summary([p["swapped"] for p in pairs]),
        "jaccard": summary([p["jaccard"] for p in pairs]),
    }


def rank_gap(seed: SeedSeries, k: int, window: int) -> dict[str, Any]:
    ranking = seed.ranking(k)
    counts = seed.counts(k)
    shown = ranking[:window]
    frequencies = [counts[t] / k for t in shown]
    n = seed.n
    f_n = counts[ranking[n - 1]] / k if len(ranking) >= n else None
    f_next = counts[ranking[n]] / k if len(ranking) > n else 0.0
    return {
        "frequencies": [{"rank": i + 1, "ticker": t, "frequency": f} for i, (t, f) in enumerate(zip(shown, frequencies))],
        "ranks_available": len(ranking),
        "f_n": f_n,
        "f_n_plus_1": f_next,
        "gap": (f_n - f_next) if f_n is not None else None,
    }


def analyze_cell(
    protocol: Mapping[str, Any],
    profile_id: str,
    selector: str,
    lots: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, SeedSeries]]:
    seeds = list(protocol["seeds"])
    run_count = protocol["ga_override"]["run_count"]
    step = protocol["curve_step"]
    window = protocol["rank_window_max"]
    prefixes = list(protocol["prefixes"])
    stop_rule = protocol["production_stop_rule"]
    series = {
        seed: SeedSeries(lot_id(profile_id, seed), selector, lots[(profile_id, seed)]["selectors"][selector], protocol)
        for seed in seeds
    }
    n = series[seeds[0]].n

    curve_ks = list(range(step, run_count + 1, step))
    by_seed: dict[str, list[dict[str, Any]]] = {}
    stops: dict[str, int | None] = {}
    for seed, s in series.items():
        points = []
        for k in curve_ks:
            cv = s.cv(k)
            if abs(cv - _production_cv(s.fitness[:k])) > TOLERANCE:
                raise IntegrityError(s.where, f"fitness_cv at k={k} differs from core.metrics.coefficient_of_variation")
            head = s.fitness[:k]
            points.append(
                {
                    "k": k,
                    "fitness_cv": cv,
                    "jaccard_mean": s.jm(k),
                    "fitness_mean": mean(head),
                    "fitness_min": min(head),
                    "fitness_max": max(head),
                }
            )
        production_cv, production_jm = _production_stability(selector, s.runs)
        last = points[-1]
        if abs(last["fitness_cv"] - production_cv) > TOLERANCE or abs(last["jaccard_mean"] - production_jm) > TOLERANCE:
            raise IntegrityError(
                s.where, f"{selector} stability at k={run_count} differs from the production selector's"
            )
        by_seed[str(seed)] = points
        metrics = {k: (s.cv(k), s.jm(k)) for k in range(stop_rule["min_runs"], run_count + 1)}
        stops[str(seed)] = adaptive_stop_at(metrics, run_count, stop_rule, selector)

    curve_summary = []
    for index, k in enumerate(curve_ks):
        row: dict[str, Any] = {"k": k}
        for metric in ("fitness_cv", "jaccard_mean", "fitness_mean", "fitness_min", "fitness_max"):
            row[metric] = summary([by_seed[str(seed)][index][metric] for seed in seeds], ("mean", "min", "max"))
        curve_summary.append(row)

    for seed, s in series.items():
        if s.consensus(run_count) != sorted(s.selected):
            raise IntegrityError(
                s.where, f"{selector} consensus recomputed at {run_count} runs differs from selected_tickers"
            )

    frequencies, consensus, turnover, within, gaps, grid = [], [], [], [], [], []
    for k in prefixes:
        frequencies.append(
            {
                "k": k,
                "by_seed": {
                    str(seed): {
                        t: {"count": c, "frequency": c / k} for t, c in sorted(s.counts(k).items())
                    }
                    for seed, s in series.items()
                },
            }
        )
        consensus.append({"k": k, "by_seed": {str(seed): s.consensus(k) for seed, s in series.items()}})
        turnover.append(seed_turnover(series, seeds, k, n))
        values = {str(seed): swapped(n, s.consensus(k), s.consensus(run_count)) for seed, s in series.items()}
        within.append({"k": k, "by_seed": values, **summary(list(values.values()), ("mean", "median", "max"))})
        seed_gaps = {str(seed): rank_gap(s, k, window) for seed, s in series.items()}
        gap_values = [g["gap"] for g in seed_gaps.values() if g["gap"] is not None]
        beyond = {}
        for rank in range(n + 1, window + 1):
            values_at = [
                g["frequencies"][rank - 1]["frequency"] if len(g["frequencies"]) >= rank else 0.0
                for g in seed_gaps.values()
            ]
            beyond[str(rank)] = median(values_at)
        gaps.append(
            {
                "k": k,
                "by_seed": seed_gaps,
                "median_gap": median(gap_values) if gap_values else None,
                "median_frequency_by_rank": beyond,
            }
        )
        grid.append(zone_grid(series, seeds, k, n, window, protocol["decision"]["zone"]["delta_grid"], selector))

    target = protocol["decision"]["target"]
    target_index = curve_ks.index(target["prefix"])
    reachability = {
        "target_jaccard": reachability_verdict(
            {str(seed): by_seed[str(seed)][target_index]["jaccard_mean"] for seed in seeds},
            lambda v: v >= stop_rule["target_jaccard"],
        ),
        "target_cv": reachability_verdict(
            {str(seed): by_seed[str(seed)][target_index]["fitness_cv"] for seed in seeds},
            lambda v: v <= stop_rule["target_cv"],
        ),
    }
    cell = {
        "n_assets": n,
        "curves": {"by_seed": by_seed, "summary": curve_summary},
        "frequencies": frequencies,
        "consensus": consensus,
        "seed_turnover": turnover,
        "within_seed_turnover": within,
        "rank_gap": gaps,
        "adaptive_stop": {"by_seed": stops},
        "zone_grid": grid,
        "reachability": reachability,
    }
    return cell, series


def zone_grid(
    series: Mapping[int, SeedSeries],
    seeds: Sequence[int],
    k: int,
    n: int,
    window: int,
    deltas: Sequence[float],
    selector: str,
) -> dict[str, Any]:
    rows = []
    for K in zone_k_values(n, window):
        for delta in deltas:
            turnovers = simulate_pairs(series, seeds, k, K, delta, selector)
            rows.append({"K": K, "delta": delta, "turnover_mean": mean(turnovers), "turnover_max": max(turnovers)})
    baseline = simulate_pairs(series, seeds, k, 0, 0.0, selector)
    return {
        "k": k,
        "rows": rows,
        "baseline": {"K": 0, "turnover_mean": mean(baseline), "turnover_max": max(baseline)},
    }


def simulate_pairs(
    series: Mapping[int, SeedSeries], seeds: Sequence[int], k: int, K: int, delta: float, selector: str
) -> list[int]:
    turnovers = []
    for incumbent_seed, new_seed in _ordered_pairs(seeds):
        new = series[new_seed]
        _portfolio, turnover = zone_portfolio(
            series[incumbent_seed].consensus(k), new.ranking(k), new.counts(k), k, K, delta, selector
        )
        turnovers.append(turnover)
    return turnovers


def reachability_verdict(values: Mapping[str, float], met: Any) -> dict[str, Any]:
    hits = [bool(met(v)) for v in values.values()]
    verdict = "reachable" if all(hits) else "unreachable" if not any(hits) else "mixed"
    return {"verdict": verdict, "by_seed": dict(values)}


def combine_verdicts(verdicts: Sequence[str]) -> str:
    if all(v == "reachable" for v in verdicts):
        return "reachable"
    if all(v == "unreachable" for v in verdicts):
        return "unreachable"
    return "mixed"


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------


def run_count_rule(
    S: Mapping[int, float],
    M: Mapping[int, float],
    candidates: Sequence[int],
    epsilon: float,
    statistics: Sequence[str] = ("mean", "median"),
) -> dict[str, Any]:
    """Criterion 25 for one cell: flattened flag and ``k_min``."""

    last = candidates[-1]
    flattened = len(candidates) < 2 or S[candidates[-2]] - S[last] <= epsilon + TOLERANCE
    k_min = None
    if flattened:
        curves = {"mean": S, "median": M}
        for k in candidates:
            if all(curves[stat][k] <= curves[stat][last] + epsilon + TOLERANCE for stat in statistics):
                k_min = k
                break
    return {"flattened": flattened, "k_min": k_min}


def combine_run_count(results: Sequence[Mapping[str, Any]]) -> int | str:
    if any(not r["flattened"] for r in results):
        return DATA_INSUFFICIENT
    return max(r["k_min"] for r in results)


def final_run_count(full_value: int | str, left_out: Mapping[str, int | str]) -> int | str:
    values = [full_value, *left_out.values()]
    if any(v == DATA_INSUFFICIENT for v in values):
        return DATA_INSUFFICIENT
    return max(values)


def recommend_run_count(
    protocol: Mapping[str, Any], cells: Mapping[str, tuple[Mapping[int, SeedSeries], int]]
) -> dict[str, Any]:
    rule = protocol["decision"]["run_count"]
    candidates = list(rule["candidates"])
    seeds = list(protocol["seeds"])

    def evaluate(subset: Sequence[int]) -> tuple[int | str, dict[str, Any]]:
        per_cell, results = {}, []
        for cell_id, (series, n) in cells.items():
            S, M, curve = {}, {}, []
            for k in candidates:
                values = [swapped(n, series[a].consensus(k), series[b].consensus(k)) for a, b in _pairs(subset)]
                S[k], M[k] = mean(values), median(values)
                curve.append({"k": k, "S": S[k], "M": M[k]})
            result = run_count_rule(S, M, candidates, rule["epsilon_swaps"], rule["statistics"])
            results.append(result)
            per_cell[cell_id] = {"curve": curve, **result}
        return combine_run_count(results), per_cell

    full_value, full_cells = evaluate(seeds)
    left_out: dict[str, int | str] = {}
    left_cells: dict[str, dict[str, Any]] = {}
    if rule["leave_one_seed_out"] == "max":
        for seed in seeds:
            value, per_cell = evaluate([s for s in seeds if s != seed])
            left_out[str(seed)] = value
            left_cells[str(seed)] = per_cell
    per_cell = {
        cell_id: {
            "full": full_cells[cell_id],
            "leave_one_seed_out": {seed: left_cells[seed][cell_id] for seed in left_cells},
        }
        for cell_id in cells
    }
    return {
        "value": final_run_count(full_value, left_out),
        "full_value": full_value,
        "leave_one_seed_out": left_out,
        "per_cell": per_cell,
    }


def recommend_targets(
    protocol: Mapping[str, Any], reachability: Mapping[str, Mapping[str, Any]], run_count_value: int | str
) -> dict[str, Any]:
    targets = protocol["decision"]["target"]["targets"]
    result: dict[str, Any] = {}
    for name in ("target_jaccard", "target_cv"):
        per_cell = {cell_id: cell[name]["verdict"] for cell_id, cell in reachability.items()}
        result[name] = {"verdict": combine_verdicts(list(per_cell.values())), "per_cell": per_cell}
    keep = all(result[name]["verdict"] == "reachable" for name in targets)
    result["action"] = "keep" if keep else "use_fixed_run_count"
    fallback = protocol["ga_override"]["run_count"]
    result["fixed_run_count"] = None if keep else (fallback if run_count_value == DATA_INSUFFICIENT else run_count_value)
    return result


def k_star(ranks: Sequence[float], quantile: float, n: int, window: int) -> int | None:
    value = inverted_cdf(ranks, quantile)
    if value == math.inf or value > window:
        return None
    return max(int(value), n)


def delta_star(differences: Sequence[Fraction], quantile: float, step: float) -> float:
    return round_up(inverted_cdf(differences, quantile), step)


def frequency_differences(
    series: Mapping[int, SeedSeries], seeds: Sequence[int], k: int
) -> list[Fraction]:
    values = []
    for a, b in _pairs(seeds):
        ca, cb = series[a].counts(k), series[b].counts(k)
        for ticker in sorted(set(series[a].consensus(k)) | set(series[b].consensus(k))):
            values.append(abs(Fraction(ca.get(ticker, 0), k) - Fraction(cb.get(ticker, 0), k)))
    return values


def incumbent_ranks(series: Mapping[int, SeedSeries], seeds: Sequence[int], k: int) -> list[float]:
    ranks: list[float] = []
    for incumbent_seed, new_seed in _ordered_pairs(seeds):
        rank = {t: i + 1 for i, t in enumerate(series[new_seed].ranking(k))}
        ranks.extend(rank.get(t, math.inf) for t in series[incumbent_seed].consensus(k))
    return ranks


def recommend_zone(
    protocol: Mapping[str, Any],
    cells: Mapping[str, tuple[Mapping[int, SeedSeries], int]],
    run_count_value: int | str,
) -> dict[str, Any]:
    zone = protocol["decision"]["zone"]
    seeds = list(protocol["seeds"])
    window = protocol["rank_window_max"]
    at_prefix = protocol["ga_override"]["run_count"] if run_count_value == DATA_INSUFFICIENT else int(run_count_value)
    result: dict[str, Any] = {}
    for selector in SELECTORS:
        per_cell: dict[str, dict[str, Any]] = {}
        for cell_id, (series, n) in cells.items():
            if not cell_id.endswith(f"/{selector}"):
                continue
            star = k_star(incumbent_ranks(series, seeds, at_prefix), zone["quantile"], n, window)
            delta = delta_star(frequency_differences(series, seeds, at_prefix), zone["quantile"], zone["delta_round_up"])
            per_cell[cell_id] = {"k_star": star, "k_offset": (star - n) if star is not None else None, "delta": delta}
        any_null = any(item["k_star"] is None for item in per_cell.values())
        k_offset = None if any_null else max(item["k_offset"] for item in per_cell.values())
        delta = None if any_null else max(item["delta"] for item in per_cell.values())
        for cell_id, (series, n) in cells.items():
            if cell_id not in per_cell:
                continue
            baseline = simulate_pairs(series, seeds, at_prefix, 0, 0.0, selector)
            per_cell[cell_id]["at_k0"] = {"turnover_mean": mean(baseline), "turnover_max": max(baseline)}
            if k_offset is None:
                per_cell[cell_id]["at_recommended"] = None
            else:
                turnovers = simulate_pairs(series, seeds, at_prefix, n + k_offset, delta, selector)
                per_cell[cell_id]["at_recommended"] = {
                    "K": n + k_offset,
                    "delta": delta,
                    "turnover_mean": mean(turnovers),
                    "turnover_max": max(turnovers),
                }
        result[selector] = {"k_offset": k_offset, "delta": delta, "at_prefix": at_prefix, "per_cell": per_cell}
    return result


def production_run_match(
    protocol: Mapping[str, Any], lots: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[str, bool]:
    declared = protocol["production_run"]
    lot = lots[(declared["profile"], declared["seed"])]
    return {
        selector: set(lot["selectors"][selector]["selected_tickers"]) == set(declared[selector])
        for selector in SELECTORS
    }


def selector_score(record: Mapping[str, Any]) -> float:
    """Profile score the selectors see (``SELECTOR_SCORE_WEIGHTS``), for the limitations."""

    from app.services.premium_policy import SELECTOR_SCORE_WEIGHTS

    if "anchor" in record:
        from app.services.profile import ANCHOR_SCORES

        score = float(ANCHOR_SCORES[record["anchor"]])
        values = {"score": score, **{key: score for key in SELECTOR_SCORE_WEIGHTS if key != "score"}}
    else:
        values = {"score": float(record["suitability_score"]), **{k: float(v) for k, v in record["dimensions"].items()}}
    return math.fsum(weight * values[key] for key, weight in SELECTOR_SCORE_WEIGHTS.items())


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    lots: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    series_by_cell: dict[str, tuple[Mapping[int, SeedSeries], int]] = {}
    for profile_id in protocol["profiles"]:
        for selector in SELECTORS:
            cell_id = f"{profile_id}/{selector}"
            cell, series = analyze_cell(protocol, profile_id, selector, lots)
            cells[cell_id] = cell
            series_by_cell[cell_id] = (series, cell["n_assets"])
    run_count = recommend_run_count(protocol, series_by_cell)
    reachability = {cell_id: cell["reachability"] for cell_id, cell in cells.items()}
    first = next(iter(lots.values()))
    code = {key: first["code"].get(key) for key in ("commit", "code_fingerprint", "code_trees", "experiment_fingerprint", "experiment_trees")}
    return {
        "schema": REPORT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "protocol_commit": first.get("protocol_commit"),
        "protocol_committed_at": first.get("protocol_committed_at"),
        "mode": first["mode"],
        "snapshot": first["snapshot"],
        "code": code,
        "lots": {
            lot_id(profile_id, seed): {
                "commit": lot["code"].get("commit"),
                "policy_sha256": lot.get("policy_sha256"),
                "runs": {selector: len(lot["selectors"][selector]["runs"]) for selector in SELECTORS},
            }
            for (profile_id, seed), lot in lots.items()
        },
        "profiles": {
            profile_id: {
                "selector_score": selector_score(declared["record"]),
                "n_assets": dict(declared["expected_n_assets"]),
            }
            for profile_id, declared in protocol["profiles"].items()
        },
        "cells": cells,
        "production_run_match": production_run_match(protocol, lots),
        "recommendation": {
            "run_count": run_count,
            "adaptive_targets": recommend_targets(protocol, reachability, run_count["value"]),
            "zone": recommend_zone(protocol, series_by_cell, run_count["value"]),
        },
    }


def build_cost_report(
    protocol: Mapping[str, Any],
    lots: Mapping[tuple[str, int], Mapping[str, Any]],
    lots_dir: Path,
) -> dict[str, Any]:
    rows = []
    machines: dict[str, Any] = {}
    for (profile_id, seed) in lots:
        where = lot_id(profile_id, seed)
        cost = _read_json(cost_path(lots_dir, profile_id, seed), where)
        try:
            stages = cost["stages"]
            row = {
                "lot": where,
                "profile_id": profile_id,
                "seed": seed,
                "stocks": {"wall_s": stages["stocks"]["wall_s"], "cpu_s": stages["stocks"]["cpu_s"]},
                "fiis": {"wall_s": stages["fiis"]["wall_s"], "cpu_s": stages["fiis"]["cpu_s"]},
                "total": {"wall_s": cost["wall_s"], "cpu_s": cost["cpu_s"]},
                "started_at": cost["started_at"],
                "finished_at": cost["finished_at"],
                "jobs": cost["jobs"],
                "machine_fingerprint": cost["machine"]["fingerprint"],
            }
        except (KeyError, TypeError) as exc:
            raise IntegrityError(where, f"cost file misses {exc}") from exc
        machines[cost["machine"]["fingerprint"]] = {
            key: cost["machine"].get(key)
            for key in ("fingerprint", "cpu_model", "physical_cores", "logical_cpus", "mem_total_kb", "os", "kernel", "python", "libraries")
        }
        rows.append(row)
    summary_file = _read_json(Path(lots_dir) / "run-summary.json", "run-summary")
    try:
        experiment = {
            "experiment_wall_s": summary_file["experiment_wall_s"],
            "jobs": summary_file["jobs"],
            "first_started_at": summary_file["first_started_at"],
            "last_finished_at": summary_file["last_finished_at"],
        }
    except (KeyError, TypeError) as exc:
        raise IntegrityError("run-summary", f"misses {exc}") from exc
    mode = next(iter(lots.values()))["mode"]
    return {
        "schema": COST_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "mode": mode,
        "label": mode,
        "lots": rows,
        "cpu_hours_total": math.fsum(row["total"]["cpu_s"] for row in rows) / 3600,
        "experiment": experiment,
        "machines": [machines[key] for key in sorted(machines)],
    }


def run_report(protocol_path: Path, lots_dir: Path, out_dir: Path) -> dict[Path, str]:
    """Build the three report files; raise before writing anything on failure."""

    protocol, protocol_sha256 = load_protocol(protocol_path)
    lots = load_lots(protocol, protocol_sha256, lots_dir)
    report = build_report(protocol, protocol_sha256, lots)
    cost = build_cost_report(protocol, lots, lots_dir)
    out_dir = Path(out_dir)
    return {
        out_dir / REPORT_FILES[0]: dumps(report),
        out_dir / REPORT_FILES[1]: dumps(cost),
        out_dir / REPORT_FILES[2]: render_markdown(protocol, report, cost),
    }


# --------------------------------------------------------------------------
# Markdown (Portuguese, generated from the JSON)
# --------------------------------------------------------------------------


def _num(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _versions(cost: Mapping[str, Any]) -> str:
    parts = []
    for machine in cost["machines"]:
        libraries = machine.get("libraries") or {}
        parts.append(
            f"Python {machine.get('python')}, numpy {libraries.get('numpy')}, pandas {libraries.get('pandas')}"
        )
    return "; ".join(parts) or "-"


def render_markdown(protocol: Mapping[str, Any], report: Mapping[str, Any], cost: Mapping[str, Any]) -> str:
    mode = report["mode"]
    rec = report["recommendation"]
    rule = protocol["decision"]["run_count"]
    zone_rule = protocol["decision"]["zone"]
    stop = protocol["production_stop_rule"]
    run_count = protocol["ga_override"]["run_count"]
    title = "Convergência do multi-run" + (" (SMOKE - não é medição)" if mode == "smoke" else " (medido)")
    lines = [
        f"# {title}",
        "",
        "Relatório gerado por `scripts/multirun_convergence.py report` a partir do protocolo e dos lotes; "
        "nenhum número é escrito à mão.",
        "",
        f"- modo: `{mode}`",
        f"- protocolo: `{report['protocol_id']}` sha256 `{report['protocol_sha256']}`, commit "
        f"`{report['protocol_commit']}` ({report['protocol_committed_at']})",
        f"- snapshot: `{report['snapshot'].get('manifest_id')}` manifest_sha256 `{report['snapshot'].get('manifest_sha256')}`",
        f"- código: commit `{report['code'].get('commit')}`, code_fingerprint `{report['code'].get('code_fingerprint')}`, "
        f"experiment_fingerprint `{report['code'].get('experiment_fingerprint')}`",
        f"- seeds: {', '.join(str(s) for s in protocol['seeds'])}; execuções por seletor: {run_count}; "
        f"prefixos: {', '.join(str(k) for k in protocol['prefixes'])}",
        "",
        "## Recomendações",
        "",
        f"### run_count: **{rec['run_count']['value']}**",
        "",
        f"Regra: por célula, `S(k)` e `M(k)` são a média e a mediana de trocados entre os pares de seeds no "
        f"prefixo k; a célula achatou quando `S({rule['candidates'][-2] if len(rule['candidates']) > 1 else rule['candidates'][-1]}) - "
        f"S({rule['candidates'][-1]}) <= {_num(float(rule['epsilon_swaps']), 2)}`; `k_min` é o menor candidato "
        f"({', '.join(str(k) for k in rule['candidates'])}) com S e M até {_num(float(rule['epsilon_swaps']), 2)} acima "
        f"do último; o valor é o maior `k_min` entre as células, `data_insufficient` se alguma não achatou, e a "
        f"recomendação é o máximo entre a regra cheia e cada seed deixada de fora.",
        "",
        f"- regra cheia: {rec['run_count']['full_value']}",
    ]
    for seed, value in rec["run_count"]["leave_one_seed_out"].items():
        lines.append(f"- sem a seed {seed}: {value}")
    lines.append("")
    rows = []
    for cell_id, data in rec["run_count"]["per_cell"].items():
        curve = ", ".join(f"{p['k']}: S {_num(p['S'], 2)} / M {_num(p['M'], 2)}" for p in data["full"]["curve"])
        rows.append([cell_id, curve, _num(data["full"]["flattened"]), _num(data["full"]["k_min"])])
    lines += _table(["célula", "curva (cheia)", "achatou", "k_min"], rows)
    targets = rec["adaptive_targets"]
    lines += [
        "",
        f"### Alvos do critério adaptativo: **{targets['action']}**",
        "",
        f"Regra: a {protocol['decision']['target']['prefix']} execuções, `target_jaccard` {_num(float(stop['target_jaccard']), 2)} "
        f"(`jaccard_mean >=`) e `target_cv` {_num(float(stop['target_cv']), 2)} (`fitness_cv <=`) são `reachable` numa "
        "célula quando todas as seeds atingem, `unreachable` quando nenhuma, `mixed` no resto; `keep` só quando os dois "
        "vereditos são `reachable` em todas as células, senão `use_fixed_run_count`.",
        "",
        f"- target_jaccard: {targets['target_jaccard']['verdict']}",
        f"- target_cv: {targets['target_cv']['verdict']}",
        f"- fixed_run_count: {_num(targets['fixed_run_count'])}",
        "",
        "### Zona de proteção (K, δ)",
        "",
        f"Regra (AD-022): no prefixo recomendado, `K*` é o quantil {_num(float(zone_rule['quantile']), 2)} (CDF invertida) "
        "dos rankings de cada titular no lote da outra seed (pares ordenados; sem ranking conta como infinito; abaixo de n "
        f"sobe para n; acima de {protocol['rank_window_max']} é nulo); `δ*` é o mesmo quantil de `|f_i(t) - f_j(t)|` nos "
        f"pares de seeds, exato a partir das contagens e arredondado para cima em {_num(float(zone_rule['delta_round_up']), 2)}; "
        "vale o maior entre os perfis.",
        "",
    ]
    rows = []
    for selector in SELECTORS:
        item = rec["zone"][selector]
        rows.append([selector, _num(item["k_offset"]), _num(item["delta"], 2), item["at_prefix"]])
    lines += _table(["seletor", "k_offset", "δ", "prefixo"], rows)
    lines.append("")
    rows = []
    for selector in SELECTORS:
        for cell_id, item in rec["zone"][selector]["per_cell"].items():
            at = item["at_recommended"]
            rows.append(
                [
                    cell_id,
                    _num(item["k_star"]),
                    _num(item["delta"], 2),
                    f"{_num(item['at_k0']['turnover_mean'], 2)} / {item['at_k0']['turnover_max']}",
                    f"{_num(at['turnover_mean'], 2)} / {at['turnover_max']}" if at else "-",
                ]
            )
    lines += _table(["célula", "K*", "δ*", "giro K=0 (média / máx)", "giro no ponto recomendado (média / máx)"], rows)

    match = report["production_run_match"]
    declared = protocol["production_run"]
    lines += [
        "",
        "## Execução de produção de 2026-11-02",
        "",
        f"O lote `{declared['profile']}` com a seed {declared['seed']} refaz a execução `{declared['id']}` sem o "
        "critério adaptativo. Mesmos tickers que os usuários receberam: "
        f"ações {_num(match['stocks'])}, FIIs {_num(match['fiis'])}.",
    ]

    for cell_id, cell in report["cells"].items():
        n = cell["n_assets"]
        lines += ["", f"## Célula {cell_id} (n = {n})", "", "### Curvas (média entre seeds, [mín, máx])", ""]
        lines += _table(
            ["k", "fitness_cv", "jaccard_mean"],
            (
                [
                    row["k"],
                    f"{_num(row['fitness_cv']['mean'], 4)} [{_num(row['fitness_cv']['min'], 4)}, {_num(row['fitness_cv']['max'], 4)}]",
                    f"{_num(row['jaccard_mean']['mean'], 4)} [{_num(row['jaccard_mean']['min'], 4)}, {_num(row['jaccard_mean']['max'], 4)}]",
                ]
                for row in cell["curves"]["summary"]
            ),
        )
        stops = cell["adaptive_stop"]["by_seed"]
        lines += ["", "Parada adaptativa simulada (primeiro k em que a produção pararia): "
                  + ", ".join(f"seed {seed}: {_num(at)}" for seed, at in stops.items())]
        final = cell["rank_gap"][-1]
        lines += ["", f"### Frequências a {final['k']} execuções (top {protocol['rank_window_max']} por seed)", ""]
        seeds = list(final["by_seed"])
        rows = []
        for rank in range(1, protocol["rank_window_max"] + 1):
            row: list[object] = [rank]
            for seed in seeds:
                freqs = final["by_seed"][seed]["frequencies"]
                row.append(f"{freqs[rank - 1]['ticker']} {_num(freqs[rank - 1]['frequency'], 3)}" if len(freqs) >= rank else "-")
            rows.append(row)
        lines += _table(["rank", *(f"seed {s}" for s in seeds)], rows)
        lines += ["", "### Giro entre seeds e dentro da seed (trocados)", ""]
        lines += _table(
            ["k", "entre seeds média", "mediana", "mín", "máx", "Jaccard médio", "dentro da seed média", "mediana", "máx"],
            (
                [
                    t["k"],
                    _num(t["swapped"]["mean"], 2),
                    _num(t["swapped"]["median"], 2),
                    t["swapped"]["min"],
                    t["swapped"]["max"],
                    _num(t["jaccard"]["mean"], 3),
                    _num(w["mean"], 2),
                    _num(w["median"], 2),
                    w["max"],
                ]
                for t, w in zip(cell["seed_turnover"], cell["within_seed_turnover"])
            ),
        )
        lines += ["", "### Folga de ranking (f_n - f_(n+1))", ""]
        lines += _table(
            ["k", *(f"seed {s}" for s in seeds), "mediana"],
            (
                [
                    g["k"],
                    *(
                        f"{_num(g['by_seed'][s]['f_n'], 3)} / {_num(g['by_seed'][s]['f_n_plus_1'], 3)} / {_num(g['by_seed'][s]['gap'], 3)}"
                        for s in seeds
                    ),
                    _num(g["median_gap"], 3),
                ]
                for g in cell["rank_gap"]
            ),
        )
        selector = cell_id.rsplit("/", 1)[1]
        at_prefix = rec["zone"][selector]["at_prefix"]
        grid = next(g for g in cell["zone_grid"] if g["k"] == at_prefix)
        shown_k = sorted({K for K in (n, n + 2, n + 5, n + 10, protocol["rank_window_max"]) if K <= protocol["rank_window_max"]})
        lines += ["", f"### Zona simulada a {at_prefix} execuções (giro médio / máximo nos pares ordenados)", ""]
        deltas = list(protocol["decision"]["zone"]["delta_grid"])
        by_point = {(row["K"], row["delta"]): row for row in grid["rows"]}
        rows = [["0 (consenso puro)", *(f"{_num(grid['baseline']['turnover_mean'], 2)} / {grid['baseline']['turnover_max']}" for _ in deltas)]]
        for K in shown_k:
            rows.append(
                [K, *(f"{_num(by_point[(K, d)]['turnover_mean'], 2)} / {by_point[(K, d)]['turnover_max']}" for d in deltas)]
            )
        lines += _table(["K", *(f"δ {_num(float(d), 3)}" for d in deltas)], rows)
        item = rec["zone"][selector]["per_cell"][cell_id]
        at = item["at_recommended"]
        lines += [
            "",
            f"Ponto recomendado: K = {at['K'] if at else '-'}, δ = {_num(at['delta'], 2) if at else '-'}: giro "
            f"{_num(at['turnover_mean'], 2) if at else '-'} / {at['turnover_max'] if at else '-'}, contra "
            f"{_num(item['at_k0']['turnover_mean'], 2)} / {item['at_k0']['turnover_max']} sem zona (K = 0).",
        ]

    label = "medido" if cost["label"] == "measured" else "smoke"
    lines += ["", f"## Custo ({label})", ""]
    lines += _table(
        ["lote", "ações wall (s)", "ações CPU (s)", "FIIs wall (s)", "FIIs CPU (s)", "total wall (s)", "total CPU (s)"],
        (
            [
                row["lot"],
                _num(float(row["stocks"]["wall_s"]), 1),
                _num(float(row["stocks"]["cpu_s"]), 1),
                _num(float(row["fiis"]["wall_s"]), 1),
                _num(float(row["fiis"]["cpu_s"]), 1),
                _num(float(row["total"]["wall_s"]), 1),
                _num(float(row["total"]["cpu_s"]), 1),
            ]
            for row in cost["lots"]
        ),
    )
    experiment = cost["experiment"]
    lines += [
        "",
        f"- CPU total: {_num(cost['cpu_hours_total'], 2)} CPU-horas ({label})",
        f"- tempo de parede do experimento: {_num(float(experiment['experiment_wall_s']), 1)} s com `jobs` {experiment['jobs']}",
        "- máquina: " + "; ".join(
            f"{m.get('cpu_model')} ({m.get('physical_cores')} núcleos / {m.get('logical_cpus')} threads), fingerprint `{m.get('fingerprint')}`"
            for m in cost["machines"]
        ),
    ]

    scores = {pid: data["selector_score"] for pid, data in report["profiles"].items()}
    lowest = min(scores.values())
    lines += [
        "",
        "## Limitações",
        "",
        "- Um único snapshot: isto é o piso do giro entre seeds, não o giro mês a mês, que a fase 2 mede no #46.",
        f"- {len(protocol['seeds'])} seeds: os {len(protocol['seeds']) * (len(protocol['seeds']) - 1) // 2} pares não são independentes.",
        "- A zona simulada é o modelo da porta 3 (AD-022); o #46 precisa reproduzi-lo ou re-simular sobre estes lotes.",
        "- Nenhuma afirmação fora da amostra: o relatório mede estabilidade, não qualidade.",
        "- Outras configurações do GA (população, gerações, mutação) não foram medidas.",
        "- Cobertura de perfis: selector score "
        + ", ".join(f"{_num(score, 6)} para `{pid}`" for pid, score in scores.items())
        + f"; nenhum perfil com selector score abaixo de {_num(lowest, 6)} (cerca de 0,70), e nenhum na metade "
        "conservador-moderado (selector score 0,5 ou menos), foi medido, enquanto o #46 aplica o maior `k_offset` e δ a todo perfil.",
        f"- Os lotes estão presos ao build que os produziu ({_versions(cost)}, copiados dos arquivos de custo); a "
        "aritmética do relatório é stdlib e não depende dele.",
        "",
    ]
    return "\n".join(lines)
