#!/usr/bin/env python3
"""Multi-run convergence experiment (issue #9): full lots per profile and seed.

``run`` executes the production Premium pipeline (``run_premium_optimization`` with
the adapter's default engines) once per (profile, seed) declared in the
pre-registered protocol, with ``adaptive_mode`` false and ``run_count`` from the
protocol in the experiment's own policy copy, and stores every GA run of both
selectors in a lot (``lots/<profile_id>/seed-<seed>.json``) plus its measured cost
(``seed-<seed>.cost.json``). Each (profile, seed) runs in a fresh Python process;
at most ``--jobs`` run at once. Observation-only proxies around
``pipelines.multi_run.run_single_execution`` and ``fii_selection._run_fii_execution``
record each run; production code is not modified.

``report`` rebuilds ``multirun-convergence.json``, ``-cost.json`` and ``.md`` from
the protocol and the lots (``scripts/convergence_analysis.py``).

Exit codes:
    0  everything was written
    1  at least one child failed (its lot is not written; the others are)
    2  invalid protocol or arguments, or a resolved n_assets other than declared
    3  the registry does not hold the declared snapshot
    4  lot integrity (see convergence_analysis); an invalid existing lot is never overwritten
    6  measured run with the protocol untracked or modified, or modified code paths
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "api", PROJECT_ROOT / "py", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import convergence_analysis as analysis  # noqa: E402
from convergence_analysis import IntegrityError, ProtocolError  # noqa: E402

REPO_ROOT = PROJECT_ROOT
EXPERIMENT_PATHS = ("scripts/multirun_convergence.py", "scripts/convergence_analysis.py")
# Door 4: a measured run refuses to start when any of these is modified.
CODE_PATHS = (
    "api/app",
    "py",
    "scripts/multirun_convergence.py",
    "scripts/convergence_analysis.py",
    "scripts/benchmark_premium.py",
    "scripts/bench_measure.py",
)
SMOKE_GA = {"population": 20, "generations": 5}
PREFIX = "multirun_convergence"

EXIT_OK = 0
EXIT_CHILD_FAILED = 1
EXIT_USAGE = 2
EXIT_SNAPSHOT = 3
EXIT_INTEGRITY = 4
EXIT_PREREGISTRATION = 6


class ChildError(RuntimeError):
    """A child stops with ``code``; ``where`` names the field, selector or lot."""

    def __init__(self, code: int, where: str, reason: str) -> None:
        super().__init__(f"{where}: {reason}")
        self.code = code
        self.where = where
        self.reason = reason


def _error(where: str, reason: str) -> None:
    print(f"{PREFIX}: {where}: {reason}", file=sys.stderr, flush=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Git: pre-registration (door 4) and fingerprints
# --------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def _git_out(*args: str) -> str | None:
    completed = _git(*args)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _repo_relative(path: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(Path(REPO_ROOT).resolve()).as_posix()
    except ValueError:
        return None


def preregistration_problems(protocol_path: Path) -> list[str]:
    """Why a measured run may not start (empty when it may)."""

    rel = _repo_relative(protocol_path)
    if rel is None:
        return [f"protocol {protocol_path} is outside the repository"]
    if _git("ls-files", "--error-unmatch", rel).returncode != 0:
        return [f"protocol {rel} is not committed"]
    status = _git("status", "--porcelain", "--", rel, *CODE_PATHS)
    if status.returncode != 0:
        return [f"git status failed: {status.stderr.strip()}"]
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    if changed:
        return [f"uncommitted changes: {', '.join(line[3:] for line in changed[:5])}"]
    return []


def protocol_provenance(protocol_path: Path) -> dict[str, str | None]:
    rel = _repo_relative(protocol_path)
    if rel is None:
        return {"protocol_commit": None, "protocol_committed_at": None}
    return {
        "protocol_commit": _git_out("log", "-1", "--format=%H", "--", rel),
        "protocol_committed_at": _git_out("log", "-1", "--format=%cI", "--", rel),
    }


def experiment_info() -> dict[str, Any]:
    trees = {path: _git_out("rev-parse", f"HEAD:{path}") for path in EXPERIMENT_PATHS}
    fingerprint = (
        hashlib.sha256(json.dumps(trees, sort_keys=True).encode()).hexdigest() if all(trees.values()) else None
    )
    return {"experiment_trees": trees, "experiment_fingerprint": fingerprint}


def code_info() -> dict[str, Any]:
    import benchmark_premium

    git = benchmark_premium.git_info()
    return {
        "commit": git["commit"],
        "code_fingerprint": git["code_fingerprint"],
        "code_trees": git["code_trees"],
        **experiment_info(),
    }


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def profile_record(record: Mapping[str, Any]) -> SimpleNamespace:
    """Profile v1 record from the protocol's declaration."""

    import benchmark_premium

    if "anchor" in record:
        return benchmark_premium.anchor_profile_record(record["anchor"])
    return SimpleNamespace(
        schema_version=1,
        version=1,
        suitability_score=float(record["suitability_score"]),
        raw_score=float(record["raw_score"]),
        dimensions={key: float(value) for key, value in record["dimensions"].items()},
        generic_profile=record["generic_profile"],
        answers={"restricoes": list(record["restrictions"])},
        restrictions_json=None,
        rules_json=[],
        warnings_json=[],
    )


def resolve_policy(record: Mapping[str, Any], manifest: Any, seed: int) -> dict[str, Any]:
    """Resolve the Premium policy as ``PremiumRecommendationService`` does."""

    from app.services.premium_policy import POLICY_VERSION, resolve_recommendation_policy
    from app.services.premium_recommendation import _manifest_hashes

    cutoff = manifest.cutoff_date
    return resolve_recommendation_policy(
        profile_record(record),
        rules_version=POLICY_VERSION,
        plan="premium",
        source_snapshot_ids=(manifest.manifest_id,),
        source_snapshot_hashes=_manifest_hashes(manifest),
        cutoff_date=cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff),
        random_seed=int(seed),
    ).to_dict()


def check_n_assets(policy: Mapping[str, Any], expected: Mapping[str, int]) -> None:
    for selector in analysis.SELECTORS:
        resolved = int(policy[selector]["n_assets"])
        if resolved != int(expected[selector]):
            raise ChildError(
                EXIT_USAGE,
                "expected_n_assets",
                f"resolved {selector} n_assets {resolved}, declared {expected[selector]}",
            )


def apply_override(policy: dict[str, Any], ga_override: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    for selector in analysis.SELECTORS:
        ga = policy[selector]["system_ga_config"]
        ga["adaptive_mode"] = False
        ga["run_count"] = int(ga_override["run_count"])
        if smoke:
            ga.update(SMOKE_GA)
    return policy


# --------------------------------------------------------------------------
# Child: one (profile, seed)
# --------------------------------------------------------------------------


def _observe(result: Mapping[str, Any], portfolio: Any) -> dict[str, Any]:
    picked: dict[str, dict[str, Any]] = {}
    for row in portfolio.to_dict("records"):
        sector = row.get("SETOR")
        score = float(row["SCORE"])
        picked[str(row["TICKER"])] = {
            # Production scores some stocks NaN and the GA may pick them; JSON has no NaN.
            "score": score if math.isfinite(score) else None,
            "sector": None if sector is None or sector != sector else str(sector),
        }
    return {
        "run_id": int(result["run_id"]),
        "seed": int(result["seed"]),
        "tickers": [str(t) for t in result["tickers"]],
        "fitness": float(result["fitness"]),
        "hhi": float(result["hhi"]),
        "generations_run": int(result.get("generations_run") or 0),
        "picked": picked,
    }


@contextmanager
def capture_selector_runs() -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Record every stock and FII GA run (observation only; restored on exit).

    The stock loop calls ``run_single_execution`` and ``run_fii_selection``
    builds its runner from ``_run_fii_execution`` at call time, so module-level
    proxies see each run; both return the original objects unchanged.
    """

    import fii_selection
    from pipelines import multi_run

    captured: dict[str, list[dict[str, Any]]] = {"stocks": [], "fiis": []}
    original_stock = multi_run.run_single_execution
    original_fii = fii_selection._run_fii_execution

    def stock_proxy(*args: Any, **kwargs: Any) -> Any:
        result, portfolio = original_stock(*args, **kwargs)
        captured["stocks"].append(_observe(result, portfolio))
        return result, portfolio

    def fii_proxy(*args: Any, **kwargs: Any) -> Any:
        result, portfolio = original_fii(*args, **kwargs)
        captured["fiis"].append(_observe(result, portfolio))
        return result, portfolio

    multi_run.run_single_execution = stock_proxy
    fii_selection._run_fii_execution = fii_proxy
    try:
        yield captured
    finally:
        multi_run.run_single_execution = original_stock
        fii_selection._run_fii_execution = original_fii


def check_selector_runs(selector: str, captured: Sequence[Mapping[str, Any]], run_count: int, result: Any) -> None:
    """Criterion 34: exactly run_count runs, in order, matching the selector's ``n_runs``."""

    ids = [run["run_id"] for run in captured]
    if ids != list(range(run_count)):
        raise ChildError(
            EXIT_CHILD_FAILED,
            selector,
            f"captured {len(ids)} runs, expected run_id 0..{run_count - 1} in order",
        )
    if not isinstance(result, Mapping) or "n_runs" not in result:
        raise ChildError(EXIT_CHILD_FAILED, selector, "selector result has no n_runs")
    if int(result["n_runs"]) != len(captured):
        raise ChildError(
            EXIT_CHILD_FAILED,
            selector,
            f"selector reported n_runs {result['n_runs']}, captured {len(captured)}",
        )


def merge_candidates(selector: str, runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for run in runs:
        for ticker, info in run["picked"].items():
            known = candidates.setdefault(ticker, dict(info))
            if known != info:
                raise ChildError(
                    EXIT_CHILD_FAILED,
                    selector,
                    f"inconsistent candidate scores for {ticker}: {known} vs {info} in run {run['run_id']}",
                )
    return candidates


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _stage_costs(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, stage in stages.items():
        item = dict(stage)
        item["cpu_s"] = float(stage.get("cpu_user_s", 0.0)) + float(stage.get("cpu_system_s", 0.0))
        result[name] = item
    return result


def _loadavg() -> list[float] | None:
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def run_child(
    protocol_path: Path,
    context: Mapping[str, Any],
    registry: Path,
    lots_dir: Path,
    profile_id: str,
    seed: int,
    *,
    jobs: int,
    smoke: bool,
    engines: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[Path, str]:
    """Run one (profile, seed) and return the lot and cost files to write."""

    import benchmark_premium
    from bench_measure import RssSampler, StageMeter, compose_stages, read_proc_status_kb, ru_maxrss_kb, validate_rss

    started_at = _utcnow()
    load_start = _loadavg()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()

    protocol, protocol_sha256 = analysis.load_protocol(protocol_path)
    if protocol_sha256 != context["protocol_sha256"]:
        raise ChildError(EXIT_INTEGRITY, "protocol", "changed while the run was starting")
    declared = protocol["profiles"][profile_id]
    run_count = int(protocol["ga_override"]["run_count"])

    declaration_path = _declaration_path(protocol)
    try:
        manifest, snapshot = benchmark_premium.verify_snapshot(
            benchmark_premium.load_declaration(declaration_path), registry
        )
    except benchmark_premium.SnapshotDeclarationError as exc:
        raise ChildError(EXIT_SNAPSHOT, "snapshot", str(exc)) from exc

    policy = resolve_policy(declared["record"], manifest, seed)
    check_n_assets(policy, declared["expected_n_assets"])
    apply_override(policy, protocol["ga_override"], smoke)
    selector_inputs = benchmark_premium.selector_inputs_from_manifest(manifest, snapshot)

    from app.adapters.premium_optimization import run_premium_optimization

    engines = dict(engines or benchmark_premium.default_engines())
    calls: dict[str, dict[str, Any]] = {}

    def observed(name: str, engine: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = engine(*args, **kwargs)
            calls[name] = {"result": result, "kwargs": kwargs}
            return result

        return wrapper

    workspace = Path(tempfile.mkdtemp(prefix=f"multirun-{profile_id}-{seed}-"))
    print(f"{PREFIX}: {analysis.lot_id(profile_id, seed)}: workspace {workspace}", file=sys.stderr, flush=True)
    sampler = RssSampler().start()
    meter = StageMeter(sampler)
    try:
        with capture_selector_runs() as captured:
            with meter.measure("total"):
                try:
                    output = run_premium_optimization(
                        policy,
                        manifest,
                        workspace,
                        float(protocol["capital_brl"]),
                        selector_inputs=selector_inputs,
                        snapshot_loader=meter.wrap("snapshot_load", engines["snapshot_loader"]),
                        allocation_engine=meter.wrap("allocation", engines["allocation_engine"]),
                        stock_engine=meter.wrap("stocks", observed("stocks", engines["stock_engine"])),
                        fii_engine=meter.wrap("fiis", observed("fiis", engines["fii_engine"])),
                        seed=int(seed),
                    )
                except Exception as exc:  # noqa: BLE001 - any pipeline failure fails the child
                    cause = exc.__cause__ or exc
                    raise ChildError(EXIT_CHILD_FAILED, "pipeline", f"{exc} ({cause!r})") from exc
    finally:
        sampler.stop()
        shutil.rmtree(workspace, ignore_errors=True)

    selectors: dict[str, Any] = {}
    for selector in analysis.SELECTORS:
        if selector not in calls:
            raise ChildError(EXIT_CHILD_FAILED, selector, "selector did not run (zero class weight)")
        result = calls[selector]["result"]
        runs = captured[selector]
        check_selector_runs(selector, runs, run_count, result)
        kwargs = calls[selector]["kwargs"]
        config = kwargs.get("config", kwargs.get("selection_config"))
        selectors[selector] = {
            "n_assets": int(policy[selector]["n_assets"]),
            "selection_config": _json_copy(config),
            "run_count": run_count,
            "selected_tickers": [str(item["ticker"]) for item in output.get(selector, [])],
            "candidates": merge_candidates(selector, runs),
            "runs": [{key: value for key, value in run.items() if key != "picked"} for run in runs],
        }

    lot = {
        "schema": analysis.LOT_SCHEMA,
        "mode": context["mode"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "protocol_commit": context["protocol_commit"],
        "protocol_committed_at": context["protocol_committed_at"],
        "profile_id": profile_id,
        "seed": int(seed),
        "policy_sha256": benchmark_premium._sha256_json(policy),
        "snapshot": context["snapshot"],
        "code": context["code"],
        "selectors": selectors,
    }
    where = analysis.lot_id(profile_id, seed)
    try:
        analysis.check_lot(lot, where, protocol, protocol_sha256, profile_id, seed)
    except IntegrityError as exc:
        raise ChildError(EXIT_CHILD_FAILED, exc.where, exc.reason) from exc

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    stages = _stage_costs(compose_stages(meter.stages))
    sampled = sampler.peak_kb()
    process_peak = ru_maxrss_kb()
    vm_hwm = read_proc_status_kb("VmHWM")
    cost = {
        "lot": where,
        "mode": context["mode"],
        "started_at": started_at,
        "finished_at": _utcnow(),
        "wall_s": time.perf_counter() - wall_start,
        "cpu_s": (usage_end.ru_utime - usage_start.ru_utime) + (usage_end.ru_stime - usage_start.ru_stime),
        "stages": stages,
        "rss_peak_kb": {
            "sampled": sampled,
            "ru_maxrss": process_peak,
            "vm_hwm": vm_hwm,
            **validate_rss(sampled, process_peak, vm_hwm),
        },
        "jobs": int(jobs),
        "load_avg_start": load_start,
        "load_avg_end": _loadavg(),
        "machine": benchmark_premium.machine_info(),
    }
    return {
        analysis.cost_path(lots_dir, profile_id, seed): analysis.dumps(cost),
        analysis.lot_path(lots_dir, profile_id, seed): analysis.dumps(lot),
    }


def _declaration_path(protocol: Mapping[str, Any]) -> Path:
    path = Path(protocol["snapshot_declaration"])
    return path if path.is_absolute() else Path(REPO_ROOT) / path


# --------------------------------------------------------------------------
# Parent: scheduling
# --------------------------------------------------------------------------


def schedule(
    tasks: Sequence[Any],
    jobs: int,
    spawn: Callable[[Any], Any],
    *,
    poll_interval: float = 1.0,
) -> list[dict[str, Any]]:
    """Run ``spawn(task)`` processes with at most ``jobs`` alive; wait for all."""

    pending = list(tasks)
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < jobs:
            task = pending.pop(0)
            running.append(
                {"task": task, "process": spawn(task), "started_at": _utcnow(), "start": time.perf_counter()}
            )
        alive = []
        for item in running:
            code = item["process"].poll()
            if code is None:
                alive.append(item)
                continue
            finished.append(
                {
                    "task": item["task"],
                    "returncode": code,
                    "started_at": item["started_at"],
                    "finished_at": _utcnow(),
                    "start": item["start"],
                    "end": time.perf_counter(),
                }
            )
        running = alive
        if running:
            time.sleep(poll_interval)
    return finished


def check_existing_lot(
    path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    context: Mapping[str, Any],
    profile_id: str,
    seed: int,
) -> None:
    """Raise ``IntegrityError`` unless the existing lot may be reused as is."""

    where = analysis.lot_id(profile_id, seed)
    lot = analysis.check_lot(
        analysis._read_json(path, where), where, protocol, protocol_sha256, profile_id, seed
    )
    for label, value, expected in (
        ("mode", lot.get("mode"), context["mode"]),
        ("snapshot", lot.get("snapshot"), context["snapshot"]),
        ("code_fingerprint", lot["code"].get("code_fingerprint"), context["code"]["code_fingerprint"]),
        ("experiment_fingerprint", lot["code"].get("experiment_fingerprint"), context["code"]["experiment_fingerprint"]),
    ):
        if value != expected:
            raise IntegrityError(where, f"existing lot has another {label} than HEAD; it is not overwritten")
    if not analysis.cost_path(path.parent.parent, profile_id, seed).is_file():
        raise IntegrityError(where, "existing lot has no cost file")


def build_context(protocol_path: Path, protocol_sha256: str, snapshot: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """What every child copies into its lot, computed once by the parent."""

    return {
        "mode": mode,
        "protocol_sha256": protocol_sha256,
        **protocol_provenance(protocol_path),
        "snapshot": {
            "manifest_id": snapshot["manifest_id"],
            "manifest_sha256": snapshot["manifest_sha256"],
            "sources": dict(snapshot["sources"]),
        },
        "code": code_info(),
    }


def _spawn_child(args: argparse.Namespace, context_path: Path) -> Callable[[tuple[str, int]], Any]:
    def spawn(task: tuple[str, int]) -> subprocess.Popen:
        profile_id, seed = task
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_child",
            "--protocol", str(Path(args.protocol).resolve()),
            "--registry", str(Path(args.registry).resolve()),
            "--lots-dir", str(Path(args.lots_dir).resolve()),
            "--profile", profile_id,
            "--seed", str(seed),
            "--jobs", str(args.jobs),
            "--context", str(context_path),
        ]
        if args.smoke:
            command.append("--smoke")
        return subprocess.Popen(command, stdout=subprocess.DEVNULL)

    return spawn


def command_run(args: argparse.Namespace, spawn_factory: Callable[..., Callable[[Any], Any]] = _spawn_child) -> int:
    import benchmark_premium

    if args.jobs < 1:
        _error("--jobs", "must be at least 1")
        return EXIT_USAGE
    try:
        protocol, protocol_sha256 = analysis.load_protocol(args.protocol)
    except ProtocolError as exc:
        _error(exc.field, exc.reason)
        return EXIT_USAGE
    mode = "smoke" if args.smoke else "measured"
    if not args.smoke:
        problems = preregistration_problems(args.protocol)
        if problems:
            _error("pre-registration", "; ".join(problems))
            return EXIT_PREREGISTRATION
    try:
        _manifest, snapshot = benchmark_premium.verify_snapshot(
            benchmark_premium.load_declaration(_declaration_path(protocol)), args.registry
        )
    except benchmark_premium.SnapshotDeclarationError as exc:
        _error("snapshot", str(exc))
        return EXIT_SNAPSHOT
    context = build_context(args.protocol, protocol_sha256, snapshot, mode)

    lots_dir = Path(args.lots_dir)
    tasks: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []
    for profile_id in protocol["profiles"]:
        for seed in protocol["seeds"]:
            path = analysis.lot_path(lots_dir, profile_id, seed)
            if path.exists():
                try:
                    check_existing_lot(path, protocol, protocol_sha256, context, profile_id, seed)
                except IntegrityError as exc:
                    _error(exc.where, exc.reason)
                    return EXIT_INTEGRITY
                skipped.append((profile_id, seed))
            else:
                tasks.append((profile_id, seed))
    for profile_id, seed in skipped:
        print(f"{PREFIX}: {analysis.lot_id(profile_id, seed)}: valid lot exists, skipped", file=sys.stderr)
    if not tasks:
        return EXIT_OK

    scratch = Path(tempfile.mkdtemp(prefix="multirun-context-"))
    try:
        context_path = scratch / "context.json"
        context_path.write_text(analysis.dumps(context), encoding="utf-8")
        finished = schedule(tasks, args.jobs, spawn_factory(args, context_path), poll_interval=args.poll_interval)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    failed = [item for item in finished if item["returncode"] != 0]
    for item in failed:
        profile_id, seed = item["task"]
        _error(analysis.lot_id(profile_id, seed), f"child exited with {item['returncode']}; no lot written")
    statuses = {task: "skipped" for task in skipped}
    statuses.update({item["task"]: ("failed" if item["returncode"] else "written") for item in finished})
    summary = {
        "protocol_id": protocol["protocol_id"],
        "mode": mode,
        "jobs": args.jobs,
        "experiment_wall_s": max(item["end"] for item in finished) - min(item["start"] for item in finished),
        "first_started_at": min(item["started_at"] for item in finished),
        "last_finished_at": max(item["finished_at"] for item in finished),
        "lots": [
            {"lot": analysis.lot_id(profile_id, seed), "status": statuses[(profile_id, seed)]}
            for profile_id in protocol["profiles"]
            for seed in protocol["seeds"]
        ],
    }
    analysis.write_files_atomically({lots_dir / "run-summary.json": analysis.dumps(summary)})
    return EXIT_CHILD_FAILED if failed else EXIT_OK


def command_child(args: argparse.Namespace, engines: Mapping[str, Callable[..., Any]] | None = None) -> int:
    where = f"{args.profile}:{args.seed}"
    try:
        context = json.loads(Path(args.context).read_text(encoding="utf-8"))
        files = run_child(
            Path(args.protocol),
            context,
            Path(args.registry),
            Path(args.lots_dir),
            args.profile,
            int(args.seed),
            jobs=args.jobs,
            smoke=args.smoke,
            engines=engines,
        )
    except ChildError as exc:
        _error(where, f"{exc.where}: {exc.reason}")
        return exc.code
    except ProtocolError as exc:
        _error(where, f"{exc.field}: {exc.reason}")
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - any other failure fails the child
        _error(where, repr(exc))
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return EXIT_CHILD_FAILED
    analysis.write_files_atomically(files)
    return EXIT_OK


def command_report(args: argparse.Namespace) -> int:
    try:
        files = analysis.run_report(Path(args.protocol), Path(args.lots_dir), Path(args.out_dir))
    except ProtocolError as exc:
        _error(exc.field, exc.reason)
        return EXIT_USAGE
    except IntegrityError as exc:
        _error(exc.where, exc.reason)
        return EXIT_INTEGRITY
    analysis.write_files_atomically(files)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run every missing (profile, seed) lot")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--lots-dir", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=5)
    run.add_argument("--smoke", action="store_true", help="population 20, 5 generations; lots marked smoke")
    run.add_argument("--poll-interval", type=float, default=1.0, help=argparse.SUPPRESS)

    report = commands.add_parser("report", help="build the report from the protocol and the lots")
    report.add_argument("--protocol", type=Path, required=True)
    report.add_argument("--lots-dir", type=Path, required=True)
    report.add_argument("--out-dir", type=Path, required=True)

    child = commands.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--protocol", type=Path, required=True)
    child.add_argument("--registry", type=Path, required=True)
    child.add_argument("--lots-dir", type=Path, required=True)
    child.add_argument("--profile", required=True)
    child.add_argument("--seed", type=int, required=True)
    child.add_argument("--jobs", type=int, required=True)
    child.add_argument("--context", type=Path, required=True)
    child.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    if not sys.platform.startswith("linux"):
        _error("platform", "Linux only (/proc and resource)")
        return EXIT_USAGE
    if args.command == "run":
        return command_run(args)
    if args.command == "report":
        return command_report(args)
    return command_child(args)


if __name__ == "__main__":
    sys.exit(main())
