#!/usr/bin/env python3
"""Measured performance baseline for the two-stage Premium pipeline (issue #8).

Runs ``run_premium_optimization`` for the conservador, moderado and arrojado
anchor profiles on one declared, hash-checked snapshot, with timed wrappers
injected in place of ``snapshot_loader``, ``allocation_engine``,
``stock_engine`` and ``fii_engine``. Production code is not modified: the
wrappers call the adapter's own default engines.

Per stage it records wall time, CPU time (process and rusage, self and
children), peak RSS sampled from ``/proc/self/status`` every second, and for
the stock and FII selectors the effective ``n_runs`` and generations executed.
``composition`` is the total minus the engine stages.

Linux only (``/proc`` and ``resource``).

Exit codes:
    0  success (and, with --baseline, no regression)
    1  the pipeline failed
    2  usage error
    3  the snapshot on disk is not the declared one
    4  RSS measurement is inconsistent with the process peak
    5  wall-time regression above the threshold versus --baseline
    6  --baseline is not comparable (code, snapshot, seed, machine or mode differ)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "api", PROJECT_ROOT / "py", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bench_measure import (  # noqa: E402 - needs the sys.path entries above
    ENGINE_STAGES,
    RSS_INTERVAL_S,
    RssSampler,
    StageMeter,
    capture_fii_runs,
    compose_stages,
    fii_inspect,
    read_proc_status_kb,
    ru_maxrss_kb,
    stock_inspect,
    validate_rss,
)

SCHEMA = "premium-benchmark-v1"
PROFILES = ("conservador", "moderado", "arrojado")
DEFAULT_DECLARATION = PROJECT_ROOT / "docs" / "validation" / "premium-performance-snapshot.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "snapshots"
DEFAULT_SEED = 42
DEFAULT_CAPITAL_BRL = 100_000.0
DEFAULT_THRESHOLD = 0.20
# Paths whose content defines "the same code" for baseline comparability.
CODE_PATHS = ("api/app", "py", "scripts/benchmark_premium.py", "scripts/bench_measure.py")

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2
EXIT_SNAPSHOT = 3
EXIT_RSS = 4
EXIT_REGRESSION = 5
EXIT_NOT_COMPARABLE = 6

class SnapshotDeclarationError(RuntimeError):
    """The snapshot on disk does not match the declared snapshot."""


# --------------------------------------------------------------------------
# Snapshot declaration
# --------------------------------------------------------------------------


def load_declaration(path: Path) -> dict[str, Any]:
    try:
        declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotDeclarationError(f"cannot read snapshot declaration {path}: {exc}") from exc
    missing = [key for key in ("manifest_id", "manifest_sha256", "sources") if key not in declaration]
    if missing:
        raise SnapshotDeclarationError(f"snapshot declaration {path} misses {missing}")
    return declaration


def verify_snapshot(declaration: Mapping[str, Any], registry: Path) -> tuple[object, dict[str, Any]]:
    """Fail unless the registry holds exactly the declared snapshot."""

    from snapshot_manifest import SnapshotManifest, SnapshotManifestError, sha256_path, validate_manifest

    manifest_path = Path(registry) / f"{declaration['manifest_id']}.json"
    try:
        manifest = SnapshotManifest.load(manifest_path)
    except SnapshotManifestError as exc:
        raise SnapshotDeclarationError(str(exc)) from exc
    if manifest.manifest_sha256 != declaration["manifest_sha256"]:
        raise SnapshotDeclarationError(
            f"manifest_sha256 mismatch for {manifest_path}: declared "
            f"{declaration['manifest_sha256']}, found {manifest.manifest_sha256}"
        )
    declared_sources = declaration["sources"]
    actual: dict[str, str] = {}
    for role in ("allocation", "stock", "fii"):
        if role not in declared_sources:
            raise SnapshotDeclarationError(f"snapshot declaration misses source {role}")
        source = manifest.sources.get(role)
        if source is None:
            raise SnapshotDeclarationError(f"manifest has no {role} source")
        path = source.resolved_path()
        try:
            digest = sha256_path(path)
        except SnapshotManifestError as exc:
            raise SnapshotDeclarationError(str(exc)) from exc
        if digest != declared_sources[role]:
            raise SnapshotDeclarationError(
                f"SHA-256 mismatch for {role} ({path}): declared {declared_sources[role]}, found {digest}"
            )
        actual[role] = digest
    try:
        validated = validate_manifest(manifest)
    except SnapshotManifestError as exc:
        raise SnapshotDeclarationError(f"manifest failed validation: {exc}") from exc
    report = {
        "manifest_id": validated.manifest_id,
        "manifest_sha256": validated.manifest_sha256,
        "manifest_path": str(manifest_path),
        "cutoff_date": validated.cutoff_date.isoformat(),
        "sources": actual,
        "selector_paths": {
            "stocks": str(validated.selector_source_for("stock").resolved_path()),
            "fiis": str(validated.selector_source_for("fii").resolved_path()),
        },
    }
    return validated, report


def selector_inputs_from_manifest(manifest: Any, snapshot: Mapping[str, Any]) -> Any:
    """Frozen selector CSVs from the manifest, in place of live ``refresh_inputs``."""

    from app.services.status_invest_inputs import SourceSnapshot, StatusInvestInputs

    def source(role: str, key: str) -> SourceSnapshot:
        return SourceSnapshot(
            path=Path(snapshot["selector_paths"][key]),
            provider="statusinvest",
            retrieved_at=str(manifest.created_at),
            sha256=snapshot["sources"][role],
        )

    return StatusInvestInputs(stocks=source("stock", "stocks"), fiis=source("fii", "fiis"))


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


def anchor_profile_record(profile: str) -> SimpleNamespace:
    """Profile v1 revision at the product's anchor score for ``profile``."""

    from app.services.profile import ANCHOR_SCORES

    score = float(ANCHOR_SCORES[profile])
    return SimpleNamespace(
        schema_version=1,
        version=1,
        suitability_score=score,
        raw_score=score,
        dimensions={key: score for key in ("apetite", "capacidade", "liquidez", "conhecimento")},
        generic_profile=profile,
        answers={"restricoes": ["nenhuma"]},
        restrictions_json=None,
        rules_json=[],
        warnings_json=[],
    )


def build_policy(profile: str, manifest: Any, seed: int, *, smoke: bool = False) -> dict[str, Any]:
    """Resolve the Premium policy exactly as ``PremiumRecommendationService`` does."""

    from app.services.premium_policy import POLICY_VERSION, resolve_recommendation_policy
    from app.services.premium_recommendation import _manifest_hashes

    cutoff = manifest.cutoff_date
    policy = resolve_recommendation_policy(
        anchor_profile_record(profile),
        rules_version=POLICY_VERSION,
        plan="premium",
        source_snapshot_ids=(manifest.manifest_id,),
        source_snapshot_hashes=_manifest_hashes(manifest),
        cutoff_date=cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff),
        random_seed=int(seed),
    ).to_dict()
    if smoke:
        for section in ("stocks", "fiis"):
            policy[section]["system_ga_config"].update(
                {"population": 20, "generations": 5, "run_count": 1, "adaptive_mode": False, "min_runs": 1}
            )
    return policy


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# One profile
# --------------------------------------------------------------------------


def default_engines() -> dict[str, Callable[..., Any]]:
    from app.adapters import premium_optimization as adapter

    return {
        "snapshot_loader": adapter._default_snapshot_loader,
        "allocation_engine": adapter._default_allocation_engine,
        "stock_engine": adapter._default_stock_engine,
        "fii_engine": adapter._default_fii_engine,
    }


def run_profile(
    profile: str,
    policy: Mapping[str, Any],
    manifest: Any,
    selector_inputs: Any,
    workspace: Path,
    *,
    seed: int,
    capital_brl: float = DEFAULT_CAPITAL_BRL,
    engines: Mapping[str, Callable[..., Any]] | None = None,
    fii_capture: Callable[[], Any] = capture_fii_runs,
    sampler: RssSampler | None = None,
    process_peak_kb: Callable[[], int] = ru_maxrss_kb,
    hwm_kb: Callable[[], int | None] = lambda: read_proc_status_kb("VmHWM"),
) -> dict[str, Any]:
    """Run one Premium optimization with every engine stage measured."""

    from app.adapters.premium_optimization import run_premium_optimization

    engines = dict(engines or default_engines())
    own_sampler = sampler is None
    sampler = sampler or RssSampler()
    if own_sampler:
        sampler.start()
    meter = StageMeter(sampler)
    try:
        with meter.measure("total"):
            result = run_premium_optimization(
                policy,
                manifest,
                Path(workspace),
                float(capital_brl),
                selector_inputs=selector_inputs,
                snapshot_loader=meter.wrap("snapshot_load", engines["snapshot_loader"]),
                allocation_engine=meter.wrap("allocation", engines["allocation_engine"]),
                stock_engine=meter.wrap("stocks", engines["stock_engine"], inspect=stock_inspect),
                fii_engine=meter.wrap(
                    "fiis", engines["fii_engine"], inspect=fii_inspect, around=fii_capture
                ),
                seed=int(seed),
            )
    finally:
        if own_sampler:
            sampler.stop()

    stages = compose_stages(meter.stages)
    sampled_peak = sampler.peak_kb()
    process_peak = int(process_peak_kb())
    vm_hwm = hwm_kb()
    rss = {
        "interval_s": sampler.interval,
        "samples": len(sampler.snapshot()),
        "max_gap_s": sampler.max_gap_s(),
        "sampled_peak_kb": sampled_peak,
        "ru_maxrss_kb": process_peak,
        "vm_hwm_kb": vm_hwm,
        "ru_maxrss_children_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
    }
    rss.update(validate_rss(sampled_peak, process_peak, vm_hwm))

    engine_stats: dict[str, Any] = {
        "allocation": {"ga": False, "note": "ADR 0003: equal-weight allocation, no GA"},
    }
    for section in ("stocks", "fiis"):
        ga = dict(policy[section]["system_ga_config"])
        stats = dict(stages[section].pop("engine", {}) or {})
        stats.update(
            {
                "ga": True,
                "run_count_configured": ga.get("run_count"),
                "generations_configured": ga.get("generations"),
                "population_configured": ga.get("population"),
                "adaptive_mode": ga.get("adaptive_mode"),
                "min_runs": ga.get("min_runs"),
            }
        )
        engine_stats[section] = stats
    return {
        "profile": profile,
        "seed": int(seed),
        "capital_brl": float(capital_brl),
        "policy_sha256": _sha256_json(policy),
        "stages": stages,
        "engines": engine_stats,
        "rss": rss,
        "output": {
            "stocks": [item["ticker"] for item in result.get("stocks", [])],
            "fiis": [item["ticker"] for item in result.get("fiis", [])],
            "classes": {item.get("key", str(i)): item.get("target_weight") for i, item in enumerate(result.get("classes", []))},
        },
    }


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def git_info() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    code_status = _git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS)
    trees = {path: _git("rev-parse", f"HEAD:{path}") for path in CODE_PATHS}
    fingerprint = (
        hashlib.sha256(json.dumps(trees, sort_keys=True).encode()).hexdigest()
        if all(trees.values())
        else None
    )
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "code_dirty": bool(code_status) if code_status is not None else None,
        "code_trees": trees,
        "code_fingerprint": fingerprint,
    }


def _cpuinfo() -> tuple[str | None, int | None]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    model = None
    cores: set[tuple[str, str]] = set()
    physical = core = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "physical id":
            physical = value
        elif key == "core id":
            core = value
        elif not line.strip() and core is not None:
            cores.add((physical or "0", core))
            physical = core = None
    if core is not None:
        cores.add((physical or "0", core))
    return model, (len(cores) or None)


def _os_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        return None
    return None


def machine_info() -> dict[str, Any]:
    model, physical_cores = _cpuinfo()
    mem_total_kb = read_proc_status_kb("MemTotal", Path("/proc/meminfo"))
    info = {
        "cpu_model": model,
        "physical_cores": physical_cores,
        "logical_cpus": os.cpu_count(),
        "mem_total_kb": mem_total_kb,
        "os": _os_release(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    info["fingerprint"] = _sha256_json(
        {key: info[key] for key in ("cpu_model", "logical_cpus", "mem_total_kb", "os", "kernel", "arch", "python")}
    )
    return info


# --------------------------------------------------------------------------
# Baseline comparison
# --------------------------------------------------------------------------


def compare_to_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compare per-profile total wall time; ``status`` drives the exit code."""

    problems: list[str] = []
    for label, getter in (
        ("schema", lambda r: r.get("schema")),
        ("mode", lambda r: r.get("mode")),
        ("code fingerprint", lambda r: (r.get("git") or {}).get("code_fingerprint")),
        ("snapshot manifest_sha256", lambda r: (r.get("snapshot") or {}).get("manifest_sha256")),
        ("snapshot sources", lambda r: (r.get("snapshot") or {}).get("sources")),
        ("seed", lambda r: r.get("seed")),
        ("machine fingerprint", lambda r: (r.get("machine") or {}).get("fingerprint")),
    ):
        left, right = getter(baseline), getter(current)
        if left is None or right is None or left != right:
            problems.append(f"{label} differs: baseline={left!r} current={right!r}")
    for label, report in (("baseline", baseline), ("current", current)):
        if (report.get("git") or {}).get("code_dirty") is not False:
            problems.append(f"{label} was measured on a dirty or unknown code tree")
    rows = []
    for profile, measured in sorted((current.get("profiles") or {}).items()):
        reference = (baseline.get("profiles") or {}).get(profile)
        if reference is None:
            problems.append(f"profile {profile} is missing from the baseline")
            continue
        if reference.get("policy_sha256") != measured.get("policy_sha256"):
            problems.append(f"profile {profile} policy differs from the baseline")
        base_wall = float(reference["stages"]["total"]["wall_s"])
        wall = float(measured["stages"]["total"]["wall_s"])
        ratio = (wall / base_wall - 1.0) if base_wall > 0 else None
        rows.append(
            {
                "profile": profile,
                "baseline_wall_s": base_wall,
                "current_wall_s": wall,
                "delta_ratio": ratio,
                "regression": ratio is not None and ratio > threshold,
            }
        )
    if problems:
        status = "not_comparable"
    elif any(row["regression"] for row in rows):
        status = "regression"
    else:
        status = "ok"
    return {"status": status, "threshold": threshold, "problems": problems, "profiles": rows}


def comparison_exit_code(comparison: Mapping[str, Any]) -> int:
    return {
        "ok": EXIT_OK,
        "regression": EXIT_REGRESSION,
        "not_comparable": EXIT_NOT_COMPARABLE,
    }[comparison["status"]]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Premium benchmark ({report.get('mode')})",
        "",
        f"- commit: `{report['git'].get('commit')}` (dirty: {report['git'].get('dirty')}, code dirty: {report['git'].get('code_dirty')})",
        f"- snapshot: `{report['snapshot']['manifest_id']}` manifest_sha256 `{report['snapshot']['manifest_sha256']}`",
        f"- seed: {report['seed']}; RSS interval: {RSS_INTERVAL_S:.0f} s",
        f"- machine: {report['machine'].get('cpu_model')}, {report['machine'].get('physical_cores')} cores / "
        f"{report['machine'].get('logical_cpus')} threads, {_fmt((report['machine'].get('mem_total_kb') or 0) / 1048576)} GiB, "
        f"{report['machine'].get('os')}, Python {report['machine'].get('python')}",
        f"- command: `{report['command']}`",
        "",
        "| perfil | etapa | wall (s) | CPU (s) | CPU filhos (s) | CPU/wall | pico RSS amostrado (MiB) | n_runs efetivo | gerações (total / média) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile, measured in report["profiles"].items():
        for stage in (*ENGINE_STAGES, "composition", "total"):
            data = measured["stages"][stage]
            engine = measured["engines"].get(stage, {})
            gens = (
                f"{_fmt(engine.get('generations_total'), 0)} / {_fmt(engine.get('generations_mean'), 1)}"
                if engine.get("ga")
                else "n/a"
            )
            peak = data.get("rss_peak_sampled_kb")
            lines.append(
                f"| {profile} | {stage} | {_fmt(data['wall_s'])} | {_fmt(data['cpu_seconds'])} | "
                f"{_fmt(data['cpu_children_seconds'])} | {_fmt(data.get('cpu_utilization'))} | "
                f"{_fmt(peak / 1024 if peak else None, 1)} | "
                f"{_fmt(engine.get('n_runs_effective')) if engine.get('ga') else 'n/a'} | {gens} |"
            )
        rss = measured["rss"]
        lines.append(
            f"| {profile} | RSS check | | | | | sampled {_fmt((rss['sampled_peak_kb'] or 0) / 1024, 1)} vs ru_maxrss "
            f"{_fmt(rss['ru_maxrss_kb'] / 1024, 1)} / VmHWM {_fmt((rss['vm_hwm_kb'] or 0) / 1024, 1)} ({rss['status']}) | | |"
        )
    comparison = report.get("comparison")
    if comparison:
        lines += ["", f"Baseline comparison: **{comparison['status']}** (threshold {comparison['threshold']:.0%})"]
        lines += [f"- {problem}" for problem in comparison["problems"]]
        for row in comparison["profiles"]:
            lines.append(
                f"- {row['profile']}: {_fmt(row['baseline_wall_s'])} s -> {_fmt(row['current_wall_s'])} s "
                f"({_fmt((row['delta_ratio'] or 0) * 100, 1)}%){' REGRESSION' if row['regression'] else ''}"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated subset of conservador,moderado,arrojado")
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION, help="declared snapshot hashes (JSON)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="snapshot registry holding <manifest_id>.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL_BRL, help="investable capital in BRL")
    parser.add_argument("--output-dir", type=Path, required=False, help="where summary.json, summary.md and <profile>.json are written")
    parser.add_argument("--baseline", type=Path, help="baseline summary.json; exit 5 on a wall-time regression above --threshold")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="allowed wall-time regression ratio (default 0.20)")
    parser.add_argument("--compare-only", type=Path, metavar="SUMMARY", help="compare an existing summary.json with --baseline without running")
    parser.add_argument("--isolation", choices=("process", "none"), default="process", help="run each profile in a fresh Python process (default) so ru_maxrss is per profile")
    parser.add_argument("--workspace", type=Path, help="keep engine workspaces here instead of a temporary directory")
    parser.add_argument("--smoke", action="store_true", help="wiring check with a tiny GA (1 run, 5 generations); not a measurement")
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _profiles(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in PROFILES]
    if not values or unknown:
        raise ValueError(f"unknown profiles: {unknown or raw!r}")
    return values


def _measure_in_process(profile: str, args: argparse.Namespace, manifest: Any, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    policy = build_policy(profile, manifest, args.seed, smoke=args.smoke)
    selector_inputs = selector_inputs_from_manifest(manifest, snapshot)
    if args.workspace:
        workspace = Path(args.workspace) / profile
        return run_profile(profile, policy, manifest, selector_inputs, workspace, seed=args.seed, capital_brl=args.capital)
    with tempfile.TemporaryDirectory(prefix=f"premium-bench-{profile}-") as tmp:
        return run_profile(profile, policy, manifest, selector_inputs, Path(tmp), seed=args.seed, capital_brl=args.capital)


def _measure_in_child(profile: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    child_output = output_dir / f".{profile}.child.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profiles", profile,
        "--isolation", "none",
        "--declaration", str(args.declaration),
        "--registry", str(args.registry),
        "--seed", str(args.seed),
        "--capital", str(args.capital),
        "--child-output", str(child_output),
    ]
    if args.workspace:
        command += ["--workspace", str(args.workspace)]
    if args.smoke:
        command.append("--smoke")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(f"profile {profile} exited with {completed.returncode}")
    measured = json.loads(child_output.read_text(encoding="utf-8"))
    child_output.unlink()
    return measured


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        print("benchmark_premium: Linux only (/proc and resource)", file=sys.stderr)
        return EXIT_USAGE

    if args.compare_only:
        if not args.baseline:
            print("--compare-only requires --baseline", file=sys.stderr)
            return EXIT_USAGE
        current = json.loads(args.compare_only.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparison = compare_to_baseline(current, baseline, args.threshold)
        print(json.dumps(comparison, indent=2))
        return comparison_exit_code(comparison)

    try:
        profiles = _profiles(args.profiles)
    except ValueError as exc:
        print(f"benchmark_premium: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not args.child_output and not args.output_dir:
        print("benchmark_premium: --output-dir is required", file=sys.stderr)
        return EXIT_USAGE

    # Snapshot identity is checked before any measurement starts.
    try:
        manifest, snapshot = verify_snapshot(load_declaration(args.declaration), args.registry)
    except SnapshotDeclarationError as exc:
        print(f"benchmark_premium: snapshot is not the declared one: {exc}", file=sys.stderr)
        return EXIT_SNAPSHOT

    if args.child_output:
        try:
            measured = _measure_in_process(profiles[0], args, manifest, snapshot)
        except Exception as exc:  # noqa: BLE001 - reported to the parent via the exit code
            print(f"benchmark_premium: profile {profiles[0]} failed: {exc!r}", file=sys.stderr)
            return EXIT_RUN_FAILED
        _write_json(args.child_output, measured)
        return EXIT_OK

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "smoke" if args.smoke else "measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([Path(sys.executable).name, *sys.argv]) if argv is None else shlex.join(["benchmark_premium.py", *argv]),
        "git": git_info(),
        "machine": machine_info(),
        "snapshot": snapshot,
        "seed": args.seed,
        "capital_brl": args.capital,
        "isolation": args.isolation,
        "rss_interval_s": RSS_INTERVAL_S,
        "profiles": {},
    }
    for profile in profiles:
        print(f"benchmark_premium: running {profile} ...", file=sys.stderr, flush=True)
        try:
            if args.isolation == "process":
                measured = _measure_in_child(profile, args, output_dir)
            else:
                measured = _measure_in_process(profile, args, manifest, snapshot)
        except Exception as exc:  # noqa: BLE001
            print(f"benchmark_premium: {exc}", file=sys.stderr)
            return EXIT_RUN_FAILED
        report["profiles"][profile] = measured
        _write_json(output_dir / f"{profile}.json", measured)
        print(
            f"benchmark_premium: {profile} total {measured['stages']['total']['wall_s']:.1f} s, "
            f"RSS {measured['rss']['status']}",
            file=sys.stderr,
            flush=True,
        )

    exit_code = EXIT_OK
    if any(item["rss"]["status"] == "invalid" for item in report["profiles"].values()):
        exit_code = EXIT_RSS
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        report["comparison"] = compare_to_baseline(report, baseline, args.threshold)
        if exit_code == EXIT_OK:
            exit_code = comparison_exit_code(report["comparison"])
    _write_json(output_dir / "summary.json", report)
    (output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    if report.get("comparison"):
        print(json.dumps(report["comparison"], indent=2), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
