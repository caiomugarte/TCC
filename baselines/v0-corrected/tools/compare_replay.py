#!/usr/bin/env python3
"""compare_replay.py - the opt-in 80-run legacy-vs-corrected comparison (VC-20..VC-22, VC-27).

A run is one (profile, seed) pair: 20 seeds x 4 frozen profiles = 80 runs, run_ids 0-19
= seeds 42-61 under the frozen `seed_formula` (`seed_base + run_id`, seed_base 42). Per
run BOTH drivers execute in the SAME declared environment - subprocesses of
`sys.executable`, one spawn per (profile, run_id), each in its own scratch root - and
the proof records, per driver separately, `reported_fitness`, `recomputed_fitness`,
`tickers` (sorted) and `hhi` (design.md D-a).

The falsifiable expectation table (spec.md) is data in EXPECTATIONS below; the checker
recomputes every expectation from `runs[]` and any violation is reported as
`unexpected divergence`, naming the expectation row, and exits non-zero - a violation is
never folded into "expected" (VC-21). The top-level summary carries the COMPUTED run
denominator so the gate cannot silently change scope (VC-20).

Opt-in only: nothing executes unless TCC_V0_CORRECTED_REPLAY=1 (VC-22); without the
flag every mode exits 2 before any driver spawns, so the default test run stays fast.

Modes:

- default: execute both drivers over the 80 runs, check E1-E4, tie every corrected row
  to the committed reference (VC-29 live tie, run_id 0-19), and record the proof at
  --proof (first run records itself; an existing record must match exactly - the
  proof-as-record contract - and is never overwritten).
- `--check-proof PATH`: run the checker over an existing proof payload without
  executing anything (the VC-27 negative-fixture path).

Exit codes: 0 pass; 1 unexpected divergence / tie failure / record mismatch;
2 the opt-in flag is unset (zero execution); 3 runtime error.
"""

import sys

sys.dont_write_bytecode = True  # before any bundle import (AD-013)

import argparse
import ast
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path


def _load_sibling(name):
    """The bundle's own sibling tool, loaded from this file's directory under a unique name."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v0_corrected_replay_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vd = _load_sibling("_verdict")

SCHEMA_VERSION = 1
PROFILES = ("conservador", "moderado", "arrojado", "caio")
RUN_ID_START = 0
N_RUN_IDS = 20
SEED_BASE = 42
TOL = 1e-9  # VC-01/VC-29 recompute tolerance

ENV_FLAG = "TCC_V0_CORRECTED_REPLAY"
TOOLS_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOLS_DIR.parent
LEGACY_BUNDLE = BUNDLE_DIR.parent / "v0-legacy"
REPO_ROOT = BUNDLE_DIR.parents[1]
DEFAULT_PROOF = (
    REPO_ROOT / ".specs" / "features" / "p0-02-v0-corrected" / "proofs" / "replay_20seeds.json"
)
LEGACY_DRIVER = "_legacy_driver.py"   # per-bundle: <bundle>/tools/_legacy_driver.py
CORRECTED_DRIVER = "_corrected_driver.py"

EXIT_OK = 0
EXIT_DIVERGENCE = 1
EXIT_NO_OPTIN = 2
EXIT_RUNTIME = 3

TOP_LEVEL_KEYS = (
    "schema_version",
    "denominator",
    "run_id_range",
    "seeds",
    "profiles",
    "env",
    "expectations",
    "runs",
)
RUN_KEYS = ("profile", "run_id", "seed", "legacy", "corrected", "ticker_sets_equal")
DRIVER_KEYS = ("reported_fitness", "recomputed_fitness", "tickers", "hhi")
EXPECTATION_KEYS = ("name", "held", "numerator", "denominator")
ENV_KEYS = ("python_VV", "numpy", "pandas")


class ReplayError(Exception):
    """The replay cannot run or its driver output is unusable."""


# ---------------------------------------------------------------------------
# The falsifiable expectation table (spec.md) as data. `fold` names a function in
# FOLDS mapping runs[] to a numerator; `threshold` decides `held` over the computed
# denominator: ("all", _) = numerator == denominator; ("min", k) = numerator >= k.
# ---------------------------------------------------------------------------

EXPECTATIONS = (
    {
        "name": "E1",
        "claim": "reported fitness (legacy) == reported fitness (corrected), bit-identical, on 80/80 runs",
        "fold": "reported_bit_identity",
        "threshold": ("all", None),
        "violation": "The relocation changed more than the capture; the 'same protocol' claim is false",
    },
    {
        "name": "E2",
        "claim": "recomputed fitness (corrected) == reported fitness (corrected) on 100% of rows",
        "fold": "corrected_recompute_agreement",
        "threshold": ("all", None),
        "violation": "The corrected bundle still mispairs fitness and chromosome",
    },
    {
        "name": "E3",
        "claim": "ticker sets (legacy) != ticker sets (corrected) on more than half of the runs (>= 41/80)",
        "fold": "ticker_set_divergence",
        "threshold": ("min", 41),
        "violation": "The defect would not change what is returned, undermining its materiality",
    },
    {
        "name": "E4",
        "claim": "recomputed fitness (legacy) != reported fitness (legacy) on >= 1 run",
        "fold": "legacy_recompute_divergence",
        "threshold": ("min", 1),
        "violation": "The legacy artifacts would recompute clean, contradicting the 120/120 measurement",
    },
)


def _reported_bit_identity(runs):
    """E1: runs whose two reported fitness values are bit-identical (packed float64)."""
    return sum(
        1
        for run in runs
        if vd.float_bitwise_equal(
            run["legacy"]["reported_fitness"], run["corrected"]["reported_fitness"]
        )
    )


def _corrected_recompute_agreement(runs):
    """E2: runs whose corrected recomputed fitness equals its reported fitness (1e-9)."""
    return sum(
        1
        for run in runs
        if _within(run["corrected"]["recomputed_fitness"], run["corrected"]["reported_fitness"])
    )


def _ticker_set_divergence(runs):
    """E3: runs whose legacy and corrected ticker sets differ."""
    return sum(
        1
        for run in runs
        if not vd.tickers_equal(run["legacy"]["tickers"], run["corrected"]["tickers"])
    )


def _legacy_recompute_divergence(runs):
    """E4: runs whose legacy recomputed fitness differs from its reported fitness (1e-9)."""
    return sum(
        1
        for run in runs
        if not _within(run["legacy"]["recomputed_fitness"], run["legacy"]["reported_fitness"])
    )


FOLDS = {
    "reported_bit_identity": _reported_bit_identity,
    "corrected_recompute_agreement": _corrected_recompute_agreement,
    "ticker_set_divergence": _ticker_set_divergence,
    "legacy_recompute_divergence": _legacy_recompute_divergence,
}


def _within(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def expectation_rows(runs):
    """The per-expectation summary rows, recomputed from the run rows (never copied)."""
    rows = []
    denominator = len(runs)
    for expectation in EXPECTATIONS:
        numerator = FOLDS[expectation["fold"]](runs)
        mode, bound = expectation["threshold"]
        if mode == "all":
            held = numerator == denominator
        else:
            held = numerator >= bound
        rows.append(
            {
                "name": expectation["name"],
                "held": bool(held),
                "numerator": int(numerator),
                "denominator": int(denominator),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Proof construction (design.md D-a schema)
# ---------------------------------------------------------------------------


def capture_env():
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python_VV": "Python " + sys.version,
        "numpy": version("numpy"),
        "pandas": version("pandas"),
    }


def build_proof(runs, env):
    """The replay_20seeds.json payload: computed denominator + E1-E4 fold + run rows."""
    ordered = _canonical_order(runs)
    run_ids = sorted({int(run["run_id"]) for run in ordered})
    seeds = sorted({int(run["seed"]) for run in ordered})
    profiles = [p for p in PROFILES if any(run["profile"] == p for run in ordered)]
    return {
        "schema_version": SCHEMA_VERSION,
        "denominator": len(ordered),
        "run_id_range": [run_ids[0], run_ids[-1]] if run_ids else [],
        "seeds": seeds,
        "profiles": profiles,
        "env": dict(env),
        "expectations": expectation_rows(ordered),
        "runs": ordered,
    }


def _canonical_order(runs):
    def key(run):
        profile = run.get("profile")
        order = PROFILES.index(profile) if profile in PROFILES else len(PROFILES)
        return (order, str(profile), int(run.get("run_id", -1)))

    return [dict(run) for run in sorted(runs, key=key)]


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


def check_schema(proof):
    """Structural problems of a proof payload ( [] = the payload is well-formed ).

    The scope is pinned, not inferred: 20 run_ids (0-19) x 4 frozen profiles, seeds
    42-61, exactly one row per (profile, run_id) - so the gate cannot silently change
    scope (VC-20).
    """
    problems = []
    if not isinstance(proof, dict):
        return ["proof schema: payload is not a JSON object"]
    missing = [key for key in TOP_LEVEL_KEYS if key not in proof]
    if missing:
        return [f"proof schema: missing top-level key(s) {missing}"]
    extra = [key for key in proof if key not in TOP_LEVEL_KEYS]
    if extra:
        problems.append(f"proof schema: unexpected top-level key(s) {extra}")
    if proof["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"proof schema: schema_version {proof['schema_version']!r} != {SCHEMA_VERSION}"
        )

    runs = proof["runs"]
    if not isinstance(runs, list):
        problems.append("proof schema: runs is not a list")
        return problems
    if proof["denominator"] != len(runs):
        problems.append(
            f"proof schema: denominator {proof['denominator']!r} != {len(runs)} run rows"
        )
    if len(runs) != len(PROFILES) * N_RUN_IDS:
        problems.append(
            f"proof schema: {len(runs)} run rows != {len(PROFILES) * N_RUN_IDS} "
            f"({N_RUN_IDS} run_ids x {len(PROFILES)} profiles)"
        )
    expected_pairs = {
        (profile, run_id)
        for profile in PROFILES
        for run_id in range(RUN_ID_START, RUN_ID_START + N_RUN_IDS)
    }
    seen_pairs = set()
    for number, run in enumerate(runs):
        label = f"runs[{number}]"
        if not isinstance(run, dict):
            problems.append(f"proof schema: {label} is not an object")
            continue
        if set(run) != set(RUN_KEYS):
            problems.append(
                f"proof schema: {label} keys {sorted(run)} != {sorted(RUN_KEYS)}"
            )
            continue
        profile, run_id, seed = run["profile"], run_id_of(run), run["seed"]
        pair = (profile, run_id)
        if pair in seen_pairs:
            problems.append(f"proof schema: duplicate run for {pair[0]} run_id {pair[1]}")
        seen_pairs.add(pair)
        if pair not in expected_pairs:
            problems.append(f"proof schema: out-of-scope run {pair[0]} run_id {pair[1]}")
        if seed != SEED_BASE + run_id:
            problems.append(
                f"proof schema: {label} seed {seed!r} != {SEED_BASE + run_id} (seed_base + run_id)"
            )
        for side in ("legacy", "corrected"):
            driver = run[side]
            if not isinstance(driver, dict) or set(driver) != set(DRIVER_KEYS):
                problems.append(
                    f"proof schema: {label}.{side} keys "
                    f"{sorted(driver) if isinstance(driver, dict) else driver!r} != {sorted(DRIVER_KEYS)}"
                )
                continue
            for key in ("reported_fitness", "recomputed_fitness", "hhi"):
                if not isinstance(driver[key], (int, float)) or isinstance(driver[key], bool):
                    problems.append(f"proof schema: {label}.{side}.{key} is not numeric")
            tickers = driver["tickers"]
            if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                problems.append(f"proof schema: {label}.{side}.tickers is not a list of strings")
        if not isinstance(run["ticker_sets_equal"], bool):
            problems.append(f"proof schema: {label}.ticker_sets_equal is not a boolean")
        elif isinstance(run["legacy"], dict) and isinstance(run["corrected"], dict):
            if set(run["legacy"]) == set(DRIVER_KEYS) and set(run["corrected"]) == set(DRIVER_KEYS):
                recomputed = vd.tickers_equal(
                    run["legacy"]["tickers"], run["corrected"]["tickers"]
                )
                if run["ticker_sets_equal"] != recomputed:
                    problems.append(
                        f"proof schema: {label}.ticker_sets_equal {run['ticker_sets_equal']} "
                        f"disagrees with the recomputed ticker sets"
                    )
    missing_pairs = expected_pairs - seen_pairs
    for profile, run_id in sorted(missing_pairs):
        problems.append(f"proof schema: missing run for {profile} run_id {run_id}")

    if proof["run_id_range"] != [RUN_ID_START, RUN_ID_START + N_RUN_IDS - 1]:
        problems.append(
            f"proof schema: run_id_range {proof['run_id_range']!r} != "
            f"{[RUN_ID_START, RUN_ID_START + N_RUN_IDS - 1]}"
        )
    expected_seeds = list(range(SEED_BASE, SEED_BASE + N_RUN_IDS))
    if proof["seeds"] != expected_seeds:
        problems.append(f"proof schema: seeds {proof['seeds']!r} != {expected_seeds}")
    if proof["profiles"] != list(PROFILES):
        problems.append(f"proof schema: profiles {proof['profiles']!r} != {list(PROFILES)}")

    if not isinstance(proof["env"], dict) or set(proof["env"]) != set(ENV_KEYS):
        problems.append(
            f"proof schema: env keys {sorted(proof['env']) if isinstance(proof['env'], dict) else proof['env']!r}"
            f" != {sorted(ENV_KEYS)}"
        )
    else:
        for key in ENV_KEYS:
            if proof["env"][key] is not None and not isinstance(proof["env"][key], str):
                problems.append(f"proof schema: env.{key} is not a string")

    expectations = proof["expectations"]
    if not isinstance(expectations, list):
        problems.append("proof schema: expectations is not a list")
    else:
        names = [row.get("name") for row in expectations if isinstance(row, dict)]
        expected_names = [row["name"] for row in EXPECTATIONS]
        if names != expected_names:
            problems.append(
                f"proof schema: expectation rows {names!r} != {expected_names}"
            )
        for row in expectations:
            if not isinstance(row, dict) or set(row) != set(EXPECTATION_KEYS):
                problems.append(
                    f"proof schema: expectation row keys "
                    f"{sorted(row) if isinstance(row, dict) else row!r} != {sorted(EXPECTATION_KEYS)}"
                )
    return problems


def run_id_of(run):
    value = run.get("run_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def check_expectations(proof):
    """E1-E4 recomputed from runs[] against the summary ( [] = every row holds ).

    Every violated expectation is reported as `unexpected divergence`, naming the row -
    it is never folded into "expected" (VC-21). A summary that disagrees with the fold
    recomputed from the run rows is a separate, equally fatal finding.
    """
    problems = check_schema(proof)
    if problems:
        return problems
    runs = proof["runs"]
    recomputed = expectation_rows(runs)
    recorded = {row["name"]: row for row in proof["expectations"]}
    for row in recomputed:
        summary = recorded.get(row["name"])
        if summary is None:
            continue  # already reported by the schema check
        for field in ("held", "numerator", "denominator"):
            if summary[field] != row[field]:
                problems.append(
                    f"summary disagrees with the recomputed fold: {row['name']} {field} "
                    f"summary={summary[field]!r} recomputed={row[field]!r}"
                )
    for expectation, row in zip(EXPECTATIONS, recomputed):
        if row["held"]:
            continue
        mode, bound = expectation["threshold"]
        needed = f"{row['denominator']}/{row['denominator']}" if mode == "all" else f">= {bound}"
        offenders = _offending_runs(runs, expectation["fold"])
        problems.append(
            f"unexpected divergence: {expectation['name']} {expectation['claim']} - "
            f"holds on {row['numerator']}/{row['denominator']} runs (needs {needed}"
            f"{offenders}); violation means: {expectation['violation']}"
        )
    return problems


def _offending_runs(runs, fold_name):
    """Short run-id list of the rows behind one fold's misses (for the divergence line)."""
    if fold_name == "reported_bit_identity":
        miss = lambda run: not vd.float_bitwise_equal(  # noqa: E731
            run["legacy"]["reported_fitness"], run["corrected"]["reported_fitness"]
        )
    elif fold_name == "corrected_recompute_agreement":
        miss = lambda run: not _within(  # noqa: E731
            run["corrected"]["recomputed_fitness"], run["corrected"]["reported_fitness"]
        )
    elif fold_name == "ticker_set_divergence":
        miss = lambda run: vd.tickers_equal(  # noqa: E731
            run["legacy"]["tickers"], run["corrected"]["tickers"]
        )
    else:
        miss = lambda run: _within(  # noqa: E731
            run["legacy"]["recomputed_fitness"], run["legacy"]["reported_fitness"]
        )
    names = [f"{run['profile']}/{run_id_of(run)}" for run in runs if miss(run)]
    if not names:
        return ""
    shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
    return f"; runs: {shown}"


# ---------------------------------------------------------------------------
# Reference tie (VC-29 live leg, run_id 0-19) and the proof-as-record contract
# ---------------------------------------------------------------------------


def check_proof_vs_reference(proof, reference_dir, tol=TOL):
    """Every corrected run row ties to the committed reference row ([] = tied).

    `tickers` as a set and `fitness`/`hhi` within tol against
    `reference/metrics_stability_<profile>.csv` for the same (profile, run_id) - the
    committed reference is provably the output of the corrected code, not a relabelled
    legacy result (VC-29).
    """
    reference_dir = Path(reference_dir)
    problems = []
    recorded = {}
    for profile in PROFILES:
        path = reference_dir / f"metrics_stability_{profile}.csv"
        try:
            recorded[profile] = {
                row["run_id"]: row for row in _read_metrics_rows(path)
            }
        except (OSError, ValueError, ReplayError) as exc:
            problems.append(f"reference tie: cannot read {path}: {exc}")
            return problems
    for run in proof.get("runs", []):
        profile, run_id = run.get("profile"), run_id_of(run)
        label = f"{profile} run_id {run_id}"
        reference_row = recorded.get(profile, {}).get(run_id)
        if reference_row is None:
            problems.append(f"reference tie: {label}: no committed reference row")
            continue
        driver_row = {
            "tickers": run["corrected"]["tickers"],
            "fitness": run["corrected"]["reported_fitness"],
            "hhi": run["corrected"]["hhi"],
        }
        for line in vd.tie_row(reference_row, driver_row, tol):
            problems.append(f"reference tie: {label}: {line}")
    return problems


def _read_metrics_rows(path):
    """The recorded run rows of one metrics_stability CSV, keyed fields parsed."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for number, row in enumerate(csv.DictReader(handle, strict=True), start=2):
            try:
                rows.append(
                    {
                        "run_id": int(row["run_id"]),
                        "seed": int(row["seed"]),
                        "tickers": list(ast.literal_eval(row["tickers"])),
                        "fitness": float(row["fitness"]),
                        "hhi": float(row["hhi"]),
                    }
                )
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                raise ReplayError(f"{path} line {number} is not a usable run row: {exc}") from exc
    return rows


def check_proof_vs_committed(proof, committed_path):
    """A later spot re-run's rows equal the committed record ([] = proof-as-record holds).

    Every run row of `proof` must match its committed counterpart exactly; committed
    rows a spot re-run did not execute are left alone. A missing record is a finding.
    """
    path = Path(committed_path)
    problems = []
    try:
        committed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"record mismatch: cannot read the committed proof at {path}: {exc}"]
    committed_rows = {
        (row.get("profile"), run_id_of(row)): row
        for row in committed.get("runs", [])
        if isinstance(row, dict)
    }
    for run in proof.get("runs", []):
        pair = (run.get("profile"), run_id_of(run))
        label = f"{pair[0]} run_id {pair[1]}"
        expected = committed_rows.get(pair)
        if expected is None:
            problems.append(f"record mismatch: {label}: no committed row to compare against")
        elif expected != run:
            problems.append(
                f"record mismatch: {label}: the re-run row differs from the committed row"
            )
    return problems


# ---------------------------------------------------------------------------
# Execution: one spawn per (profile, run_id) per driver, both in scratch roots
# ---------------------------------------------------------------------------


def run_pair(profile, run_id, legacy_root, corrected_root):
    """Both drivers for one (profile, run_id) in the same interpreter/environment.

    Each driver runs as a subprocess of `sys.executable` with cwd inside its own
    scratch root holding scratch copies of the bundle's code/, data/ and a one-run
    protocol.json - nothing is written under either bundle. Returns the run row of the
    D-a schema.
    """
    legacy_row = _spawn_driver_row(legacy_root, LEGACY_DRIVER, profile, run_id)
    corrected_row = _spawn_driver_row(corrected_root, CORRECTED_DRIVER, profile, run_id)
    legacy = _driver_view(legacy_row)
    corrected = _driver_view(corrected_row)
    return {
        "profile": profile,
        "run_id": int(run_id),
        "seed": SEED_BASE + int(run_id),
        "legacy": legacy,
        "corrected": corrected,
        "ticker_sets_equal": vd.tickers_equal(legacy["tickers"], corrected["tickers"]),
    }


def _driver_view(row):
    return {
        "reported_fitness": float(row["fitness"]),
        "recomputed_fitness": float(row["recomputed_fitness"]),
        "tickers": list(row["tickers"]),
        "hhi": float(row["hhi"]),
    }


def _spawn_driver_row(bundle, driver_name, profile, run_id):
    """One driver spawn in a fresh scratch root; returns its single run row."""
    bundle = Path(bundle)
    driver_path = bundle / "tools" / driver_name
    root = Path(tempfile.mkdtemp(prefix="v0-replay-"))
    try:
        shutil.copytree(bundle / "code", root / "code")
        shutil.copytree(bundle / "data", root / "data")
        protocol = _one_run_protocol(bundle, profile, run_id)
        (root / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, str(driver_path)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise ReplayError(
                f"{driver_path.name} exited {completed.returncode} for {profile} run_id {run_id}: {tail}"
            )
        runs_path = root / "runs.json"
        if not runs_path.is_file():
            raise ReplayError(f"{driver_path.name} wrote no runs.json for {profile} run_id {run_id}")
        payload = json.loads(runs_path.read_text(encoding="utf-8"))
        runs = payload.get("runs") or []
        if len(runs) != 1:
            raise ReplayError(
                f"{driver_path.name} produced {len(runs)} run rows for {profile} run_id {run_id}, expected 1"
            )
        return runs[0]
    except (OSError, ValueError) as exc:
        raise ReplayError(f"driver spawn failed for {profile} run_id {run_id}: {exc}") from exc
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _one_run_protocol(bundle, profile, run_id):
    """The bundle's protocol.json reshaped to execute exactly this one (profile, run)."""
    bundle = Path(bundle)
    base = json.loads((bundle / "protocol.json").read_text(encoding="utf-8"))
    perf_config = _perf_config(bundle)
    if profile not in perf_config:
        raise ReplayError(f"{bundle} code/ga.py PERFIL_CONFIG has no {profile!r}")
    cfg = perf_config[profile]
    protocol = dict(base)
    protocol["profile"] = profile
    protocol["ga"] = {
        "n_assets": cfg["n_assets"],
        "lambda": cfg["lambda"],
        "generations": cfg["generations"],
        "pop_size": cfg["pop_size"],
        "crossover_rate": base["ga"]["crossover_rate"],
        "mutation_rate": base["ga"]["mutation_rate"],
    }
    protocol["runs"] = dict(base["runs"], run_id_start=int(run_id), n_runs=1)
    return protocol


def _perf_config(bundle):
    """PERFIL_CONFIG read out of the bundle's code/ga.py - extracted, never imported."""
    path = Path(bundle) / "code" / "ga.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PERFIL_CONFIG":
                value = node.value
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "PERFIL_CONFIG" for t in node.targets):
                value = node.value
        if value is not None:
            return ast.literal_eval(value)
    raise ReplayError(f"PERFIL_CONFIG is not assigned at module level in {path}")


def execute_replay(legacy_root, corrected_root, workers=None, progress=None):
    """All 80 (profile, run_id) pairs, both drivers per pair, in canonical order."""
    jobs = [
        (profile, run_id)
        for profile in PROFILES
        for run_id in range(RUN_ID_START, RUN_ID_START + N_RUN_IDS)
    ]
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_pair, profile, run_id, legacy_root, corrected_root): (profile, run_id)
            for profile, run_id in jobs
        }
        done = 0
        for future in as_completed(futures):
            profile, run_id = futures[future]
            results[(profile, run_id)] = future.result()
            done += 1
            if progress is not None:
                progress(f"run {done}/{len(jobs)}: {profile} run_id {run_id}")
    return [results[job] for job in jobs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="compare_replay.py",
        description="Opt-in 80-run legacy-vs-corrected replay with the E1-E4 gate (VC-20..VC-22).",
    )
    parser.add_argument("--legacy", default=str(LEGACY_BUNDLE), help="legacy bundle root (default: %(default)s)")
    parser.add_argument("--bundle", default=str(BUNDLE_DIR), help="corrected bundle root (default: %(default)s)")
    parser.add_argument("--proof", default=str(DEFAULT_PROOF), help="replay proof path (default: %(default)s)")
    parser.add_argument("--reference", default=None, help="corrected reference dir (default: <bundle>/reference)")
    parser.add_argument("--check-proof", default=None, metavar="PATH",
                        help="check an existing proof payload only - no execution")
    parser.add_argument("--workers", type=int, default=None, help="concurrent run pairs (default: min(8, cpus))")
    args = parser.parse_args(argv)

    # VC-22: opt-in only - refused before anything executes (exit 2, zero execution).
    if os.environ.get(ENV_FLAG) != "1":
        print(
            f"compare_replay: refusing to run: set {ENV_FLAG}=1 to execute the "
            f"80-run comparison (nothing was executed)",
            file=sys.stderr,
        )
        return EXIT_NO_OPTIN

    if args.check_proof:
        try:
            proof = json.loads(Path(args.check_proof).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"compare_replay: cannot read {args.check_proof}: {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        problems = check_expectations(proof)
        return _finish(proof, problems, recorded=None)

    try:
        runs = execute_replay(
            Path(args.legacy), Path(args.bundle), workers=args.workers,
            progress=lambda line: print(f"compare_replay: {line}", file=sys.stderr),
        )
    except ReplayError as exc:
        print(f"compare_replay: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    proof = build_proof(runs, capture_env())

    problems = list(check_expectations(proof))
    problems += check_proof_vs_reference(proof, Path(args.reference) if args.reference else Path(args.bundle) / "reference")

    record = Path(args.proof)
    if record.is_file():
        problems += check_proof_vs_committed(proof, record)
        write = False
    else:
        write = True  # the first run records itself (proof-as-record for every later run)
    if write:
        _write_proof(record, proof)
    return _finish(proof, problems, recorded=record if not write else None)


def _finish(proof, problems, recorded):
    for problem in problems:
        print(f"compare_replay: {problem}", file=sys.stderr)
    summary = {
        "status": "VIOLATED" if problems else "HELD",
        "denominator": proof.get("denominator"),
        "expectations": proof.get("expectations"),
    }
    if recorded is not None:
        summary["record"] = str(recorded)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_DIVERGENCE if problems else EXIT_OK


def _write_proof(path, proof):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
