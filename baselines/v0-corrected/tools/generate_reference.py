#!/usr/bin/env python3
"""generate_reference.py - two-pass scratch generation of the corrected reference (D-c).

Build-time only, run under the declared environment (tcc-venv). Per pass it copies the
bundle's `code/`, `data/` and `protocol.json` into a fresh scratch root and runs the
frozen byte-identical `code/ga_multiple_runs.py` there (its import-time `OUT_DIR.mkdir`
lands in the scratch root); the frozen writer is the only writer whose output claims
"the v0 protocol plus one hunk". Then, before anything is committed:

1. `assert_complete` - all 30 run_ids x 4 profiles present (the frozen writer swallows
   per-run exceptions, so completeness is checked, never assumed), and each summary
   `n_runs` equals its CSV row count (VC-09).
2. `compare_passes` - every file of pass A vs pass B byte-compared by sha256; any
   difference is a hard failure (the determinism proof).
3. `tie_to_driver` - per (profile, run_id 0-29) one `_corrected_driver.py` spawn in its
   own scratch root, tied to the generated CSV rows through `_verdict.tie_row`
   (tickers set + hhi + fitness within 1e-9). This is also the forged-reference
   defense of VC-29: a row the driver did not produce cannot pass.

Exit codes: 0 generated + deterministic + tied (pass A bytes installed verbatim into
the destination, proof written); 1 the run completed but must not be committed
(completeness, nondeterminism or tie divergence - the proof file records the run);
3 unusable input (missing bundle pieces, spawn failure, hand-edited destination).

Every run re-seeds itself (`seed = 42 + run_id`) before executing, so scheduling
cannot change any row: the tie spawns may run concurrently (`--tie-workers`) without
touching the protocol - the 460/400/300/500-generation runs all still execute in full.
"""

import sys

sys.dont_write_bytecode = True  # before any bundle import (AD-013)

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path


def _load_sibling(name):
    """The bundle's own sibling tool, loaded from this file's directory under a unique name."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v0_corrected_gen_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vd = _load_sibling("_verdict")

SCHEMA_VERSION = 1
PROFILES = ("conservador", "moderado", "arrojado", "caio")
RUN_ID_START = 0
N_RUNS = 30
SEED_BASE = 42
PASS_IDS = ("a", "b")

TOOLS_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOLS_DIR.parent
REPO_ROOT = BUNDLE_DIR.parents[1]
DRIVER_PATH = TOOLS_DIR / "_corrected_driver.py"
DEFAULT_PROOF = REPO_ROOT / ".specs" / "features" / "p0-02-v0-corrected" / "proofs" / "generation_proof.json"

EXPECTED_OUTPUT_FILES = tuple(
    [f"metrics_stability_{p}.csv" for p in PROFILES]
    + [f"carteira_{p}_consensus.json" for p in PROFILES]
    + ["multiple_runs_summary.json"]
)


class GenerationError(Exception):
    """The generation cannot run or cannot be trusted as input."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_metrics_rows(outputs_dir, profile):
    """The recorded run rows of metrics_stability_<profile>.csv, keyed fields parsed."""
    path = Path(outputs_dir) / f"metrics_stability_{profile}.csv"
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
                raise GenerationError(f"{path} line {number} is not a usable run row: {exc}") from exc
    return rows


def assert_complete(outputs_dir):
    """Problems proving all 30 run_ids x 4 profiles were written ( [] = complete )."""
    outputs_dir = Path(outputs_dir)
    problems = []
    if not outputs_dir.is_dir():
        return [f"no outputs directory at {outputs_dir}"]
    present = sorted(p.name for p in outputs_dir.iterdir() if p.is_file())
    for name in EXPECTED_OUTPUT_FILES:
        if name not in present:
            problems.append(f"missing output file: {name}")
    for name in present:
        if name not in EXPECTED_OUTPUT_FILES:
            problems.append(f"unexpected output file: {name}")

    expected_run_ids = list(range(RUN_ID_START, RUN_ID_START + N_RUNS))
    for profile in PROFILES:
        if f"metrics_stability_{profile}.csv" not in present:
            continue
        rows = read_metrics_rows(outputs_dir, profile)
        by_run = {}
        for row in rows:
            by_run.setdefault(row["run_id"], []).append(row)
        for run_id in expected_run_ids:
            found = by_run.get(run_id, [])
            if not found:
                problems.append(f"metrics_stability_{profile}.csv is missing run_id {run_id}")
            elif len(found) > 1:
                problems.append(f"metrics_stability_{profile}.csv repeats run_id {run_id}")
            elif found[0]["seed"] != SEED_BASE + run_id:
                problems.append(
                    f"metrics_stability_{profile}.csv run_id {run_id} has seed {found[0]['seed']}, "
                    f"expected {SEED_BASE + run_id}"
                )
        for run_id in sorted(by_run):
            if run_id not in expected_run_ids:
                problems.append(f"metrics_stability_{profile}.csv has out-of-protocol run_id {run_id}")

    if "multiple_runs_summary.json" in present:
        try:
            summary = json.loads((outputs_dir / "multiple_runs_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"multiple_runs_summary.json is unreadable: {exc}")
            summary = {}
        for profile in PROFILES:
            entry = summary.get(profile)
            if not isinstance(entry, dict):
                problems.append(f"multiple_runs_summary.json has no {profile} entry")
                continue
            csv_rows = len(read_metrics_rows(outputs_dir, profile)) if f"metrics_stability_{profile}.csv" in present else 0
            n_runs = entry.get("n_runs")
            all_runs = entry.get("all_runs")
            # VC-09: n_runs equals the number of data rows in the profile's CSV
            if n_runs != csv_rows:
                problems.append(
                    f"multiple_runs_summary.json {profile} n_runs {n_runs!r} != "
                    f"{csv_rows} rows in metrics_stability_{profile}.csv"
                )
            if not isinstance(all_runs, list) or len(all_runs) != csv_rows:
                problems.append(
                    f"multiple_runs_summary.json {profile} all_runs length "
                    f"{len(all_runs) if isinstance(all_runs, list) else all_runs!r} != {csv_rows} CSV rows"
                )
    return problems


def compare_passes(pass_a, pass_b):
    """Byte-digest divergences between two passes ( [] = the passes are identical )."""
    problems = []
    digests_a, digests_b = dict(pass_a), dict(pass_b)
    for name in sorted(set(digests_a) | set(digests_b)):
        if name not in digests_a:
            problems.append(f"file only in pass b: {name}")
        elif name not in digests_b:
            problems.append(f"file only in pass a: {name}")
        elif digests_a[name] != digests_b[name]:
            problems.append(
                f"digest differs for {name}: a={digests_a[name]} b={digests_b[name]}"
            )
    return problems


def emit_proof(pass_a, pass_b, driver_rows, env=None):
    """The generation_proof.json payload: schema + double-pass digests + driver rows."""
    return {
        "schema_version": SCHEMA_VERSION,
        "env": capture_env() if env is None else dict(env),
        "passes": {"a": dict(pass_a), "b": dict(pass_b)},
        "deterministic": not compare_passes(pass_a, pass_b),
        "driver_rows": [dict(row) for row in driver_rows],
    }


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


def tie_to_driver(outputs_dir, bundle, scratch, runner=None, max_workers=None):
    """Per (profile, run_id 0-29) driver tie via `_verdict.tie_row` (D-c step 4).

    Returns (problems, driver_rows); driver_rows holds the 120 rows in canonical
    order (profile order, run_id ascending). `runner(profile, run_id)` must return
    the driver's row mapping or None; the default spawns `_corrected_driver.py` per
    pair in its own scratch root.
    """
    outputs_dir = Path(outputs_dir)
    recorded = {}
    for profile in PROFILES:
        recorded[profile] = {
            row["run_id"]: row for row in read_metrics_rows(outputs_dir, profile)
        }
    jobs = [
        (profile, run_id)
        for profile in PROFILES
        for run_id in range(RUN_ID_START, RUN_ID_START + N_RUNS)
    ]
    if runner is None:
        runner = _spawn_driver_row(bundle, scratch)
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(lambda job: runner(*job), jobs))
    else:
        rows = [runner(*job) for job in jobs]

    problems = []
    driver_rows = []
    for (profile, run_id), row in zip(jobs, rows):
        label = f"{profile} run_id {run_id}"
        if row is None:
            problems.append(f"{label}: driver produced no row")
            continue
        recorded_row = recorded[profile].get(run_id)
        if recorded_row is None:
            problems.append(f"{label}: generated reference has no row to tie against")
            continue
        for line in vd.tie_row(recorded_row, row):
            problems.append(f"{label}: {line}")
        driver_rows.append(
            {
                "profile": profile,
                "run_id": row.get("run_id", run_id),
                "seed": row.get("seed", SEED_BASE + run_id),
                "tickers": list(row.get("tickers", [])),
                "fitness": float(row.get("fitness")),
                "hhi": float(row.get("hhi")),
            }
        )
    return problems, driver_rows


def _spawn_driver_row(bundle, scratch):
    """The real runner: one `_corrected_driver.py` spawn per (profile, run_id)."""
    bundle = Path(bundle)
    scratch = Path(scratch)
    base_protocol = json.loads((bundle / "protocol.json").read_text(encoding="utf-8"))
    perf_config = _perf_config(bundle)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def runner(profile, run_id):
        root = scratch / f"tie-{profile}-{run_id}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            shutil.copytree(bundle / "code", root / "code")
            shutil.copytree(bundle / "data", root / "data")
            protocol = dict(base_protocol)
            protocol["profile"] = profile
            cfg = dict(perf_config[profile])
            protocol["ga"] = {
                "n_assets": cfg["n_assets"],
                "lambda": cfg["lambda"],
                "generations": cfg["generations"],
                "pop_size": cfg["pop_size"],
                "crossover_rate": base_protocol["ga"]["crossover_rate"],
                "mutation_rate": base_protocol["ga"]["mutation_rate"],
            }
            protocol["runs"] = dict(
                base_protocol["runs"], run_id_start=run_id, n_runs=1
            )
            (root / "protocol.json").write_text(
                json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, str(DRIVER_PATH)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return None
            runs_path = root / "runs.json"
            if not runs_path.is_file():
                return None
            payload = json.loads(runs_path.read_text(encoding="utf-8"))
            runs = payload.get("runs") or []
            return runs[0] if len(runs) == 1 else None
        except (OSError, ValueError):
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)

    return runner


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
    raise GenerationError(f"PERFIL_CONFIG is not assigned at module level in {path}")


def generate_pass(bundle, pass_id, scratch=None):
    """One full generation in a scratch root: run the frozen writer, hash its outputs.

    Returns (outputs dir, {relpath: sha256}). Without `scratch` the caller owns a
    private mkdtemp root whose path is the parent of the returned outputs dir.
    """
    bundle = Path(bundle)
    if scratch is None:
        root = Path(tempfile.mkdtemp(prefix=f"v0-corrected-gen-{pass_id}-"))
    else:
        root = Path(scratch) / pass_id
        root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle / "code", root / "code")
    shutil.copytree(bundle / "data", root / "data")
    shutil.copy2(bundle / "protocol.json", root / "protocol.json")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(root / "code" / "ga_multiple_runs.py")],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-2000:]
        raise GenerationError(f"frozen writer exited {completed.returncode}: {tail}")
    outputs = root / "outputs"
    if not outputs.is_dir():
        raise GenerationError("frozen writer produced no outputs/ directory")
    digests = {}
    for path in sorted(outputs.rglob("*")):
        if path.is_file():
            digests[path.relative_to(outputs).as_posix()] = sha256_bytes(path.read_bytes())
    return outputs, digests


def install_reference(outputs_dir, dest):
    """Copy pass A bytes verbatim into dest; refuse to launder a hand-edited dest."""
    outputs_dir = Path(outputs_dir)
    dest = Path(dest)
    files = {
        path.relative_to(outputs_dir).as_posix(): path.read_bytes()
        for path in sorted(outputs_dir.rglob("*"))
        if path.is_file()
    }
    problems = []
    if dest.exists():
        existing = {
            path.relative_to(dest).as_posix()
            for path in dest.rglob("*")
            if path.is_file()
        }
        for rel in sorted(existing - set(files)):
            problems.append(f"unexpected file already in {dest}: {rel}")
        for rel, data in sorted(files.items()):
            current = dest / rel
            if current.is_file() and current.read_bytes() != data:
                problems.append(f"{current} differs from the generated bytes (hand-edited?)")
    if problems:
        return problems
    dest.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        (dest / rel).write_bytes(data)
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="generate_reference.py",
        description="Two-pass scratch generation of the v0-corrected reference (D-c).",
    )
    parser.add_argument("--bundle", default=str(BUNDLE_DIR), help="bundle directory (default: %(default)s)")
    parser.add_argument("--dest", default=None, help="reference destination (default: <bundle>/reference)")
    parser.add_argument("--proof", default=str(DEFAULT_PROOF), help="generation_proof.json path (default: %(default)s)")
    parser.add_argument("--passes", type=int, default=2, help="generation passes (default: %(default)s)")
    parser.add_argument("--tie-workers", type=int, default=None, help="concurrent driver-tie spawns (default: min(8, cpus))")
    args = parser.parse_args(argv)

    if args.passes != 2:
        print("generate_reference: only --passes 2 is supported", file=sys.stderr)
        return 3

    bundle = Path(args.bundle).resolve()
    dest = Path(args.dest).resolve() if args.dest else bundle / "reference"
    proof_path = Path(args.proof).resolve()

    required = [
        bundle / "code" / "ga.py",
        bundle / "code" / "profiles.py",
        bundle / "code" / "ga_multiple_runs.py",
        bundle / "protocol.json",
        bundle / "data" / "processed",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"generate_reference: unusable bundle, missing {missing}", file=sys.stderr)
        return 3

    scratch = Path(tempfile.mkdtemp(prefix="v0-corrected-gen-"))
    try:
        try:
            outputs_a, digests_a = generate_pass(bundle, "a", scratch)
            outputs_b, digests_b = generate_pass(bundle, "b", scratch)
        except GenerationError as exc:
            print(f"generate_reference: {exc}", file=sys.stderr)
            return 3

        problems = assert_complete(outputs_a) + assert_complete(outputs_b)
        if problems:
            _report("incomplete generation", problems)
            _write_proof(proof_path, emit_proof(digests_a, digests_b, []))
            return 1

        divergences = compare_passes(digests_a, digests_b)
        if divergences:
            _report("nondeterministic generation", divergences)
            _write_proof(proof_path, emit_proof(digests_a, digests_b, []))
            return 1

        tie_problems, driver_rows = tie_to_driver(
            outputs_a, bundle, scratch / "tie", max_workers=args.tie_workers
        )
        proof = emit_proof(digests_a, digests_b, driver_rows)
        _write_proof(proof_path, proof)
        if tie_problems:
            _report("driver tie divergence", tie_problems)
            return 1

        install_problems = install_reference(outputs_a, dest)
        if install_problems:
            _report("destination unusable", install_problems)
            return 3

        print(
            json.dumps(
                {
                    "status": "GENERATED",
                    "dest": str(dest),
                    "files": len(digests_a),
                    "deterministic": True,
                    "driver_rows": len(driver_rows),
                    "proof": str(proof_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _report(title, problems):
    print(f"generate_reference: {title} ({len(problems)} problem(s))", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)


def _write_proof(path, proof):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
