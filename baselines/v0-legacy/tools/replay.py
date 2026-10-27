#!/usr/bin/env python3
"""replay.py - one command that reproduces the archived v0-legacy numbers, or says why not.

Order of operations, and why:

1. verify the manifest - a bundle whose bytes moved is not this bundle (cause `hash_mismatch`)
2. load `protocol.json` - every parameter and every expected value comes from it, from the
   archived code or from `reference/`; a value it needs and does not carry is
   `protocol_mismatch`
3. resolve the metrics file the protocol names; a dataset the manifest does not cover is
   `dataset_unavailable`
4. probe the third-party modules the archived `code/` imports in a child process; anything
   missing is `dependency_unavailable`, listed by name
5. build a scratch root from `code/`, `data/` and `protocol.json`, spawn the driver there, and
   read every run datum from `<scratch_root>/runs.json` and nowhere else
6. compare against `reference/`, then re-verify the manifest and report `bundle_stable`

`recomputed_fitness` is reported beside `reported_fitness` and never decides `status`: the
archived `ga.py` reports the previous generation's score against the new population, and
letting the two numbers substitute for each other would repair the baseline instead of
measuring it.

Exit codes: 0 `REPRODUCED`, 1 `MISMATCH`, 2 `UNVERIFIABLE`, 3 no trustworthy report.
Exactly one JSON object goes to stdout; diagnostics go to stderr.
"""

import argparse
import ast
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of the frozen tree
sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_manifest as vm  # noqa: E402  (sibling tool, resolved from this file's directory)

SCHEMA_VERSION = 1
DEFAULT_BUNDLE = "baselines/v0-legacy"
DEFAULT_REPORT = "outputs/baseline_v0_legacy/replay_report.json"
DEFAULT_BUDGET_SECONDS = 1800
DEFAULT_DRIVER = "tools/_legacy_driver.py"

EXIT_REPRODUCED = 0
EXIT_MISMATCH = 1
EXIT_UNVERIFIABLE = 2
EXIT_UNTRUSTWORTHY = 3

STDERR_TAIL_LIMIT = 4096
COMPARISON_KINDS = ("int", "float", "tickers", "str")


class ProtocolError(Exception):
    """A value the replay needs is absent from protocol.json, or is of an unknown shape."""


def _getter(protocol):
    def get(dotted, kind):
        node = protocol
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ProtocolError(f"protocol.json has no {dotted}")
            node = node[part]
        if kind == "int" and (not isinstance(node, int) or isinstance(node, bool)):
            raise ProtocolError(f"protocol.json {dotted} is not an integer")
        if kind == "float" and not isinstance(node, (int, float)):
            raise ProtocolError(f"protocol.json {dotted} is not a number")
        if kind == "str" and not isinstance(node, str):
            raise ProtocolError(f"protocol.json {dotted} is not a string")
        if kind == "list" and not isinstance(node, list):
            raise ProtocolError(f"protocol.json {dotted} is not a list")
        return node

    return get


def load_protocol(bundle):
    """Return the comparison settings the replay uses, read from protocol.json only."""
    path = Path(bundle) / "protocol.json"
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read {path}: {exc}") from exc
    if not isinstance(protocol, dict):
        raise ProtocolError("protocol.json is not a JSON object")

    get = _getter(protocol)
    values = {
        "profile": get("profile", "str"),
        "n_assets": get("ga.n_assets", "int"),
        "lambda": get("ga.lambda", "float"),
        "generations": get("ga.generations", "int"),
        "pop_size": get("ga.pop_size", "int"),
        "n_runs": get("runs.n_runs", "int"),
        "run_id_start": get("runs.run_id_start", "int"),
        "rtol": get("comparison.rtol", "float"),
        "atol": get("comparison.atol", "float"),
        "fields": get("comparison.fields", "list"),
        "reference_metrics": get("comparison.reference_metrics", "str"),
    }
    for spec in values["fields"]:
        if not isinstance(spec, dict) or "name" not in spec or spec.get("kind") not in COMPARISON_KINDS:
            raise ProtocolError(
                "protocol.json comparison.fields needs {name, kind} with kind in "
                + ", ".join(COMPARISON_KINDS)
            )
    values["reference_metrics"] = values["reference_metrics"].format(profile=values["profile"])
    return values


def required_modules(code_dir):
    """The third-party modules the archived code imports at module level, via ast.

    Module level only: `cleaner`, imported inside `generate_consensus_summary`, is never on
    this path. Modules that live beside the archived code are the archive itself, not a
    dependency to probe.
    """
    code_dir = Path(code_dir)
    local = {path.stem for path in code_dir.glob("*.py")}
    names = set()
    for path in sorted(code_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return sorted(
        name for name in names if name not in sys.stdlib_module_names and name not in local
    )


def probe_modules(modules):
    """Import each module in a child process; return the ones that cannot be imported."""
    missing = []
    for name in modules:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            missing.append(name)
    return missing


def resolve_driver(bundle, driver_arg):
    path = Path(driver_arg)
    return path.resolve() if path.is_absolute() else (Path(bundle) / path).resolve()


def is_archived_driver(bundle, driver):
    """True when the resolved driver is the bundle's own executor, the one that runs code/."""
    return Path(driver).resolve() == (Path(bundle) / DEFAULT_DRIVER).resolve()


def _coerce(value, kind):
    if value is None:
        return None
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return str(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def _equal(recorded, recomputed, kind, rtol, atol):
    if kind == "float":
        return math.isclose(recorded, recomputed, rel_tol=rtol, abs_tol=atol)
    return recorded == recomputed


def read_reference(metrics_path):
    """run_id -> row, from the recorded metrics CSV."""
    with open(metrics_path, newline="", encoding="utf-8") as handle:
        return {int(row["run_id"]): row for row in csv.DictReader(handle)}


def _stderr_tail(text):
    if not text:
        return ""
    return text[-STDERR_TAIL_LIMIT:]


def _unverifiable(report, cause, detail):
    report["status"] = "UNVERIFIABLE"
    report["cause"] = cause
    report["detail"] = detail


def new_report(bundle, driver):
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": None,
        "bundle": str(bundle),
        "driver": str(driver),
        "status": "UNVERIFIABLE",
        "cause": None,
        "detail": None,
        "fields": [],
        "runs": [],
        "missing_modules": [],
        "protocol": {},
        "manifest": {"status": "UNKNOWN", "files": 0, "findings": []},
        "bundle_stable": False,
        "golden_output_written": False,
        "stderr_tail": None,
    }


def execute(bundle, driver, report, budget_seconds):
    """Run the replay up to and including the comparison. Returns the exit code."""
    manifest = vm.verify_bundle(bundle)
    report["manifest"] = {
        "status": manifest["status"],
        "files": manifest["files"],
        "findings": manifest["findings"],
    }
    report["baseline_id"] = manifest.get("baseline_id")
    if manifest["status"] != "VERIFIED":
        _unverifiable(
            report,
            "hash_mismatch",
            f"manifest is {manifest['status']} before execution, so no legacy code ran",
        )
        return EXIT_UNVERIFIABLE

    listed = {entry["path"] for entry in vm.load_manifest(bundle)["files"]}

    try:
        values = load_protocol(bundle)
    except ProtocolError as exc:
        _unverifiable(report, "protocol_mismatch", str(exc))
        return EXIT_UNVERIFIABLE
    report["protocol"] = values

    metrics_rel = values["reference_metrics"]
    if metrics_rel not in listed:
        _unverifiable(
            report,
            "dataset_unavailable",
            f"{metrics_rel} is not a covered file, so the recorded outcome is not archived",
        )
        return EXIT_UNVERIFIABLE
    metrics_path = bundle / metrics_rel

    missing = []
    if is_archived_driver(bundle, driver):
        # The dependency set belongs to the archived executor. A substitute driver passed with
        # --driver declares its own imports, so probing for numpy would stop a replay that never
        # needs it - which is exactly what AC 22 asks a substitute to be able to avoid.
        missing = probe_modules(required_modules(bundle / "code"))
        report["missing_modules"] = missing
    if missing:
        _unverifiable(
            report,
            "dependency_unavailable",
            "cannot import " + ", ".join(missing) + " in this interpreter",
        )
        return EXIT_UNVERIFIABLE

    if not driver.is_file():
        _unverifiable(report, "runtime_error", f"driver not found: {driver}")
        return EXIT_UNVERIFIABLE

    scratch = Path(tempfile.mkdtemp(prefix="v0-legacy-replay-"))
    try:
        shutil.copytree(bundle / "code", scratch / "code")
        shutil.copytree(bundle / "data", scratch / "data")
        shutil.copy2(bundle / "protocol.json", scratch / "protocol.json")

        try:
            completed = subprocess.run(
                [sys.executable, str(driver)],
                cwd=str(scratch),
                capture_output=True,
                text=True,
                timeout=budget_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            report["stderr_tail"] = _stderr_tail(exc.stderr if isinstance(exc.stderr, str) else "")
            _unverifiable(
                report,
                "runtime_budget_exceeded",
                f"executor still running after {budget_seconds}s and was terminated",
            )
            return EXIT_UNVERIFIABLE
        except OSError as exc:
            _unverifiable(report, "runtime_error", f"cannot spawn {driver}: {exc}")
            return EXIT_UNVERIFIABLE

        if completed.returncode != 0:
            report["stderr_tail"] = _stderr_tail(completed.stderr)
            _unverifiable(
                report,
                "runtime_error",
                f"executor exited {completed.returncode}",
            )
            return EXIT_UNVERIFIABLE

        runs_path = scratch / "runs.json"
        if not runs_path.is_file():
            report["stderr_tail"] = _stderr_tail(completed.stderr)
            _unverifiable(report, "runtime_error", "executor produced no runs.json")
            return EXIT_UNVERIFIABLE
        try:
            payload = json.loads(runs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _unverifiable(report, "runtime_error", f"cannot read runs.json: {exc}")
            return EXIT_UNVERIFIABLE
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            _unverifiable(report, "runtime_error", "runs.json is not an object with a runs array")
            return EXIT_UNVERIFIABLE

        actual = {}
        for run in payload["runs"]:
            if not isinstance(run, dict) or not isinstance(run.get("run_id"), int):
                _unverifiable(report, "runtime_error", "runs.json holds a run without an int run_id")
                return EXIT_UNVERIFIABLE
            actual[run["run_id"]] = run

        expected = read_reference(metrics_path)
        first, count = values["run_id_start"], values["n_runs"]
        for run_id in range(first, first + count):
            if run_id not in expected:
                _unverifiable(
                    report,
                    "dataset_unavailable",
                    f"{metrics_rel} has no row for run {run_id}",
                )
                return EXIT_UNVERIFIABLE
            if run_id not in actual:
                _unverifiable(report, "runtime_error", f"executor produced no datum for run {run_id}")
                return EXIT_UNVERIFIABLE
            recorded_row = expected[run_id]
            actual_row = actual[run_id]
            for spec in values["fields"]:
                name, kind = spec["name"], spec["kind"]
                if name not in recorded_row:
                    _unverifiable(
                        report,
                        "dataset_unavailable",
                        f"{metrics_rel} has no column {name}",
                    )
                    return EXIT_UNVERIFIABLE
                if name not in actual_row:
                    _unverifiable(
                        report,
                        "runtime_error",
                        f"runs.json run {run_id} has no {name}",
                    )
                    return EXIT_UNVERIFIABLE
                recorded = _coerce(recorded_row[name], kind)
                recomputed = _coerce(actual_row[name], kind)
                if not _equal(recorded, recomputed, kind, values["rtol"], values["atol"]):
                    report["fields"].append(
                        {
                            "run_id": run_id,
                            "field": name,
                            "recorded": recorded,
                            "recomputed": recomputed,
                        }
                    )
            if "recomputed_fitness" not in actual_row:
                _unverifiable(
                    report,
                    "runtime_error",
                    f"runs.json run {run_id} has no recomputed_fitness",
                )
                return EXIT_UNVERIFIABLE
            report["runs"].append(
                {
                    "run_id": run_id,
                    "seed": actual_row.get("seed"),
                    "reported_fitness": _coerce(recorded_row["fitness"], "float"),
                    "recomputed_fitness": _coerce(actual_row["recomputed_fitness"], "float"),
                }
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if report["fields"]:
        report["status"] = "MISMATCH"
        report["cause"] = None
        report["detail"] = f"{len(report['fields'])} compared field(s) differ from reference/"
        return EXIT_MISMATCH
    report["status"] = "REPRODUCED"
    report["cause"] = None
    report["detail"] = None
    return EXIT_REPRODUCED


def final_verify(bundle, report, exit_status):
    """AC 21: re-verify after execution; a covered byte that moved is not a trustworthy report."""
    if report["manifest"]["status"] != "VERIFIED":
        return exit_status
    final = vm.verify_bundle(bundle)
    report["bundle_stable"] = final["status"] == "VERIFIED"
    if report["bundle_stable"]:
        return exit_status
    moved = ", ".join(f"{item['kind']}:{item['path']}" for item in final["findings"][:5])
    _unverifiable(report, "hash_mismatch", f"a covered byte moved during the run ({moved})")
    return EXIT_UNTRUSTWORTHY


def replay(bundle, driver_arg, budget_seconds):
    """Return (report, exit code) for one replay."""
    bundle = Path(bundle).resolve()
    driver = resolve_driver(bundle, driver_arg)
    report = new_report(bundle, driver)
    exit_status = execute(bundle, driver, report, budget_seconds)
    exit_status = final_verify(bundle, report, exit_status)
    return report, exit_status


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="replay.py",
        description="Replay the archived v0-legacy runs and compare them with reference/.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        help="bundle directory to replay (default: %(default)s)",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="where to write the JSON report (default: %(default)s)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help="executor budget before it is terminated (default: %(default)s)",
    )
    parser.add_argument(
        "--driver",
        default=DEFAULT_DRIVER,
        help="executor path, resolved against the bundle (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    report, exit_status = replay(args.bundle, args.driver, args.budget_seconds)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")

    cause = f"/{report['cause']}" if report["cause"] else ""
    print(f"replay: {report['status']}{cause} -> {report_path}", file=sys.stderr)
    if report["detail"]:
        print(f"replay: {report['detail']}", file=sys.stderr)
    print(text, end="")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
