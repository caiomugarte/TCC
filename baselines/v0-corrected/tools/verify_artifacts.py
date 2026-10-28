#!/usr/bin/env python3
"""verify_artifacts.py - the artifact verifier and status derivation (D-d/D-e).

Modes (VC-01, VC-09, VC-10, VC-11, VC-24, VC-26):

* default: `verify_reference` over `reference/` (or `--reference-dir`) - 100% row
  recompute. Every row's reported fitness must equal the frozen recomputation from
  the row's own tickers (1e-9), `hhi` must self-recompute from those tickers, every
  recorded `score_*` cell must reproduce, and each summary `n_runs` must equal the
  row count of its `metrics_stability_<profile>.csv` (VC-09). A row whose chromosome
  does not reconstruct as exactly `n` distinct rows of the frozen snapshot FAILS the
  artifact - never skipped, never counted as passing, never dropped from
  `rows_total` (VC-10, VC-26); a row carrying a ticker the frozen snapshot does not
  know is `not_verifiable`, never a silent pass (VC-11). Then `check_status` runs
  against `status.json` when that file exists (it is emitted at T13).
* `--emit-status PATH`: `derive_status_entries` enumerates every recorded
  fitness/score/consensus-membership value of `baselines/v0-legacy/reference/**`
  and the `outputs/` tree tracked at the pinned base commit, reading each artifact
  as `git show <pinned>:<path>` blobs (VC-14: the map never pins itself), stamping
  the D-d evidence fields and a verdict computed by the shared
  `_verdict.verdict_for` - the single verdict path the marking test re-checks.
* `--json`: print the machine-readable report.

Exit codes: 0 everything verified (and status consistent when checked); 1 any FAIL
or status disagreement; 3 unusable input. Every negative fixture the tests need is
a runtime mutation inside `tempfile.TemporaryDirectory()` driven through
`--reference-dir` or the importable functions - no fixture file is ever committed
(D-f).

Rows are never dropped from a denominator: `rows_total` counts every recorded row
of the artifact, and `rows_passed + rows_failed + rows_unverifiable == rows_total`
holds by construction.
"""

import sys

sys.dont_write_bytecode = True  # before any bundle import (AD-013)

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOLS_DIR.parent
REPO_ROOT = BUNDLE_DIR.parents[1]
DEFAULT_REFERENCE_DIR = BUNDLE_DIR / "reference"
DEFAULT_STATUS = BUNDLE_DIR / "status.json"
DEFAULT_BUNDLE = BUNDLE_DIR
PINNED_BASE = "181453f986cce7941addf9092e38c6e005e18844"
PROFILES = ("conservador", "moderado", "arrojado", "caio")
TOL = 1e-9
DEFECT_REF = (
    "P0-02 capture defect: py/ga.py recorded the generation best against the replaced "
    "population (issue #7)"
)
SCORE_REASON = (
    "recorded score fields do not reproduce from the frozen snapshot under declared weights"
)
UNVERIFIABLE_REASON = (
    "row chromosome or score inputs do not reconstruct from the frozen bundle snapshot"
)
DERIVED_UNVERIFIABLE_REASON = (
    "contributor values do not reconstruct from the frozen bundle snapshot at the pinned base"
)
DERIVED_NO_CONTRIBUTORS_REASON = "no mechanical contributor set exists for this recorded value"


def _load_sibling(name):
    """The bundle's own sibling tool, loaded from this file's directory under a unique name."""
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v0_corrected_verify_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vd = _load_sibling("_verdict")
ff = _load_sibling("_frozen_fitness")


class VerifyError(Exception):
    """The verifier was pointed at input it cannot audit."""


# ----------------------------------------------------------------- helpers #


def _within(a, b, tol=TOL):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_metrics_csv(data, path):
    """(columns, rows) of one run-row CSV - keyed cells parsed, loud on garbage."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
    columns = list(reader.fieldnames or [])
    rows = []
    for number, row in enumerate(reader, start=2):
        try:
            rows.append(
                {
                    "run_id": int(row["run_id"]),
                    "tickers": list(ast.literal_eval(row["tickers"])),
                    "cells": dict(row),
                }
            )
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            raise VerifyError(f"{path} line {number} is not a usable run row: {exc}") from exc
    return columns, rows


def _is_run_row_csv(path, columns):
    return (
        path.endswith(".csv")
        and "run_id" in columns
        and "tickers" in columns
        and "fitness" in columns
    )


# ------------------------------------------------- frozen evidence (D-e) #


def _evidence_for(fb, profile, tickers, recorded):
    """(RowEvidence, notes): the row's evidence, plus loud notes on why it is inexact.

    A profile the frozen snapshot cannot rank (e.g. `caio2`, `caio_last`, whose
    datasets are not in the bundle) yields `inexact`/`data_snapshot_match=False`
    evidence - not_verifiable - instead of a silent recomputation against some
    other data root (VC-11).
    """
    try:
        evidence = fb.evidence_for_row(profile, tickers, recorded)
    except ff.FrozenError as exc:
        recorded = dict(recorded)
        score_names = sorted(k for k in recorded if k.startswith("score_"))
        return (
            ff.RowEvidence(
                row_reconstruction="inexact",
                data_snapshot_match=False,
                hhi_reported=_float_or_none(recorded.get("hhi")),
                hhi_recomputed=None,
                score_fields={
                    name: (_float_or_none(recorded.get(name)), None) for name in score_names
                },
                reported=_float_or_none(recorded.get("fitness")),
                recomputed=None,
            ),
            [f"no frozen snapshot answers this row: {exc}"],
        )
    notes = _reconstruction_notes(fb, profile, tickers, evidence)
    return evidence, notes


def _reconstruction_notes(fb, profile, tickers, evidence):
    """Why a row is inexact - the loud diagnostics a FAIL must carry ([] = clean)."""
    if evidence.row_reconstruction == "exact" and evidence.data_snapshot_match:
        return []
    values = [str(t) for t in tickers]
    distinct = set(values)
    notes = []
    if len(distinct) != len(values):
        notes.append(f"duplicated ticker in the chromosome: {sorted(values)}")
    try:
        n = fb.n_assets(profile)
    except ff.FrozenError:
        n = None
    if n is not None and len(values) != n:
        notes.append(f"chromosome has {len(values)} tickers, exactly {n} distinct rows required")
    unknown = sorted(distinct - fb.universe(profile))
    if unknown:
        notes.append(f"ticker(s) absent from the frozen snapshot: {unknown}")
    if not notes:
        notes.append("chromosome does not reconstruct as exactly n distinct frozen rows")
    return notes


def _row_score_state(fb, profile, tickers, recorded):
    """(score_recompute, score pairs) over every recorded score_* cell of the row."""
    evidence, _ = _evidence_for(fb, profile, tickers, recorded)
    pairs = {name: pair for name, pair in evidence.score_fields.items()}
    return vd.score_recompute(pairs), pairs, evidence


# -------------------------------------------------- verify_reference #


def verify_reference(reference_dir, bundle, scratch):
    """100% row recompute over one reference directory (D-e contract).

    Report: {"artifacts": [{path, rows_total, rows_passed, rows_failed,
    rows_unverifiable, findings}], "status": "PASS"|"FAIL"}. A failing row stays in
    `rows_total` and is never counted as passing (VC-10, VC-26).
    """
    reference_dir = Path(reference_dir)
    bundle = Path(bundle)
    if not reference_dir.is_dir():
        raise VerifyError(f"reference directory not found: {reference_dir}")
    fb = ff.FrozenBundle.load(bundle, Path(scratch))
    artifacts = []
    status = "PASS"

    def _record(artifact):
        if artifact["findings"] or artifact["rows_failed"] or artifact["rows_unverifiable"]:
            artifact["status"] = "FAIL"
        else:
            artifact["status"] = "PASS"
        return artifact

    for profile in PROFILES:
        path = reference_dir / f"metrics_stability_{profile}.csv"
        findings = []
        rows_total = rows_passed = rows_failed = rows_unverifiable = 0
        frame_error = None
        try:
            fb.ranked(profile)  # frame-level TICKER uniqueness asserted loudly here
        except ff.FrozenError as exc:
            frame_error = f"frozen ranked frame for {profile} is unusable: {exc}"
            findings.append({"row_key": None, "problems": [frame_error]})
        if not path.is_file():
            findings.append({"row_key": None, "problems": [f"missing artifact: {path.name}"]})
            artifacts.append(_record({
                "path": path.name, "rows_total": 0, "rows_passed": 0, "rows_failed": 0,
                "rows_unverifiable": 0, "findings": findings,
            }))
            status = "FAIL"
            continue
        _, rows = _read_metrics_csv(path.read_bytes(), path.name)
        for row in rows:
            rows_total += 1  # every recorded row stays in the denominator
            recorded = row["cells"]
            if frame_error is not None:
                rows_unverifiable += 1
                findings.append({"row_key": {"run_id": row["run_id"]},
                                 "problems": [frame_error]})
                continue
            evidence, notes = _evidence_for(fb, profile, row["tickers"], recorded)
            if evidence.row_reconstruction != "exact" or not evidence.data_snapshot_match:
                rows_unverifiable += 1  # not_verifiable - never a silent pass
                findings.append({"row_key": {"run_id": row["run_id"]},
                                 "problems": notes or [UNVERIFIABLE_REASON]})
                continue
            problems = []
            if not _within(evidence.reported, evidence.recomputed):
                problems.append(
                    f"fitness reported {evidence.reported!r} != recomputed "
                    f"{evidence.recomputed!r} (tolerance {TOL})"
                )
            if not _within(evidence.hhi_reported, evidence.hhi_recomputed):
                problems.append(
                    f"hhi reported {evidence.hhi_reported!r} != self-recomputed "
                    f"{evidence.hhi_recomputed!r} from the row's own tickers"
                )
            for name, (reported, recomputed) in sorted(evidence.score_fields.items()):
                if not _within(reported, recomputed):
                    problems.append(
                        f"{name} reported {reported!r} != recomputed {recomputed!r}"
                    )
            if problems:
                rows_failed += 1
                findings.append({"row_key": {"run_id": row["run_id"]}, "problems": problems})
            else:
                rows_passed += 1
        artifacts.append(_record({
            "path": path.name, "rows_total": rows_total, "rows_passed": rows_passed,
            "rows_failed": rows_failed, "rows_unverifiable": rows_unverifiable,
            "findings": findings,
        }))

    # VC-09: summary n_runs == CSV rows - one check row per profile
    summary_path = reference_dir / "multiple_runs_summary.json"
    findings = []
    rows_total = rows_passed = rows_failed = 0
    try:
        summary = json.loads(summary_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        summary = None
        findings.append({"row_key": None,
                         "problems": [f"unreadable multiple_runs_summary.json: {exc}"]})
    for profile in PROFILES:
        rows_total += 1
        csv_path = reference_dir / f"metrics_stability_{profile}.csv"
        data_rows = 0
        if csv_path.is_file():
            _, rows = _read_metrics_csv(csv_path.read_bytes(), csv_path.name)
            data_rows = len(rows)
        declared = None
        if isinstance(summary, dict):
            declared = (summary.get(profile) or {}).get("n_runs")
        if summary is not None and declared == data_rows:
            rows_passed += 1
        else:
            rows_failed += 1
            findings.append({
                "row_key": {"profile": profile},
                "problems": [
                    f"n_runs {declared!r} != {data_rows} data rows in metrics_stability_{profile}.csv"
                ],
            })
    if summary_path.is_file():
        artifacts.append(_record({
            "path": summary_path.name, "rows_total": rows_total, "rows_passed": rows_passed,
            "rows_failed": rows_failed, "rows_unverifiable": 0, "findings": findings,
        }))
    else:
        artifacts.append(_record({
            "path": summary_path.name, "rows_total": rows_total, "rows_passed": rows_passed,
            "rows_failed": rows_failed + 1, "rows_unverifiable": 0,
            "findings": findings + [{"row_key": None,
                                     "problems": ["missing artifact: multiple_runs_summary.json"]}],
        }))

    for artifact in artifacts:
        if artifact["status"] != "PASS":
            status = "FAIL"
    return {"status": status, "artifacts": artifacts}


# ------------------------------------------------ derive_status_entries #


def _git_show(repo, rev_path):
    completed = subprocess.run(
        ["git", "show", rev_path], cwd=str(repo), capture_output=True
    )
    if completed.returncode != 0:
        raise VerifyError(f"cannot read {rev_path}: {completed.stderr.decode().strip()}")
    return completed.stdout


def _git_ls_tree(repo, rev, prefix):
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev, "--", prefix],
        cwd=str(repo),
        capture_output=True,
    )
    if completed.returncode != 0:
        raise VerifyError(f"cannot list {prefix} at {rev}: {completed.stderr.decode().strip()}")
    return [line for line in completed.stdout.decode("utf-8").splitlines() if line]


def _json_pointer(pointer):
    return pointer.replace("~", "~0").replace("/", "~1")


def _profile_from_path(path):
    name = Path(path).name
    for prefix, suffix in (
        ("metrics_stability_", ".csv"),
        ("carteira_", "_consensus.json"),
        ("comparison_", ".json"),
    ):
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix): -len(suffix)]
    return None


def _scalar_leaf(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _RowIndex:
    """D-d's RowIndex over the enumerated run rows - mechanical, auditable ids."""

    def __init__(self, run_rows, selection_run_ids):
        self._run_rows = run_rows  # [{id, path, root, profile, run_id}]
        self._selection_run_ids = selection_run_ids  # (path, locator) -> run_id

    def rows_in_file(self, path):
        return [row["id"] for row in self._run_rows if row["path"] == path]

    def rows_in(self, root, profile, run_id=None):
        out = []
        for row in self._run_rows:
            if row["root"] != root or row["profile"] != profile:
                continue
            if run_id is not None and row["run_id"] != run_id:
                continue
            out.append(row["id"])
        return out

    def root_of(self, path):
        if path.startswith("outputs/"):
            return "outputs"
        return "baselines/v0-legacy/reference"

    def profile_of(self, path, locator):
        profile = _profile_from_path(path)
        if profile is not None:
            return profile
        head = locator.lstrip("/").split("/", 1)[0]
        if not head or head.isdigit():
            return None  # a bare list element carries no profile
        return head

    def recorded_run_id(self, path, locator):
        return self._selection_run_ids.get((path, locator))


def _walk_json(node, pointer, ctx, out):
    """Enumerate recorded fitness/score/membership values and run rows by structure.

    Emits either {"kind": "run_row", "locator", "row"} for `all_runs[i]` dicts or a
    value leaf {"locator", "kind", "artifact_class", "selection_run_id", "value"}.
    Nested and aggregate `fitness` values are enumerated by JSON structure, never by
    text search (VC-13).
    """
    if isinstance(node, dict):
        in_run_row = ctx.get("in_run_row", False)
        in_fitness_object = ctx.get("in_fitness_object", False)
        own_run_id = node.get("run_id")
        selection_run_id = (
            own_run_id
            if isinstance(own_run_id, int) and not isinstance(own_run_id, bool) and not in_run_row
            else ctx.get("selection_run_id")
        )
        for key, value in node.items():
            child = f"{pointer}/{_json_pointer(str(key))}"
            if in_fitness_object and _scalar_leaf(value):
                out.append({
                    "locator": child, "kind": "derived", "value": value,
                    "artifact_class": _derived_class(selection_run_id),
                    "selection_run_id": selection_run_id,
                })
                continue
            if key == "fitness":
                if isinstance(value, dict):
                    _walk_json(
                        value, child,
                        {**ctx, "in_fitness_object": True, "list_key": key,
                         "selection_run_id": selection_run_id},
                        out,
                    )
                elif _scalar_leaf(value):
                    out.append({
                        "locator": child,
                        "kind": "fitness" if in_run_row else "derived",
                        "value": value,
                        "artifact_class": (
                            "json_run_rows" if in_run_row
                            else _derived_class(selection_run_id)
                        ),
                        "selection_run_id": None if in_run_row else selection_run_id,
                    })
                continue
            if key.startswith("score_") and _scalar_leaf(value) and in_run_row:
                out.append({
                    "locator": child, "kind": "score", "value": value,
                    "artifact_class": "json_run_rows", "selection_run_id": None,
                })
                continue
            if key == "FREQUENCY" and _scalar_leaf(value):
                out.append({
                    "locator": child, "kind": "derived", "value": value,
                    "artifact_class": "json_consensus", "selection_run_id": None,
                })
                continue
            if key == "tickers" and isinstance(value, list) and ctx.get("membership"):
                out.append({
                    "locator": child, "kind": "derived", "value": value,
                    "artifact_class": "json_consensus", "selection_run_id": None,
                })
                continue
            child_ctx = dict(ctx)
            child_ctx["list_key"] = key
            child_ctx["selection_run_id"] = selection_run_id
            if key == "consensus_portfolio":
                child_ctx["membership"] = True
            _walk_json(value, child, child_ctx, out)
    elif isinstance(node, list):
        if ctx.get("list_key") == "all_runs":
            for index, item in enumerate(node):
                child = f"{pointer}/{index}"
                if isinstance(item, dict):
                    out.append({"kind": "run_row", "locator": child, "row": item})
                else:
                    _walk_json(item, child, {**ctx, "list_key": None}, out)
            return
        for index, item in enumerate(node):
            child = f"{pointer}/{index}"
            if ctx.get("in_fitness_object") and _scalar_leaf(item):
                out.append({
                    "locator": child, "kind": "derived", "value": item,
                    "artifact_class": _derived_class(ctx.get("selection_run_id")),
                    "selection_run_id": ctx.get("selection_run_id"),
                })
                continue
            _walk_json(item, child, {**ctx, "list_key": None}, out)


def _derived_class(selection_run_id):
    if selection_run_id is not None:
        return "json_selection"
    return "json_aggregate"


def _collect_recorded_values(root):
    """Every recorded fitness/score/membership value of one parsed JSON artifact."""
    out = []
    _walk_json(root, "", {"list_key": None}, out)
    return out


def derive_status_entries(repo, bundle, pinned_commit, scratch):
    """Every recorded value of the two subject trees, evidenced from the pinned blobs.

    Subjects: `baselines/v0-legacy/reference/**` and every `outputs/` file tracked at
    `pinned_commit` (VC-13). Each artifact is read via `git show <pinned>:<path>` - the
    map never pins itself (VC-14). Entry schema is D-d's; verdicts come from the
    shared `_verdict.verdict_for` through `recompute_verdicts`, the same code path
    the marking test checks (VC-24).
    """
    repo = Path(repo)
    fb = ff.FrozenBundle.load(Path(bundle), Path(scratch))
    subjects = []
    for prefix in ("baselines/v0-legacy/reference/", "outputs/"):
        for path in _git_ls_tree(repo, pinned_commit, prefix):
            subjects.append((path, _git_show(repo, f"{pinned_commit}:{path}")))

    raw_entries = []
    run_rows = []
    selection_run_ids = {}

    for path, data in sorted(subjects):
        digest = hashlib.sha256(data).hexdigest()
        if path.endswith(".csv"):
            probe = csv.DictReader(io.StringIO(data.decode("utf-8")), strict=True)
            if not _is_run_row_csv(path, list(probe.fieldnames or [])):
                continue  # no recorded fitness/score value in scope
            columns, rows = _read_metrics_csv(data, path)
            profile = _profile_from_path(path)
            if profile is None:
                raise VerifyError(f"cannot resolve the profile of run-row CSV {path}")
            score_columns = [c for c in columns if c.startswith("score_")]
            for row in rows:
                producer, producer_evidence = vd.producer_for_row(columns, row["cells"])
                tickers = row["tickers"]
                score_recompute, pairs, evidence = _row_score_state(
                    fb, profile, tickers, row["cells"]
                )
                row_id = vd.entry_id(path, "fitness", {"run_id": row["run_id"]})
                run_rows.append({
                    "id": row_id,
                    "path": path,
                    "root": "outputs" if path.startswith("outputs/") else "baselines/v0-legacy/reference",
                    "profile": profile,
                    "run_id": row["run_id"],
                })
                base = {
                    "path": path,
                    "sha256": digest,
                    "row_key": {"run_id": row["run_id"]},
                    "producer": producer,
                    "producer_evidence": producer_evidence,
                    "row_reconstruction": evidence.row_reconstruction,
                    "data_snapshot_match": evidence.data_snapshot_match,
                    "score_recompute": score_recompute,
                    "contributors": [row_id],
                }
                raw_entries.append({
                    **base,
                    "artifact_class": "csv_run_rows",
                    "locator": "fitness",
                    "kind": "fitness",
                    "reported": _float_or_none(row["cells"].get("fitness")),
                    "recomputed": evidence.recomputed,
                })
                for name in score_columns:
                    reported, recomputed = pairs.get(name, (_float_or_none(row["cells"].get(name)), None))
                    raw_entries.append({
                        **base,
                        "artifact_class": "csv_run_rows",
                        "locator": name,
                        "kind": "score",
                        "reported": reported,
                        "recomputed": recomputed,
                    })
        elif path.endswith(".json"):
            try:
                root = json.loads(data.decode("utf-8"))
            except ValueError as exc:
                raise VerifyError(f"{path} is not parseable JSON: {exc}") from exc
            values = _collect_recorded_values(root)
            if not values:
                continue
            root_name = "outputs" if path.startswith("outputs/") else "baselines/v0-legacy/reference"
            for item in values:
                if item["kind"] == "run_row":
                    row = item["row"]
                    profile = item["locator"].lstrip("/").split("/", 1)[0]
                    producer, producer_evidence = vd.producer_for_row(None, row)
                    tickers = [str(t) for t in row.get("tickers") or []]
                    score_recompute, pairs, evidence = _row_score_state(
                        fb, profile, tickers, row
                    )
                    fitness_locator = f"{item['locator']}/fitness"
                    row_id = vd.entry_id(path, fitness_locator, None)
                    run_rows.append({
                        "id": row_id,
                        "path": path,
                        "root": root_name,
                        "profile": profile,
                        "run_id": row.get("run_id"),
                    })
                    base = {
                        "path": path,
                        "sha256": digest,
                        "row_key": None,
                        "producer": producer,
                        "producer_evidence": producer_evidence,
                        "row_reconstruction": evidence.row_reconstruction,
                        "data_snapshot_match": evidence.data_snapshot_match,
                        "score_recompute": score_recompute,
                        "contributors": [row_id],
                    }
                    raw_entries.append({
                        **base,
                        "artifact_class": "json_run_rows",
                        "locator": fitness_locator,
                        "kind": "fitness",
                        "reported": _float_or_none(row.get("fitness")),
                        "recomputed": evidence.recomputed,
                    })
                    for name in sorted(k for k in row if k.startswith("score_")):
                        reported, recomputed = pairs.get(
                            name, (_float_or_none(row.get(name)), None)
                        )
                        raw_entries.append({
                            **base,
                            "artifact_class": "json_run_rows",
                            "locator": f"{item['locator']}/{_json_pointer(name)}",
                            "kind": "score",
                            "reported": reported,
                            "recomputed": recomputed,
                        })
                    continue
                selection_run_id = item.get("selection_run_id")
                if selection_run_id is not None:
                    selection_run_ids[(path, item["locator"])] = selection_run_id
                raw_entries.append({
                    "path": path,
                    "sha256": digest,
                    "row_key": None,
                    "artifact_class": item["artifact_class"],
                    "locator": item["locator"],
                    "kind": item["kind"],
                    "producer": "unrecorded",
                    "producer_evidence": "",
                    "row_reconstruction": "not_applicable",
                    "data_snapshot_match": None,
                    "score_recompute": "not_applicable",
                    "reported": None,
                    "recomputed": None,
                    "contributors": [],
                })

    # D-d contributor mapping, then verdicts through the shared rule
    index = _RowIndex(run_rows, selection_run_ids)
    for entry in raw_entries:
        entry["contributors"] = list(
            vd.contributors_for(
                entry["kind"], entry["path"], entry["locator"], entry["row_key"], index
            )
        )
        if entry["kind"] == "derived":
            entry["row_key"] = None
    entries = recompute_verdicts(raw_entries)
    return entries


def recompute_verdicts(entries):
    """Stamp verdict/defect_ref/reason on each entry via the shared `_verdict` rule.

    Derived entries resolve their contributors' verdicts bottom-up from the same
    entry list, so a verdict is always a pure function of recorded evidence
    (VC-24) - generator and checker run this identical path.
    """
    by_id = {vd.entry_id(e["path"], e["locator"], e["row_key"]): e for e in entries}
    verdicts = {}
    problems = {}

    def _verdict_of(entry_id):
        if entry_id in verdicts:
            return verdicts[entry_id]
        entry = by_id.get(entry_id)
        if entry is None:
            raise VerifyError(f"unknown contributor id {entry_id}")
        if entry["kind"] == "derived":
            contributor_verdicts = [_verdict_of(cid) for cid in entry["contributors"]]
            value = vd.verdict_for({**entry, "contributor_verdicts": contributor_verdicts})
        else:
            value = vd.verdict_for(entry)
        verdicts[entry_id] = value
        return value

    stamped = []
    for entry in entries:
        entry_id = vd.entry_id(entry["path"], entry["locator"], entry["row_key"])
        verdict = _verdict_of(entry_id)
        out = {k: v for k, v in entry.items() if k != "contributor_verdicts"}
        out["verdict"] = verdict
        if verdict in vd.DEFECT_VERDICTS:
            out["defect_ref"] = DEFECT_REF
            out["reason"] = None
        elif verdict == "valid":
            out["defect_ref"] = None
            out["reason"] = None
        else:
            out["defect_ref"] = None
            out["reason"] = _reason_for(entry, verdict)
        errors = vd.annotation_errors(out, verdict)
        if errors:
            problems[entry_id] = errors
        stamped.append(out)
    if problems:
        raise VerifyError(f"annotation discipline broken: {problems}")
    stamped.sort(key=lambda e: (e["path"], e["locator"], json.dumps(e["row_key"], sort_keys=True)))
    return stamped


def _reason_for(entry, verdict):
    if verdict == "score_provenance_inconsistent":
        return SCORE_REASON
    if verdict == "not_verifiable_at_head":
        if entry["kind"] == "derived":
            return (
                DERIVED_NO_CONTRIBUTORS_REASON
                if not entry["contributors"]
                else DERIVED_UNVERIFIABLE_REASON
            )
        return UNVERIFIABLE_REASON
    return f"verdict {verdict}"


# ------------------------------------------------------- check_status #


def check_status(status_path, derived):
    """Problems where the recorded status.json disagrees with fresh derivation ([] = clean).

    Two legs (VC-24): recorded evidence must equal re-derived evidence per entry,
    and every recorded verdict must recompute from its own recorded evidence fields
    through the shared `verdict_for` - a verdict is never trusted on the word of
    the map.
    """
    status_path = Path(status_path)
    try:
        document = json.loads(status_path.read_bytes().decode("utf-8"))
        recorded = list(document["entries"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"status map is unreadable: {exc}"]
    problems = []
    derived_by_id = {
        vd.entry_id(e["path"], e["locator"], e["row_key"]): e for e in derived
    }
    recorded_by_id = {}
    for entry in recorded:
        entry_id = vd.entry_id(entry.get("path", ""), entry.get("locator", ""),
                               entry.get("row_key"))
        if entry_id in recorded_by_id:
            problems.append(f"entry recorded twice: {entry_id}")
        recorded_by_id[entry_id] = entry
    for entry_id in sorted(set(derived_by_id) - set(recorded_by_id)):
        problems.append(f"unrecorded value: {entry_id}")
    for entry_id in sorted(set(recorded_by_id) - set(derived_by_id)):
        problems.append(f"stale entry: {entry_id} does not resolve to a recorded value")
    for entry_id in sorted(set(derived_by_id) & set(recorded_by_id)):
        want, got = derived_by_id[entry_id], recorded_by_id[entry_id]
        for field in ("path", "artifact_class", "locator", "kind", "row_key", "sha256",
                      "producer", "producer_evidence", "row_reconstruction",
                      "data_snapshot_match", "score_recompute", "reported", "recomputed",
                      "contributors"):
            if got.get(field) != want.get(field):
                problems.append(
                    f"{entry_id}: recorded {field}={got.get(field)!r} != re-derived {want.get(field)!r}"
                )
    try:
        recomputed = recompute_verdicts(recorded)
    except (VerifyError, vd.VerdictError) as exc:
        problems.append(f"verdicts do not recompute from the recorded evidence: {exc}")
        return problems
    for entry in recomputed:
        entry_id = vd.entry_id(entry["path"], entry["locator"], entry["row_key"])
        recorded_entry = recorded_by_id.get(entry_id, {})
        if entry["verdict"] != recorded_entry.get("verdict"):
            problems.append(
                f"{entry_id}: recorded verdict {recorded_entry.get('verdict')!r} != "
                f"recomputed {entry['verdict']!r}"
            )
        for field in ("defect_ref", "reason"):
            if entry.get(field) != recorded_entry.get(field):
                problems.append(
                    f"{entry_id}: recorded {field}={recorded_entry.get(field)!r} != "
                    f"recomputed {entry.get(field)!r}"
                )
    return problems


# -------------------------------------------------------------- main #


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--emit-status", metavar="PATH", default=None)
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument("--pinned-base", default=PINNED_BASE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    scratch = tempfile.TemporaryDirectory(prefix="v0-corrected-verify-")
    try:
        report = verify_reference(
            Path(args.reference_dir), Path(args.bundle), Path(scratch.name)
        )
        if args.emit_status is not None:
            derived = derive_status_entries(
                REPO_ROOT, Path(args.bundle), args.pinned_base, Path(scratch.name)
            )
            payload = {
                "schema_version": 1,
                "pinned_base_commit": args.pinned_base,
                "entries": derived,
            }
            Path(args.emit_status).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report["status_map"] = {"emitted": args.emit_status, "entries": len(derived)}
            exit_code = 0 if report["status"] == "PASS" else 1
        else:
            status_path = Path(args.status)
            if status_path.is_file():
                derived = derive_status_entries(
                    REPO_ROOT, Path(args.bundle), args.pinned_base, Path(scratch.name)
                )
                problems = check_status(status_path, derived)
                report["status_map"] = {"checked": str(status_path), "problems": problems}
                if problems:
                    exit_code = 1
                else:
                    exit_code = 0 if report["status"] == "PASS" else 1
            else:
                # status.json is emitted at T13; before that the row verifier is the gate
                report["status_map"] = {
                    "checked": None,
                    "problems": [],
                    "note": "status.json not present - nothing to check yet",
                }
                exit_code = 0 if report["status"] == "PASS" else 1
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            print(f"verify_artifacts: {report['status']}")
            for artifact in report["artifacts"]:
                print(
                    f"  {artifact['path']}: {artifact['status']} "
                    f"rows={artifact['rows_total']} passed={artifact['rows_passed']} "
                    f"failed={artifact['rows_failed']} "
                    f"unverifiable={artifact['rows_unverifiable']}"
                )
                for finding in artifact["findings"][:5]:
                    print(f"    - {finding['row_key']}: {'; '.join(finding['problems'])}")
            status_map = report.get("status_map") or {}
            if status_map.get("problems"):
                for problem in status_map["problems"][:10]:
                    print(f"    ! {problem}")
                exit_code = 1
            elif status_map.get("note"):
                print(f"  {status_map['note']}")
        return exit_code
    except vd.VerdictError as exc:
        # MispairingError and rule failures are gate failures, never folded into a verdict
        print(f"verify_artifacts: gate failure: {exc}", file=sys.stderr)
        return 1
    except VerifyError as exc:
        print(f"verify_artifacts: unusable input: {exc}", file=sys.stderr)
        return 3
    finally:
        scratch.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
