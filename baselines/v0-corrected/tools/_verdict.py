"""_verdict.py - the single verdict path shared by generator, checker and marking test.

Pure standard library. This module owns the ordered Verdict Attribution Rule of
spec.md (VC-12) as `verdict_for`, the mechanical evidence helpers behind it, the
bit-identity and row-tie semantics of VC-25/VC-29, and `entry_id`. The generator
(`verify_artifacts.py --emit-status`) and the marking test call the SAME
`verdict_for`, so a verdict cannot drift between who writes it and who checks it.

The decision table is evaluated strictly in the spec's order. The corrected-capture
mispairing row is a gate failure (`MispairingError`), never a verdict: a row produced
by the corrected capture whose score fields all reproduce yet whose reported fitness
does not equal the recomputation contradicts the fix itself.

Annotation discipline (VC-12) is separate from the rule: `defect_ref` attaches only to
the two defect-attributed verdicts and every other non-`valid` verdict carries a
`reason`. `annotation_errors` checks that discipline; `verdict_for` computes verdicts
from evidence fields only.

The D-d contributor mapping is mechanical and auditable: `contributors_for` resolves
contributor ids through a caller-built `RowIndex` (duck-typed, methods documented
below) so the ids recorded in each entry are derivable, never asserted.
"""

import json
import math
import struct

VERDICTS = (
    "invalid_fitness",
    "invalid_beyond_fitness",
    "score_provenance_inconsistent",
    "not_verifiable_at_head",
    "valid",
)
KINDS = ("fitness", "score", "derived")
PRODUCERS = ("legacy_capture", "corrected_capture", "unrecorded")
DEFECT_VERDICTS = ("invalid_fitness", "invalid_beyond_fitness")
ROW_RECONSTRUCTIONS = ("exact", "inexact", "not_applicable")
SCORE_RECOMPUTES = ("match", "mismatch", "not_applicable")

TOL = 1e-9  # "reproduces within 1e-9" / "hhi and fitness within 1e-9" (VC-11, VC-29)

EVIDENCE_CORRECTED = "generations_run+converged_early columns present"
EVIDENCE_LEGACY = "run-row keys absent"


class VerdictError(Exception):
    """An entry's evidence cannot resolve to a verdict under the decision table."""


class MispairingError(VerdictError):
    """Corrected-capture mispairing: gate failure, never folded into a verdict."""


def producer_for_row(columns, row=None):
    """(producer, evidence) for one run row from its key set (CSV header or JSON row keys).

    `generations_run` and `converged_early` present => corrected_capture
    (writer py/pipelines/multi_run.py via py/core/optimizer.py); absent => legacy_capture
    (writer py/ga_multiple_runs.py). Applies uniformly to CSV rows and JSON run-row dicts.
    """
    keys = list(row.keys()) if columns is None else list(columns)
    if "generations_run" in keys and "converged_early" in keys:
        return "corrected_capture", EVIDENCE_CORRECTED
    return "legacy_capture", EVIDENCE_LEGACY


def producer_for_family(producers):
    """The uniform producer of a contributor family; anything else is `unrecorded`."""
    values = list(producers)
    for value in values:
        if value not in PRODUCERS:
            raise VerdictError(f"unknown producer {value!r} in contributor family")
    if len(set(values)) == 1:
        return values[0]
    return "unrecorded"


def score_recompute(per_field):
    """`match` iff EVERY recorded score field reproduces within TOL, else `mismatch`.

    `per_field` maps each recorded score field name to (reported, recomputed).
    `not_applicable` only when the row records no score field at all. A field whose
    recomputation is absent does not reproduce, so the row is `mismatch` - one metric
    matching can never buy a `match` (AD-015).
    """
    fields = dict(per_field)
    if not fields:
        return "not_applicable"
    for name, pair in fields.items():
        try:
            reported, recomputed = pair
        except (TypeError, ValueError) as exc:
            raise VerdictError(f"score field {name!r} is not a (reported, recomputed) pair") from exc
        if reported is None or recomputed is None:
            return "mismatch"
        if not _within(reported, recomputed):
            return "mismatch"
    return "match"


def row_reconstruction(n_expected, tickers, universe):
    """`exact` iff exactly n distinct tickers, all in the frozen universe; else `inexact`."""
    values = [str(t) for t in tickers]
    distinct = set(values)
    known = set(str(u) for u in universe)
    if len(values) == n_expected and len(distinct) == n_expected and distinct <= known:
        return "exact"
    return "inexact"


def contributors_for(kind, path, locator, row_key, index):
    """Mechanical contributor ids for one entry (D-d mapping, in order):

    (1) a run row's contributors are itself;
    (2) a selection recording a run_id -> the run rows with that (profile, run_id)
        in the same tree root;
    (3) aggregates inside a file that has its own run rows -> that file's run rows;
    (4) other derived leaves -> all run rows of the same profile in the same tree root
        (outputs/ vs the legacy reference tree);
    (5) no mechanical contributor set -> [] (the caller records producer `unrecorded`).

    `index` is a caller-built RowIndex with:
      rows_in_file(path) -> list[str]            ids of run rows recorded in that file
      rows_in(root, profile, run_id=None) -> list[str]  ids of run rows of that
          (root, profile), optionally narrowed to one run_id
      root_of(path) -> str | None                tree root a path belongs to
      profile_of(path, locator) -> str | None    profile from the JSON subtree key or
          the file-name suffix
      recorded_run_id(path, locator) -> int | None  run_id a selection records, else None
    """
    if kind in ("fitness", "score"):
        # (1) a run row contributes itself
        return [entry_id(path, locator, row_key)]
    run_id = row_key.get("run_id") if isinstance(row_key, dict) else None
    if run_id is None:
        # (2) selections carry the run_id they record
        run_id = index.recorded_run_id(path, locator)
    root = index.root_of(path)
    profile = index.profile_of(path, locator)
    if run_id is not None:
        return list(index.rows_in(root, profile, int(run_id)))
    in_file = list(index.rows_in_file(path))
    if in_file:
        # (3) the file's own run rows
        return in_file
    if root is None or profile is None:
        # (5) no mechanical contributor set
        return []
    # (4) conservative superset: every run row of that profile in that tree root
    return list(index.rows_in(root, profile))


def verdict_for(entry):
    """The ordered Verdict Attribution Rule (spec.md) over an entry's evidence fields.

    Reads `kind` plus the evidence fields `producer`, `row_reconstruction`,
    `data_snapshot_match`, `score_recompute`, `reported`, `recomputed` (and
    `contributor_verdicts` for `derived` entries). Raises MispairingError on the
    corrected-capture mispairing row - a gate failure that can never become a verdict.
    """
    kind = entry.get("kind")
    if kind not in KINDS:
        raise VerdictError(f"entry kind must be one of {KINDS}, got {kind!r}")
    if kind == "derived":
        return _derived_verdict(entry)

    producer = _require(entry, "producer")
    if producer not in PRODUCERS:
        raise VerdictError(f"unknown producer {producer!r}")
    reconstruction = _require(entry, "row_reconstruction")
    if reconstruction not in ROW_RECONSTRUCTIONS:
        raise VerdictError(f"unknown row_reconstruction {reconstruction!r}")
    snapshot = _require(entry, "data_snapshot_match")
    recompute = _require(entry, "score_recompute")
    if recompute not in SCORE_RECOMPUTES:
        raise VerdictError(f"unknown score_recompute {recompute!r}")
    reported = _require(entry, "reported")
    recomputed = _require(entry, "recomputed")

    # rows in the spec's order - an earlier row resolves before any later one
    if reconstruction == "inexact":
        return "not_verifiable_at_head"
    if snapshot is False:
        return "not_verifiable_at_head"
    if recompute == "mismatch":
        return "score_provenance_inconsistent"
    if kind == "score":
        # score fields: mismatch resolved above, match resolves here; nothing else is valid
        if recompute == "match":
            return "valid"
        raise VerdictError("a score entry must record score fields (score_recompute not_applicable)")
    if producer == "unrecorded":
        return "not_verifiable_at_head"
    if reported is None or recomputed is None:
        raise VerdictError("reported/recomputed are required once the rule reaches the producer rows")
    # "reported == recomputed" resolves at the VC-01 tolerance (within 1e-9), like the
    # recomputation checks themselves; VC-25's exactness is a different comparison
    if producer == "legacy_capture" and not _within(reported, recomputed):
        return "invalid_fitness"
    if producer in ("legacy_capture", "corrected_capture") and _within(reported, recomputed):
        return "valid"
    if producer == "corrected_capture" and not _within(reported, recomputed):
        raise MispairingError(
            "corrected_capture row with score_recompute match but reported != recomputed: "
            "unexpected mispairing that contradicts the fix - gate failure, never a verdict"
        )
    raise VerdictError(f"no decision-table row resolves for producer={producer!r}")


def _derived_verdict(entry):
    """Derived values resolve through their contributors' verdicts (spec derived rows)."""
    verdicts = list(entry.get("contributor_verdicts") or [])
    for verdict in verdicts:
        if verdict not in VERDICTS:
            raise VerdictError(f"unknown contributor verdict {verdict!r}")
    if not verdicts:
        # D-d case (5): no mechanical contributor set -> not_verifiable_at_head
        return "not_verifiable_at_head"
    if "invalid_fitness" in verdicts:
        return "invalid_beyond_fitness"
    if "not_verifiable_at_head" in verdicts:
        return "not_verifiable_at_head"
    if "score_provenance_inconsistent" in verdicts:
        return "score_provenance_inconsistent"
    return "valid"


def annotation_errors(entry, verdict):
    """VC-12 annotation discipline problems for one entry at one verdict ( [] = clean ).

    `defect_ref` attaches only to `invalid_fitness` / `invalid_beyond_fitness`; those
    two verdicts require it; every other non-`valid` verdict requires a `reason`.
    """
    if verdict not in VERDICTS:
        raise VerdictError(f"unknown verdict {verdict!r}")
    problems = []
    defect_ref = entry.get("defect_ref")
    reason = entry.get("reason")
    if verdict in DEFECT_VERDICTS:
        if not defect_ref:
            problems.append(f"{verdict} requires a defect_ref naming the capture defect")
    else:
        if defect_ref:
            problems.append(
                f"defect_ref attaches only to {'/'.join(DEFECT_VERDICTS)}, not to {verdict}"
            )
    if verdict not in DEFECT_VERDICTS and verdict != "valid" and not str(reason or "").strip():
        problems.append(f"{verdict} requires a reason naming its actual cause")
    return problems


def entry_id(path, locator, row_key):
    """Stable identity of one recorded value: path + locator + row key."""
    key = "null" if row_key is None else json.dumps(row_key, sort_keys=True, separators=(",", ":"))
    return f"{path}|{locator}|{key}"


def float_bitwise_equal(a, b):
    """VC-25 bit-identity: the compared cells are equal iff their packed float64 bits are."""
    return struct.pack(">d", float(a)) == struct.pack(">d", float(b))


def tickers_equal(a, b):
    """Ticker sets are equal (order-insensitive, duplicate-safe)."""
    left, right = list(a), list(b)
    return len(left) == len(right) and set(left) == set(right)


def tie_row(recorded, driver, tol=TOL):
    """VC-29 row tie: tickers set + hhi + fitness within tol. [] = pass.

    `recorded` and `driver` are mappings with `tickers`, `fitness` and `hhi`.
    Every divergence is returned as a problem line; nothing here raises.
    """
    problems = []
    for key in ("tickers", "fitness", "hhi"):
        if key not in recorded or key not in driver:
            problems.append(f"missing {key}: recorded={key in recorded} driver={key in driver}")
            return problems
    if not tickers_equal(recorded["tickers"], driver["tickers"]):
        problems.append(
            f"tickers differ: recorded={sorted(recorded['tickers'])} driver={sorted(driver['tickers'])}"
        )
    for key in ("fitness", "hhi"):
        try:
            left, right = float(recorded[key]), float(driver[key])
        except (TypeError, ValueError):
            problems.append(f"{key} is not numeric: recorded={recorded[key]!r} driver={driver[key]!r}")
            continue
        if not _within(left, right, tol):
            problems.append(f"{key} differs beyond {tol}: recorded={left!r} driver={right!r}")
    return problems


def _within(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _require(entry, name):
    if name not in entry:
        raise VerdictError(f"entry is missing evidence field {name!r}")
    return entry[name]
