# Immutable Snapshot Registry & CLI Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/snapshot-registry-lifecycle/design.md`
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `.specs/codebase/TESTING.md` (dependency-light, focused tests), `tests/` conventions (stdlib `unittest`, `sys.path` to `py/`, `tempfile` fixtures per `tests/test_snapshot_manifest.py:31`); no linter/coverage config in repo (`pyproject.toml` has no dev deps). Environment: `pytest`/`pandas`/`api/.venv` absent - stdlib only.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Registry CLI (new `py/snapshot_registry.py`) | unit | All branches; 1:1 to spec ACs; every listed edge case has a test | `tests/test_snapshot_registry.py` | `python3 -m unittest tests.test_snapshot_registry` |
| Manifest contract (extended `py/snapshot_manifest.py`) | unit | All branches; round-trip idempotency; legacy compat; state semantics | `tests/test_snapshot_registry.py`, `tests/test_snapshot_manifest.py` | `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest` |
| Docs (`docs/snapshot_registry.md`) | none | - (build gate + manual doc review) | - | build gate only |
| API integration (read-only consumers) | none | - (covered indirectly by legacy-selection tests) | - | quick gate only |

**Coverage Expectation provenance**: strong default (all branches, 1:1 to ACs) since no coverage thresholds are configured; existing tests are a floor only.

## Gate Check Commands

> Generated from codebase - confirm before Execute. Scope note: `tests/test_premium_research_contracts.py`, `tests/test_fii_selection.py`, `tests/test_multi_run_backtest.py`, `tests/test_fii_backtest_analysis.py` fail pre-existing on missing `pandas` in this environment (verified 2026-10-27) and are excluded from gates.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data` |
| Full | After tasks touching manifest selection or loader contracts | `python3 -m unittest tests.test_allocation tests.test_allocation_data tests.test_snapshot_manifest tests.test_snapshot_registry tests.test_fetch_status_invest tests.test_fetch_status_invest_fii` |
| Build | After docs/config tasks or phase completion | full gate + `python3 -m compileall -q py/snapshot_registry.py py/snapshot_manifest.py` |

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Manifest Contract Extension

Pure, additive changes to `py/snapshot_manifest.py` (no CLI yet).

```
T1 -> T2
T1 -> T3
```

### Phase 2: Registry CLI

New offline module `py/snapshot_registry.py`, one command per task.

```
T4 -> T5 -> T6 -> T7
```

### Phase 3: Audit Mapping & Documentation

Class-to-series auditability and the reproducible CLI documentation.

```
T8 -> T9
```

---

## Task Breakdown

### Phase 1: Manifest Contract Extension

#### T1: Add optional lifecycle envelope fields to SnapshotManifest

**What**: Extend `SnapshotManifest` with optional `state`, `state_changed_at`, `retire_reason`, `manifest_sha256` fields; `from_dict` rejects unknown `state` values; `as_dict` omits absent fields so `from_dict(as_dict())` stays byte-idempotent (incl. legacy manifests and unknown payload keys).
**Where**: `py/snapshot_manifest.py`
**Depends on**: None
**Reuses**: `py/snapshot_manifest.py:181` `from_dict`/`as_dict`, `tests/test_snapshot_manifest.py:31` `build_manifest_fixture`
**Requirement**: SNAP-01

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `SnapshotManifest` accepts and round-trips all four optional fields; absent fields are omitted from `as_dict()`
- [ ] Unknown `state` string raises `SnapshotManifestError`; legacy manifest dict without `state` loads unchanged
- [ ] Round-trip test: `from_dict(as_dict(m)).as_dict() == m.as_dict()` for legacy, draft, and retired payloads
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 4 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T2: Add canonical manifest content hash

**What**: Implement `manifest_canonical_sha256` (canonical JSON per design AD-002: sorted keys, compact separators, UTF-8; excludes `manifest_sha256`, `state`, `state_changed_at`, `retire_reason`) and make `validate_manifest` reject a manifest whose declared `manifest_sha256` diverges from the recomputed hash (legacy without the field unchanged).
**Where**: `py/snapshot_manifest.py`
**Depends on**: T1
**Reuses**: `py/snapshot_manifest.py:393` `validate_manifest`, `hashlib` usage in `sha256_path`
**Requirement**: SNAP-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Hash is deterministic across key order and identical for a manifest before/after lifecycle transitions
- [ ] `validate_manifest` exits the failure path (`SnapshotManifestError`) on `manifest_sha256` divergence and passes when it matches
- [ ] Legacy manifest without `manifest_sha256` still validates
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T3: Add lifecycle state semantics to registry selection

**What**: Extend `SnapshotRegistry.latest_compatible` and `latest_compatible_manifest` with `include_validated: bool = False`: never select `draft`/`retired`, select `validated` only with the flag, select state-less (legacy) manifests as today, keep sort order and no-fallback behavior.
**Where**: `py/snapshot_manifest.py`
**Depends on**: T1
**Reuses**: `py/snapshot_manifest.py:452` `SnapshotRegistry`, fixture builder pattern
**Requirement**: SNAP-08

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Selection excludes `draft` and `retired` regardless of cutoff recency
- [ ] `include_validated=True` admits `validated`; default does not
- [ ] Legacy-only fixture registry returns the same manifest as the current implementation
- [ ] No-candidate case still raises `SnapshotManifestError` with no fallback
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 4 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

---

### Phase 2: Registry CLI

#### T4: Create registry CLI skeleton and `build` command

**What**: New `py/snapshot_registry.py` with argparse dispatch, documented exit codes (0/2/3/4/5), `write_manifest_atomic` (tmp + `os.replace`, create-exclusive when not overwriting), and `build`: derive `common_dates`/`cutoff_date` from the required series date columns, hash sources via `sha256_path`/`file_sha256`, write `<registry>/<manifest_id>.json` as `draft` with `manifest_sha256`; require `--created-at`.
**Where**: `py/snapshot_registry.py`
**Depends on**: T1, T2
**Reuses**: `py/snapshot_manifest.py` (`sha256_path`, `file_sha256`, `SnapshotManifest`, `manifest_canonical_sha256`), fixture layout of `tests/test_snapshot_manifest.py:31`
**Requirement**: SNAP-03, SNAP-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `build` writes a valid `draft` manifest (passes `validate_manifest`) and exits 0
- [ ] Re-running `build` with identical inputs and `--created-at` yields a byte-identical file
- [ ] Missing `--created-at` exits 2; existing target or missing source exits 3; nothing is modified
- [ ] Registry root contains only manifest JSON (data in subdirectories) - layout test asserts the loader glob is unaffected
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 5 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T5: Implement `validate` command with divergence checks

**What**: `validate` runs `validate_manifest` + `manifest_sha256` recomputation + `cutoff_date == last common_date` check; exits 4 on any hash/date divergence (hash divergente rejeitado; missing `caio_fiis.csv` fails with no IFIX fallback); on success transitions `draft -> validated` via envelope-only rewrite (legacy: treat absent state as draft, backfill `manifest_sha256`); `published`/`retired` exit 5; failures modify nothing.
**Where**: `py/snapshot_registry.py`
**Depends on**: T4
**Reuses**: `py/snapshot_manifest.py` `validate_manifest`, `transition_state` helper from T4 skeleton
**Requirement**: SNAP-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Divergent source hash exits 4 and leaves the manifest byte-identical (issue IT-1)
- [ ] `cutoff_date` different from last `common_dates` entry exits 4 (issue AC2)
- [ ] Manifest missing `caio_fiis.csv` fails; `ifix.csv` present is never substituted (issue IT-2)
- [ ] Success path sets `state: validated` and preserves `manifest_sha256` and unknown keys
- [ ] Re-running `validate` on `validated` exits 0 with no write
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 6 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T6: Implement `publish` command with immutability guard

**What**: `publish` transitions `validated -> published` via envelope-only rewrite; refuses any other source state; an already-`published` manifest exits 3 with a byte-identical file; content fields covered by `manifest_sha256` never change.
**Where**: `py/snapshot_registry.py`
**Depends on**: T5
**Reuses**: `transition_state`, `rewrite_envelope`, `write_manifest_atomic` from T4/T5
**Requirement**: SNAP-05

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Publishing a `validated` manifest exits 0 and only envelope fields change (issue AC1)
- [ ] Second `publish` exits 3 and the file bytes are unchanged (issue IT-3)
- [ ] `publish` on `draft`/`retired` exits 5 with no write
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T7: Implement `retire` command (state-only, never deletes)

**What**: `retire` transitions `published|validated -> retired`, records optional `--reason`, is terminal (re-retire exits 5), and never deletes or rewrites any non-envelope field or data file.
**Where**: `py/snapshot_registry.py`
**Depends on**: T6
**Reuses**: `transition_state`, `rewrite_envelope` from T4/T5
**Requirement**: SNAP-06

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `retire` on `published` sets `state: retired` (+ reason) and exits 0
- [ ] No file in the registry is deleted (data files and manifest remain on disk)
- [ ] `retire` on `draft`/`retired` exits 5 with no write
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

---

### Phase 3: Audit Mapping & Documentation

#### T8: Expose auditable class-to-series mapping

**What**: `class_series_map(manifest)` returns the deterministic class -> `{path, sha256, provider}` map (fiis -> `caio_fiis.csv`, no fallback) and is printed by `build`/`validate` (`--json` machine-readable, stable key order).
**Where**: `py/snapshot_registry.py`
**Depends on**: T5
**Reuses**: `SnapshotManifest.source_for`, `ALLOCATION_FILES` (`py/snapshot_manifest.py:26`) - referenced only, no refactor
**Requirement**: SNAP-09

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Mapping covers every `supported_classes` entry plus `ptax` and the explicit selector sources
- [ ] `fiis` resolves to `caio_fiis.csv`; absence fails (no IFIX fallback)
- [ ] Output is byte-stable across runs on the same manifest
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit
**Gate**: quick

#### T9: Write reproducible registry documentation

**What**: `docs/snapshot_registry.md`: registry layout (design AD-001), all four commands with exact invocations, the exit code table, the state machine table, the canonical hash definition (AD-002), and the prohibitions (no network, no overwrite of published artifacts, no deletion on `retire`, no series fallback, single-operator writes).
**Where**: `docs/snapshot_registry.md`
**Depends on**: T4, T5, T6, T7, T8
**Reuses**: `design.md` tables (state machine, exit codes, layout)
**Requirement**: SNAP-10

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Every documented command matches the implemented flags and exit codes (checked against `--help` output)
- [ ] A reader can reproduce a fixture `build -> validate -> publish -> retire` run from the doc alone
- [ ] Prohibitions section lists all five prohibitions
- [ ] Gate check passes: `python3 -m compileall -q py/snapshot_registry.py py/snapshot_manifest.py`

**Tests**: none
**Gate**: build

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1:  T1 -> T2
          T1 -> T3
Phase 2:  T4 -> T5
          T5 -> T6
          T6 -> T7
Phase 3:  T8 -> T9
```

Execution is strictly sequential - there is no intra-phase parallelism. A single agent (or batch worker) works one task at a time, in order.

**Suggested batches (>8 tasks - sub-agents are NOT dispatched by the SPEC stage; recorded for the IMPLEMENT stage):**

| Batch | Phases | Tasks | Count |
| ----- | ------ | ----- | ----- |
| A | Phase 1 + Phase 2 | T1-T7 | 7 |
| B | Phase 3 | T8-T9 | 2 |

Batches run sequentially; each worker executes its tasks in order (implement -> gate -> atomic commit) and reports a compact summary. The always-on Verifier runs after the final task.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Lifecycle envelope fields | 1 file (dataclass + parsers) | ✅ Granular |
| T2: Canonical manifest hash | 1 function + 1 validation hook | ✅ Granular |
| T3: Selection state semantics | 1 method (+ module-level wrapper) | ✅ Granular |
| T4: CLI skeleton + build | 1 new file, 1 command + 2 helpers | ✅ Granular |
| T5: validate command | 1 command | ✅ Granular |
| T6: publish command | 1 command | ✅ Granular |
| T7: retire command | 1 command | ✅ Granular |
| T8: class-to-series mapping | 1 function + CLI output hook | ✅ Granular |
| T9: documentation | 1 file | ✅ Granular |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | none | (no incoming arrow) | ✅ Match |
| T2 | T1 | `T1 -> T2` (Phase 1) | ✅ Match |
| T3 | T1 | `T1 -> T3` (Phase 1) | ✅ Match |
| T4 | T1, T2 (cross-phase; arrows exempt from intra-phase parity) | `T4 -> T5` outgoing | ✅ Match |
| T5 | T4 | `T4 -> T5` (Phase 2) | ✅ Match |
| T6 | T5 | `T5 -> T6` (Phase 2) | ✅ Match |
| T7 | T6 | `T6 -> T7` (Phase 2) | ✅ Match |
| T8 | T5 (cross-phase; exempt) | `T8 -> T9` outgoing | ✅ Match |
| T9 | T4, T5, T6, T7 (cross-phase; exempt), T8 | `T8 -> T9` (Phase 3) | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Manifest contract | unit | unit | ✅ OK |
| T2 | Manifest contract | unit | unit | ✅ OK |
| T3 | Manifest contract | unit | unit | ✅ OK |
| T4 | Registry CLI | unit | unit | ✅ OK |
| T5 | Registry CLI | unit | unit | ✅ OK |
| T6 | Registry CLI | unit | unit | ✅ OK |
| T7 | Registry CLI | unit | unit | ✅ OK |
| T8 | Registry CLI | unit | unit | ✅ OK |
| T9 | Docs | none | none | ✅ OK (matrix says none) |
