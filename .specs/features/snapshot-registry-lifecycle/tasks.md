# Immutable Snapshot Registry & CLI Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/snapshot-registry-lifecycle/design.md`
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `.specs/codebase/TESTING.md` - it states there is **no maintained pytest/unittest suite** and suggests pandas-based testing, which is unavailable in this environment, so this plan follows the `tests/` conventions instead (stdlib `unittest`, `sys.path` to `py/`, `tempfile` fixtures per `tests/test_snapshot_manifest.py:31`); no linter/coverage config in repo (`pyproject.toml` has no dev deps). Environment: `pytest`/`pandas`/`api/.venv` absent - stdlib only.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Registry CLI (new `py/snapshot_registry.py`) | unit | All branches; 1:1 to spec ACs; every listed edge case has a test | `tests/test_snapshot_registry.py` | `python3 -m unittest tests.test_snapshot_registry` |
| Manifest contract (extended `py/snapshot_manifest.py`) | unit | All branches; round-trip idempotency; legacy compat; state semantics | `tests/test_snapshot_registry.py`, `tests/test_snapshot_manifest.py` | `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest` |
| Docs (`docs/snapshot_registry.md`) | none | - (build gate + executable doc-content checks; reader walkthrough marked `manual`) | - | build gate + doc content checks (T9) |
| API integration (read-only consumers) | none | - (covered indirectly by legacy-selection tests) | - | quick gate only |

**Coverage Expectation provenance**: strong default (all branches, 1:1 to ACs) since no coverage thresholds are configured; existing tests are a floor only.

## Gate Check Commands

> Generated from codebase - confirm before Execute. Scope note: `tests/test_premium_research_contracts.py`, `tests/test_fii_selection.py`, `tests/test_multi_run_backtest.py`, `tests/test_fii_backtest_analysis.py` fail pre-existing on missing `pandas` in this environment (verified 2026-10-27) and are excluded from gates.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After tasks with unit tests only | `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data` |
| Full | After tasks touching manifest selection or loader contracts | `python3 -m unittest tests.test_allocation tests.test_allocation_data tests.test_snapshot_manifest tests.test_snapshot_registry tests.test_fetch_status_invest tests.test_fetch_status_invest_fii` |
| Build | After docs/config tasks or phase completion | **full gate +** `python3 -m compileall -q py/snapshot_registry.py py/snapshot_manifest.py` (T9 runs both halves) |

> `tests/test_snapshot_registry.py` is created (minimal harness) by T1, so the quick gate is runnable - and green - from T1 on (review F2). The Build gate is exactly "full gate + compileall"; T9's done-when invokes both halves (review F12).

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

**What**: Extend `SnapshotManifest` with optional `state`, `state_changed_at`, `retire_reason`, `manifest_sha256` fields; `from_dict` rejects unknown `state` values; `as_dict` omits absent fields so `from_dict(as_dict())` stays byte-idempotent (incl. legacy manifests and unknown payload keys). Also CREATE `tests/test_snapshot_registry.py` (minimal stdlib `unittest` harness + `tempfile` fixture helpers in the style of `tests/test_snapshot_manifest.py:31`) so the quick gate is runnable from T1 on; every new test of this feature lives in that file (review F2).
**Where**: `py/snapshot_manifest.py`
**Depends on**: None
**Reuses**: `py/snapshot_manifest.py:181` `from_dict`/`as_dict`, `tests/test_snapshot_manifest.py:31` `build_manifest_fixture`
**Requirement**: SNAP-01

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `SnapshotManifest` accepts and round-trips all four optional fields; absent fields are omitted from `as_dict()`
- [x] Unknown `state` string raises `SnapshotManifestError`; legacy manifest dict without `state` loads unchanged
- [x] Round-trip test: `from_dict(as_dict(m)).as_dict() == m.as_dict()` for legacy, draft, and retired payloads
- [x] `tests/test_snapshot_registry.py` exists with the harness and the T1 tests, so `python3 -m unittest tests.test_snapshot_registry` runs green from T1 onward
- [x] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [x] Test count: ≥ 4 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

#### T2: Add canonical manifest content hash

**What**: Implement `manifest_canonical_sha256` as the SINGLE canonicalization point (design AD-002, review F1): normalize the input via `SnapshotManifest.from_dict(raw).as_dict()` (a `SnapshotManifest` instance is passed as `manifest.as_dict()`), strip `manifest_sha256`/`state`/`state_changed_at`/`retire_reason`, hash `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`. Unknown payload keys are NOT part of the hash (the `as_dict()` projection discards them). Make `validate_manifest` reject a manifest whose declared `manifest_sha256` diverges from this recomputation (legacy without the field unchanged).
**Where**: `py/snapshot_manifest.py`
**Depends on**: T1
**Reuses**: `py/snapshot_manifest.py:393` `validate_manifest`, `hashlib` usage in `sha256_path`
**Requirement**: SNAP-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Hash is deterministic across key order and identical for a manifest before/after lifecycle transitions (`test_manifest_hash_survives_transitions`)
- [x] `as_dict() → json → from_dict → validate_manifest` recomputes the same hash, including a `manifest_path` injected exactly as `api/app/services/premium_recommendation.py:87-88` persists it (`test_manifest_hash_survives_premium_rehydration`)
- [x] `validate_manifest` exits the failure path (`SnapshotManifestError`) on `manifest_sha256` divergence and passes when it matches
- [x] Legacy manifest without `manifest_sha256` still validates
- [x] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [x] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

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

- [x] Selection excludes `draft` and `retired` regardless of cutoff recency
- [x] `include_validated=True` admits `validated`; default does not
- [x] Legacy-only fixture registry returns the same manifest as the current implementation
- [x] No-candidate case still raises `SnapshotManifestError` with no fallback
- [x] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [x] Test count: ≥ 4 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

---

### Phase 2: Registry CLI

#### T4: Create registry CLI skeleton and `build` command

**What**: New `py/snapshot_registry.py` with argparse dispatch, documented exit codes (0/2/3/4/5), `--manifest-id` validation (`^[A-Za-z0-9][A-Za-z0-9._-]*$`, never `metadata`, must equal the loaded `manifest_id`; violations exit 2 - review F7), `write_manifest_atomic` (fixed disk serializer `json.dumps(..., sort_keys=True, indent=2, ensure_ascii=False)` + trailing `\n`; temp file created in the target's OWN directory - review F15 - then `os.replace`; create-exclusive when not overwriting), and `build`: require `--created-at` and inputs (`--snapshot-dir`, `--stock-universe`, `--fii-universe`) resolving INSIDE the registry root (escape after `Path.resolve()` ⇒ exit 2 - review F5, nit 3; `data/<snapshot_name>/` is the conventional layout); emit exactly the `SnapshotManifest.as_dict()` form (single `allocation` source with `metadata.files`, explicit `stock`/`fii` selector sources, registry-root-relative POSIX paths) with `supported_classes` derived from `ASSET_CLASSES` (`py/allocation_config.py:14-20`) in fixed order; fix each source's `provider` as a literal constant per source role and `metadata` exactly `{"files": {...}}` (`allocation`) or `{}` (selectors) - NEVER copy fetch `metadata.json` content (its absolute paths would break AC3 byte-reproducibility - review N5); derive `common_dates` as the sorted intersection of the five class series exactly as `load_premium_snapshot_bundle` consumes them (`international_equity`/`crypto` already multiplied by `ptax`), `cutoff_date` = last date, exit 4 when the intersection is empty (review F3); hash sources via `sha256_path`/`file_sha256`; write `<registry>/<manifest_id>.json` as `draft` with `manifest_sha256`.
**Where**: `py/snapshot_registry.py`
**Depends on**: T1, T2
**Reuses**: `py/snapshot_manifest.py` (`sha256_path`, `file_sha256`, `SnapshotManifest`, `manifest_canonical_sha256`), `py/allocation_data.py:290-315` (`load_premium_snapshot_bundle` rows check), fixture layout of `tests/test_snapshot_manifest.py:31`
**Requirement**: SNAP-03, SNAP-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `build` writes a valid `draft` manifest (passes `validate_manifest`) and exits 0
- [x] Building the same `manifest_id` with identical inputs and `--created-at` in TWO different registries yields byte-identical files (`test_build_is_byte_reproducible_across_registries` - review F4; a repeated `build` in one registry is exit 3 by design)
- [x] `common_dates` equals the row dates `load_premium_snapshot_bundle` returns for a fixture whose `ptax`/`crypto` series carry extra dates (`test_build_common_dates_match_premium_bundle_rows`); an empty intersection exits 4
- [x] Missing `--created-at` exits 2; `--manifest-id` `../../x` / `metadata` / != loaded id exit 2 (`test_manifest_id_rejects_traversal_and_reserved_name`); input escaping the registry root exits 2; existing target or missing source exits 3; nothing is modified
- [x] A failing `os.replace` leaves the original manifest byte-identical (`test_failed_replace_keeps_original_manifest`)
- [x] Source `provider`/`metadata` are the fixed literals per source role (`metadata.files` / `{}`) with no absolute-path leak from fetch `metadata.json` (`test_build_source_metadata_is_fixed_and_absolute_path_free` - review N5)
- [x] The new module's source declares no network facility (`urllib|socket|requests|http` scan is empty) (`test_cli_source_has_no_network_imports` - issue AC18, review N7)
- [x] Registry root contains only manifest JSON (data in subdirectories) - layout test asserts the loader glob is unaffected
- [x] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [x] Test count: ≥ 9 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

#### T5: Implement `validate` command with divergence checks

**What**: `validate` runs `validate_manifest` + `manifest_sha256` recomputation + `cutoff_date == last common_date` check; exits 4 on any hash/date divergence or on a required source of the manifest under audit being absent from the manifest OR missing/unreadable on disk (hash divergente rejeitado; missing `caio_fiis.csv` fails with no IFIX fallback; review F10 + ruling N1: in `validate`/`validate --check` every required-source failure is 4 - after `publish` the data files are part of the audited artifact); on success transitions `draft -> validated` via envelope-only rewrite (legacy: treat absent state as draft, backfill `manifest_sha256`); `published`/`retired` exit 5; failures modify nothing. Also implement `validate --check` (flag on `validate`, NOT a subcommand - review F6): read-only audit that runs the FULL `validate_manifest` verification set - skipping only the lifecycle-state gate ("hashes match" is necessary, not sufficient - review N2) - in ANY lifecycle state and exits 0 / 4 on any verification failure without writing or transitioning.
**Where**: `py/snapshot_registry.py`
**Depends on**: T4
**Reuses**: `py/snapshot_manifest.py` `validate_manifest`, `transition_state` helper from T4 skeleton
**Requirement**: SNAP-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Divergent source hash exits 4 and leaves the manifest byte-identical (issue IT-1) (`test_validate_rejects_divergent_source_hash`)
- [ ] `cutoff_date` different from last `common_dates` entry exits 4 (issue AC2) (`test_validate_rejects_cutoff_date_divergence`)
- [ ] Manifest missing `caio_fiis.csv` fails with exit 4; `ifix.csv` present is never substituted (issue IT-2) (`test_validate_requires_caio_fiis_without_ifix_fallback`)
- [ ] A required source file missing/unreadable on disk (e.g. `caio_fiis.csv` unlinked, as `tests/test_allocation_data.py:45-57` builds it) exits 4 in BOTH `validate` and `validate --check`, modifying nothing (`test_validate_missing_source_file_exits_4` - review ruling N1)
- [ ] `validate --check` runs the full `validate_manifest` set: a manifest with consistent source hashes but a contract violation (e.g. `fiis` absent, `cutoff_date` incoherent) exits 4 in `--check` exactly as in `validate` (`test_check_runs_full_validation_not_just_hashes` - review N2)
- [ ] Success path sets `state: validated` and preserves `manifest_sha256` and unknown keys
- [ ] Re-running `validate` on `validated` exits 0 with no write
- [ ] `validate --check` exits 0 on a published fixture and 4 after a content field is tampered with, in every state, writing nothing (`test_verify_published_detects_tampering`)
- [ ] `validate` on `published`/`retired` exits 5 and leaves the file byte-identical (`test_validate_on_published_or_retired_exits_5_without_writes`)
- [ ] A legacy manifest without `manifest_sha256` gets it backfilled on its first `validate` (`test_validate_backfills_manifest_sha256_on_legacy_manifest`)
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 10 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

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
- [ ] Second `publish` exits 3 and the file bytes are unchanged (issue IT-3) (`test_publish_never_overwrites_published_manifest`)
- [ ] `publish` on `draft`/`retired` exits 5 with no write
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 3 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

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
- [ ] Content fields (normalized projection) are byte-identical before and after `retire`, and unknown keys in the file are preserved (issue AC11/AC12) (`test_retire_preserves_content_fields_byte_for_byte`)
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 4 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

---

### Phase 3: Audit Mapping & Documentation

#### T8: Expose auditable class-to-series mapping

**What**: `class_series_map(manifest)` returns the deterministic class -> `{path, sha256, provider}` map (fiis -> `caio_fiis.csv`, no fallback) where `path` is registry-root-relative POSIX (re-relativized - `source_for` returns ABSOLUTE paths for derived series) and `sha256` is the `file_sha256` of the series FILE itself, never the `allocation` directory digest `source_for` hands back for derived classes (review N4); derived entries carry `derived_from: "allocation"`. Printed by `validate --json`, combinable with `--check` (`validate --check --json`, 0/4 semantics, zero writes) so the map is printable in EVERY lifecycle state including `published`/`retired` (review N3; review F11: `--json` is a `validate` flag only - `build` does not print the map).
**Where**: `py/snapshot_registry.py`
**Depends on**: T5
**Reuses**: `SnapshotManifest.source_for`, `ALLOCATION_FILES` (`py/snapshot_manifest.py:26`) - referenced only, no refactor
**Requirement**: SNAP-09

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Mapping covers every `supported_classes` entry plus `ptax` and the explicit selector sources - and ONLY those: `ifix.csv`/`ibovespa.csv` (always written by `py/fetch_allocation_snapshot.py:523-532`) are not class inputs and are never mapped (review F14)
- [ ] `fiis` resolves to `caio_fiis.csv`; absence fails (no IFIX fallback)
- [ ] Every entry's `sha256` is the `file_sha256` of the resolved file itself; `fiis`/`brazilian_stocks` entries never expose the `allocation` directory digest and carry `derived_from: "allocation"` (`test_class_series_map_hashes_the_file_not_the_directory` - review N4)
- [ ] `path` values are registry-root-relative POSIX even for derived series (`test_class_series_map_uses_registry_relative_paths` - review N4)
- [ ] `validate --check --json` prints the same map for a `published` (and `retired`) fixture with 0/4 semantics and zero writes (`test_check_json_prints_map_in_published_state` - review N3)
- [ ] Output is byte-stable across runs on the same manifest
- [ ] Gate check passes: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`
- [ ] Test count: ≥ 6 new tests pass (no silent deletions)

**Tests**: unit (`tests.test_snapshot_registry`)
**Gate**: quick (`python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data`)

#### T9: Write reproducible registry documentation

**What**: `docs/snapshot_registry.md`: registry layout (design AD-001), all four commands with exact invocations (`build`/`validate`/`publish`/`retire`, `validate --check`, `validate --json`, `validate --check --json`), the exit code table (incl. the 3/4 split of review F10 as ruled in N1: exit 3 = missing CLI target / refused overwrite / `build` input, exit 4 = the audited artifact fails any required-source check), the state machine table (incl. the legacy row fix of review F9), the canonical hash definition (AD-002, `as_dict()` projection; `supported_classes` order pinned by `build` from `ASSET_CLASSES`), and the prohibitions (no network, no overwrite of published artifacts, no deletion on `retire`, no series fallback, single-operator writes).
**Where**: `docs/snapshot_registry.md`
**Depends on**: T4, T5, T6, T7, T8
**Reuses**: `design.md` tables (state machine, exit codes, layout)
**Requirement**: SNAP-10

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Doc-content check passes (executable): `python3 -c "import pathlib,sys; t=pathlib.Path('docs/snapshot_registry.md').read_text(encoding='utf-8'); required=['build','validate --check','validate --check --json','validate --json','publish','retire','exit 2','exit 3','exit 4','exit 5','os.replace','caio_fiis.csv','no network','never overwrite','never deletes','no fallback','single-operator']; missing=[s for s in required if s not in t]; sys.exit('missing: %s' % missing if missing else 0)"`
- [ ] Flag parity check passes (executable): `for c in build validate publish retire; do python3 py/snapshot_registry.py "$c" --help >/dev/null || exit 1; done` and every documented flag appears in the corresponding `--help` output
- [ ] Every documented exit code matches the implemented table (executable: the doc-content check asserts the five exit-code spellings above)
- [ ] Reader walkthrough of a fixture `build -> validate -> publish -> retire` run from the doc alone: **manual** (not automated; explicitly out of the gate)
- [ ] Build gate passes (full gate + compileall): `python3 -m unittest tests.test_allocation tests.test_allocation_data tests.test_snapshot_manifest tests.test_snapshot_registry tests.test_fetch_status_invest tests.test_fetch_status_invest_fii` and `python3 -m compileall -q py/snapshot_registry.py py/snapshot_manifest.py`

**Tests**: none (doc artifact; executable doc-content + flag-parity checks in Done when, walkthrough marked `manual`)
**Gate**: build (full gate + `python3 -m compileall -q py/snapshot_registry.py py/snapshot_manifest.py`)

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
