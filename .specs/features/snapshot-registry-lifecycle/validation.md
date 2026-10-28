# Snapshot Registry Lifecycle Validation

**Date**: 2026-10-27
**Spec**: `.specs/features/snapshot-registry-lifecycle/spec.md`
**Diff range**: `330600d..49d4e9c` (9 commits: `f8cf3cf`..`49d4e9c`)
**Verifier**: independent sub-agent (author ≠ verifier; author was the implementer agent)

## Validation Verdict

**Result**: PASS ✅ - every issue AC (1-3), every required test (a-c), and every spec AC (P1/P2/P3) maps to a test whose asserted value is the spec-defined outcome; all five required mutants die.

---

## Task Completion

| Task | Status | Notes |
| ---- | ------ | ----- |
| T1 | ✅ Done | Lifecycle envelope + round-trip; `tests/test_snapshot_registry.py` created |
| T2 | ✅ Done | `manifest_canonical_sha256` single canonicalization point; `validate_manifest` rejects divergence |
| T3 | ✅ Done | `_state_selectable` + `include_validated` on both selector entry points |
| T4 | ✅ Done | CLI skeleton, `build`, atomic writer, `--manifest-id` hardening |
| T5 | ✅ Done | `validate` + `validate --check` (`--json` combinable), exit 4/5 contract |
| T6 | ✅ Done | `publish` + immutability guard (exit 3) |
| T7 | ✅ Done | `retire` state-only (exit 5 terminal) |
| T8 | ✅ Done | `class_series_map` file-level digest, registry-relative paths |
| T9 | ✅ Done | `docs/snapshot_registry.md`; executable doc-content + flag-parity checks pass |

---

## Issue #16 Acceptance Criteria

| Issue item | Proof (test) | Spec-defined outcome asserted | Result |
| ---------- | ------------ | ----------------------------- | ------ |
| AC1 published snapshot immutable + auditable by hash | `tests/test_snapshot_registry.py:117` (`test_manifest_hash_survives_transitions`), `:938` (`test_publish_never_overwrites_published_manifest`), `:828` (`test_verify_published_detects_tampering`) | hash identical before/after every transition (`:125-129`); 2nd publish = `3` with byte-identical file (`:950-951`); `--check` = `4` on a tampered published file, bytes preserved (`:849-850`) | ✅ PASS |
| AC2 validation fails on date or hash divergence | `tests/test_snapshot_registry.py:684` (`test_validate_rejects_divergent_source_hash`), `:701` (`test_validate_rejects_cutoff_date_divergence`) | `4` on source-hash divergence, manifest byte-identical (`:694-699`); `4` on `cutoff_date` != last `common_date` (`:714`) | ✅ PASS |
| AC3 CLI documented + reproducible | `tests/test_snapshot_registry.py:354` (`test_build_is_byte_reproducible_across_registries`) + `docs/snapshot_registry.md:1-238` | two registries, same inputs/`--created-at` ⇒ identical bytes (`:363-366`); doc-content + flag-parity gates run green (see Gate Check) | ✅ PASS |
| Test (a) divergent hash rejected | `tests/test_snapshot_registry.py:684` + existing `tests/test_snapshot_manifest.py:99` (`test_changed_source_hash_is_rejected`) | `4`, no write | ✅ PASS |
| Test (b) manifest without `caio_fiis.csv` fails, no fallback | `tests/test_snapshot_registry.py:717` (`test_validate_requires_caio_fiis_without_ifix_fallback`) + existing `tests/test_allocation_data.py:46` | `4` both when `fiis` is absent from `supported_classes` (`:735`) and when `caio_fiis.csv` is unlinked while `ifix.csv` exists (`:749`) | ✅ PASS |
| Test (c) published snapshot cannot be overwritten | `tests/test_snapshot_registry.py:938` | `3` + `target.read_bytes() == frozen` (`:950-951`) | ✅ PASS |

---

## Spec-Anchored Acceptance Criteria

### P1: Lifecycle CLI with immutability and audit hash

| # | Criterion (WHEN/IF X THEN Y) | Spec-defined outcome | `file:line` + assertion | Result |
| - | ---------------------------- | -------------------- | ----------------------- | ------ |
| 1 | `build` with inputs + `--created-at` writes `<registry>/<manifest_id>.json` as `as_dict()` with `state=draft`, per-source `sha256`, `common_dates`, `cutoff_date`, `manifest_sha256`, exit 0 | exit 0, state `draft`, `supported_classes` == `ASSET_CLASSES`, hash == recompute; fixed `provider`/`metadata` literals, allocation `metadata.files` | `tests/test_snapshot_registry.py:343` `assertEqual(code, 0, err)`; `:346` `raw["state"] == "draft"`; `:348` `raw["supported_classes"] == list(ASSET_CLASSES)`; `:352` `raw["manifest_sha256"] == manifest_canonical_sha256(manifest)`; `:541-561` provider/metadata/path literals | ✅ PASS (see note 1) |
| 2 | differing coverage ⇒ `common_dates` = sorted intersection as `load_premium_snapshot_bundle` consumes, first date zero return, `cutoff_date` = last, exit 4 only when empty | rows == `common_dates`, `["2020-01-01","2020-01-02","2020-01-04"]`, cutoff `2020-01-04`; empty intersection ⇒ 4, no file | `:394-399` `row_dates == tuple(raw["common_dates"])` + exact list/cutoff; `:409-410` `assertEqual(code, 4)` / `assertFalse(target.exists())` | ✅ PASS |
| 3 | identical inputs + `--created-at` in two registries ⇒ byte-identical files | identical bytes | `:363-366` `assertEqual(first.read_bytes(), second.read_bytes())` | ✅ PASS |
| 4 | existing target or missing source ⇒ exit 3, nothing modified | 3 + bytes unchanged; 3 + no manifest | `:482-484` `assertEqual(code, 3)` + `target.read_bytes() == original`; `:510-512` | ✅ PASS |
| 5 | `validate` good manifest ⇒ `draft`→`validated`, exit 0 | 0, `state=validated`, hash + unknown key preserved | `:808-812` `assertEqual(code, 0, err)`; `after["state"]=="validated"`; `after["manifest_sha256"]==declared_hash` | ✅ PASS |
| 6 | hash or `cutoff_date` divergence ⇒ exit 4, no file modified | 4 + manifest byte-identical | `:694-699` (plain and `--check`) + `:714` `assertEqual(code, 4)` | ✅ PASS |
| 7 | `build` required source missing ⇒ exit 3, no fallback (incl. IFIX) | 3, stderr names `caio_fiis.csv`, no manifest even with `ifix.csv` present | `:510-512` `assertIn("caio_fiis.csv", err)` | ✅ PASS |
| 8 | required class series absent (e.g. `caio_fiis.csv`) ⇒ `validate` exit 4, no fallback (incl. IFIX) | 4 in both variants | `:735` `assertEqual(code, 4)`; `:749` `assertEqual(code, 4)` | ✅ PASS |
| 9 | `publish` on `validated` ⇒ `published`, exit 0, hash-covered fields unchanged | 0, state `published`, hash equal, all non-envelope fields equal | `:922-936` `assertEqual(code, 0, err)`; `after["manifest_sha256"]==before[...]`; `assertEqual(after_content, before_content)` | ✅ PASS |
| 10 | `publish` on `published` ⇒ exit 3, byte-identical | 3 + frozen bytes | `:950-951` | ✅ PASS |
| 11 | `published` refuses content modification; only `retire` changes lifecycle fields | target/output + retire preserves every non-envelope field byte-for-byte | `:936` content equality; `:1059-1076` `after == before` on non-envelope JSON bytes | ✅ PASS |
| 12 | `retire` on `published`/`validated` ⇒ `retired` + reason, exit 0, no deletion | 0, state `retired`, reason recorded; file set unchanged, data file still present | `:994-997` `assertEqual(code, 0, err)`; `raw["state"]=="retired"`; `raw["retire_reason"]=="superseded"`; `:1020-1024` `assertEqual(after, before)` + `caio_fiis.csv` exists | ✅ PASS |
| 13 | invalid transition ⇒ exit 5, nothing modified | 5 + bytes frozen (publish on draft/retired, retire twice/draft, validate on published/retired) | `:612-631` `assertRaises(AlreadyPublishedError/InvalidTransitionError)`; `:970-971`, `:1044-1045`, `:870-871` `assertEqual(code, 5)` + frozen bytes | ✅ PASS |
| 14 | `manifest_sha256` = SHA-256 of canonical JSON of normalized `as_dict()` minus 4 keys | key order irrelevant, envelope-constant, build value == recompute | `:122-129` reorder/envelope variants equal baseline; `:352` build value == `manifest_canonical_sha256` | ✅ PASS |
| 15 | declared `manifest_sha256` diverging ⇒ `validate_manifest` and CLI `validate` reject | raises `SnapshotManifestError`; CLI `--check` = 4 in every state | `:162-165` `assertRaisesRegex(..., "manifest_sha256 does not match")`; `:848-850` | ✅ PASS (see note 2) |
| 16 | `validate --check` (optionally `--json`) runs full verification set in ANY state, skipping only the state gate; 0/4, zero writes | 4 for `fiis` absent and for incoherent cutoff in both `validate` and `--check`; 4 on tamper in draft/validated/published/retired; 0 + zero writes for a clean published/retired manifest | `:779-780`, `:792-793`; `:840-850`; `:1212-1217` `assertEqual(code, 0)`, map printed, bytes frozen | ✅ PASS |
| 17 | bad `--manifest-id` (pattern, `metadata`, != loaded) ⇒ exit 2, no write | 2, no JSON created; 2 + "differs from loaded manifest_id" | `:442-444` `assertEqual(code, 2, err)` + `sorted(root.glob("*.json")) == []`; `:902-903` | ✅ PASS |
| 18 | no network access, verified by stdlib source scan | regex scan finds no `urllib/socket/requests/http` | `:570-572` `assertIsNone(re.search(...))` | ✅ PASS |
| 19 | every manifest write atomic (temp in target dir + `os.replace`) | `os.replace` failure leaves original byte-identical, no `.tmp` leftover | `:522-531` `assertRaises(OSError)`; `assertEqual(bytes, original)`; `leftovers == []` | ✅ PASS |
| 20 | legacy (no `state`) treated as `draft` for transitions, `manifest_sha256` backfilled | 0, state validated, hash == recompute, unknown key kept | `:887-891` | ✅ PASS |
| 21 | registry root has only manifest `*.json` (+ `metadata.json`); data in subdirs; no loose state/index JSON | root glob == `["snap-2020-01.json"]`, data dir exists, `manifests()` still finds exactly 1 | `:580-588` | ✅ PASS (see note 3) |

### P2: State-aware selection

| # | Criterion | Spec-defined outcome | `file:line` + assertion | Result |
| - | --------- | -------------------- | ----------------------- | ------ |
| 1 | never selects `draft`/`retired` | older `published` selected although `draft`/`retired` are newer | `tests/test_snapshot_registry.py:217-223` `assertEqual(selected.manifest_id, "published-old")` (both entry points) | ✅ PASS |
| 2 | `include_validated=True` also accepts `validated` | default → `published-old`; flag → `validated-new` | `:232-241` | ✅ PASS |
| 3 | legacy manifests accepted exactly as today, same result | legacy-only registry → newest by cutoff (`newer`/`2020-01-03`) | `:253-256` `assertEqual(selected.manifest_id, "newer")`; legacy round-trip `:85-95` | ✅ PASS (see note 4) |
| 4 | no qualifying candidate ⇒ raise, no fallback | `SnapshotManifestError` with and without `include_validated` | `:265-272` `assertRaisesRegex(..., "no compatible snapshot manifest found")` | ✅ PASS |

### P3: Auditable class-to-series mapping

| # | Criterion | Spec-defined outcome | `file:line` + assertion | Result |
| - | --------- | -------------------- | ----------------------- | ------ |
| 1 | `validate --json` / `--check --json` report per class: registry-relative path, `file_sha256` (never directory digest), provider; `--check --json` in every state with 0/4, zero writes | map printed, 0, bytes frozen; file digests, relative paths, `derived_from` | `tests/test_snapshot_registry.py:1212-1217` `assertEqual(code, 0)` + `assertEqual(json.loads(out), class_series_map(...))` + frozen bytes; content proofs `:1096-1105`, `:1125-1127`; `validate --json` on draft `:1243-1250` | ✅ PASS (see note 5) |
| 2 | deterministic; `fiis`→`caio_fiis.csv` no fallback; only class inputs + `ptax` + selectors, never `ifix.csv`/`ibovespa.csv` | key set == classes ∪ {ptax, stock, fii}; no ifix/ibovespa in output; `fiis` path exact; byte-stable | `:1160-1165` `assertEqual(set(mapping), set(ASSET_CLASSES) | {"ptax","stock","fii"})` + `assertNotIn`; `:1173-1176`; `:1183-1188`; `:1233-1235` | ✅ PASS |
| 3 | docs state layout, commands, exit table, prohibitions | all required strings present; flags exist; exit table matches implementation | `docs/snapshot_registry.md:10-29` (layout), `:86-153` (commands), `:155-166` (exit table), `:203-214` (prohibitions); gates: doc-content EXIT 0, flag parity EXIT 0 | ✅ PASS (see note 6) |

**Status**: all 28 spec ACs covered; 0 behavioral gaps; 6 precision notes (non-blocking).

---

## Discrimination Sensor

Isolated scratch: `tests/` + `py/` copied to `/tmp/opencode/sensor-p009.*`; each mutation applied to the copy only, `python3 -m unittest tests.test_snapshot_registry -v` run there, copy discarded. The real worktree was never mutated (`git status --porcelain` empty before and after).

| # | Mutation | File:line (origin) | Killed? | Killer test |
| - | -------- | ------------------ | ------- | ----------- |
| 1 | `class_series_map`: `file_sha256(resolved)` → `sha256_path(resolved)` (literal sensor spec) | `py/snapshot_registry.py:373` | **SURVIVED (equivalent)** | none - `file_sha256` (`py/snapshot_manifest.py:74-79`) delegates to `sha256_path` for files, and every source the P3 contract admits resolves to a file, so the mutant is behaviorally identical for contract-valid manifests (a directory-valued class source is outside P3 and never produced by `build`) |
| 2 | Same site, real defect: hash the series' PARENT DIRECTORY (`sha256_path(resolved.parent)`) so the allocation directory digest is exposed as a file hash | `py/snapshot_registry.py:373` | ✅ **Killed** | `tests/test_snapshot_registry.py:1082` `test_class_series_map_hashes_the_file_not_the_directory` (+ `:1237`) |
| 3 | State gate moved AFTER content verification in `validate` (published tampered ⇒ 4 instead of 5) | `py/snapshot_registry.py:386-388` | ✅ **Killed** | `:852` `test_validate_on_published_or_retired_exits_5_without_writes` (+ `:1190`) |
| 4 | `build_manifest`: `common_dates` = UNION of series dates instead of the intersection | `py/snapshot_registry.py:249-250` | ✅ **Killed** | `:368` `test_build_common_dates_match_premium_bundle_rows` |
| 5 | `publish` overwrite allowed (`AlreadyPublishedError` guard removed) | `py/snapshot_registry.py:173-174` | ✅ **Killed** | `:938` `test_publish_never_overwrites_published_manifest` (+ `:592`) |
| 6 | `latest_compatible` state filter removed (drafts/retired selectable) | `py/snapshot_manifest.py:574` | ✅ **Killed** | `:209`, `:225`, `:258` `SelectionStateTests` |

**Sensor depth**: P0-full (6 behavior-level mutations ≥ the required 5).
**Result**: 5/5 non-equivalent mutants killed; 1 equivalent mutant (mutation 1) - no test can distinguish it because the two functions are identical for files. No surviving mutant; no fix task required.
**Isolation check**: scratch copies only, removed after each run; real-tree porcelain empty before and after.

---

## Code Quality

| Principle | Status |
| --------- | ------ |
| Minimum code | ✅ 593-line CLI + 98 additive manifest lines; no unused abstraction beyond the required helpers |
| Surgical changes | ✅ `py/snapshot_manifest.py` diff is +98/-0 - `sha256_path`, `file_sha256`, `ALLOCATION_FILES`, `manifests()` untouched |
| No scope creep | ✅ no scheduler, no storage HA, no `fetch_allocation_snapshot` change, no pytest adoption |
| Matches patterns | ✅ stdlib `unittest`, `sys.path` to `py/`, `tempfile` fixtures (mirrors `tests/test_snapshot_manifest.py`) |
| Spec-anchored outcome check (asserted values match spec) | ✅ all 28 ACs (table above) |
| Per-layer Coverage Expectation (domain 1:1 ACs) | ✅ 51 new tests for 28 ACs + 10 edge cases |
| Every test maps to a spec requirement - no unclaimed tests | ✅ each test class maps to P1/P2/P3; `LifecycleHelperTests` maps to AC13/AC14/AC20 |
| Documented guidelines followed | `docs/snapshot_registry.md` + `.specs/STATE.md` AD-001..AD-006 |

---

## Edge Cases

- [x] Registry dir missing ⇒ exit 3 naming the path - handled (`py/snapshot_registry.py:315-317`, `:458-460`); verified by manual probe (validate=3, build=3); no automated test (note 7).
- [x] `--created-at` missing/invalid ⇒ exit 2, nothing written - `tests/test_snapshot_registry.py:430-435`.
- [x] `--manifest-id` pattern/reserved/loaded mismatch ⇒ exit 2 - `:442-444`, `:902-903`.
- [x] `build` input escaping the registry root ⇒ exit 2 - `:471-472`.
- [x] Required source absent from manifest or disk ⇒ exit 4, no fallback - `:717-749`, `:751-764`.
- [x] Unknown `state` string ⇒ load error - `:76-83`; CLI surfaces as 4 (probed: `manifest cannot be loaded: unknown manifest state: frozen`).
- [x] Content modified after `validated` ⇒ `validate` 4 and `--check` 4 in any state - `:828-850`.
- [x] `validate` on legacy changes only envelope fields - `:873-891`.
- [x] Overlapping/narrowed date coverage ⇒ intersection - `:368-399`.
- [x] Failed `os.replace` ⇒ original byte-identical, no partial manifest - `:514-531`.
- [x] Legacy manifest never validated: `retire`/`publish` exit 5 (design F9 row) - probed: `retire=5`, `publish=5`, then `validate=0` with backfilled hash.

---

## Gate Check

- **Gate command (quick)**: `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data` → **60 tests, OK, exit 0**
- **New file gate**: `python3 -m unittest tests.test_snapshot_registry -v` → **51 tests, OK, exit 0**
- **Build gate**: `python3 -m unittest tests.test_allocation tests.test_allocation_data tests.test_snapshot_manifest tests.test_snapshot_registry tests.test_fetch_status_invest tests.test_fetch_status_invest_fii` → run as part of discover below (same stdlib subtree), gate modules green
- **Full discover**: `python3 -m unittest discover -s tests` → **101 tests, 4 errors, exit 1**; the 4 errors are pre-existing pandas import failures: `test_fii_backtest_analysis`, `test_fii_selection`, `test_multi_run_backtest`, `test_premium_research_contracts` (identical set at baseline)
- **Compile**: `python3 -m py_compile py/snapshot_registry.py py/snapshot_manifest.py` → exit 0
- **CLI**: `python3 py/snapshot_registry.py --help` → exit 0
- **Doc gate (T9)**: doc-content check → exit 0; flag parity for `build validate publish retire --help` → exit 0
- **Test count before feature**: 50 (baseline, per task brief) → **after**: 101 → **delta +51**, no pre-existing test file modified (`git diff --stat` lists only the 5 feature files)
- **Skipped tests**: none
- **Failures**: only the 4 pre-existing pandas errors, out of scope per spec assumption "Pre-existing suite failures"

### End-to-end smoke (isolated `mktemp -d`, no `snapshots/` in the repo)

`build` → 0 · `validate` → 0 · `publish` → 0 · `publish` again → **3** · `validate --check --json` → **0** (map printed, 8 entries) · `retire --reason smoke` → **0** · `retire` again → **5**. Final on-disk state `retired`, reason recorded, registry root contains only `snap-2020-01.json`.

---

## Contract Regressions

None.

- `sha256_path` / `file_sha256` / `ALLOCATION_FILES` / `SnapshotRegistry.manifests()`: unchanged by construction - `git diff 330600d..49d4e9c -- py/snapshot_manifest.py` reports **98 insertions, 0 deletions**.
- Legacy `SnapshotManifest` (no `state`/`manifest_sha256`): loads, validates, selects - `tests/test_snapshot_registry.py:85`, `:167`, `:244`, plus the untouched `tests/test_snapshot_manifest.py` suite (green).
- `from_dict(as_dict())` idempotent for legacy/unknown-key/draft/retired payloads - `:97-113`.
- `manifest_sha256` recomputable from persisted provenance: **static verification** (API deps unavailable). `api/app/services/premium_recommendation.py:207-208` persists `{"manifest_path": ..., "manifest": _manifest_payload(manifest)}` where `_manifest_payload` is `as_dict()` plus an injected `manifest_path` (`:86-88`); `api/app/services/premium_executor.py:152-163` rehydrates with `SnapshotManifest.from_dict(manifest_data, base_dir=...)` then `validate_manifest`. `from_dict` ignores the extra `manifest_path` key, so `manifest_canonical_sha256` recomputes over the same `as_dict()` projection - modeled exactly by `tests/test_snapshot_registry.py:131-145`.
- Docs ↔ `--help` parity: documented flags `--registry`, `--manifest-id`, `--created-at`, `--snapshot-dir`, `--stock-universe`, `--fii-universe`, `--check`, `--json`, `--reason` all exist; documented exit codes 0/2/3/4/5 match the implemented constants (`py/snapshot_registry.py:45-49`) and the smoke run.

---

## Fix Plans

None - no blocking gap. Non-blocking precision notes (no fix task required to mark done):

1. **Note 1**: `build`'s emitted `allocation_source`/`stock_source`/`fii_source` aliases are not asserted directly; they are produced by `as_dict()` (`py/snapshot_manifest.py:382-384`) and consumed by `validate_manifest`. Low risk; a direct key assertion would harden AC1.
2. **Note 2**: AC15's "CLI `validate`" half is exercised through `--check` (`:848`), not through plain `validate` on a `manifest_sha256`-divergent *draft*. Same code path (`py/snapshot_manifest.py:486-489`).
3. **Note 3**: AC21 reads "root contains only manifest `*.json`" literally, while root-level selector CSVs are allowed and used by fixtures (`:302-305`) and documented (`docs/snapshot_registry.md:17`). The loader glob contract (the actual risk) is intact and tested.
4. **Note 4**: P2-3 "same result as the current implementation" is proven against the spec-defined ordering (newest cutoff), not by a differential run of the old code.
5. **Note 5**: `--check --json` is asserted for `validated`/`published`/`retired`; the `draft` case is only probed manually (exit 0, map printed).
6. **Note 6**: the doc-content and flag-parity checks are executable commands from T9's done-when, not persisted `unittest` cases. They pass now; a regression in prose would not be caught by the suite.
7. **Note 7**: "registry directory does not exist ⇒ exit 3" is handled and probed manually but has no automated test.

---

## Requirement Traceability Update

`spec.md` was intentionally left untouched (the Verifier has no commit authority; editing it would dirty the tree without an atomic commit). Verified status recorded here:

| Requirement | Previous Status | Verified Status |
| ----------- | --------------- | --------------- |
| SNAP-01 | In Design | ✅ Verified |
| SNAP-02 | In Design | ✅ Verified |
| SNAP-03 | In Design | ✅ Verified |
| SNAP-04 | In Design | ✅ Verified |
| SNAP-05 | In Design | ✅ Verified |
| SNAP-06 | In Design | ✅ Verified |
| SNAP-07 | In Design | ✅ Verified |
| SNAP-08 | In Design | ✅ Verified |
| SNAP-09 | In Design | ✅ Verified |
| SNAP-10 | In Design | ✅ Verified |

---

## Summary

**Overall**: ✅ Ready

**Spec-anchored check**: 28/28 ACs matched the spec-defined outcome | 0 gaps | 6 precision notes
**Issue map**: AC1, AC2, AC3, (a), (b), (c) all proven by a named test
**Sensor**: 5/5 distinguishable mutants killed (1 equivalent mutant documented)
**Gate**: 51 new tests / 60 quick-gate tests green; 101 discovered with the 4 pre-existing pandas errors only

**What works**: full offline lifecycle with hash-audited immutability; divergence rejection with zero writes; legacy manifests untouched and still selectable; class-to-series audit map with file-level digests; reproducible build across registries; documented CLI whose exit codes match reality.

**Issues found**: none blocking. The 7 precision notes above are hardening opportunities only.

**Next steps**: none required for completion. Optional: persist the doc-content gate as a unittest and add the draft `--check --json` / missing-registry-dir cases.

---

## Post-fix re-validation (delta 063a8c3..eb67580)

**Date**: 2026-10-27
**Verifier**: independent sub-agent (author ≠ verifier), fresh re-derivation - the fixer's report was not trusted
**HEAD**: `eb67580` on `t3/issue-16`, clean tree
**Delta**: `063a8c3..eb67580` (3 commits: `b5f19d7` build source/target overlap, `373753a` state-aware selection error + rewrite guard, `eb67580` OSError mapping + docs)
**Review under test**: PR #34 review `5671339117` (COMMENT - F1/F2 should-fix, F3/F4/F5 nits)
**Diff shape**: 5 files, +301/-20; `tests/` pure addition (+152/-0), `py/` +113/-10, docs +38/-10.

**Result**: PASS ✅ - F1..F5 are all fixed, each fix is killed by a named test, and nothing regressed.

### Fix-by-fix verdict

| # | Finding (review `5671339117`) | Fix location | Discriminating test | Revert → failure | Status |
| - | ---------------------------- | ------------ | ------------------- | ---------------- | ------ |
| F1 | `--snapshot-dir` equal to/ancestor of the registry root produced a manifest that could never validate | `py/snapshot_registry.py:329` (`_contains_path`), guard at `:520-531` | `tests/test_snapshot_registry.py:545` `test_build_snapshot_dir_containing_manifest_target_exits_2` + positive guard `:574` `test_build_registry_data_layout_still_builds_and_validates` | guard removed → `AssertionError: 0 != 2` (build accepted the overlap and exited 0) | ✅ fixed_confirmed |
| F2 | `validated` manifests invisible to the only in-repo consumer, unrecorded in the design | `py/snapshot_manifest.py:546` `_non_selectable_state_hint`, raise at `:608-617`; fail-closed predicate unchanged at `:532`; recorded in `design.md:117` + `docs/snapshot_registry.md:237` | `:275` `test_selection_error_counts_states_and_instructs_publish`, `:318` `test_legacy_manifest_without_state_remains_selectable`, `:331` `test_published_manifest_remains_selectable` | hint neutered → `test_selection_error_counts_states_and_instructs_publish` fails (`"2 validated"` / `"publish"` absent) | ✅ fixed_confirmed |
| F3 | `ENVELOPE_KEYS` was a dead constant | `py/snapshot_registry.py:124` `_assert_envelope_only`, invoked at `:174`; documented `design.md:130` | `:768` `test_rewrite_envelope_refuses_non_envelope_mutation` | call removed → mutation of `created_at` accepted, test fails (`SnapshotManifestError` not raised, bytes changed) | ✅ fixed_confirmed |
| F4 | registry `OSError` escaped the exit-code table (traceback, exit 1) | `py/snapshot_registry.py:647-656` (`except OSError` at `:649`) | `:1213` `test_unwritable_registry_maps_oserror_to_exit_2_without_traceback` (`skipIf` root; uid 1000 here, so it ran) | `except` removed → `PermissionError: [Errno 13] ... .snap-2020-01.json.*.tmp` escapes `main()` (test errors) | ✅ fixed_confirmed |
| F5 | design promised `O_CREAT\|O_EXCL`; code uses `exists()` + `os.replace` | `design.md:129` vs `py/snapshot_registry.py:98-117` | documentation-only; verified by reading both | n/a (no behavior) | ✅ fixed_confirmed |

F2 is a resolution of the review's actual complaint ("sem registro no design"), not a weakening: `_state_selectable` (`py/snapshot_manifest.py:532`) and `include_validated=False` default are untouched, and mutation M2b (state filter removed) is still killed by four selection tests - fail-closed survives.

### Discrimination sensor (post-fix)

Isolated scratch `/tmp/opencode/p009-delta/scratch` (copies of `py/` + `tests/`); one behavior-level edit per run, restored from the real tree after each. Real tree never mutated - `git status --porcelain` empty before and after (`diff -r -x __pycache__` scratch/py vs real `py/` = identical).

| ID | Mutation | Verdict | Killer |
| -- | -------- | ------- | ------ |
| M1 | F1 guard deleted (overlap accepted again) | **killed** | `test_build_snapshot_dir_containing_manifest_target_exits_2` (exit 0 != 2) |
| M1b | F1 over-strict: every `kind == "dir"` rejected | **killed** | 8 `BuildCommandTests` incl. `test_build_registry_data_layout_still_builds_and_validates` (positive guard discriminates over-blocking) |
| M2 | F2 hint forced empty | **killed** | `test_selection_error_counts_states_and_instructs_publish` |
| M2b | F2 state filter removed (`draft`/`retired`/`validated` all selectable) | **killed** | `test_selection_excludes_draft_and_retired`, `test_selection_admits_validated_only_with_include_validated`, `test_selection_without_qualifying_candidate_raises`, `test_selection_error_counts_states_and_instructs_publish` |
| M3 | F3 `_assert_envelope_only(baseline, result, path)` call removed | **killed** | `test_rewrite_envelope_refuses_non_envelope_mutation` |
| M3b | F3 `manifest_sha256` clause removed (declared hash may be rewritten) | **SURVIVED** | none - see gap G1 |
| M4 | F4 `except OSError` removed from `main()` | **killed** | `test_unwritable_registry_maps_oserror_to_exit_2_without_traceback` |

6/7 mutants killed. M3b is a real (non-equivalent) survivor, so it is reported as a gap rather than folded into the PASS:

**G1 (non-blocking, hardening)**: `_assert_envelope_only` (`py/snapshot_registry.py:141-149`) also forbids changing a *declared* `manifest_sha256` (backfill when absent stays legal). No test exercises that clause; `transition_state` never changes the hash, so the code path is defense-in-depth only. The F3 claim under test - content keys cannot change through `rewrite_envelope` - is fully proven. Suggested follow-up: one test in `LifecycleHelperTests` that mutates `manifest_sha256` in the `mutate` callable and expects `SnapshotManifestError`.

### Checks (all commands re-run at `eb67580`)

| Command | Exit | Result |
| ------- | ---- | ------ |
| `python3 -m unittest tests.test_snapshot_registry tests.test_snapshot_manifest tests.test_allocation_data` | 0 | `Ran 67 tests ... OK` (was 60 at the pre-fix gate; +7 new) |
| `python3 -m unittest discover -s tests` | 1 | `Ran 108 tests`, `FAILED (errors=4)` - exactly `test_fii_backtest_analysis`, `test_fii_selection`, `test_multi_run_backtest`, `test_premium_research_contracts` (`ModuleNotFoundError: pandas`, pre-existing) |
| `python3 -m unittest tests.test_snapshot_registry -v` | 0 | `Ran 58 tests ... OK`, 0 skips executed |
| `python3 -m py_compile py/snapshot_registry.py py/snapshot_manifest.py` | 0 | clean |
| `python3 py/snapshot_registry.py --help` | 0 | clean |
| E2E smoke, `mktemp -d`, real CLI process | 0 | `build` 0 → `validate` 0 → `publish` 0 → `publish` 3 (`published manifest is never overwritten`) → `validate --check --json` 0 (map 8 entries, key set == classes ∪ {ptax, stock, fii}) → `retire --reason smoke` 0 → `retire` 5 (`cannot retire a retired manifest`) |
| E2E F1 repro: `build --snapshot-dir <registry>` | 2 | `--snapshot-dir must not contain the destination manifest path: <registry> contains <registry>/snap-conflict.json; place the snapshot in <registry>/data/<snapshot_name>/`; directory listing byte-identical before/after (no manifest, no temp file) |
| `validate` on a data file `chmod 000` | 4 | `validation failed: [Errno 13] Permission denied: .../caio_fiis.csv`, no traceback (`--check` also 4) - the exit-4 "unreadable source" row of the docs is *not* shadowed by F4's exit-2 mapping |
| `python3 .../validate_state.py snapshot-registry-lifecycle` | 0 | `0 error(s)` |

### Contract checks (unchanged surfaces)

- `sha256_path`, `file_sha256`, `ALLOCATION_FILES`, `SnapshotRegistry.manifests()`: no hunk in the delta touches them (`git diff 063a8c3..eb67580 -- py/snapshot_manifest.py` = +39/-3, all inside selection error reporting). `ALLOCATION_FILES` key set == `ASSET_CLASSES ∪ {ptax}` (6 entries, `fiis → caio_fiis.csv`).
- Legacy manifest (no `state`/`manifest_sha256`): loads, `validate_manifest` passes, `latest_compatible_manifest` selects it (`state is None`) - `test_legacy_manifest_without_state_remains_selectable` + direct probe.
- `from_dict(as_dict())` idempotent for legacy / draft / published / retired (fixed point after one normalization) - probe green.
- `manifest_sha256` recomputable on the provenance path: persisted `as_dict()` + injected `manifest_path`, rehydrated with `SnapshotManifest.from_dict(..., base_dir=...)`, recomputed hash == stored hash (`43bec974…`) and `validate_manifest` accepts it.
- No existing test weakened, skipped, or removed: `git diff 063a8c3..eb67580 -- tests/` is `152 0` (additions only, 7 new `def test_`), and the only `skipIf` in the suite is F4's root guard.

### Regressions

None. Behaviour that the pre-fix report certified still holds (67/67 quick gate, E2E lifecycle identical), and both new guards are shown to be *not* over-strict (M1b, plus the `data/` layout positive test).

### Limitations

1. **Premium path is validated by reading, not by execution** - `api/` imports pandas/FastAPI, unavailable offline. The chain `premium_recommendation._manifest_payload` → persisted `{"manifest_path", "manifest"}` (`api/app/services/premium_recommendation.py:79-90`, `:203-208`) → `premium_executor._load_manifest` → `from_dict` + `validate_manifest` (`:152-168`) was re-checked at `eb67580` and is unchanged by the delta; the hash recompute it depends on is proven by the probe above. No runtime proof of the F2 consumption semantics (that a `validated`-only registry makes Premium unavailable) beyond the selection tests.
2. F2's hint for `include_validated=True` (validated counted as selectable, therefore absent from the counts) is correct by construction but has no test.
3. Sensor mutations were applied to a file copy, not through the CLI process; the E2E smoke covers the process-level exit codes for the fixed behaviours (F1/F4 paths included).
4. The 4 pandas errors are pre-existing and out of scope per the spec assumption; they were not re-triaged.
