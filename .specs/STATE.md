# STATE

## Decisions

### AD-001
- **Decision**: Snapshot registry layout is fixed: manifest files are the only `*.json` at the registry root (`<registry>/<manifest_id>.json`, plus the loader-tolerated `metadata.json`); snapshot data lives in subdirectories; lifecycle state lives inside the manifest envelope - never in loose state/index files at the root.
- **Reason**: `SnapshotRegistry.manifests()` globs `*.json` at the registry root and skips only `metadata.json`, so any loose JSON there would be parsed as a manifest and break `api/app/services/premium_recommendation.py` selection; a single source of truth also matches the issue wording "manifest legado sem `state`".
- **Trade-off**: `retire` rewrites the manifest file (envelope fields only), so immutability is defined over content fields, not over file bytes.
- **Scope**: `py/snapshot_manifest.py`, `py/snapshot_registry.py`, `snapshots/` consumers (Premium API), future snapshot features (#11, #15).
- **Date**: 2026-10-27
- **Status**: active

### AD-002
- **Decision**: The manifest audit hash `manifest_sha256` is SHA-256 over canonical JSON (`sort_keys=True`, `separators=(",", ":")`, UTF-8) of the NORMALIZED manifest projection `SnapshotManifest.from_dict(raw).as_dict()` minus `manifest_sha256` itself and the lifecycle envelope (`state`, `state_changed_at`, `retire_reason`); `created_at` is included, `build` requires an explicit `--created-at`, and the canonicalization lives in one helper (`manifest_canonical_sha256`).
- **Reason**: The hash must be recomputable wherever the manifest is rehydrated. The Premium path (`api/app/services/premium_executor.py:162` → `SnapshotManifest.from_dict(manifest_data, base_dir=…)` → `validate_manifest`, fed by `api/app/services/premium_recommendation.py:87-88` which persists `as_dict()` with `manifest_path` injected) only sees the dataclass, so a hash over raw JSON bytes would reject every rehydrated run; hashing the `as_dict()` projection recomputes identically for file loads, dataclass instances, and persisted provenance (review finding F1).
- **Trade-off**: Tampering with `state`, or with keys outside the `as_dict()` projection (unknown extras in legacy files), is not hash-detected - guarded by the CLI transition table, selection semantics, and documentation; operators must supply `--created-at` explicitly. "Normalized" means the schema projection only: `as_dict()` does not sort `supported_classes` - its order determinism comes from `build` deriving it from `ASSET_CLASSES` (AD-004).
- **Scope**: `py/snapshot_manifest.py` hash/validation surface, `py/snapshot_registry.py` CLI, audit tooling over manifests.
- **Date**: 2026-10-27 (revised spec review round 2; round 3 fixes the `premium_executor.py:162` citation and the `as_dict()`/`supported_classes` wording)
- **Status**: active

### AD-003
- **Decision**: Registry selection semantics by lifecycle state: `draft`/`retired` are never selectable; `validated` is selectable only with an explicit `include_validated=True`; manifests without `state` (legacy) remain selectable and produce identical results to the pre-change implementation.
- **Reason**: Requirement of issue #16 (P0-09) plus strict backward compatibility for existing manifests consumed through `latest_compatible_manifest`.
- **Trade-off**: A legacy manifest is treated as `draft` for CLI transitions but as selectable for API runs until an operator runs `validate` - a deliberate split to avoid breaking live selection.
- **Scope**: `SnapshotRegistry.latest_compatible` / `latest_compatible_manifest`, `api/app/services/premium_recommendation.py`.
- **Date**: 2026-10-27
- **Status**: active

### AD-004
- **Decision**: The manifest emit/canonical on-disk form is fixed: `build` writes exactly the `SnapshotManifest.as_dict()` projection (single `allocation` source with `metadata.files`, explicit `stock`/`fii` selector sources and their `*_source` aliases); every manifest write uses `json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)` + trailing `\n`; `supported_classes` is derived from `ASSET_CLASSES` (`py/allocation_config.py:14-20`) in fixed order; each source's `provider` is a fixed literal per source role and `metadata` is exactly `{"files": {...}}` (`allocation`) or `{}` (selector sources) - fetch `metadata.json` content is never copied into the manifest; source paths are registry-root-relative POSIX paths and every `build` input must resolve INSIDE the registry root (escape after `Path.resolve()` ⇒ exit 2; `data/<snapshot_name>/` is the conventional layout); byte-reproducibility is proven by building the same `manifest_id` in two distinct registries and comparing bytes.
- **Reason**: `SnapshotSource.resolved_path` resolves relative to the manifest's directory, so a path recorded outside the registry produces a manifest that fails its own `validate` (the necessary invariant is "inside the registry root", not "under `data/`" - review nit); the consumer contract (`py/allocation_data.py:238-250`, fixture `tests/test_snapshot_manifest.py:54-68`) expects the `allocation` + `metadata.files` shape; fetch `metadata.json` carries absolute paths (`portfolio_path`, `portfolio_artifacts` - `py/fetch_allocation_snapshot.py:538,584-590`) that would break AC3 byte-reproducibility across registries (review N5); `as_dict()` does not sort `supported_classes`, so order determinism lives here (review nit); a repeated `build` in one registry is exit 3 by design (immutability), so the reproducibility proof needs two registries (review findings F4, F5).
- **Trade-off**: Operators must place inputs inside the registry root (conventionally `data/<snapshot_name>/`) before building; envelope rewrites preserve unknown keys but those are not covered by `manifest_sha256`; fetch-side provenance beyond `metadata.files` stays in the snapshot directory's own `metadata.json` and is not part of the manifest.
- **Scope**: `py/snapshot_registry.py` (`build`, `write_manifest_atomic`), `docs/snapshot_registry.md`, fixture strategy for tests.
- **Date**: 2026-10-27 (revised spec review round 3)
- **Status**: active

### AD-005
- **Decision**: Validation exit-code and audit-path contract: every required-source verification failure of a manifest under audit - a required series absent from the manifest OR its file/derived artifact missing or unreadable on disk - exits **4** in `validate` AND `validate --check` (review ruling N1); exit **3** is reserved for missing CLI targets (registry dir, `<registry>/<manifest-id>.json`), refused overwrite, and `build` source inputs (AC7). `validate --check` runs the FULL `validate_manifest` verification set minus the lifecycle-state gate (hashes matching is necessary, not sufficient) and combines with `--json` (0/4 semantics, zero writes), which is the only way to print the class-to-series map in `published`/`retired` states.
- **Reason**: 3 = "the operator pointed at something that does not exist"; 4 = "the artifact under audit does not hold up" - after `publish` the data files are part of the audited artifact, so losing them is tampering (coherent with AC16 = 0/4 and issue IT-2, whose fixture unlinks `caio_fiis.csv` from disk); a `--check` that verifies less than `validate` could report a green audit on an invalid manifest (review N2); P3 promises review of a *published* snapshot's inputs, which plain `validate` refuses with exit 5 (review N3).
- **Trade-off**: The 3/4 split is no longer "missing file vs contract divergence" at the file level - a missing audited source file is exit 4, which operators must learn from the docs; `--check` becomes as strict as `validate` for reads.
- **Scope**: `py/snapshot_registry.py` (`validate`, `validate --check`, `validate --json`), `docs/snapshot_registry.md` exit table.
- **Date**: 2026-10-27
- **Status**: active

### AD-006
- **Decision**: Provenance honesty in emitted and reported data: `class_series_map` reports each class's `file_sha256` (the digest of the series FILE itself) and a registry-root-relative POSIX `path`, adding `derived_from: "allocation"` for derived series - it must never expose the `allocation` directory digest (`source_for` returns `sha256=allocation.sha256` + absolute `path` for derived series, `py/snapshot_manifest.py:312-318`) as a file hash; `build` fixes `provider` as a literal constant per source role and `metadata` to `{"files": {...}}`/`{}`, never copying fetch `metadata.json`.
- **Reason**: An audit artifact must say what it means: labeling a directory digest as a series' `sha256` is verifiable but misleading (review N4), and absolute paths or fetch-side metadata make two registries produce different manifest bytes, breaking AC3 reproducibility (review N5).
- **Trade-off**: The map's `sha256` no longer coincides with the `allocation` source hash the manifest stores; auditors cross-check derived files via the allocation directory hash recorded in `sources.allocation` (kept authoritative in `validate_manifest`, `py/snapshot_manifest.py:425-434`).
- **Scope**: `py/snapshot_registry.py` (`class_series_map`, `build`), `docs/snapshot_registry.md`, P3 contract.
- **Date**: 2026-10-27
- **Status**: active

### AD-007
- **Decision**: P0-02 marks affected artifacts invalid instead of regenerating anything under `outputs/`: `baselines/v0-corrected/status.json` maps every affected artifact and field to a verdict (`invalid_fitness`, `invalid_beyond_fitness`, `score_provenance_inconsistent`, `not_verifiable_at_head`, `valid`) plus evidence fields and the defect reference; production `outputs/carteira_caio_consensus.json` stays byte-identical while the note states that the production consensus was produced by the corrected optimizer over a 150-run set the archive does not contain (its FREQUENCY values are exact counts over 150: 38, 34, 33, 25, 22, 21, 21, 20, 20, 17, and sibling `outputs/metrics_stability_caio.csv` carries `generations_run`/`converged_early`) while the archived consensus came from the defective capture - the membership difference is a provenance difference, not evidence that the live recommendation is defective (round-2 correction: the earlier claim that "both came from defective chromosomes" was false - the production file is NOT from a defective capture); a test enforces that every strict `fitness` locator in `baselines/v0-legacy/reference/**` and the pinned `outputs/` tree appears exactly once in `status.json`.
- **Reason**: The issue allows "regenerar OU marcar"; marking is lower risk, reversible, and AC2 demands it. Any future touch under `outputs/` requires first proving no consumer globs it plus human sign-off.
- **Trade-off**: `outputs/` keeps hosting invalid numbers; correctness depends on readers honoring `status.json`; digests are pinned to git blob bytes at the pinned base commit (self-referential pinning avoided).
- **Scope**: `outputs/`, `baselines/v0-corrected/status.json`, the marking tests, the methodological note.
- **Date**: 2026-10-27 (revised spec review round 2: consensus provenance wording corrected per B2)
- **Status**: active

### AD-008
- **Decision**: `baselines/v0-legacy/**` covered subtrees stay byte-untouched (17 files / 632,981 B pinned by `tests/test_v0_legacy_manifest.py`); all 52 P0-01 tests stay green; `verify_manifest.py` stays VERIFIED; the frozen defective `code/ga.py` is never repaired; the only v0-legacy edit is the manifest-excluded `README.md` gaining the INVALID banner with the defect reference.
- **Reason**: The frozen bundle is the measurement instrument the P0-02 replay compares against; repairing or adding files under it would destroy the measurement and break the pinned manifest test (EXCLUDED_PATHS pins the exclusion list, not README bytes).
- **Trade-off**: Readers of `baselines/v0-legacy/` alone still see defective numbers; the banner and `status.json` cross-reference are the only in-place warnings available without breaking byte-pins.
- **Scope**: `baselines/v0-legacy/`, `tests/test_v0_legacy_manifest.py` invariants, P0-02 marking design.
- **Date**: 2026-10-27
- **Status**: active

### AD-009
- **Decision**: `v0-corrected` is self-contained and byte-proven, not prose-asserted: copies verified by sha256 against `v0-legacy/baseline.json`; `code/ga.py` = v0-legacy bytes + exactly one hunk (relocation of `population = new_pop` after the best-record block) proven by applying an exact (old_bytes, new_bytes) canonical transform in memory and asserting byte equality - the same pair applied to `git show <base>:py/ga.py` must yield the corrected live `py/ga.py` byte-for-byte, so no other semantic change (strict `>`, RNG call order, roulette, crossover, mutation, seeding) can sneak in (round 2: the pair replaces the earlier patch-with-placeholder, which was not applyable); no `caio_last` config; early stopping stays disabled; 30 runs; seed rule `seed_base + run_id`; same consensus rule; same profiles/weights proven by AST extraction (`PERFIL_CONFIG`, `GROUPS`, `PROFILE_WEIGHTS`, `NCROSS_RATE`, `MUT_RATE`) and deep-equal `protocol.json` except `baseline_id` + `derived_from`.
- **Reason**: "Same configs and protocol" must be checkable in CI; prose claims drift, bytes and ASTs do not.
- **Trade-off**: The bundle duplicates data and code (~1 MB+); every protocol change upstream requires building a new bundle instead of editing in place.
- **Scope**: `baselines/v0-corrected/**`, `tests/test_v0_corrected_bundle.py`, equality gates.
- **Date**: 2026-10-27 (revised spec review round 2: exact byte pair + live-tree byte-equality proof)
- **Status**: active

### AD-010
- **Decision**: `v0-corrected` anchors to an explicitly declared environment (durable pin inside the bundle: interpreter `python -VV`, numpy/pandas versions, full `pip freeze`, install command, plus the caveat that the proofs are reproducible only while those wheels remain obtainable); the methodological note makes no reproducibility claim beyond the committed proofs and states that archived v0-legacy values are not reachable in the declared environment for caio; the 20-seed legacy-vs-corrected comparison runs both drivers in the SAME declared environment, which is what makes it internally valid. Root-cause wording (round-2 correction per N9): the archived caio values were produced by a scoring configuration/version the archive does not record - the archived data and the RNG alone do not explain them (P0-01: declared weights reproduce the control SCOREs 15/15, 10/10, 10/10 within 1e-9 but caio 0/7, missing by up to 0.42, on the archived dataset itself - plan.md:20-31, verification.md:372; round-2 re-measurement: caio score_median recomputes 30/30 for control profiles and mismatches 30/30 for caio). Supporting replay evidence: conservador seed 42 replays bit-identical to archived run 0 (4.480605754022773 / hhi 0.16 / same tickers) while caio seed 42 replays deterministically to 6.244563/hhi 0.16 vs archived 4.604746/hhi 0.26. The frozen caio file's 202 rows vs 139 at HEAD is a live-dataset fact, NOT the explanation of the archive's own caio values (previous wording conflated the two).
- **Reason**: The archive does not reproduce for caio here; claiming archive reproducibility would be unprovable. The same-environment comparison keeps the legacy-vs-corrected deltas attributable to the hunk alone.
- **Trade-off**: The pinned environment may itself become unavailable (venv `/tmp/opencode/tcc-venv` is ephemeral); archived numbers stay unreproducible and the caio scoring provenance gap remains officially unattributed in the note beyond the measured evidence.
- **Scope**: `baselines/v0-corrected/environment.txt` + `requirements.txt`, methodological note, `compare_replay.py`, expectations table.
- **Date**: 2026-10-27 (revised spec review round 2: root-cause wording corrected per N9; environment pin content per N14)
- **Status**: active

### AD-011
- **Decision**: Fix only the legacy live path `py/ga.py` (record-block relocation, no other semantic change); `py/core/optimizer.py` untouched; no early stopping, no derived seed, no per-company dedupe, no performance work, no engine rewrite.
- **Reason**: Issue scope; `py/core/optimizer.py:257-263,301-305` already captures the chromosome at evaluation time; the defect is only the legacy record block placement.
- **Trade-off**: Other legacy oddities (roulette fragility, `-inf` handling) survive; they are protocol, not defect, and changing them would break the "same protocol" claim.
- **Scope**: `py/ga.py`, `baselines/v0-corrected/code/ga.py`.
- **Date**: 2026-10-27
- **Status**: active

### AD-012
- **Decision**: The property test is a deterministic enumeration (generations x pop_size x seeds over small synthetic frames incl. NaN SCORE and duplicate SETORs), not hypothesis-based; it instruments the REAL `run_ga` (monkeypatched `ga.fitness` recorder, generations chunked by `pop_size`) and loads both `ga` modules via `importlib.util.spec_from_file_location` under unique names - it never re-implements the loop - and it never feeds infeasible masks (`mask.sum() != n`) through the frozen module's selection path, because any `-inf` score makes the frozen probs computation all-NaN and `np.random.choice` raises; the slow legacy-vs-corrected replay (20 seeds x 4 frozen profiles = 80 runs per driver) is opt-in via `TCC_V0_CORRECTED_REPLAY=1` and its JSON proof is committed under `.specs/features/p0-02-v0-corrected/proofs/`.
- **Reason**: No hypothesis (or pytest) exists in any available interpreter; the full-protocol replay must not gate the fast suite. Round-2 budget correction (N8): measured single-run cost is 14.5 s conservador / 24.7 s moderado / 32.4 s caio / 37.7 s arrojado (~109 s per seed across 4 profiles), so 160 driver executions are ~75 min measured wall-clock (the earlier "~13 min" was ~5x understated); the replay is accepted as a 1-2 h opt-in run in the completion gate.
- **Trade-off**: Enumeration samples a bounded space instead of generating random cases; the opt-in replay can be forgotten, so the committed proof is required evidence at completion.
- **Scope**: `tests/test_v0_corrected_property.py`, `baselines/v0-corrected/tools/compare_replay.py`, proofs dir.
- **Date**: 2026-10-27 (revised spec review round 2: budget corrected per N8; instrumentation contract per N12; frozen-module feasibility constraint per O6)
- **Status**: active

### AD-013
- **Decision**: Every tool or test importing bundle code sets `sys.dont_write_bytecode = True` before the import and runs with cwd inside a scratch root; bundle modules with import-time side effects (`ga_multiple_runs.py:24-25` runs `OUT_DIR.mkdir` at IMPORT time and writes at :338/:362/:382) are imported only under that scratch root; nothing may write under `code/`, `data/` or `reference/` of either bundle.
- **Reason**: Importing frozen code writes `__pycache__/*.pyc` under `baselines/`, which turns the frozen bundle into "DRIFT - unexpected" in its manifest gate (`replay.py:38` already sets the flag - round-2 citation correction: the earlier "replay.py:45" was line drift); the import-time `mkdir`/writes reach outside `__pycache__`, so bytecode suppression alone is not enough (N7).
- **Trade-off**: No bytecode caching for bundle imports (slightly slower test imports); discipline is enforced by a gate that re-runs both manifest verifiers after the suite and by the scratch-root requirement on generators.
- **Scope**: All tests/tools that import `baselines/*/code/*`; the v0-corrected generators.
- **Date**: 2026-10-27 (revised spec review round 2: citation fix per N15, import-time side effects per N7)
- **Status**: active

### AD-014
- **Decision**: New tests/code respect `tests/test_v0_legacy_isolation.py:24-34`: no `"baselines/"` string or bundle-module imports in `py/`, `api/` or root `*.py`; no `__init__.py`/`setup.py`/`pyproject.toml` anywhere under `baselines/`; new bundle tooling lives under `baselines/v0-corrected/tools/` and tests under `tests/`.
- **Reason**: P0-01 gate forbids production code reaching into bundles and forbids packaging markers under `baselines/`; violating it breaks the existing suite.
- **Trade-off**: Bundle code cannot be imported from production paths even for shared helpers, keeping duplication between `py/` and bundles.
- **Scope**: `py/`, `api/`, `tests/`, `baselines/v0-corrected/tools/`.
- **Date**: 2026-10-27
- **Status**: active

### AD-015
- **Decision**: The invalidation verdicts are DATA, not prose: every `status.json` entry records the evidence fields `producer` (`legacy_capture`/`corrected_capture`/`unrecorded`; CSVs derive it mechanically from the `generations_run` column written by `py/pipelines/multi_run.py:45,123-124` through `py/core/optimizer.py:330-331` when present, `py/ga_multiple_runs.py:362` when absent; JSONs record it per file with mechanical evidence), `row_reconstruction`, `data_snapshot_match`, `score_recompute`, `reported`, `recomputed`, and a verdict computed by the Verdict Attribution Rule decision table (`invalid_fitness`, `invalid_beyond_fitness`, `score_provenance_inconsistent`, `not_verifiable_at_head`, `valid`); `score_recompute` is defined over EVERY recorded score field of the row (`score_mean`, `score_median`, `score_std` and any other `score_*` leaf) at 1e-9, so one metric matching cannot buy a `match`; `data_snapshot_match == false` resolves to `not_verifiable_at_head` before any producer rule; caio's fitness fields get `score_provenance_inconsistent` (fitness not verifiable), not `invalid_beyond_fitness` and not `invalid_fitness`; the defect reference is recorded only where the rule attributes the defect, every other non-`valid` verdict carrying its own `reason`; a test recomputes every entry's verdict from its own recorded fields and fails on disagreement, and a corrected-capture row fails the gate outright only when every recorded score field reproduces (i.e. `score_recompute == match`) and `reported != recomputed` - a corrected-capture row whose score fields drifted is `score_provenance_inconsistent`, because the table resolves `score_recompute == mismatch` first.
- **Reason**: Round-2 blocking finding B1: a verdict vocabulary with no discriminator is unfalsifiable - the caio score mismatch cannot be attributed to the capture defect (the snapshot itself does not explain it) and the `outputs/` fixed-path files carry corrected-producer columns (`generations_run`/`converged_early`), so a blanket "affected by the P0-02 defect" mapping is unsupported; attribution must be recomputable from evidence.
- **Trade-off**: The verdict model is stricter than the issue's minimum - some values land in `not_verifiable_at_head`/`score_provenance_inconsistent` where a bolder spec would say "invalid"; marking work grows (evidence per entry) but every verdict becomes checkable.
- **Scope**: `baselines/v0-corrected/status.json`, `tests/test_v0_corrected_status.py`, `verify_artifacts.py`, the methodological note's wording.
- **Date**: 2026-10-27 (new at spec review round 2)
- **Status**: active

## Handoff

- **Feature**: p0-02-v0-corrected (Issue #7 / P0-02) - `.specs/features/p0-02-v0-corrected/`
- **Phase / Task**: SPECIFY complete (29 requirements, `validate_spec.py` exit 0). Independent SPEC review: round 1 `CHANGES_REQUIRED blocking=6 nonblocking=15`, round 2 `CHANGES_REQUIRED blocking=2 nonblocking=7`; the two residual blocking findings (F1 reference↔driver closure, F2 verdict-field/`score_recompute` scope) were closed by the orchestrator in a third pass which added VC-29 and the bounded divergence table. DESIGN/TASKS not started.
- **Completed**: `spec.md` (P1 corrected bundle + VC-01..VC-11, VC-25/VC-26/VC-29; P2 invalidation with the Verdict Attribution Rule + VC-12..VC-16/VC-24/VC-28; P3 note + 80-run replay + VC-17..VC-23/VC-27), falsifiable E1-E4 expectation table over 80 runs (seeds 42-61), pinned base commit `181453f`, canonical (old_bytes,new_bytes) transform, and AD-007..AD-015 in this file.
- **In-progress**: none
- **Next step**: human approves the SPEC (TLC checkpoint), then DESIGN (bundle layout + tool contracts + test design) and TASKS, then EXECUTE. Implementation order that keeps the gates honest: `py/ga.py` hunk + byte-equality test, then bundle code/data copy + manifest, then generators + `reference/`, then `verify_artifacts.py` + property enumeration, then `status.json` + marking tests, then the note, then the opt-in 80-run replay (~75 min measured).
- **Blockers**: SPEC awaits human approval. Environment constraint unchanged: numpy/pandas exist only in the ephemeral `/tmp/opencode/tcc-venv` (no hypothesis, no pytest); stdlib `unittest` is the suite runner.
- **Previous handoff**: snapshot-registry-lifecycle (Issue #16 / P0-09) reached SPEC complete with review round 3 closed and IMPLEMENT pending on branch `t3/issue-16` (`cf4e8ad`, `b736875`); its artifacts remain at `.specs/features/snapshot-registry-lifecycle/`.
- **Uncommitted files**: the SPEC-stage commit carries `spec.md` + this file.
- **Branch**: t3/issue-7-auto-route
