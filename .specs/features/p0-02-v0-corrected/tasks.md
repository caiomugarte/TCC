# v0-corrected (P0-02) Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/p0-02-v0-corrected/spec.md` (29 requirements, VC-01..VC-29)
**Design**: `.specs/features/p0-02-v0-corrected/design.md` (resolutions D-a..D-g)
**Status**: Draft
**Pinned base commit**: `181453f986cce7941addf9092e38c6e005e18844`
**Binding decisions**: AD-007..AD-015 in `.specs/STATE.md`

---

## Frozen Constraints (every task must uphold; a violation is a defect in the task)

- `baselines/v0-legacy/` covered bytes stay untouched (17 files / 632,981 B); its manifest gate stays `VERIFIED`; the 52 P0-01 tests stay green; the only v0-legacy edit is the manifest-excluded `README.md` banner.
- `tests/test_v0_legacy_isolation.py` rules: no `"baselines/"` string constant (docstrings included) and no bundle import in `py/`, `api/` or root `*.py`; no `__init__.py`/`setup.py`/`pyproject.toml` anywhere under `baselines/`.
- Every bundle importer sets `sys.dont_write_bytecode = True` before the import and runs with cwd inside a scratch root (AD-013; `ga_multiple_runs.py:24-25` writes at import time).
- stdlib `unittest` is the runner; numpy/pandas only in `/tmp/opencode/tcc-venv`; no hypothesis, no pytest (AD-012).
- `outputs/` is never regenerated or modified; its byte-identity at the pinned base is enforced by test (AD-007).
- Zero committed fixture files: every negative fixture is a runtime mutation inside `tempfile.TemporaryDirectory()` (D-f).
- `environment.txt` + `requirements.txt` are durable; the `/tmp` venv and wheel availability are explicitly NOT reproducible later (D-g).

## Frozen Human Decisions (taken by the orchestrator; do not re-litigate)

1. **The five design additions beyond spec.md's Deliverables sizing are ACCEPTED** — `tools/generate_reference.py`, `tools/_verdict.py`, `tools/_frozen_fitness.py`, `proofs/generation_proof.json`, `tests/test_v0_zz_manifest_gates.py`. They are mapped in the "Spec Deltas" table below so the divergence from spec.md's Deliverables section is auditable.
2. **VC-25's escalation path is accepted**: if any control-profile row's corrected `fitness` diverges bit-wise from the archive, the run reports a **blocking finding for a human** — it is never folded into the divergence table. Therefore reference generation is placed EARLY (Phase 2, before the marking layer of Phase 4) so this risk is discovered in the first execution batch, not at the end.

## Spec Deltas (auditable divergence from spec.md Deliverables)

| # | spec.md Deliverables says | This task list adds / changes | Why | Status |
| --- | --- | --- | --- | --- |
| 1 | `tools/`: `build_manifest.py`, `verify_manifest.py`, `verify_artifacts.py`, `compare_replay.py`, `_corrected_driver.py` | adds `tools/generate_reference.py` | D-c: two-pass scratch generation + determinism compare + per-row driver tie | ACCEPTED (human decision 1) |
| 2 | — | adds `tools/_verdict.py` | D-d: the single verdict function shared by generator and marking test | ACCEPTED (human decision 1) |
| 3 | — | adds `tools/_frozen_fitness.py` | D-e: frozen recomputation importing bundle code only | ACCEPTED (human decision 1) |
| 4 | `proofs/`: `replay_20seeds.json`, `reference_divergences.json` | adds `proofs/generation_proof.json` | D-b/D-c: 120-row driver tie + double-pass digests | ACCEPTED (human decision 1) |
| 5 | `tests/`: 5 modules | adds `tests/test_v0_zz_manifest_gates.py` | VC-23/VC-28: both manifest gates after the full suite | ACCEPTED (human decision 1) |
| 6 | `baseline.json` implied "with the bundle skeleton" | `baseline.json` is built at T8, not at T5 | It covers `code/`, `data/` AND `reference/`; `reference/` exists only after generation, and `build_manifest.py` refuses overwrite (D-c step 5 makes the manifest the post-step of generation) | TASKS ordering |
| 7 | design step 4 places `_verdict.py` after generation (step 3) | `tools/_verdict.py` is built at T6, before generation | `generate_reference.py`'s row tie calls `_verdict.tie_row` (D-f); this resolves the design's own step-3/step-4 circularity | TASKS ordering |

## Baseline Hygiene (precondition, verified before this plan was written)

The worktree at `5f7a7e6` was found with `baselines/v0-legacy/code/__pycache__/` pollution (two `.pyc` files from a prior measurement run that imported bundle code without bytecode suppression) which reddened 21 of the 52 P0-01 tests and made `verify_manifest.py` report `DRIFT`. The cache was removed; `python3 -m unittest discover -s tests -p 'test_v0_legacy_*.py'` now reports `Ran 52 tests ... OK` and `verify_manifest.py` reports `VERIFIED 17`. Any importer that skips `sys.dont_write_bytecode` recreates this pollution and reopens the defect - this is exactly what T19's gate defends.

---

## Test Coverage Matrix

> Generated from codebase sampling (`tests/test_v0_legacy_*.py` = 52 tests as the floor and style model; `tests/test_*.py` layout), project guidelines (spec.md Test List, AD-012/AD-013/AD-014 in `.specs/STATE.md`, `tests/test_v0_legacy_isolation.py`), and the spec ACs. No `AGENTS.md`, `CONTRIBUTING.md`, `Makefile`, CI workflow or linter config exists in this repository (P0-01 checks.md: "Build: none - this repository configures no build, no type check and no lint"); the strong defaults apply on top of the spec's own Test List. Confirm before Execute.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command | Interpreter |
| --- | --- | --- | --- | --- | --- |
| Live legacy path `py/ga.py` | unit | Byte-equality against the canonical transform of `git show 181453f:py/ga.py` (VC-03): 0-byte delta, `old_bytes` exactly once, mutation negative | `tests/test_v0_corrected_bundle.py` | Quick | system python3 (stdlib) |
| Frozen bundle copies (`code/`, `data/processed/`) | unit | Byte/sha256 identity vs `v0-legacy` (VC-02, VC-04), 1:1 with the frozen manifest digests | `tests/test_v0_corrected_bundle.py` | Quick | system python3 (stdlib hashing) |
| Protocol `protocol.json` | unit | Deep-equal minus `baseline_id`/`derived_from`, AST constants equal, no `caio_last` (VC-05) + negatives | `tests/test_v0_corrected_bundle.py` | Quick | system python3 (stdlib ast/json) |
| Manifest tooling (`build_manifest.py`, `verify_manifest.py`) | unit | Closed subtrees, exclusions with reasons, `missing`/`modified`/`unexpected` findings, exit codes (VC-04, VC-28 machinery) + negatives | `tests/test_v0_corrected_bundle.py` | Quick | system python3 (stdlib, tempfile) |
| Verdict module (`tools/_verdict.py`) | unit | Whole decision table in order incl. the `MispairingError` gate row; bitwise-float and row-tie semantics (VC-12, VC-25, VC-29) | `tests/test_v0_corrected_status.py` | Quick | system python3 (stdlib) |
| Generation pipeline (`generate_reference.py`, `_corrected_driver.py`) | unit + build integration | Completeness (30 run_ids x 4 profiles), determinism compare, proof schema, forged-row tie; end-to-end double pass at T8 | `tests/test_v0_corrected_bundle.py` + generation command | Quick + Build | stdlib tests; tcc-venv for the run |
| Frozen recomputation (`_frozen_fitness.py`) | unit | Exact frozen semantics `sum(SCORE) - lambda*hhi*n` with pandas skipna (`n*score_mean` structurally impossible), hhi/score recompute, RowEvidence (VC-11) | `tests/test_v0_corrected_artifacts.py` | Full | tcc-venv (numpy/pandas) |
| Artifact verifier (`verify_artifacts.py`) | integration | 100% rows reported==recomputed, denominators never shrink, evidence re-derivation (VC-01, VC-09, VC-10, VC-24, VC-26) + runtime-mutation negatives | `tests/test_v0_corrected_artifacts.py` | Full | tcc-venv |
| Property enumeration (test-only) | unit | Instrumented REAL `run_ga`; NaN SCORE, tied SCORE, duplicate SETOR, even pop_size parity; legacy >= 1 violated generation; feasible masks only (VC-06..VC-08) | `tests/test_v0_corrected_property.py` | Full | tcc-venv (SKIP with reason elsewhere) |
| Marking layer (`status.json`) | unit | Coverage/uniqueness (5 + 15 strict `fitness` locators), git-blob digests + `outputs/` byte-identity, verdict recompute from evidence (VC-12..VC-15, VC-24) | `tests/test_v0_corrected_status.py` | Quick | system python3 (stdlib) |
| Documentation & env pin (banner, note, `environment.txt`) | unit | Verbatim statement lists, reproducibility limits, env content (VC-16..VC-19) | `tests/test_v0_corrected_status.py` | Quick | system python3 (stdlib) |
| Replay checker (`compare_replay.py` + proof schema) | unit | E1-E4 fold from `runs[]`, denominator/seed schema, env gate exit 2, injected-violation fixture (VC-20..VC-22, VC-27) | `tests/test_v0_corrected_replay.py` | Quick | system python3 (stdlib) |
| Replay execution (opt-in) | e2e | 80 runs both drivers in the declared environment; payload == committed proof (VC-20; live VC-29 tie run_id 0-19) | `tools/compare_replay.py` | Replay | tcc-venv (~75 min) |
| Suite trailer | e2e (subprocess gates) | Both manifest gates exit 0 with zero unexpected files after the full suite (VC-23, VC-28) | `tests/test_v0_zz_manifest_gates.py` | Quick | system python3 (subprocess) |
| Generated evidence (`reference/`, `proofs/*.json`, `baseline.json`) | none (gate-only) | Determinism + driver tie proven by the generator gates; pinned and re-checked by the test rows above | - | Build | tcc-venv (generation) |

Co-location rule: every task that creates a layer above writes its own tests in the same task; `Tests: none` is not used anywhere in this plan.

## Gate Check Commands

> Generated from the codebase and spec.md's Test List - confirm before Execute.

| Gate Level | When to Use | Command | Interpreter |
| --- | --- | --- | --- |
| Quick | after every task | `python3 -m unittest discover -s tests -p 'test_v0_*.py'` | system python3 3.14.4 (stdlib only; numpy/pandas tests SKIP with explicit reason; includes the frozen 52) |
| Quick target | while implementing one task | `python3 -m unittest discover -s tests -p '<task test module>.py'` | system python3 |
| Full | after tasks with execution/recompute tests (T7-T12, T18) | `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'` | tcc-venv (numpy 2.5.4 / pandas 3.1.0) |
| Build | after each phase and at completion (v0-corrected legs exist from T8/T11; before that run the v0-legacy gate alone) | `python3 baselines/v0-legacy/tools/verify_manifest.py && python3 baselines/v0-corrected/tools/verify_manifest.py && /tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/verify_artifacts.py` | system python3 + tcc-venv |
| Replay | exactly once, at T18 (opt-in, ~75 min measured) | `TCC_V0_CORRECTED_REPLAY=1 /tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/compare_replay.py` | tcc-venv |

There is no lint/build step to run (the repository configures none). The interpreter split is strict: stdlib gates run on system `python3`; every execution/recomputation test runs under `/tmp/opencode/tcc-venv/bin/python` and SKIPs with an explicit reason when that interpreter is gone - accepted, not hidden (D-g).

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next begins, and tasks within a phase execute in order. Each phase is one batch (A-E). At Execute, 19 tasks pack into 5 whole-phase batches, so the sub-agent offer applies (one worker per batch, batches sequential; the ~75-min replay stays a single opt-in task inside batch E).

### Phase 1 (batch A): Live fix, frozen skeleton, manifest tooling

The one semantic hunk in the live tree, the byte-proven bundle skeleton, and the manifest contract. Everything here is stdlib and fast.

```
T1 -> T2 -> T3 -> T4
T3 -> T5
```

### Phase 2 (batch B): Verdict core, reference generation, VC-25 early verdict

`reference/` is generated EARLY, before the marking layer (human decision 2): the control-profile bit-identity risk (90 rows of new evidence, measured only on spot-checks) is discovered in the FIRST execution batch. A control-row divergence is a blocking finding for a human, never a table entry.

```
T5 -> T6 -> T7 -> T8 -> T9
T5 -> T8
```

### Phase 3 (batch C): Verifier, property enumeration, negative fixtures

The recomputation engine, the artifact verifier with its `--emit-status` generator, and the deterministic property enumeration. All negative fixtures are runtime mutations in tempdirs (D-f).

```
T3 -> T10 -> T11
T6 -> T11
T8 -> T11
T2 -> T12
```

### Phase 4 (batch D): status.json, marking tests, banner, method note

The verdict map and its checker, then the three documentation artifacts with their verbatim substring checks.

```
T11 -> T13 -> T14 -> T16
T13 -> T15
```

### Phase 5 (batch E): Replay comparison, opt-in proof, completion gates

The checker first (so its failure branch is tested), then the one opt-in 80-run replay (~75 min) with its committed proof, then the suite-trailer gates and the completion run.

```
T7 -> T17 -> T18 -> T19
T6 -> T17
T8 -> T18
T8 -> T19
```

---

## Task Breakdown

### Phase 1 tasks - live fix and frozen skeleton

### T1: Relocate the record block in the live `py/ga.py`

**What**: Apply the canonical (old_bytes, new_bytes) transform from spec.md's Deliverables to `py/ga.py` - `population = new_pop` moves after the best-recording block, nothing else changes - and prove the result is byte-equal to the transform of `git show 181453f:py/ga.py`.
**Where**: `py/ga.py`
**Depends on**: None
**Files touched**: `py/ga.py`, `tests/test_v0_corrected_bundle.py`
**Reuses**: the exact byte pair from spec.md "Canonical transform"; `tests/test_v0_legacy_*.py` style (stdlib `unittest`, descriptive names)
**Closes**: VC-03 (AC3)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `py/ga.py` equals `old->new` applied to `git show 181453f:py/ga.py` byte-for-byte (0-byte delta), so strict `>`, RNG call order, roulette, crossover, mutation and seeding stay untouched
- [x] Tests (3) green - names below; a one-byte mutation of the worktree file fails the equality leg
- [x] Quick gate green: `python3 -m unittest discover -s tests -p 'test_v0_*.py'` (52 P0-01 + 3 new)

**Tests**: unit (stdlib; derived from VC-03's acceptance criteria, `tests/test_v0_corrected_bundle.py`):

- `test_canonical_old_bytes_occur_exactly_once_in_the_pinned_live_file`
- `test_transformed_pinned_blob_equals_the_worktree_ga_py_byte_for_byte`
- `test_one_byte_mutation_fails_the_byte_equality`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `fix(ga): move population = new_pop after the best-record block`

---

### T2: Freeze the corrected code copy in the bundle

**What**: Create `baselines/v0-corrected/code/ga.py` as the canonical transform applied to `baselines/v0-legacy/code/ga.py` bytes, and prove byte equality (the one deliberate difference of the bundle).
**Where**: `baselines/v0-corrected/code/ga.py`
**Depends on**: T1
**Files touched**: `baselines/v0-corrected/code/ga.py`, `tests/test_v0_corrected_bundle.py`
**Reuses**: the canonical byte pair; the T1 test legs' transform helper
**Closes**: VC-02 (AC3)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `baselines/v0-corrected/code/ga.py` equals `old->new` applied to `baselines/v0-legacy/code/ga.py` byte-for-byte (0-byte delta); `old_bytes` occurs exactly once in the legacy file
- [x] Tests (3) green; quick gate green (52 P0-01 + 6 new)

**Tests**: unit (derived from VC-02, `tests/test_v0_corrected_bundle.py`):

- `test_corrected_code_ga_equals_the_transform_of_the_legacy_bundle_file`
- `test_old_bytes_occur_exactly_once_in_the_legacy_bundle_file`
- `test_transform_replacement_never_touches_the_rng_or_tie_semantics` (byte equality over the untouched regions)

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the corrected frozen code copy`

---

### T3: Copy the frozen code and data byte-identically

**What**: Copy `code/profiles.py`, `code/ga_multiple_runs.py` and the five `data/processed/` files from `v0-legacy` into `v0-corrected` with zero byte delta, verified by sha256 against `baselines/v0-legacy/baseline.json`.
**Where**: `baselines/v0-corrected/code/profiles.py`
**Depends on**: T2
**Files touched**: `baselines/v0-corrected/code/profiles.py`, `baselines/v0-corrected/code/ga_multiple_runs.py`, `baselines/v0-corrected/data/processed/` (5 files), `tests/test_v0_corrected_bundle.py`
**Reuses**: `baselines/v0-legacy/baseline.json` digests as the oracle
**Closes**: VC-04 (AC3; manifest pin at T5/T8)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] All seven copies are byte-identical to their `v0-legacy` counterparts and their sha256 match the `v0-legacy/baseline.json` entries
- [x] Tests (3) green; quick gate green (52 P0-01 + 9 new)
- [x] Nothing is written under `baselines/v0-legacy/` (frozen constraint)

**Tests**: unit (derived from VC-04, `tests/test_v0_corrected_bundle.py`):

- `test_profiles_and_ga_multiple_runs_are_byte_identical_to_the_legacy_copies`
- `test_every_data_processed_file_matches_the_v0_legacy_manifest_sha256`
- `test_every_corrected_copy_has_zero_byte_delta_against_its_counterpart`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): copy the frozen code and data byte-identically`

---

### T4: Derive the corrected protocol.json

**What**: Create `baselines/v0-corrected/protocol.json` = the `v0-legacy` protocol deep-equal except `baseline_id` and a `derived_from` object, with the four frozen configs only (no `caio_last`).
**Where**: `baselines/v0-corrected/protocol.json`
**Depends on**: T3
**Files touched**: `baselines/v0-corrected/protocol.json`, `tests/test_v0_corrected_bundle.py`
**Reuses**: `baselines/v0-legacy/protocol.json`; `tests/test_v0_legacy_protocol.py`'s AST-extraction approach (extract, never import)
**Closes**: VC-05 (AC3)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Deep-equal to the legacy protocol minus `baseline_id`/`derived_from`; `derived_from` names `v0-legacy` and the pinned base
- [x] AST-extracted `PERFIL_CONFIG`, `GROUPS`, `PROFILE_WEIGHTS`, `NCROSS_RATE`, `MUT_RATE` from both bundles' `code/` are equal, four frozen configs only
- [x] Tests (3) green; quick gate green (52 P0-01 + 12 new)

**Tests**: unit (derived from VC-05, `tests/test_v0_corrected_bundle.py`):

- `test_protocol_is_deep_equal_except_baseline_id_and_derived_from`
- `test_ast_extracted_constants_equal_the_legacy_values_without_caio_last`
- `test_an_extra_protocol_key_is_rejected_by_the_deep_equal_check`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the derived protocol with AST-equality proof`

---

### T5: Add the manifest build/verify tooling

**What**: Create `tools/build_manifest.py` and `tools/verify_manifest.py` under `baselines/v0-corrected/tools/` with the same closed schema as `v0-legacy` (schema_version 1) - covered subtrees `code/`, `data/`, `reference/`; exclusions `baseline.json`, `protocol.json`, `README.md`, `tools/`, `status.json`, `environment.txt`, `requirements.txt`, each with a reason - and prove the behavior on synthetic tempdir bundles. `baseline.json` itself is NOT built here (spec delta #6: `reference/` does not exist until T8).
**Where**: `baselines/v0-corrected/tools/verify_manifest.py`
**Depends on**: T3
**Files touched**: `baselines/v0-corrected/tools/build_manifest.py`, `baselines/v0-corrected/tools/verify_manifest.py`, `tests/test_v0_corrected_bundle.py`
**Reuses**: `baselines/v0-legacy/tools/{build_manifest,verify_manifest}.py` as the behavioral model (adapted copies; no cross-bundle import)
**Closes**: VC-04 (manifest verification), VC-28 (closure machinery)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `build_manifest.build()` emits `files[]` with `path`, `source_path`, `sha256`, `git_blob_sha1`, `bytes`, `mode`, `role` and refuses overwrite (exit 3)
- [x] `verify_manifest.verify_bundle()` reports `missing`/`modified`/`unexpected` by path and kind, exits 0 `VERIFIED` / 1 `DRIFT` / 3 untrustworthy, `DEFAULT_BUNDLE = baselines/v0-corrected`
- [x] A file appearing later under an excluded path (e.g. `status.json`) is NOT `unexpected`
- [x] Tests (4) green on synthetic `TemporaryDirectory()` bundles only; quick gate green (52 P0-01 + 16 new = 68; the earlier "+13" was stale arithmetic)

**Tests**: unit (derived from VC-04/VC-28 mechanics, `tests/test_v0_corrected_bundle.py`):

- `test_manifest_covers_three_subtrees_and_names_the_seven_exclusions_with_reasons`
- `test_missing_modified_and_unexpected_files_are_reported_by_path_and_kind`
- `test_verified_bundle_exits_zero_and_broken_manifest_exits_three`
- `test_excluded_paths_appearing_later_are_not_unexpected`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the manifest build and verify tooling`

---

### Phase 2 tasks - verdict core, generation, VC-25 early verdict

### T6: Add the shared verdict and row-tie module

**What**: Create `baselines/v0-corrected/tools/_verdict.py` (pure stdlib): the ordered Verdict Attribution Rule as `verdict_for`, the mechanical evidence helpers, `MispairingError` for the corrected-capture mispairing row, `float_bitwise_equal`, `tickers_equal`, `tie_row`, `entry_id` - the single verdict path shared by generator and checker (D-d).
**Where**: `baselines/v0-corrected/tools/_verdict.py`
**Depends on**: T5
**Files touched**: `baselines/v0-corrected/tools/_verdict.py`, `tests/test_v0_corrected_status.py`
**Reuses**: the decision table verbatim from spec.md "Verdict Attribution Rule"; the module contracts from design.md
**Closes**: VC-12 (rule + mispairing gate), VC-25 (bit-identity semantics), VC-29 (row-tie semantics)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Every decision-table row resolves in the spec's order for `fitness`, `score` and `derived` kinds; `data_snapshot_match == false` resolves before any producer rule
- [x] The corrected-capture mispairing row raises `MispairingError` - a gate failure, never a verdict
- [x] `defect_ref` attaches only to `invalid_fitness`/`invalid_beyond_fitness`; every other non-`valid` verdict requires a `reason`
- [x] Tests (6) green on system python3 (pure stdlib); quick gate green (52 P0-01 + 22 new = 74)

**Tests**: unit (derived from VC-12/VC-25/VC-29 semantics, `tests/test_v0_corrected_status.py`):

- `test_decision_table_rows_resolve_in_order_for_fitness_score_and_derived`
- `test_data_snapshot_mismatch_resolves_before_every_producer_rule`
- `test_corrected_capture_mispairing_raises_instead_of_becoming_a_verdict`
- `test_defect_ref_only_attaches_to_the_two_defect_verdicts`
- `test_bitwise_float_equality_compares_packed_bits`
- `test_tie_row_rejects_forged_fitness_and_mismatched_hhi`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the shared verdict and row-tie module`

---

### T7: Add the corrected driver and the reference generator

**What**: Create `tools/_corrected_driver.py` (per-(profile, seed) executor mirroring `_legacy_driver.run_all`, scratch cwd, writes only `runs.json`) and `tools/generate_reference.py` (two-pass scratch generation of the frozen writer, completeness assert, determinism byte-compare, per-row driver tie via `tie_row`, `generation_proof.json` emission) per the D-b/D-c contracts.
**Where**: `baselines/v0-corrected/tools/generate_reference.py`
**Depends on**: T6
**Files touched**: `baselines/v0-corrected/tools/_corrected_driver.py`, `baselines/v0-corrected/tools/generate_reference.py`, `tests/test_v0_corrected_bundle.py`
**Reuses**: `_verdict.tie_row`; `baselines/v0-legacy/tools/_legacy_driver.py` as the RNG-call-order mirror; the frozen `code/ga_multiple_runs.py` as the only writer
**Closes**: VC-29 (driver-tie mechanism), VC-01 (generation-time recomputation tie), VC-09 (completeness of run rows)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `generate_reference.py` refuses to commit on nondeterminism or tie divergence (exit 1) and reports unusable input with exit 3
- [x] Completeness: all 30 run_ids x 4 profiles asserted (the frozen writer swallows per-run exceptions; completeness is checked, not assumed)
- [x] The generator's tie defends the forged reference: a row the driver did not produce fails
- [x] Tests (4) green (pure functions over synthetic tempdirs, stdlib); Full gate green under tcc-venv: `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'`

**Tests**: unit (derived from VC-01/VC-09/VC-29 criteria, `tests/test_v0_corrected_bundle.py`):

- `test_assert_complete_rejects_a_missing_run_id_or_profile`
- `test_compare_passes_catches_a_single_byte_digest_difference`
- `test_emit_proof_records_the_schema_and_120_driver_rows`
- `test_tie_to_driver_flags_a_row_the_stub_driver_did_not_produce`

**Gate**: `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the corrected driver and reference generator`

---

### T8: Generate the reference and commit its proofs

**What**: Run the two-pass generation (~2 x 50 min, tcc-venv), commit `baselines/v0-corrected/reference/` (9 files, pass A bytes verbatim) and `proofs/generation_proof.json` (double-pass digests + 120 driver rows), then pin the covered subtrees by building `baseline.json` (the D-c post-step).
**Where**: `baselines/v0-corrected/reference/`
**Depends on**: T7, T5
**Files touched**: `baselines/v0-corrected/reference/` (9 files), `.specs/features/p0-02-v0-corrected/proofs/generation_proof.json`, `baselines/v0-corrected/baseline.json`, `tests/test_v0_corrected_bundle.py`
**Reuses**: T7's generator; T5's `build_manifest.py`
**Closes**: VC-01 (120-row generation tie), VC-29 (committed reference == driver output), VC-04 (manifest pin)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Generator exit 0: both passes byte-identical (sha256 per file), 30 run_ids x 4 profiles complete, 120/120 driver-tie rows pass (tickers set, `fitness`, `hhi`)
- [x] `reference/` holds pass A bytes verbatim - never hand-edited; `baseline.json` pins them with `source_path: generated:tools/generate_reference.py#pass-a`
- [x] Tests (3) green: the fast reference <-> `generation_proof.json` tie for all 120 rows
- [x] Build gate green: `python3 baselines/v0-legacy/tools/verify_manifest.py && python3 baselines/v0-corrected/tools/verify_manifest.py`

**Tests**: unit (derived from VC-01/VC-29 criteria, `tests/test_v0_corrected_bundle.py`):

- `test_generation_proof_ties_every_reference_row_to_the_driver_for_120_rows`
- `test_generation_proof_passes_digests_match_the_committed_reference_bytes`
- `test_baseline_manifest_pins_the_generated_reference`

**Gate**: `/tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/generate_reference.py --bundle baselines/v0-corrected && python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): generate the reference with its generation proof`

---

### T9: Record the VC-25 early verdict and the divergence table

**What**: Compare the committed reference to the archive per (run_id, seed): the three control profiles must be bit-identical (`struct.pack('>d')`) **or within 4 ULP** (AD-016, human ruling after the batch-B blocking finding) with every non-identical row recorded in `proofs/control_ulp_drift.json`; a control row beyond 4 ULP reports a **blocking finding for a human** (never a table entry); caio divergences are recorded with (archived, corrected) values in `proofs/reference_divergences.json` (caio rows only).
**Where**: `.specs/features/p0-02-v0-corrected/proofs/`
**Depends on**: T8
**Files touched**: `.specs/features/p0-02-v0-corrected/proofs/reference_divergences.json`, `.specs/features/p0-02-v0-corrected/proofs/control_ulp_drift.json`, `tests/test_v0_corrected_bundle.py`
**Reuses**: `_verdict.float_bitwise_equal` + a ULP-distance helper; the generated reference; the archive CSVs read-only
**Closes**: VC-25 (AC3), VC-29 (forged-reference negative, D-f)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Control-profile `fitness` is bit-identical to the archive per (run_id, seed) for 78 of the **90** control rows (3 control profiles x 30 runs) and within 4 ULP for the rest, each of those recorded with (archived, corrected, ULP distance) in `control_ulp_drift.json`; a control row beyond 4 ULP stops the batch with a blocking finding for a human
- [x] `reference_divergences.json` holds exactly the divergent caio rows with (archived, corrected) values and no control row; any unrecorded divergence or stale entry fails the gate
- [x] Tests (6) green, including the two escalation-branch negatives (control row beyond 4 ULP fails outright; forged reference row fails the tie) and the ULP-drift table completeness check
- [x] Quick gate green (52 P0-01 + 35 new)

**Tests**: unit (derived from VC-25/VC-29 criteria, `tests/test_v0_corrected_bundle.py`):

- `test_control_profile_fitness_is_bit_identical_or_within_4_ulp_of_the_archive_per_run_id_seed`
- `test_control_ulp_drift_table_records_exactly_the_non_identical_control_rows`
- `test_divergence_table_holds_caio_rows_only_and_exactly_the_divergent_ones`
- `test_stale_divergence_entry_fails_the_gate`
- `test_control_profile_divergence_beyond_4_ulp_fails_outright_instead_of_becoming_an_entry`
- `test_forged_reference_row_with_legacy_chromosomes_fails_the_row_tie`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): record the divergence table and early verdict`

---

### Phase 3 tasks - verifier, property enumeration, negative fixtures

### T10: Add the frozen fitness recomputation module

**What**: Create `tools/_frozen_fitness.py`: `FrozenBundle` importing only the bundle's frozen `code/ga.py` and `code/profiles.py` via `importlib.util.spec_from_file_location` under unique names, `sys.dont_write_bytecode = True` first, cwd inside a scratch root; frozen semantics `sel["SCORE"].sum()` (pandas skipna) minus `lambda * hhi * n` over ranked frames rebuilt from the bundle's `data/processed/`; `n * score_mean` is structurally impossible (D-e).
**Where**: `baselines/v0-corrected/tools/_frozen_fitness.py`
**Depends on**: T3
**Files touched**: `baselines/v0-corrected/tools/_frozen_fitness.py`, `tests/test_v0_corrected_artifacts.py`
**Reuses**: the frozen `code/ga.py::fitness` and `code/profiles.py::build_scores` as the only semantics source
**Closes**: VC-11 (AC1)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `fitness_of`/`hhi_of`/`score_fields_of`/`evidence_for_row` implement the frozen semantics exactly (skipna sum; NaN-SCORE rows distinguishable from `n*mean`)
- [x] Imports set `sys.dont_write_bytecode = True` and run under a scratch cwd; nothing is written under `code/`, `data/`, `reference/`
- [x] Tests (4) green under tcc-venv (SKIP with explicit reason on system python3)

**Tests**: unit (derived from VC-11 criteria, `tests/test_v0_corrected_artifacts.py`):

- `test_frozen_fitness_equals_the_skipna_score_sum_minus_lambda_hhi_n`
- `test_n_times_score_mean_is_never_the_recomputation`
- `test_hhi_and_score_fields_recompute_from_the_rows_own_tickers`
- `test_evidence_for_row_reports_reconstruction_snapshot_and_field_pairs`

**Gate**: `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the frozen fitness recomputation module`

---

### T11: Add the artifact verifier and status derivation

**What**: Create `tools/verify_artifacts.py` per the D-d/D-e contracts: `verify_reference` (100% row recompute, hhi self-recompute, `n_runs` == CSV rows, rows never dropped from the denominator), `derive_status_entries` (git-blob reads at the pinned base; mechanical producer/contributor mapping; verdict via `_verdict.verdict_for`), `check_status`, and modes `--emit-status`/`--reference-dir`/`--json`.
**Where**: `baselines/v0-corrected/tools/verify_artifacts.py`
**Depends on**: T10, T6, T8
**Files touched**: `baselines/v0-corrected/tools/verify_artifacts.py`, `tests/test_v0_corrected_artifacts.py`
**Reuses**: `_frozen_fitness.RowEvidence`, `_verdict.verdict_for`; runtime-mutation fixtures per D-f (no committed fixture files)
**Closes**: VC-01 (AC1), VC-09, VC-10 (AC1), VC-26 (AC1), VC-24 (evidence re-derivation), VC-11 (fixture legs)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `verify_artifacts.py` exits 0 over `reference/`: 100% of 120 rows have reported == recomputed (1e-9) and hhi self-recompute passes
- [x] FAIL branches proven by runtime fixtures in `TemporaryDirectory()`: mutated ticker set (size != n), duplicated TICKER, live-only ticker, NaN-SCORE semantics
- [x] A failing row stays in `rows_total` - never dropped, never counted as passing
- [x] Tests (6) green under tcc-venv; Full gate green

**Tests**: integration (derived from VC-01/VC-09/VC-10/VC-11/VC-24/VC-26 criteria, `tests/test_v0_corrected_artifacts.py`):

- `test_every_reference_row_reports_fitness_equal_to_recomputed`
- `test_summary_n_runs_equals_the_csv_row_count`
- `test_mutated_ticker_set_fails_the_artifact_and_keeps_the_row_in_the_denominator`
- `test_duplicated_ticker_fails_loudly_instead_of_mis_reconstructing`
- `test_live_only_ticker_yields_not_verifiable_not_a_silent_pass`
- `test_derived_status_entries_rederive_evidence_from_the_pinned_blobs`

**Gate**: `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the artifact verifier and status derivation`

---

### T12: Add the deterministic property enumeration

**What**: Create `tests/test_v0_corrected_property.py`: instrument the REAL `run_ga` (`ga.fitness` monkeypatched as a recorder, generations recovered by chunking the call log by `pop_size`) with both frozen `ga` modules loaded via `spec_from_file_location` under unique names; deterministic enumeration over small synthetic frames (NaN SCORE, tied SCORE, duplicate SETOR, even pop_size parity 200/250/300/276); the legacy module must show at least one violated generation (defect signature); only feasible masks (`mask.sum() == n`) enter the selection path.
**Where**: `tests/test_v0_corrected_property.py`
**Depends on**: T2
**Files touched**: `tests/test_v0_corrected_property.py`
**Reuses**: `baselines/v0-corrected/code/ga.py` and `baselines/v0-legacy/code/ga.py` as the two subjects (never a re-implemented loop)
**Closes**: VC-06 (T1), VC-07 (T1), VC-08 (T1)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] For every enumerated generation/seed the recorded (fitness, chromosome) is a member of that generation's evaluated pairs; the winner is the evaluated individual at the argmax index of its own generation
- [x] The legacy module shows >= 1 violated generation, recorded as the expected defect signature
- [x] Tests (5) green under tcc-venv (SKIP with explicit reason on system python3); Full gate green

**Tests**: unit (derived from VC-06/VC-07/VC-08 criteria, `tests/test_v0_corrected_property.py`):

- `test_recorded_pair_belongs_to_the_evaluated_pairs_of_its_generation`
- `test_enumeration_covers_nan_score_ties_duplicate_setor_and_pop_parity`
- `test_population_length_stays_pop_size_every_generation`
- `test_legacy_module_shows_a_violated_generation_as_defect_signature`
- `test_infeasible_masks_are_never_fed_through_the_selection_path`

**Gate**: `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `test(v0-corrected): add the deterministic property enumeration`

---

### Phase 4 tasks - status.json, marking, banner, note

### T13: Emit status.json and the marking checks

**What**: Run `verify_artifacts.py --emit-status` to commit `baselines/v0-corrected/status.json` (one entry per recorded value with the evidence fields, `reported`/`recomputed`, verdict via `_verdict.verdict_for`, `sha256` from `git show 181453f:<path>`; `status.json` never enumerates itself), and complete `tests/test_v0_corrected_status.py` with the coverage/digest/verdict-recompute checks (D-d).
**Where**: `baselines/v0-corrected/status.json`
**Depends on**: T11
**Files touched**: `baselines/v0-corrected/status.json`, `tests/test_v0_corrected_status.py`
**Reuses**: the T6 rule tests in the same module; `_verdict.verdict_for` (generator and checker share the code path)
**Closes**: VC-12 (AC2), VC-13 (AC2), VC-14 (AC2), VC-15 (AC2), VC-24 (AC2)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Every strict `fitness` locator in `baselines/v0-legacy/reference/**` (5 of 9 files) and the pinned `outputs/` tree (15 of 160 tracked files) appears exactly once per recorded value; nested/aggregate `fitness` values enumerated by JSON structure
- [x] Every digest equals the git-blob computation at `181453f`; the working-tree `outputs/` bytes still match (AD-007 enforced as test); legacy-reference digests cross-check against `v0-legacy/baseline.json`
- [x] Every verdict recomputes from its recorded evidence via `verdict_for`; the marking test and the generator cannot drift (same code path)
- [x] `python3 baselines/v0-legacy/tools/verify_manifest.py` reports `VERIFIED 17` and the 52 P0-01 tests pass unmodified
- [x] Tests (11 total in the module: 6 from T6 + 5 new) green; quick gate green

**Tests**: unit (derived from VC-12/VC-13/VC-14/VC-15/VC-24 criteria, `tests/test_v0_corrected_status.py`):

- `test_every_strict_fitness_locator_appears_exactly_once`
- `test_every_entry_digest_matches_the_git_blob_at_the_pinned_base`
- `test_working_tree_outputs_bytes_still_equal_the_pinned_digests`
- `test_every_verdict_recomputes_from_its_recorded_evidence`
- `test_legacy_manifest_gate_is_verified_with_seventeen_files`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): emit status.json with the marking checks`

---

### T14: Declare the pinned environment durably

**What**: Create `baselines/v0-corrected/environment.txt` (`python -VV` output, numpy and pandas versions, full `pip freeze` of the generating venv, working install command, wheels caveat) and `requirements.txt` (`numpy==2.5.4`, `pandas==3.1.0`), with their content checks (D-g, VC-19).
**Where**: `baselines/v0-corrected/environment.txt`
**Depends on**: T13
**Files touched**: `baselines/v0-corrected/environment.txt`, `baselines/v0-corrected/requirements.txt`, `tests/test_v0_corrected_status.py`
**Reuses**: the marking module's documentation-check style
**Closes**: VC-19 (AC4)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `environment.txt` contains the interpreter stamp, numpy/pandas versions, `pip freeze`, `python3 -m venv <dir> && <dir>/bin/pip install -r requirements.txt`, and the caveat that proofs reproduce only while those wheels remain obtainable
- [x] The venv itself is NOT committed; the note-level "not reproducible later" claim is T16's
- [x] Tests (2) green; quick gate green

**Tests**: unit (derived from VC-19 criteria, `tests/test_v0_corrected_status.py`):

- `test_environment_declares_python_numpy_pandas_freeze_and_install_command`
- `test_environment_carries_the_wheels_obtainability_caveat`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): declare the pinned environment durably`

---

### T15: Banner the v0-legacy README

**What**: Add the INVALID banner to `baselines/v0-legacy/README.md` in the "Two numbers that do not agree, on purpose" area: the archived fitness values are invalid under the P0-02 capture defect, and `baselines/v0-corrected/status.json` is the verdict map (VC-16). The only v0-legacy edit; the file is manifest-excluded.
**Where**: `baselines/v0-legacy/README.md`
**Depends on**: T13
**Files touched**: `baselines/v0-legacy/README.md`, `tests/test_v0_corrected_status.py`
**Reuses**: the banner substring check style; EXCLUDED_PATHS in `tests/test_v0_legacy_manifest.py` (verified: no P0-01 test asserts README content)
**Closes**: VC-16 (AC2)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Banner names the P0-02 capture defect and points at `baselines/v0-corrected/status.json`; no covered v0-legacy byte moves
- [x] The 52 P0-01 tests still pass unmodified (README content is asserted only in the new test)
- [x] Tests (2) green; quick gate green

**Tests**: unit (derived from VC-16 criteria, `tests/test_v0_corrected_status.py`):

- `test_legacy_readme_banner_names_the_defect_and_the_status_map`
- `test_banner_edit_leaves_every_covered_v0_legacy_byte_untouched`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `docs(v0-legacy): banner the archived fitness as invalid`

---

### T16: Write the methodological note

**What**: Write `baselines/v0-corrected/README.md` (reader's guide + methodological note) with the VC-17 invalidation-scope statement list and the VC-18 reproducibility limits verbatim: reported fitness invalid wherever the legacy capture recorded a generation best; return interpretation unchanged (returns/tickers/hhi stand on their own self-consistency); `score_provenance_inconsistent` and `not_verifiable_at_head` rows are NOT attributed to the capture defect; no blanket invalidation of historical backtests; the caio run-0 replay numbers; the production-consensus provenance paragraph; and no reproducibility claim beyond the committed proofs.
**Where**: `baselines/v0-corrected/README.md`
**Depends on**: T14
**Files touched**: `baselines/v0-corrected/README.md`, `tests/test_v0_corrected_status.py`
**Reuses**: the verbatim statement lists from spec.md VC-17/VC-18; the T14 environment declarations
**Closes**: VC-17 (AC4), VC-18 (AC4)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Every VC-17 statement and every VC-18 reproducibility limit is present verbatim (substring tests enforce); no over-claiming sentence survives
- [x] The note cites `status.json` as the verdict map and `proofs/replay_20seeds.json` as the comparison of record
- [x] Tests (2) green; quick gate green

**Tests**: unit (derived from VC-17/VC-18 criteria, `tests/test_v0_corrected_status.py`):

- `test_note_states_the_invalidation_scope_statements_verbatim`
- `test_note_states_the_reproducibility_limits_verbatim`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `docs(v0-corrected): write the methodological note`

---

### Phase 5 tasks - replay, opt-in proof, completion gates

### T17: Add the replay comparison checker

**What**: Create `tools/compare_replay.py` (stdlib only; drivers run as subprocesses under `sys.executable` in scratch roots): `run_pair`, `build_proof` (per run per driver: `reported_fitness`, `recomputed_fitness`, `tickers`, `hhi`; top-level `denominator: 80`, run_id range, seed list, profile list, per-expectation numerators, environment stamp), `check_expectations` (E1-E4 recomputed from `runs[]`), `check_proof_vs_reference`, `check_proof_vs_committed`, and the env gate (exit 2 without `TCC_V0_CORRECTED_REPLAY=1`).
**Where**: `baselines/v0-corrected/tools/compare_replay.py`
**Depends on**: T7, T6
**Files touched**: `baselines/v0-corrected/tools/compare_replay.py`, `tests/test_v0_corrected_replay.py`
**Reuses**: `_legacy_driver.py`/`_corrected_driver.py` as the two subjects; `_verdict.tie_row`; the E1-E4 table from spec.md
**Closes**: VC-21 (T3), VC-22 (T3), VC-27 (T3)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Any violated expectation reports `unexpected divergence`, names the row and exits non-zero - never folded into "expected"
- [x] The env gate refuses execution before any driver spawns (exit 2) when `TCC_V0_CORRECTED_REPLAY` is unset
- [x] Tests (3) green on synthetic in-memory/tempdir proofs (stdlib, no execution); quick gate green

**Tests**: unit (derived from VC-21/VC-22/VC-27 criteria, `tests/test_v0_corrected_replay.py`):

- `test_check_expectations_recomputes_e1_e4_from_the_run_rows`
- `test_env_gate_refuses_execution_without_the_replay_flag`
- `test_injected_expectation_violation_reports_unexpected_divergence_and_exits_nonzero`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `feat(v0-corrected): add the replay comparison checker`

---

### T18: Run the opt-in 80-run replay and commit its proof

**What**: **Opt-in, ~75 min measured (budget 1-2 h), run exactly once.** With `TCC_V0_CORRECTED_REPLAY=1`, run both drivers for 20 seeds x 4 profiles (run_id 0-19 = seeds 42-61) in the declared environment; commit `proofs/replay_20seeds.json` with the 80-run denominator and per-expectation numerators; add the fast fold/schema/reference-tie legs to `tests/test_v0_corrected_replay.py`.
**Where**: `.specs/features/p0-02-v0-corrected/proofs/replay_20seeds.json`
**Depends on**: T17, T8
**Files touched**: `.specs/features/p0-02-v0-corrected/proofs/replay_20seeds.json`, `tests/test_v0_corrected_replay.py`
**Reuses**: T17's checker; the committed reference for the VC-29 live tie
**Closes**: VC-20 (T3; live VC-29 tie run_id 0-19 inside the replay, live VC-25 confirmation)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] The proof JSON records per run both drivers' reported and recomputed fitness, ticker sets and hhi; the summary carries `denominator: 80` and the E1-E4 numerators
- [x] E1-E4 hold over the computed denominator, or the run reports `unexpected divergence` and the finding goes to a human (never a silent fold)
- [x] A later spot re-run's output equals the committed rows (proof-as-record contract)
- [x] Tests (3) green (proof-schema, fold-vs-summary, corrected-row tie); quick gate green

**Tests**: unit (derived from VC-20 criteria, `tests/test_v0_corrected_replay.py`):

- `test_committed_proof_schema_carries_the_80_denominator_runs_and_env_stamp`
- `test_proof_fold_matches_its_summary_numerators_over_the_denominator`
- `test_proof_corrected_rows_tie_to_the_committed_reference_for_run_id_0_19`

**Gate**: `TCC_V0_CORRECTED_REPLAY=1 /tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/compare_replay.py && python3 -m unittest discover -s tests -p 'test_v0_*.py'`
**Commit**: `test(v0-corrected): commit the 80-run replay proof`

---

### T19: Add the suite-trailer manifest gates and run completion

**What**: Create `tests/test_v0_zz_manifest_gates.py` (the `zz` name sorts last under `unittest discover -p 'test_v0_*.py'`): after the full suite it runs both `verify_manifest.py` gates and asserts exit 0 with zero unexpected files, plus a static check that every bundle importer sets `sys.dont_write_bytecode` first. Then run the completion gates.
**Where**: `tests/test_v0_zz_manifest_gates.py`
**Depends on**: T18, T8
**Files touched**: `tests/test_v0_zz_manifest_gates.py`
**Reuses**: both `verify_manifest.py` CLIs as subprocesses
**Closes**: VC-23 (AC2), VC-28 (AC2)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Tests (3) green; both gates exit 0 after the full suite with zero unexpected files in either bundle
- [x] Completion gates green: Quick + Full + Build rows of the Gate Check Commands table (52 P0-01 unchanged; ~70 new tests total)
- [x] `outputs/` byte-identity holds; `baselines/v0-legacy/` covered bytes untouched

**Tests**: e2e (derived from VC-23/VC-28 criteria, `tests/test_v0_zz_manifest_gates.py`):

- `test_both_manifest_gates_exit_zero_after_the_full_suite`
- `test_no_unexpected_files_exist_under_either_bundle`
- `test_every_bundle_import_sets_dont_write_bytecode_before_the_import`

**Gate**: `python3 -m unittest discover -s tests -p 'test_v0_*.py' && python3 baselines/v0-legacy/tools/verify_manifest.py && python3 baselines/v0-corrected/tools/verify_manifest.py && /tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/verify_artifacts.py`
**Commit**: `test(v0-corrected): add the final manifest gates`

---

## Phase Execution Map

Visual representation of task ordering. Phases run in sequence, and tasks within a phase run in order:

```
Phase 1 (batch A):  T1  T2  T3  T4  T5
Phase 2 (batch B):  T6  T7  T8  T9
Phase 3 (batch C):  T10  T11  T12
Phase 4 (batch D):  T13  T14  T15  T16
Phase 5 (batch E):  T17  T18  T19
```

Execution is strictly sequential - there is no intra-phase parallelism. A single agent (or batch worker) works one task at a time, in order. Batch E holds the one opt-in ~75-min replay (T18) as its own task, run exactly once with its proof committed.

---

## Task Granularity Check

| Task | Scope | Status |
| --- | --- | --- |
| T1: relocate record block | 1 file change + its proof tests | Granular |
| T2: corrected code copy | 1 file (the one deliberate difference) | Granular |
| T3: identical copies | 1 cohesive deliverable: the byte-identical rest (7 files, pure copies, one oracle) | Granular (cohesive copy set) |
| T4: derived protocol | 1 file | Granular |
| T5: manifest tooling | 1 cohesive contract (2 tool files, one schema, shared tests) | Granular (cohesive) |
| T6: verdict module | 1 module (pure functions) | Granular |
| T7: generation pipeline | 1 cohesive pipeline (executor + orchestrator, one tie contract) | Granular (cohesive) |
| T8: generated reference + proofs | 1 cohesive deliverable: the generated evidence set | Granular (generated artifacts, gate-verified) |
| T9: divergence table + early verdict | 1 proof artifact + its checks | Granular |
| T10: frozen recomputation | 1 module | Granular |
| T11: artifact verifier | 1 module (one verifier contract) | Granular |
| T12: property enumeration | 1 test module | Granular |
| T13: status.json + marking checks | 1 cohesive deliverable: the verdict map and its checker | Granular (cohesive) |
| T14: environment pin | 1 cohesive deliverable: the declared environment (2 files) | Granular (cohesive) |
| T15: README banner | 1 file edit | Granular |
| T16: methodological note | 1 file | Granular |
| T17: replay checker | 1 module | Granular |
| T18: opt-in replay + proof | 1 run + 1 proof artifact (isolated on purpose - see human decision 2 and AD-012) | Granular (single execution) |
| T19: manifest gates | 1 test module (suite trailer) | Granular |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| --- | --- | --- | --- |
| T1 | None | (no incoming arrows) | Match |
| T2 | T1 | T1 -> T2 | Match |
| T3 | T2 | T2 -> T3 | Match |
| T4 | T3 | T3 -> T4 | Match |
| T5 | T3 | T3 -> T5 | Match |
| T6 | T5 | T5 -> T6 | Match |
| T7 | T6 | T6 -> T7 | Match |
| T8 | T7, T5 | T7 -> T8, T5 -> T8 | Match |
| T9 | T8 | T8 -> T9 | Match |
| T10 | T3 | T3 -> T10 | Match |
| T11 | T10, T6, T8 | T10 -> T11, T6 -> T11, T8 -> T11 | Match |
| T12 | T2 | T2 -> T12 | Match |
| T13 | T11 | T11 -> T13 | Match |
| T14 | T13 | T13 -> T14 | Match |
| T15 | T13 | T13 -> T15 | Match |
| T16 | T14 | T14 -> T16 | Match |
| T17 | T7, T6 | T7 -> T17, T6 -> T17 | Match |
| T18 | T17, T8 | T17 -> T18, T8 -> T18 | Match |
| T19 | T18, T8 | T18 -> T19, T8 -> T19 | Match |

All dependency edges point backward or within a phase - no forward-phase dependency exists (T5->T6, T3->T10, T2->T12, T6->T11, T8->T11, T11->T13, T7->T17, T6->T17, T8->T18, T8->T19 are cross-phase but backward).

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| --- | --- | --- | --- | --- |
| T1 | Live legacy path | unit | unit | OK |
| T2 | Frozen bundle copies | unit | unit | OK |
| T3 | Frozen bundle copies | unit | unit | OK |
| T4 | Protocol | unit | unit | OK |
| T5 | Manifest tooling | unit | unit | OK |
| T6 | Verdict module | unit | unit | OK |
| T7 | Generation pipeline | unit + build integration | unit (+ Full gate; e2e at T8 gate) | OK |
| T8 | Generated evidence | none (gate-only) + tie legs | unit (tie legs) | OK (superset) |
| T9 | Generated evidence + tests | unit (tie legs) | unit | OK |
| T10 | Frozen recomputation | unit | unit | OK |
| T11 | Artifact verifier | integration | integration | OK |
| T12 | Property enumeration | unit | unit | OK |
| T13 | Marking layer | unit | unit | OK |
| T14 | Documentation & env pin | unit | unit | OK |
| T15 | Documentation & env pin | unit | unit | OK |
| T16 | Documentation & env pin | unit | unit | OK |
| T17 | Replay checker | unit | unit | OK |
| T18 | Replay execution + proof | none (gate-only) + fold legs | unit (fold legs) + opt-in e2e | OK (superset) |
| T19 | Suite trailer | e2e | e2e | OK |

No `Tests: none` anywhere; no test deferral across tasks - each task's tests assert its own acceptance criteria.
