# p0-02-v0-corrected (P0-02) Validation

**Date**: 2026-10-28
**Spec**: `.specs/features/p0-02-v0-corrected/spec.md` (29 requirements, VC-01..VC-29)
**Diff range**: `181453f986cce7941addf9092e38c6e005e18844..9b280dd` (spec/design/tasks commits + the 19 task commits T1..T19 + `9b280dd`)
**Pinned base commit**: `181453f` (merge of PR #35, per spec "Pinned Base Commit")
**Verifier**: independent sub-agent (author ≠ verifier). The Verifier wrote its own checker (`/tmp/opencode/indep_check.py`, 49 assertions, shares no code with the feature's tools and does not import `tools/_verdict.py` — it re-implements the Verdict Attribution Rule independently) plus its own pandas recompute (`/tmp/opencode/indep_recompute.py`); the discrimination sensor ran in a full copy of the tree at `/tmp/opencode/sensor` (deleted afterwards; repo untouched, `git status --porcelain` identical to the pre-sensor baseline — empty).

---

## Gate Check (all commands run by the Verifier, not taken on trust)

| # | Gate command | Result | Exit |
| --- | --- | --- | --- |
| 1 | `python3 -m unittest discover -s tests -p 'test_v0_*.py'` | Ran 125 tests, OK, 16 skipped | 0 |
| 2 | `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'` (numpy 2.5.4 / pandas 3.1.0) | Ran 125 tests, OK, 1 skipped | 0 |
| 3 | `python3 baselines/v0-legacy/tools/verify_manifest.py` | `status: VERIFIED`, `files: 17`, no findings | 0 |
| 4 | `python3 baselines/v0-corrected/tools/verify_manifest.py` | `status: VERIFIED`, `files: 17`, no findings | 0 |
| 5 | `/tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/verify_artifacts.py` | PASS: 120 rows (30+30+30+30) passed=120 failed=0 unverifiable=0, plus `multiple_runs_summary.json` 4/4 | 0 |
| 6 | `python3 -m unittest discover -s tests -p 'test_v0_legacy_*.py'` | Ran 52 tests, OK (P0-01 floor unchanged) | 0 |

**Test integrity**: 52 tests before the feature → 125 in the same pattern after (73 new tests across `test_v0_corrected_bundle.py` (29), `test_v0_corrected_artifacts.py` (10), `test_v0_corrected_property.py` (5), `test_v0_corrected_replay.py` (9), `test_v0_corrected_status.py` (17), `test_v0_zz_manifest_gates.py` (3)). No test deleted; no assertion loosened (spot-checked `ArchiveComparisonTests` — it recomputes 78/12 from raw CSVs, does not mirror the proof's summary).
**Skips (justified)**: 16 on system python3 = 15 numpy/pandas-dependent tests (property enumeration + frozen-fitness/verifier legs, reason "numpy/pandas unavailable in this interpreter") + 1 opt-in live spot re-run (`TCC_V0_CORRECTED_REPLAY=1`, by design VC-22). 1 skip under tcc-venv = the same opt-in spot re-run.

---

## Task Completion (T1..T19, one atomic commit each)

| Task | Status | Notes |
| --- | --- | --- |
| T1 live `py/ga.py` hunk | ✅ Done | commit `f17f72d`; verified by Verifier-proven byte transform |
| T2 corrected `code/ga.py` | ✅ Done | `a7430b6` |
| T3 byte-identical copies (7 files) | ✅ Done | `e3e561d` |
| T4 derived `protocol.json` | ✅ Done | `15c48d8` |
| T5 manifest build/verify tooling | ✅ Done | `f9e14ba` |
| T6 `_verdict.py` | ✅ Done | `92aed3a` |
| T7 `_corrected_driver.py` + `generate_reference.py` | ✅ Done | `9f8bcbf` |
| T8 reference + `generation_proof.json` + `baseline.json` | ✅ Done | `d34e673` |
| T9 divergence table + ULP-drift table | ✅ Done | `768f65f` (+ AD-016 amendments `31e8d8b`, `cd8450c`) |
| T10 `_frozen_fitness.py` | ✅ Done | `57380ac` |
| T11 `verify_artifacts.py` | ✅ Done | `93f79ac` |
| T12 property enumeration | ✅ Done | `b0105ad` |
| T13 `status.json` + marking checks | ✅ Done | `64532c8` |
| T14 `environment.txt`/`requirements.txt` | ✅ Done | `19bb564` |
| T15 v0-legacy README banner | ✅ Done | `8c78177` (only v0-legacy edit; file manifest-excluded) |
| T16 methodological note | ✅ Done | `9b90b65` |
| T17 `compare_replay.py` | ✅ Done | `ed70f53` |
| T18 80-run replay proof | ✅ Done | `a3ef8b8` |
| T19 suite-trailer manifest gates | ✅ Done | `8da408e` |
| proof-as-record contract | ✅ Done | `9b280dd` |

---

## Spec-Anchored Acceptance Criteria (VC-01..VC-29)

Evidence format: `file:line or re-derivation` — ✅ PASS means **proven by command or independent recomputation in this session**, not by reading the implementer's claims.

| VC | Requirement (spec anchor) | Evidence (command / file:line + assertion) | Result |
| --- | --- | --- | --- |
| VC-01 | 100% of corrected reference rows: reported == recomputed (AC1) | Gate 5: `verify_artifacts.py` exit 0, 120/120 rows passed; **independent recompute**: `indep_recompute.py` (frozen `ga.fitness` + `pf.build_scores` loaded directly from the bundle, hand-rolled row loop) → `checked 120/120; mismatches=0` | ✅ PASS |
| VC-02 | corrected `code/ga.py` == canonical transform of legacy bundle bytes (AC3) | `indep_check.py`: `legacy_ga.replace(OLD,NEW)==cor_ga` byte-for-byte (0-byte delta); `old_bytes` occurs exactly once; `tests/test_v0_corrected_bundle.py:112-134` green | ✅ PASS |
| VC-03 | live `py/ga.py` == transform of `git show 181453f:py/ga.py`, 0-byte delta (AC3) | `indep_check.py`: identical byte-for-byte; sensor (a): reverting the hunk in /tmp copy kills `test_transformed_pinned_blob_equals_the_worktree_ga_py_byte_for_byte` (`tests/test_v0_corrected_bundle.py:91`) | ✅ PASS |
| VC-04 | 7 copied code/data files byte-identical to v0-legacy per `baseline.json` sha256 (AC3) | `indep_check.py`: all 7 sha256 match legacy `baseline.json` AND legacy bytes AND corrected bytes (incl. parquet via `git show`); manifest gate 4 `VERIFIED files:17` | ✅ PASS |
| VC-05 | protocol deep-equal minus `baseline_id`/`derived_from`; AST configs equal; four frozen configs, no `caio_last` (AC3) | `indep_check.py`: deep-equal passes; AST-extracted `PERFIL_CONFIG/GROUPS/PROFILE_WEIGHTS/NCROSS_RATE/MUT_RATE` equal across both bundles; set == {conservador, moderado, arrojado, caio} | ✅ PASS |
| VC-06 | recorded (fitness, chromosome) ∈ evaluated pairs of its generation, instrumented real `run_ga` (T1) | `tests/test_v0_corrected_property.py` green under tcc-venv (Gate 2, 5 tests incl. `test_recorded_pair_belongs_to_the_evaluated_pairs_of_its_generation`); instrumenting verified by reading Real-`run_ga` monkeypatch, `spec_from_file_location` unique names | ✅ PASS |
| VC-07 | deterministic enumeration covers NaN SCORE, ties, duplicate SETORs, even pop_size 200/250/300/276 (T1) | `test_enumeration_covers_nan_score_ties_duplicate_setor_and_pop_parity`, `test_population_length_stays_pop_size_every_generation` green (Gate 2) | ✅ PASS |
| VC-08 | legacy module shows >= 1 violated generation; feasible masks only (T1) | `test_legacy_module_shows_a_violated_generation_as_defect_signature` + `test_infeasible_masks_are_never_fed_through_the_selection_path` green (Gate 2) | ✅ PASS |
| VC-09 | `n_runs` == CSV row count per profile (T2) | `test_summary_n_runs_equals_the_csv_row_count` green (Gate 1 stdlib / Gate 2); Gate 5 reports `multiple_runs_summary.json rows=4 passed=4` | ✅ PASS |
| VC-10 | irreconstructible row → FAIL, never dropped from the 100% denominator (AC1) | Sensor (b): mutated reference row `moderado run_id 7` → `verify_artifacts` prints `FAIL rows=30 passed=29 failed=1 unverifiable=0`, exit 1 (row retained); `test_mutated_ticker_set_fails_the_artifact_and_keeps_the_row_in_the_denominator` green | ✅ PASS |
| VC-11 | frozen fitness semantics (`sum(SCORE)` skipna − λ·hhi·n), bundle data roots, never `n*mean`; live-only ticker → `not_verifiable` (AC1) | Verifier's independent recompute implements exactly these semantics via frozen code; `test_n_times_score_mean_is_never_the_recomputation`, `test_live_only_ticker_yields_not_verifiable_not_a_silent_pass` green (Gate 2) | ✅ PASS |
| VC-25 | control rows bit-identical-or-≤4ULP vs archive (78/90 identical, 12 recorded); divergence table caio-only, exact; beyond-bound control row fails outright (AC3, AD-016) | `indep_check.py` **recomputed ULP distances from raw CSVs with its own bit-packing**: found exactly 12 non-identical control rows (max 2 ULP — within 4), `control_ulp_drift.json` records exactly that set with equal values; `reference_divergences.json` == exactly the 30 divergent caio rows, zero control rows; `tests/test_v0_corrected_bundle.py:913-999` recompute the same and assert the escalation branch (`test_control_profile_divergence_beyond_4_ulp_fails_outright_instead_of_becoming_an_entry`) | ✅ PASS |
| VC-29 | every replay run's committed reference row == `_corrected_driver` row; forged reference fails (AC3) | `indep_check.py`: all 80 replay corrected rows tie to committed `reference/` (tickers set, bitwise fitness, hhi ≤1e-9) + all 120 `generation_proof` driver rows tie to the reference; sensor (b) digest/tie failures; `test_forged_reference_row_with_legacy_chromosomes_fails_the_row_tie` green | ✅ PASS |
| VC-26 | ticker-set size != n fixture keeps the row in the denominator (AC1) | `test_mutated_ticker_set_fails_the_artifact_and_keeps_the_row_in_the_denominator` green (Gate 2); sensor (b) observed the same branch live | ✅ PASS |
| VC-12 | every affected value mapped to a verdict computed by the Verdict Attribution Rule, evidence fields recorded, defect_ref/reason discipline (AC2) | `indep_check.py`: 4,421 entries; **every verdict recomputed by the Verifier's own independent decision-table implementation** — zero disagreements; defect_ref only on `invalid_fitness`/`invalid_beyond_fitness`, every other non-valid verdict carries `reason` | ✅ PASS |
| VC-13 | strict `fitness` locator coverage: 9 legacy-reference files (5 bearing) + 160 pinned `outputs/` files (15 bearing), exactly once, structural enumeration (AC2) | `tests/test_v0_corrected_status.py:384-431` green (asserts 5 + 15 bearing files, `recorded == expected`, exact-once); Verifier independently re-enumerated from pinned blobs: every `fitness` leaf covered exactly once (4,421 unique entry ids). Note: `summary_ga.json`'s `/{profile}/fitness` leaves are covered as kind `derived` entries with identical locators — this is the design-mandated classification (design.md D-d locator table: non-run-row fitness leaves are `derived`); coverage is exact-once either way | ✅ PASS (kind-classification note above, no finding) |
| VC-14 | every digest == sha256 of `git show 181453f:<path>`; outputs byte-identity tripwire (AC2) | `indep_check.py`: all 4,421 entries' sha256 recomputed from git blob bytes at `181453f` — all match (status.json never pins itself); working-tree `outputs/` bytes independently re-hashed against pinned digests — identical | ✅ PASS |
| VC-15 | v0-legacy gates stay green: VERIFIED files 17; 52 P0-01 tests pass unmodified (AC2) | Gates 3 and 6: `VERIFIED 17` exit 0; 52 tests OK | ✅ PASS |
| VC-16 | banner names P0-02 capture defect + points at `status.json` (AC2) | `indep_check.py` grep + manual read of `baselines/v0-legacy/README.md`: "> **INVALID (P0-02).** The archived `fitness` values are invalid under the P0-02 capture defect." + "`baselines/v0-corrected/status.json` is the verdict map"; `test_v0_corrected_status.py` banner tests green | ✅ PASS |
| VC-24 | marking test recomputes every verdict from recorded evidence; verifier re-derives evidence fields (AC2) | `test_every_verdict_recomputes_from_its_recorded_evidence` green; **sensor (c)**: an invalid_fitness→valid flip kills exactly that test; Verifier's independent table re-derivation over 4,421 entries: 0 disagreements; `test_derived_status_entries_rederive_evidence_from_the_pinned_blobs` green (Gate 2) | ✅ PASS |
| VC-28 | generators/test import bundle code only under scratch roots; nothing written under code/, data/, reference/; zero unexpected files after the full suite (AC2) | `tests/test_v0_zz_manifest_gates.py` (3 tests, green, sorts last): both gates exit 0 with zero unexpected files after the suite; `indep_check.py`: no `__pycache__`/`.pyc` under `baselines/` at rest | ✅ PASS |
| VC-17 | note states invalidation scope verbatim (AC4) | Read `baselines/v0-corrected/README.md` "Invalidation scope" — all four statements present verbatim; substring tests green (`test_note_states_the_invalidation_scope_statements_verbatim`) | ✅ PASS |
| VC-18 | note states reproducibility limits verbatim (AC4) | All four limit statements present verbatim; **each measured claim re-verified independently**: FREQUENCY values 0.253333/0.226667/0.22/0.166667/0.146667/0.14×3/0.133333×2/0.113333 == exact counts 38,34,33,25,22,21,21,20,20,17 over 150 in `outputs/carteira_caio_consensus.json`; caio run 0 proof row = legacy/corrected reported 6.244563491908496, legacy hhi 0.16, corrected hhi 0.18, archived 4.604745955851804/hhi 0.26; 78/90+12 (max 2 ULP, 4.3e-16 relative) recomputed by Verifier; 120/120 in-environment (~1) recomputed from `runs[]` | ✅ PASS |
| VC-19 | durable env pin inside the bundle (AC4) | `indep_check.py`: `environment.txt` carries the `Python 3.14.4 ... [GCC 15.2.0]` stamp, numpy 2.5.4/pandas 3.1.0, pip freeze block, `python3 -m venv ... && pip install -r requirements.txt`, wheels caveat; `requirements.txt` pins exact versions; both proof JSONs' env stamps match (`Python 3.14.4`, 2.5.4, 3.1.0) | ✅ PASS |
| VC-20 | 80 runs = 20 seeds × 4 frozen profiles, run_id 0-19 = seeds 42-61; proof records per-run both drivers + top-level computed denominator (T3) | `indep_check.py`: denominator 80, 80 run rows, seeds exactly 42..61, run_ids 0..19, 4 profiles, both drivers per row with reported/recomputed/tickers/hhi; env stamps match `environment.txt` | ✅ PASS |
| VC-21 | any violated expectation → `unexpected divergence` + non-zero exit (T3) | Sensor (d): flipping one run row makes `check_expectations` return `['unexpected divergence: E1 ... holds on 79/80 ... runs: conservador/3; violation means: ...']`; `test_injected_expectation_violation_reports_unexpected_divergence_and_exits_nonzero` green asserts the CLI exit path | ✅ PASS |
| VC-22 | replay executes only with `TCC_V0_CORRECTED_REPLAY=1` (T3) | Verifier ran `python3 baselines/v0-corrected/tools/compare_replay.py` without the flag → **exit 2** before any execution; `test_env_gate_refuses_execution_without_the_replay_flag` green; the one skipped test is the opt-in spot re-run by design | ✅ PASS |
| VC-23 | every bundle import sets `sys.dont_write_bytecode` first; zero unexpected files after full suite (AC2) | `test_every_bundle_import_sets_dont_write_bytecode_before_the_import` + `test_no_unexpected_files_exist_under_either_bundle` + `test_both_manifest_gates_exit_zero_after_the_full_suite` green; static scan independently confirmed | ✅ PASS |
| VC-27 | injected expectation-row violation → non-zero exit, failure branch executed (T3) | Same evidence as VC-21 (sensor + green test `test_injected_expectation_violation_reports_unexpected_divergence_and_exits_nonzero`) | ✅ PASS |
| E1-E4 | falsifiable expectation table over the 80-run denominator | `indep_check.py` **re-folded E1-E4 from `runs[]`**, not from the summary: E1 = 80/80 bit-identical (packed-bits check), E2 = 80/80, E3 = 80/80 (>41 threshold), E4 = 80/80 (≥1); each matches the proof's recorded numerators | ✅ PASS |

**Status**: ✅ All 29 VCs covered with evidence; 0 spec-precision gaps. No VC's only evidence is a prose assertion: each maps to a runnable test (coverage honesty spot-checked — e.g. `ArchiveComparisonTests` recomputes the 78/12 split from raw CSV bytes rather than trusting `control_ulp_drift.json`; the property tests instrument the real `run_ga`, they do not re-implement the loop).

---

## Discrimination Sensor (all mutations in `/tmp/opencode/sensor`, deleted afterwards; real worktree porcelain identical to pre-sensor baseline — empty)

| # | Mutation (file, in scratch copy) | Command run | Observed failure | Verdict |
| --- | --- | --- | --- | --- |
| a | Reverted the `py/ga.py` hunk (canonical `new_bytes` → `old_bytes`, hunk present exactly once) | `python3 -m unittest discover -s tests -p 'test_v0_corrected_bundle.py'` | `FAILED (failures=1)` — `test_transformed_pinned_blob_equals_the_worktree_ga_py_byte_for_byte` (`tests/test_v0_corrected_bundle.py:91`), byte diff showing the reverted block ordering | ✅ Killed |
| b | `metrics_stability_moderado.csv` run_id 7 fitness mutated +1e-3 (`4.3237942047732805→4.324794204773281`) | bundle unittest module + `/tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/verify_artifacts.py` | `verify_artifacts: FAIL`, exit 1, ` moderado rows=30 passed=29 failed=1 (row kept in denominator)` + 7 bundle tests red (generation-proof digest, baseline.json pin, ULP gate, driver tie, stale-entry branch) | ✅ Killed |
| c | `status.json`: `invalid_fitness` (arrojado run 0 fitness) flipped → `valid` | `python3 -m unittest discover -s tests -p 'test_v0_corrected_status.py'` | `FAILED (failures=1)` — `test_every_verdict_recomputes_from_its_recorded_evidence` (the checker recomputes the verdict from the recorded evidence fields — not self-satisfying) | ✅ Killed |
| d | `proofs/replay_20seeds.json`: conservador run_id 3 (seed 45) legacy `reported_fitness` +1e-6 | `python3 -m unittest discover -s tests -p 'test_v0_corrected_replay.py'` + direct `check_expectations()` on the mutated proof | `FAILED (failures=2)` — `test_proof_fold_matches_its_summary_numerators_over_the_denominator`, `test_proof_as_record_matches_exactly_and_never_overwrites_the_record`; checker output: `unexpected divergence: E1 ... holds on 79/80 (runs: conservador/3)` | ✅ Killed |

**Sensor depth**: P0-targeted manual run per the task order (4 behavior-level mutations across the live fix, the generated reference, the verdict map, and the replay proof — the four highest-risk surfaces).
**Result**: 4/4 killed — the suite is discriminating. **PASS ✅**

---

## Code Quality

| Principle | Status |
| --- | --- |
| No features beyond what was asked | ✅ (spec deltas table in tasks.md accounts for all 5 additions beyond spec Deliverables, accepted by frozen human decision 1) |
| No abstractions for single-use code | ✅ |
| No unnecessary "flexibility" added | ✅ |
| Only touched files required for task | ✅ (`git diff --name-status 181453f..HEAD` shows only `py/ga.py`, `baselines/v0-legacy/README.md`, `baselines/v0-corrected/**`, `tests/test_v0_corrected_*` + `test_v0_zz_*`, `proofs/**`, `.specs/**`; `outputs/`, `py/core/` untouched by name-status) |
| Didn't "improve" unrelated code | ✅ (the hunk is the only semantic change; byte-proof kills anything sneaky) |
| Matches existing patterns/style | ✅ (stdlib unittest, descriptive names per test-v0-legacy style) |
| Spec-anchored outcome check | ✅ (see AC table; every test's asserted value targets the spec-defined numeric outcome) |
| Coverage expectation met | ✅ (1:1 VC mapping; negative branches VC-10/26/27/21 have real failure fixtures) |
| Every test maps to a spec requirement | ✅ |
| Guidelines | none — strong defaults applied (repo has no lint/CI; consistent with tasks.md Test Coverage Matrix) |

---

## Edge Cases

- [x] NaN SCORE skipna semantics (fixture + enumeration include NaN rows; `n*mean` structurally impossible — VC-11 verified functionally by the Verifier's recompute)
- [x] Duplicate SCORE ties / argmax first-max (property test instrument for VC-06)
- [x] pop_size parity (asserted per generation chunk)
- [x] Empty/invalid mask never recorded and never fed through selection (VC-08/VC-10)
- [x] Duplicate TICKER reconstruction ambiguity → loud FAIL (fixture)
- [x] `__pycache__`/import-time writes fenced (dont_write_bytecode + scratch roots + trailer gates)
- [x] outputs/ provenance mix handled by the evidence-driven Verdict Attribution Rule (not blanket marking) — independently recomputed 0 disagreements over 4,421 verdicts

---

## Isolation and Hygiene (Verifier's own checks)

- No `"baselines/"` string constant in `py/**`, `api/**`, root `*.py` — scan clean (AD-014).
- No `__init__.py`/`setup.py`/`pyproject.toml` under `baselines/` — scan clean.
- No `__pycache__`/`.pyc` under `baselines/` at rest — scan clean; both manifest gates VERIFIED (zero unexpected files).
- `git status --porcelain -- outputs/` empty; working-tree `outputs/` bytes re-hashed against the pinned digests — identical (AD-007).
- No scratch/temp dirs committed under the repo.

---

## Requirement Traceability Update

| Requirement | Previous | New |
| --- | --- | --- |
| VC-01..VC-29 | Design (spec pending) | ✅ Verified |

---

## Summary

**Overall**: ✅ **PASS** — feature is READY.

**Spec-anchored check**: 29/29 VCs evidenced with file:line or independent recomputation; 0 spec-precision gaps, 0 unverifiable VCs.
**Sensor**: 4/4 mutations killed.
**Gates**: 6/6 green (125 + 125 tests, both manifests VERIFIED 17, evaluator 120/120, legacy 52 unchanged).

**What works**: exactly-one-hunk byte proof (live + bundle); 100% recomputable corrected reference; complete evidence-backed verdict map (4,421 entries re-derived independently with 0 disagreements); frozen baseline untouched; honest methodological note with every measured claim re-verified; 80-run proof folds exactly from its run rows over seeds 42-61; opt-in gating proven.

**Accepted limitations (not findings)**:
1. The Verifier did NOT re-execute the ~75-min opt-in replay or the ~2×50-min generation pass. VC-20/VC-29's proof↔code coherence rests on the committed proofs plus committed `generation_proof.json` (double-pass byte determinism + 120-row driver tie), the proof-as-record tests, the engine-sensitive negative fixtures, and sensor (b)/(d) showing row-level forgery is caught. A human can re-open these with the two committed commands at any time while the pinned environment exists.
2. The `/tmp/opencode/tcc-venv` interpreter and its wheels are ephemeral; execution-tier tests SKIP with reason once it is gone (accepted by D-g/AD-010 and stated in `environment.txt` and the note).
3. The 4-ULP archive bound is a measured margin (max observed 2 ULP), not a guarantee — AD-016 already records this trade-off.
4. `summary_ga.json` fitness leaves are classified kind `derived` (contributors-based downgrade) per the approved design's locator table — an interpretation consistent with VC-13's exact-once coverage; flagged here for the record, not as a gap.

**Next steps**: escalate to the closing step of the TLC flow; no fix tasks.
