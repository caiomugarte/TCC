# v0-corrected (P0-02) Specification

## Problem Statement

The legacy genetic algorithm pairs a recorded best fitness with the wrong chromosome: `py/ga.py:115` scores the old population, `py/ga.py:129` replaces `population` with `new_pop`, and `py/ga.py:132-135` then indexes the REPLACED population with the old scores' argmax. Every artifact produced by that path reports a fitness that does not belong to the portfolio it returns (issue #7, P0-02; re-measured at round 2: 120/120 rows in `baselines/v0-legacy/reference/metrics_stability_*.csv` have reported != recomputed while `hhi` recomputes from each row's own tickers 120/120). The capture bug is already fixed in `py/core/optimizer.py:257-263,301-305`, but the published artifacts remain invalid and the live legacy path `py/ga.py` still carries the defect. P0-02 builds a corrected, self-contained `v0-corrected` baseline that differs from `v0-legacy` by exactly the capture fix, marks the affected artifacts with a machine-readable, evidence-backed verdict map, and publishes a methodological note bounding the invalidation.

## Goals

- [ ] `v0-corrected` reference artifacts report a fitness that recomputes from their own returned chromosome in 100% of rows (issue AC1).
- [ ] Every affected legacy artifact is marked per artifact and per field with an evidence-backed verdict and a reference to the defect (issue AC2).
- [ ] `v0-corrected` is provably the `v0-legacy` protocol plus exactly one hunk - same configs, same protocol, byte-proven against both the bundle and the live tree (issue AC3).
- [ ] A methodological note explains the scope of the invalidation without over-claiming (issue AC4).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| --- | --- |
| Early stopping, derived seeds, per-company dedupe, performance work, engine rewrite | Issue "Fora de escopo"; the hunk is the only semantic change allowed (AD-011) |
| Regenerating or modifying anything under `outputs/` | Marking is lower risk and reversible; AC2 demands marking (AD-007). Any change there needs human sign-off first |
| Editing `baselines/v0-legacy/` covered subtrees (`code/`, `data/`, `reference/`) or adding files under `baselines/v0-legacy/` | The frozen bundle is the measurement instrument (AD-008); its manifest test pins 17 files / 632,981 bytes |
| Fixing `baselines/v0-legacy/code/ga.py` | The frozen defect is what the replay comparison measures; repairing it destroys the measurement |
| Making the legacy archive reproducible | The caio scoring provenance gap is unexplained (P0-01: declared weights reproduce control SCOREs but caio 0/7); the feature must not claim reproducibility it cannot prove (AD-010) |
| Re-ranking or re-selecting the production sleeve; changing the interpretation of portfolio returns | The defect invalidates reported fitness, not backtest returns; see Non-claims |
| Touching `py/core/optimizer.py` | Already correct (chromosome captured at evaluation time); AD-011 fixes only `py/ga.py` |
| Hypothesis-based property testing | No hypothesis available in any interpreter (AD-012) |

**Rollback (N11):** reverting the feature commit is complete and sufficient - the feature edits only `py/ga.py` (exactly one hunk), adds `baselines/v0-corrected/**`, `tests/test_v0_corrected_*.py`, and the manifest-excluded `baselines/v0-legacy/README.md` banner, and never rewrites `outputs/`, so committed artifacts are unchanged and revert restores the exact prior state. One consequence is by design, not a defect: once merged, the fixed `py/ga.py` changes what NEW GA runs return for every caller (`py/ga_multiple_runs.py:60`, `py/consolida_ga.py:10`, `py/build_portfolios_summary.py:92` all call `run_ga`) - future runs will select different (correct) portfolios; reverting restores the defective selection behavior.

---

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here - nothing is left silently unclear.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| D1 / AD-007: regenerate vs mark affected artifacts | Mark only; `outputs/` stays byte-identical; `baselines/v0-corrected/status.json` maps every affected artifact and field to a verdict (vocabulary: `invalid_fitness`, `invalid_beyond_fitness`, `score_provenance_inconsistent`, `not_verifiable_at_head`, `valid`) plus evidence fields and a defect reference, and tests enforce coverage and verdict correctness (VC-12, VC-13, VC-24) | The issue allows "regenerar OU marcar"; marking is lower risk, reversible, and AC2 demands it | y (frozen by orchestrator) |
| D2 / AD-008: `baselines/v0-legacy/**` covered subtrees | Untouched (17 files / 632,981 B); `verify_manifest.py` stays VERIFIED; all 52 P0-01 tests stay green; the only v0-legacy edit is the excluded `README.md` banner | The frozen bundle is the evidence base; `tests/test_v0_legacy_manifest.py:22-24,51-58` pins the file/byte counts and exclusions | y |
| D3 / AD-009: how "same configs and protocol" is proven | Mechanically, not by prose: the canonical (old_bytes, new_bytes) transform applied in memory to `v0-legacy/code/ga.py` must yield `v0-corrected/code/ga.py` byte-equal, and the same pair applied to `git show 181453f:py/ga.py` must yield the corrected `py/ga.py` byte-equal (round-2: the transform is an exact byte pair, not a patch with placeholders); AST-extracted `PERFIL_CONFIG`/`GROUPS`/`PROFILE_WEIGHTS`/`NCROSS_RATE`/`MUT_RATE` equal; `protocol.json` deeply equal except `baseline_id` + `derived_from` | Prose assertions drift; byte and AST proofs are checkable in CI | y (revised round 2: exact byte pair) |
| D4 / AD-010: environment and reproducibility claims | v0-corrected anchors to an explicitly declared environment (pinned file with `python -VV`, numpy/pandas versions, `pip freeze`, and an install command, inside the bundle); the note makes no reproducibility claim beyond the committed proofs and states those proofs hold only while the pinned interpreter and wheels remain obtainable; the 20-seed comparison runs both drivers in the same declared environment | Measured (re-verified round 2): conservador seed 42 replays bit-identical to archived run 0 (4.480605754022773 / hhi 0.16 / same tickers) while caio seed 42 replays to 6.244563 / hhi 0.16 vs archived 4.604746 / hhi 0.26 (deterministic). The caio gap is NOT dataset drift and NOT RNG drift: P0-01 proved the archived caio SCORE does not follow the declared weights on the archived dataset itself (caio 0/7, missing by up to 0.42, while control SCOREs reproduce 15/15, 10/10, 10/10 - plan.md:20-31, verification.md:372) - i.e. the archive's caio values come from a scoring configuration/version the archive does not record, and the archived data and the RNG alone do not explain them | y (wording corrected round 2 per N9) |
| D5 / AD-011: what gets fixed in the live tree | Only `py/ga.py` (record-block relocation); `py/core/optimizer.py` untouched | Matches issue scope; optimizer already correct | y |
| D6 / AD-012: property-test technology and replay cost | Deterministic enumeration (generations x pop_size x seeds over small synthetic frames incl. NaN SCORE and duplicate SETORs); the legacy-vs-corrected replay (20 seeds x 4 frozen profiles = 80 runs per driver) is opt-in via `TCC_V0_CORRECTED_REPLAY=1` and its JSON proof is committed; measured single-run cost 14.5 s conservador / 24.7 s moderado / 32.4 s caio / 37.7 s arrojado (~109 s per seed across 4 profiles; 160 driver executions = ~75 min measured wall-clock, budgeted 1-2 h) so the replay is accepted as a long opt-in run in the completion gate | No hypothesis/pandas-free property framework exists; full-protocol runs cost minutes and must not gate the fast suite | y (budget corrected round 2 per N8) |
| D7 / AD-013: bundle import hygiene | Every tool/test importing bundle code sets `sys.dont_write_bytecode = True` before the import, and bundle modules with import-time side effects (`ga_multiple_runs.py:24-25` creates `outputs/` at import) are imported only with cwd inside a scratch root; nothing may write under `code/`, `data/`, `reference/` | Importing frozen code writes `__pycache__/*.pyc` and turns the frozen bundle into "DRIFT - unexpected" (`replay.py:38` already sets the flag); the `OUT_DIR.mkdir` runs at import time | y (extended round 2 per N7) |
| D8 / AD-014: isolation test | New code respects `tests/test_v0_legacy_isolation.py:24-34`: no `"baselines/"` string constant (docstrings included) or bundle-module imports in `py/`, `api/`, root `*.py`; no `__init__.py`/`setup.py`/`pyproject.toml` anywhere under `baselines/`; the `py/ga.py` edit is a pure statement move and adds no comment or string | P0-01 gate; violating it breaks the existing suite | y |
| "120/120 rows invalid" measurement | Adopted as fact with provenance: TRIAGE measured it, the draft author re-measured it, and round 2 re-measured it independently (30/30 per profile x 4 profiles, reported != recomputed from the row's own tickers over the frozen bundle data under declared weights; e.g. conservador run 0 reported 4.480605754022773 vs recomputed 0.8156341406041691; `hhi` recomputes from row tickers 120/120) | Three independent measurements agree on the mismatch for every row | y |
| Defect semantics for the expectation table | Reported fitness is bit-identical before/after the relocation (strict `>` and RNG call order unchanged); the returned chromosome changes | Logical consequence of the hunk plus measured 2/2 seeds in spec spot-check, 5/5 full-protocol seeds by TRIAGE, and the round-2 conservador/caio seed-42 replays | y |
| caio rows: what the honest verdict is | Fitness verdict `score_provenance_inconsistent` (fitness not verifiable) and the same verdict for the score columns - NOT `invalid_beyond_fitness` and NOT `invalid_fitness`: a score_median mismatch the data snapshot itself does not explain cannot be attributed to the capture defect (measured round 2: score_median recomputes 30/30 for conservador/moderado/arrojado and mismatches 30/30 for caio on the frozen data under declared weights) | B1 counter-example (i); P0-01 recorded the same split (control SCOREs reproduce, caio 0/7) | y (revised round 2) |
| Attribution discipline (AD-015) | Verdicts are DATA, not prose: verdict = f(producer code path, data-snapshot match, recomputation) via the decision table under "Verdict Attribution Rule", evidence fields recorded per entry, and a test recomputes every verdict from its recorded evidence so a mis-assigned verdict reddens the suite (VC-24) | B1 remedy; an unverifiable verdict vocabulary is unfalsifiable | y (new round 2) |
| Interpreter split | System `python3` (3.14.4, stdlib only) runs manifest gates and structural tests; `/tmp/opencode/tcc-venv/bin/python` (3.14.4, numpy 2.5.4 / pandas 3.1.0) runs every execution and recomputation test; tests needing numpy/pandas SKIP with an explicit reason when unavailable | Measured environment: system python3 has no numpy/pandas; the venv is the only interpreter with them and is ephemeral, which is why the bundle declares its environment durably | y |
| Discussed gray areas | None triggered | All decision points were frozen (D1-D8) or resolved by measurement; no user-facing behavior, persistence, or external calls in scope | y |

**Open questions:** none - all resolved or logged above (required before the spec is confirmed).

---

## User Stories

### P1: Corrected reference bundle with honest fitness ⭐ MVP

**User Story**: As a thesis reader, I want a `v0-corrected` baseline whose reported fitness belongs to the portfolio it returns, so that I can cite corrected numbers without discarding the v0 protocol.

**Why P1**: This is the defect fix itself; everything else (marking, note, replay) exists to bound and explain it.

**Acceptance Criteria** (each line is one EARS pattern):

1. The `v0-corrected` reference artifacts SHALL report, for 100% of their rows, a `fitness` value that equals the fitness recomputed from that row's returned chromosome under the frozen fitness function and frozen data. **[VC-01 | AC1]**
2. The file `baselines/v0-corrected/code/ga.py` SHALL equal `baselines/v0-legacy/code/ga.py` with exactly the canonical (old_bytes, new_bytes) transform applied in memory - the relocation of `population = new_pop` to after the best-recording block - proven by byte equality after the replacement. **[VC-02 | AC3]**
3. WHEN the live legacy path is corrected, `py/ga.py` SHALL equal the canonical transform applied to `git show 181453f:py/ga.py` and differ from it by 0 bytes - so strict `>`, RNG call order, roulette selection, crossover, mutation and seeding stay untouched - proven by byte equality in `tests/test_v0_corrected_bundle.py`. **[VC-03 | AC3]**
4. The `v0-corrected` copies of `code/profiles.py`, `code/ga_multiple_runs.py` and `data/processed/*` SHALL be byte-identical to their `v0-legacy` counterparts, verified by sha256 against `baselines/v0-legacy/baseline.json`. **[VC-04 | AC3]**
5. The `v0-corrected` `protocol.json` SHALL be deeply equal to `v0-legacy/protocol.json` except `baseline_id` and a `derived_from` object, and the AST-extracted `PERFIL_CONFIG`, `GROUPS`, `PROFILE_WEIGHTS`, `NCROSS_RATE` and `MUT_RATE` SHALL equal the v0-legacy values with the four frozen configs only (no `caio_last`). **[VC-05 | AC3]**
6. WHEN the corrected GA records a generation best, the recorded pair (fitness, chromosome) SHALL be a member of that generation's evaluated (fitness, chromosome) pairs, so the winning chromosome always belongs to the generation of the winning fitness, for every enumerated generation and seed - asserted by instrumenting the REAL `run_ga` (`ga.fitness` monkeypatched as a recorder, generations recovered by chunking the call log by `pop_size`) with both `ga` modules loaded via `importlib.util.spec_from_file_location` under unique module names, never by re-implementing the loop. **[VC-06 | T1]**
7. The property-test enumeration SHALL be deterministic (no hypothesis) and SHALL cover synthetic frames containing NaN SCORE rows, duplicate SCORE ties, duplicate SETORs, and the even `pop_size` parity of the four frozen configs (200/250/300/276). **[VC-07 | T1]**
8. IF the same property is evaluated against the frozen `baselines/v0-legacy/code/ga.py`, THEN the enumeration SHALL observe at least one violated generation and record it as an expected defect signature, proving the test discriminates the defect - and the enumeration SHALL keep every chromosome fed through `run_ga` feasible (`mask.sum() == n`), because any `-inf` score makes the frozen probs computation all-NaN and `np.random.choice` raises instead of exposing a violation. **[VC-08 | T1]**
9. WHEN a corrected multi-run summary is written, each profile's `n_runs` SHALL equal the number of data rows in its `metrics_stability_<profile>.csv`. **[VC-09 | T2]**
10. IF an artifact row's chromosome cannot be reconstructed as exactly `n` distinct rows from the bundle's frozen data, THEN the artifact verifier SHALL report that artifact as FAIL and SHALL NOT count the row as passing or drop it from the denominator. **[VC-10 | AC1]**
11. The fitness recomputation SHALL use the frozen fitness semantics - `sel["SCORE"].sum()` with pandas skipna behavior minus `lambda * hhi * n` - over data roots resolved from the bundle path (never the live `data/`), and SHALL NOT substitute `n * score_mean`; a fixture row carrying a live-only ticker SHALL yield `not_verifiable`, never a silent pass. **[VC-11 | AC1]**
12. The `fitness` column of `baselines/v0-corrected/reference/metrics_stability_<p>.csv` SHALL be bit-identical per (run_id, seed) to `baselines/v0-legacy/reference/metrics_stability_<p>.csv` for every row whose scores reproduce from the frozen snapshot under declared weights (the three control profiles; a direct consequence of the hunk - measured round 2: conservador seed 42 replays bit-identical to archived run 0), and every remaining divergence SHALL appear with (archived, corrected) values in `proofs/reference_divergences.json`; any unrecorded divergence, or any stale table entry that no longer diverges, SHALL fail the gate. **The table is bounded:** only `caio`-profile rows may appear in it, because caio alone has the proven score-provenance gap; a control-profile row whose `fitness` diverges from the archive SHALL fail the gate outright rather than be recorded as a divergence. **[VC-25 | AC3]**
13. For every replay run (profile, run_id 0-19, seed 42-61) the committed corrected reference row - `tickers` as a set, `hhi` and `fitness` within 1e-9 - SHALL equal the row `baselines/v0-corrected/tools/_corrected_driver.py` produces for that same (profile, seed), so the committed reference is provably the output of the corrected code under the frozen protocol and not a relabelled legacy result. A forged reference that keeps legacy chromosomes and only rewrites `fitness` SHALL fail this requirement. **[VC-29 | AC3]**
14. IF a test fixture mutates an artifact row's ticker set to size != n, THEN the artifact verifier SHALL report FAIL for that artifact and the test SHALL assert the row remains in the 100% denominator, proving VC-10's failure branch executes. **[VC-26 | AC1]**

**Independent Test**: Run the deterministic enumeration (VC-06 to VC-08) and the artifact verifier (VC-01, VC-09 to VC-11, VC-26) in the declared environment; each passes or fails on its own without the marking or note work.

---

### P2: Machine-readable invalidation of the defective artifacts

**User Story**: As an auditor, I want every artifact whose numbers are affected by the defect marked with an evidence-backed verdict - per artifact and per field, with a defect reference - so that no invalid number is cited by accident and no verdict is a bare assertion.

**Why P2**: AC2 demands the marking; without it the corrected bundle coexists with silently invalid numbers, and without the attribution evidence the verdicts themselves would be unfalsifiable.

**Acceptance Criteria**:

1. The file `baselines/v0-corrected/status.json` SHALL map every affected recorded value to a verdict (`invalid_fitness`, `invalid_beyond_fitness`, `score_provenance_inconsistent`, `not_verifiable_at_head`, or `valid`) computed by the Verdict Attribution Rule, with the evidence fields `producer`, `row_reconstruction`, `score_recompute`, `data_snapshot_match`, `reported` and `recomputed` recorded per entry; a defect reference naming the P0-02 capture defect SHALL be recorded on every verdict that the rule attributes to that defect (`invalid_fitness`, `invalid_beyond_fitness`), and every other non-`valid` verdict SHALL carry a `reason` naming its actual cause instead of borrowing the defect's name. **[VC-12 | AC2]**
2. WHEN the marking test enumerates artifacts, its subject SHALL be both `baselines/v0-legacy/reference/**` (9 covered files; 5 carry strict `fitness` locators) and every `outputs/` artifact tracked at the pinned base commit (160 tracked files; 15 carry strict `fitness` locators - 7 `metrics_stability_*.csv`, 6 `comparison_*.json`, `multiple_runs_summary.json`, `summary_ga.json`); every strict `fitness` locator SHALL appear exactly once per recorded value as an entry `{path, locator (JSON Pointer to the scalar leaf, or CSV column), row_key (run_id, CSV rows only), sha256, verdict}`, nested and aggregate `fitness` values SHALL be enumerated by JSON structure, not text search, and every entry SHALL resolve to a file present in the pinned base tree. **[VC-13 | AC2]**
3. The digest of every enumerated artifact SHALL be the SHA-256 of the file's bytes at the pinned base commit (`git show 181453f:<path>`), including `baselines/v0-legacy/reference/**` digests cross-checked against `v0-legacy/baseline.json`, and the tripwire SHALL recompute from git - so `status.json` never pins itself - and SHALL fail on any digest mismatch. **[VC-14 | AC2]**
4. The `baselines/v0-legacy` gates SHALL stay green: `verify_manifest.py` exits 0 with `status: VERIFIED` and `files: 17`, and the 52 P0-01 tests pass unmodified. **[VC-15 | AC2]**
5. The `baselines/v0-legacy/README.md` banner SHALL state that the archived fitness values are invalid under the P0-02 capture defect and SHALL reference `baselines/v0-corrected/status.json` as the verdict map. **[VC-16 | AC2]**
6. The marking test SHALL recompute every entry's verdict from its own recorded evidence fields (`producer`, `row_reconstruction`, `score_recompute`, `data_snapshot_match`, `reported`, `recomputed`) via the Verdict Attribution Rule and SHALL fail on any disagreement, and the artifact verifier SHALL re-derive those evidence fields from the artifacts themselves - including `data_snapshot_match`, which SHALL be `true` only when the row's recorded episode/score inputs are reproducible from the frozen bundle snapshot. **[VC-24 | AC2]**
7. WHEN any generator or test produces or imports bundle artifacts, it SHALL run with cwd inside a scratch root, SHALL NOT write under `code/`, `data/` or `reference/` of either bundle, and SHALL NOT trigger `ga_multiple_runs.py`'s import-time `OUT_DIR.mkdir` outside that scratch root; after the full suite both manifest gates SHALL report zero unexpected files. **[VC-28 | AC2]**

**Independent Test**: Run the marking test alone: it enumerates `baselines/v0-legacy/reference/**` and the pinned `outputs/` tree, checks coverage/uniqueness against `status.json`, verifies the git-pinned digests, and recomputes every verdict from its evidence fields - independent of the corrected bundle's results.

**Verdict Attribution Rule (data, not prose).** Each entry's verdict is a pure function of its evidence fields: `producer` (`legacy_capture` | `corrected_capture` | `unrecorded` - for CSVs mechanically derived from the `generations_run` column, written by `py/pipelines/multi_run.py:45,123-124` through `py/core/optimizer.py:330-331` when present and by `py/ga_multiple_runs.py:362` when absent; for JSONs recorded per file with its mechanical evidence), `row_reconstruction` (`exact`: exactly `n` distinct tickers all present in the frozen data | `inexact`), `data_snapshot_match` (`true` only when the row's score inputs reconstruct from the frozen bundle snapshot; `false` otherwise - this field answers "do the row's input rows resolve in the snapshot", and is orthogonal to whether the recorded score values match, which `score_recompute` decides), `score_recompute` (`match` | `mismatch` | `not_applicable` - for every row that records score fields, `match` SHALL mean that EVERY recorded score field of that row (`score_mean`, `score_median`, `score_std` and any other recorded `score_*` column or JSON leaf) reproduces within 1e-9 from the frozen snapshot under the declared weights, and `mismatch` SHALL mean that at least one of them does not; `not_applicable` is reserved for entries that record no score field), `reported`, `recomputed`.

| Locator kind | Condition (in order) | Verdict |
| --- | --- | --- |
| fitness | `row_reconstruction == inexact` | `not_verifiable_at_head` |
| fitness | `data_snapshot_match == false` | `not_verifiable_at_head` |
| fitness | `score_recompute == mismatch` | `score_provenance_inconsistent` (fitness not verifiable) |
| fitness | `producer == unrecorded` | `not_verifiable_at_head` |
| fitness | `producer == legacy_capture` and `reported != recomputed` | `invalid_fitness` |
| fitness | `producer == legacy_capture` or `corrected_capture`, `reported == recomputed` | `valid` |
| fitness | `producer == corrected_capture` and `reported != recomputed` | gate FAILS (unexpected mispairing - contradicts the fix; never folded into a verdict) |
| score columns / score fields | `inexact` / `data_snapshot_match == false` / recompute `mismatch` / recompute `match` | `not_verifiable_at_head` / `not_verifiable_at_head` / `score_provenance_inconsistent` / `valid` |
| derived (aggregates, selections, consensus membership) | any contributor `invalid_fitness` | `invalid_beyond_fitness` |
| derived | else any contributor `not_verifiable_at_head` | `not_verifiable_at_head` |
| derived | else any contributor `score_provenance_inconsistent` | `score_provenance_inconsistent` |
| derived | all contributors `valid` | `valid` |

---

### P3: Methodological note and 20-seed replay proof

**User Story**: As a thesis reader, I want a methodological note that explains exactly what the invalidation covers - and a recorded comparison of legacy vs corrected on 20 seeds x 4 profiles - so that I can bound the damage honestly.

**Why P1 is not enough**: Correct numbers do not tell a reader which old numbers are still usable; AC4 and the replay are the honesty layer.

**Acceptance Criteria**:

1. The methodological note SHALL explain the invalidation scope via a statement list checked verbatim by substring tests: reported fitness is invalid wherever the legacy capture recorded a generation best; portfolio-return interpretation is unchanged and returns/tickers/hhi stand on their own measured self-consistency; rows whose recorded score fields do not reproduce from the frozen snapshot under the declared weights are marked `score_provenance_inconsistent` and rows whose producer or chromosome cannot be reconstructed are marked `not_verifiable_at_head` - neither is attributed to the capture defect; and no blanket invalidation of historical backtests is presumed. **[VC-17 | AC4]**
2. The note SHALL state the reproducibility limits verbatim by substring tests: archived v0-legacy values are not reachable in the declared environment where measured (caio run 0 replays to fitness 6.244563 - legacy and corrected agree - with hhi 0.16 from the legacy driver against the archived 4.604746 with hhi 0.26, the corrected driver's hhi for that run being 0.18); the production `outputs/carteira_caio_consensus.json` was produced by the corrected optimizer over a 150-run set the archive does not contain (its FREQUENCY values are exact counts over 150: 38, 34, 33, 25, 22, 21, 21, 20, 20, 17, and its sibling `outputs/metrics_stability_caio.csv` carries `generations_run`/`converged_early`) while the archived consensus came from the defective capture - so the membership difference is a provenance difference, not evidence that the live recommendation is defective; and no reproducibility claim is made beyond the committed proofs. **[VC-18 | AC4]**
3. The `v0-corrected` bundle SHALL declare the generating environment durably inside `baselines/v0-corrected/`: interpreter `python -VV` output, numpy and pandas versions, a full `pip freeze`, and a working install command, plus the caveat that the proofs are reproducible only while those wheels remain obtainable. **[VC-19 | AC4]**
4. WHEN the comparison runs, a run SHALL be one (profile, seed) pair and the scope SHALL be 20 seeds x 4 frozen profiles = 80 runs, with the seed set pinned to `run_id` 0-19 and therefore seeds 42-61 under the frozen `seed_formula` (`seed_base + run_id`, `seed_base` 42, per `protocol.json`) - no other 20 of the 30 protocol runs may substitute; both drivers SHALL run in the SAME declared environment and the proof JSON committed under `.specs/features/p0-02-v0-corrected/proofs/` SHALL record per run both drivers' reported and recomputed fitness plus ticker-set equality, and its top-level summary SHALL carry the computed run denominator so the gate cannot silently change scope. **[VC-20 | T3]**
5. IF any row of the falsifiable expectation table is violated over the computed denominator, THEN the comparison SHALL report `unexpected divergence` and exit non-zero, never folding the violation into "expected". **[VC-21 | T3]**
6. The comparison SHALL execute only when `TCC_V0_CORRECTED_REPLAY=1` is set, so the default test run stays fast. **[VC-22 | T3]**
7. WHEN any tool or test imports bundle code, it SHALL set `sys.dont_write_bytecode = True` before the import, and after the full suite both manifest gates SHALL still report zero unexpected files. **[VC-23 | AC2]**
8. IF a test fixture injects one expectation-row violation into the comparison checker, THEN the checker SHALL report `unexpected divergence` and exit non-zero, proving VC-21's failure branch executes. **[VC-27 | T3]**

**Independent Test**: Read the note against the issue's AC4 wording (the substring tests enforce the statement list) and run the opt-in replay; the proof JSON either matches the expectation table over its recorded denominator (pass) or names the violated expectation (fail).

---

## Pinned Base Commit

Every `git show`-pinned digest and every tree count stated in this spec - 160 tracked `outputs/` files, 15 with a strict `fitness` locator, 9 `baselines/v0-legacy/reference/**` files, 5 of them fitness-bearing, and the byte-equality baseline for `py/ga.py` - is taken at commit **`181453f986cce7941addf9092e38c6e005e18844`** (the feature's base, i.e. the merge of PR #35 that delivered P0-01). Requirements that say "the pinned base commit" mean exactly that SHA; a different base invalidates the counts and must be re-derived, never silently substituted.

---

## Falsifiable Expectation Table (80-run replay, v0-legacy vs v0-corrected)

Unit: **a run = one (profile, seed) pair; 20 seeds x 4 frozen profiles = 80 runs**, run_ids 0-19 = seeds 42-61 (VC-20). Both drivers run in the same declared environment. Rows 1-4 were measured on full-protocol replays during the SPEC rounds and re-verified independently in review round 2 (conservador seed 42 bit-identity; caio seed 42 divergence); the coverage of those spot-checks is exactly what the 80-run proof is meant to widen. Any violation is `unexpected divergence` = FAIL (VC-21). Thresholds are stated over the 80-run denominator and the proof summary carries that denominator.

| # | Expectation over the 80 runs | Provenance of the expectation | Violation means |
| --- | --- | --- | --- |
| E1 | reported fitness (legacy) == reported fitness (corrected), bit-identical, on 80/80 runs | TRIAGE 5/5 + spot-check 2/2 + round-2 conservador seed-42 replay bit-identical to archive; mechanical consequence of the hunk (strict `>`, RNG order unchanged) | The relocation changed more than the capture; the "same protocol" claim is false |
| E2 | recomputed fitness (corrected) == reported fitness (corrected) on 100% of rows | Spot-check exact on conservador run 0 (4.480605754022773 both); follows from VC-06 | The corrected bundle still mispairs fitness and chromosome |
| E3 | ticker sets (legacy) != ticker sets (corrected) on more than half of the runs (>= 41/80; mechanism predicts 80/80) | TRIAGE "differ on most runs"; spot-check 2/2 differ (e.g. run 0 legacy `BRFS3..USIM5` vs corrected `BBSE3..UNIP6`) | The defect would not change what is returned, undermining its materiality |
| E4 | recomputed fitness (legacy) != reported fitness (legacy) on >= 1 run (mechanism predicts most runs) | TRIAGE >= 1 of 5; 120/120 artifact rows mismatch (round-2 re-measurement); spot-check run 0: reported 4.480605754022773 vs recomputed 0.8156341406041691 | The legacy artifacts would recompute clean, contradicting the 120/120 measurement |

Rows are recorded per run in the proof JSON; the summary records the computed denominator and which of E1-E4 held over it.

---

## Edge Cases and Failure Modes (design against)

- **NaN SCORE rows** (measured over frozen ranked frames: 15/107 conservador, 30/186 moderado, 57/260 arrojado, 42/202 caio): pandas `sum` skips NaN, so recomputation must apply the frozen function's semantics exactly. Recomputing as `n * score_mean` is wrong whenever any selected row has NaN SCORE and SHALL never be used (VC-11). The enumeration MUST include NaN SCORE rows (VC-07).
- **Duplicate SCORE ties**: sort order and `argmax` first-max resolution become part of the protocol. The enumeration MUST include tied frames and assert the recorded winner is the evaluated individual at the argmax index of its own generation (VC-06, VC-07).
- **pop_size parity**: `range(pop_size // 2)` pairs produce `2 * (pop_size // 2)` children; an odd `pop_size` would shrink the population and desynchronize `np.random.choice(pop_size, p=probs)`. The four frozen configs are even (200/250/300/276); the live fifth config `caio_last` is also even (230), so parity holds there too. The enumeration MUST assert the population length stays `pop_size` every generation so the recorded index always addresses an evaluated individual (VC-07).
- **Empty / invalid mask**: `mask.sum() != n` yields `-inf` and such a chromosome must never be recorded as a generation best. On the verification side, a row whose chromosome cannot be reconstructed as exactly `n` distinct rows FAILS the artifact - it is never skipped and never removed from the 100% denominator (VC-10, VC-26). On the execution side the frozen probs computation turns any `-inf` score into all-NaN probabilities (measured round 2: `np.random.choice` raises `probabilities contain NaN`), so the property test SHALL NOT feed infeasible masks through `run_ga`'s selection path; infeasible masks are exercised only against the pure `fitness()` function and the verifier fixture (VC-08).
- **Duplicate TICKER rows in a ranked frame** would make sorted-ticker reconstruction ambiguous; the verifier SHALL assert TICKER uniqueness per ranked frame and FAIL loudly instead of silently mis-reconstructing (VC-10).
- **`__pycache__` and import-time write traps**: importing bundle code writes `*.pyc` under `baselines/`, and `ga_multiple_runs.py:24-25` creates `outputs/` at IMPORT time - either turns a frozen bundle into "DRIFT - unexpected" in its manifest gate. Every importer sets `sys.dont_write_bytecode = True` and runs inside a scratch root; both manifest gates re-run after the suite (VC-23, VC-28).
- **Data/score provenance drift** (measured round 2): `outputs/` rows mix corrected-producer files (`generations_run`/`converged_early` present) with legacy-producer files, and their `score_median` disagrees with frozen-snapshot recomputation 115/150 (conservador) and 150/150 (caio) - provenance drift dominates there, so a blanket "affected by the P0-02 defect" mapping is forbidden and each entry's verdict comes from its evidence fields (VC-12, VC-24). The frozen caio data has 202 rows vs 139 at HEAD `data/processed/fundamentals_clean_caio.csv` - a live-tree fact about the dataset, NOT the explanation of the caio score gap, which P0-01 proved is a scoring-configuration gap on the archived dataset itself (caio 0/7 weights).

---

## What This Feature Does NOT Claim

- It does not make the legacy archive reproducible. Where archived v0-legacy values cannot be reached in the declared environment (caio rows measured MISMATCH), the note says so and attributes the gap to a scoring configuration/version the archive does not record - the archived data and the RNG alone do not explain it (P0-01: declared weights reproduce control SCOREs 15/15, 10/10, 10/10 but caio 0/7, missing by up to 0.42).
- It does not change the interpretation of portfolio returns or any backtest outcome beyond the reported-fitness defect. For the control profiles only the fitness column is declared `invalid_fitness`; caio's fitness is declared `score_provenance_inconsistent` (not verifiable); returns, tickers and hhi stand or fall on their own measured self-consistency (hhi recomputes from row tickers 120/120).
- It does not re-rank, re-select, or re-balance the production sleeve. `outputs/carteira_caio_consensus.json` stays byte-identical; its membership differs from the archived consensus as a provenance difference - the production file was produced by the corrected optimizer over a 150-run set the archive does not contain (FREQUENCY denominators exact over 150), while the archived consensus came from the defective capture - and that difference is not evidence that the live recommendation is defective.
- It does not repair `baselines/v0-legacy/code/ga.py`; the frozen defect remains the comparison subject.

---

## Test List (mapped to ACs; interpreter per test)

| # | Test / gate | Maps to | Command | Interpreter |
| --- | --- | --- | --- | --- |
| 1 | Full suite (structural + P0-01 baseline) | all | `python3 -m unittest discover -s tests -p 'test_v0_*.py'` | system python3 3.14.4 (stdlib); numpy/pandas tests SKIP with reason |
| 2 | Full suite incl. execution tests (property enumeration, recompute) | VC-01, VC-06 to VC-08, VC-11 | `/tmp/opencode/tcc-venv/bin/python -m unittest discover -s tests -p 'test_v0_*.py'` | tcc-venv (numpy 2.5.4 / pandas 3.1.0) |
| 3 | v0-legacy manifest gate | VC-15 | `python3 baselines/v0-legacy/tools/verify_manifest.py` | system python3 (stdlib) |
| 4 | v0-corrected manifest gate + scratch-root/no-write check | VC-04, VC-19, VC-23, VC-28 | `python3 baselines/v0-corrected/tools/verify_manifest.py` | system python3 (stdlib) |
| 5 | Artifact verifier (100% row recompute + FAIL-branch fixture) | VC-01, VC-10, VC-11, VC-26 | `/tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/verify_artifacts.py` | tcc-venv |
| 6 | 20-seed x 4-profile legacy-vs-corrected comparison (opt-in, ~75 min measured) | VC-20 to VC-22 | `TCC_V0_CORRECTED_REPLAY=1 /tmp/opencode/tcc-venv/bin/python baselines/v0-corrected/tools/compare_replay.py` | tcc-venv |
| 7 | Property test (instrumented real `run_ga` + negative control) | VC-06 to VC-08 | `tests/test_v0_corrected_property.py` inside commands 1-2 | tcc-venv (skips on system python3) |
| 8 | Artifact/summary consistency (`n_runs` == CSV rows) | VC-09 | `tests/test_v0_corrected_artifacts.py` inside commands 1-2 | system python3 (stdlib csv/json) |
| 9 | Marking coverage + git-blob digests + verdict recompute + banner/note statement checks | VC-12 to VC-14, VC-16 to VC-18, VC-24 | `tests/test_v0_corrected_status.py` inside commands 1-2 | system python3 (stdlib) |
| 10 | Byte/AST/protocol equality proofs incl. `py/ga.py` byte-equality vs `git show 181453f` | VC-02 to VC-05 | `tests/test_v0_corrected_bundle.py` inside commands 1-2 | system python3 (stdlib hashing/ast/json) |
| 11 | Reference-vs-archive fitness identity + bounded divergence table + corrected-driver tie for replay seeds | VC-25, VC-29 | `tests/test_v0_corrected_bundle.py` inside commands 1-2 | system python3 (stdlib csv/json); the driver tie runs under tcc-venv |
| 12 | Replay checker negative fixture (injected expectation violation) | VC-27 | `tests/test_v0_corrected_replay.py` inside commands 1-2 | system python3 (stdlib) |

Gate for completion: commands 1-5 green, command 6 run once with its proof committed (budgeted 1-2 h opt-in), both manifest gates exit 0.

---

## Deliverables (for Design-phase sizing)

**Edited (existing files):**

- `py/ga.py` - the record-block relocation via the canonical (old_bytes, new_bytes) pair (VC-03)
- `baselines/v0-legacy/README.md` - INVALID banner in the "Two numbers that do not agree, on purpose" area, referencing the defect and `status.json` (VC-16; file is manifest-excluded, no byte pin breaks)
- `.specs/STATE.md` - AD-007..AD-015 (done in Specify)

**Created:**

- `baselines/v0-corrected/README.md` - reader's guide + methodological note with the verbatim statement list (VC-17, VC-18)
- `baselines/v0-corrected/status.json` - per-artifact/per-field verdict map with evidence fields (VC-12, VC-24)
- `baselines/v0-corrected/protocol.json` - v0-legacy protocol + `baseline_id` + `derived_from` (VC-05)
- `baselines/v0-corrected/environment.txt` + `requirements.txt` - `python -VV`, numpy/pandas versions, `pip freeze`, install command, wheels caveat (VC-19)
- `baselines/v0-corrected/code/ga.py` - frozen bytes + canonical transform (VC-02)
- `baselines/v0-corrected/code/profiles.py`, `baselines/v0-corrected/code/ga_multiple_runs.py` - byte-identical copies (VC-04)
- `baselines/v0-corrected/data/processed/` - 5 byte-identical copies (VC-04)
- `baselines/v0-corrected/reference/` - corrected results: `metrics_stability_{conservador,moderado,arrojado,caio}.csv`, `multiple_runs_summary.json`, `carteira_*_consensus.json` (VC-01, VC-09, VC-25)
- `baselines/v0-corrected/baseline.json` - own manifest (VC-04, command 4)
- `baselines/v0-corrected/tools/` - `build_manifest.py`, `verify_manifest.py`, `verify_artifacts.py`, `compare_replay.py`, `_corrected_driver.py` (VC-01, VC-10, VC-20, VC-28)
- `tests/test_v0_corrected_bundle.py`, `tests/test_v0_corrected_property.py`, `tests/test_v0_corrected_artifacts.py`, `tests/test_v0_corrected_status.py`, `tests/test_v0_corrected_replay.py` (commands 7-12)
- `.specs/features/p0-02-v0-corrected/proofs/replay_20seeds.json` - committed replay proof with the 80-run denominator (VC-20)
- `.specs/features/p0-02-v0-corrected/proofs/reference_divergences.json` - committed divergence table (VC-25)

**Canonical transform for VC-02 / VC-03** (exact byte pair, LF-terminated; `old_bytes` occurs exactly once in `py/ga.py` and in `baselines/v0-legacy/code/ga.py` - verified at round 2 - so the same pair applies to both, and the transform region is byte-identical between the two files):

`old_bytes`:

```text
        population = new_pop

        # regista melhor
        gen_best_idx = int(scores.argmax())
        if scores[gen_best_idx] > best_fitness:
            best_fitness = scores[gen_best_idx]
            best_chrom = population[gen_best_idx].copy()
```

`new_bytes`:

```text
        # regista melhor
        gen_best_idx = int(scores.argmax())
        if scores[gen_best_idx] > best_fitness:
            best_fitness = scores[gen_best_idx]
            best_chrom = population[gen_best_idx].copy()

        population = new_pop
```

Proof obligations: (a) `baselines/v0-corrected/code/ga.py == old->new applied to baselines/v0-legacy/code/ga.py` byte-for-byte (0-byte delta); (b) the corrected `py/ga.py == old->new applied to git show 181453f:py/ga.py` byte-for-byte (0-byte delta), which kills any sneaky change to probs, `>`, or seeding.

---

## Requirement Traceability

| Requirement ID | Story | Issue trace | Phase | Status |
| -------------- | ----- | ----------- | ----- | ------ |
| VC-01 | P1: corrected bundle | AC1 | Design | Pending |
| VC-02 | P1: corrected bundle | AC3 | Design | Pending |
| VC-03 | P1: corrected bundle | AC3 | Design | Pending |
| VC-04 | P1: corrected bundle | AC3 | Design | Pending |
| VC-05 | P1: corrected bundle | AC3 | Design | Pending |
| VC-06 | P1: corrected bundle | Test 1 (property) | Design | Pending |
| VC-07 | P1: corrected bundle | Test 1 (property) | Design | Pending |
| VC-08 | P1: corrected bundle | Test 1 (property) | Design | Pending |
| VC-09 | P1: corrected bundle | Test 2 (artifact) | Design | Pending |
| VC-10 | P1: corrected bundle | AC1 | Design | Pending |
| VC-11 | P1: corrected bundle | AC1 | Design | Pending |
| VC-25 | P1: corrected bundle | AC3 | Design | Pending |
| VC-29 | P1: corrected bundle | AC3 | Design | Pending |
| VC-26 | P1: corrected bundle | AC1 | Design | Pending |
| VC-12 | P2: invalidation marking | AC2 | Design | Pending |
| VC-13 | P2: invalidation marking | AC2 | Design | Pending |
| VC-14 | P2: invalidation marking | AC2 | Design | Pending |
| VC-15 | P2: invalidation marking | AC2 | Design | Pending |
| VC-16 | P2: invalidation marking | AC2 | Design | Pending |
| VC-24 | P2: invalidation marking | AC2 | Design | Pending |
| VC-28 | P2: invalidation marking | AC2 | Design | Pending |
| VC-17 | P3: note + replay | AC4 | Design | Pending |
| VC-18 | P3: note + replay | AC4 | Design | Pending |
| VC-19 | P3: note + replay | AC4 | Design | Pending |
| VC-20 | P3: note + replay | Test 3 (replay) | Design | Pending |
| VC-21 | P3: note + replay | Test 3 (replay) | Design | Pending |
| VC-22 | P3: note + replay | Test 3 (replay) | Design | Pending |
| VC-23 | P3: note + replay | AC2 | Design | Pending |
| VC-27 | P3: note + replay | Test 3 (replay) | Design | Pending |

**Coverage:** 29 total, 0 mapped to tasks, 29 unmapped ⚠️ (Tasks phase comes after human approval). Issue AC1 traced by VC-01/10/11/26; AC2 by VC-12..16/24/28 (+VC-23); AC3 by VC-02..05/25/29; AC4 by VC-17/18/19; Test 1 by VC-06..08; Test 2 by VC-09; Test 3 by VC-20..22/27. No orphan ACs; no issue AC or test left untraced. ID map (round 2): VC-01..VC-23 all retained (no renumbering; text revised on VC-02, VC-03, VC-05, VC-06, VC-08, VC-11, VC-12, VC-13, VC-14, VC-17, VC-18, VC-19, VC-20, VC-21, VC-23; VC-19 retagged AC3->AC4, VC-23 retagged AC3->AC2); VC-24..VC-28 added for the attribution rule (B1), reference<->code<->protocol closure (B6), the two negative fixtures (N5) and generator hygiene (N7). No requirement was replaced, so no old->new renumbering applies. Closing review round 2 (F1, F2): VC-12/VC-24 now enumerate `row_reconstruction`, `score_recompute` and `data_snapshot_match` with a defined column scope for `score_recompute`; the Verdict Attribution Rule gained the `data_snapshot_match == false` discriminator above the corrected-capture FAIL row; VC-25 gained the bounded divergence table (caio rows only); and **VC-29** ties the committed reference to `_corrected_driver.py` output for every replay run, so a relabelled legacy reference fails.

---

## Success Criteria

- [ ] `verify_artifacts.py` exits 0 over `v0-corrected/reference/`: 100% of rows have reported == recomputed fitness (VC-01).
- [ ] `status.json` covers every strict `fitness` locator in `baselines/v0-legacy/reference/**` and the pinned `outputs/` tree exactly once, digests equal the git-blob computation at the pinned base, and every verdict recomputes from its evidence fields (VC-12 to VC-14, VC-24).
- [ ] Both manifest gates exit 0 and the P0-01 suite (52 tests) stays green (VC-15, VC-23, VC-28).
- [ ] The bundle equality proofs pass, including `py/ga.py` byte-equality against the canonical transform of `git show 181453f:py/ga.py` (VC-02 to VC-05).
- [ ] The corrected reference fitness is bit-identical per (run_id, seed) to the legacy reference for the three control profiles, the divergence table contains caio rows only and exactly the divergent ones, and every replay run's committed row equals the corrected driver's row (VC-25, VC-29).
- [ ] The 80-run proof JSON matches expectation rows E1-E4 over its recorded denominator and is committed (VC-20, VC-21).
