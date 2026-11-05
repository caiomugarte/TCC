# Protection Zone Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/protection-zone/spec.md`
**Design**: `.specs/features/protection-zone/design.md`
**Status**: Draft (awaiting orchestrator review)
**Branch**: `t3/issue-46` (PR base `develop`). Merge with a merge commit, never squash (AD-023: #9's lot commits must stay reachable).

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `.specs/codebase/TESTING.md` (deterministic functions on small synthetic frames, no network), `.specs/codebase/CONVENTIONS.md`, `.specs/project/STATE.md` AD-0002 (a policy-version bump lands with its pin-test update). No `AGENTS.md`, `CONTRIBUTING.md`, `REVIEW.md`, CI workflow, coverage threshold or linter config exists. Runner: stdlib `unittest` (pytest is not installed in the venv). Strong defaults applied.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Core domain (`py/core/protection_zone.py`) | unit | All branches; 1:1 to PZ-01..PZ-15; every spec edge case; property-like cases listed in Test Strategy | `tests/test_protection_zone.py` | `$PY -m unittest tests.test_protection_zone` |
| AD-022 oracle over committed lots | unit (data-backed) | PZ-13: synthetic equivalence grid + the four recommended-point turnovers | `tests/test_protection_zone_equivalence.py` | `$PY -m unittest tests.test_protection_zone_equivalence` |
| Selectors (`py/pipelines/multi_run.py`, `py/fii_selection.py`) | unit | Frame equality without incumbents (rows, order, columns, values, attrs); zone and HHI-fallback paths; result keys | `tests/test_protection_zone_selectors.py` | `$PY -m unittest tests.test_protection_zone_selectors` |
| API policy (`premium_policy.py`) | unit | PZ-21..PZ-23 incl. v2 rejection and Basic byte-identity | `api/tests/test_premium_policy.py` | API suite |
| API repository (`recommendations.py`) | integration (SQLite session, as the existing tests) | Account scoping, plan, status, ordering, exclusion of current run | `api/tests/test_recommendation_repository.py` | API suite |
| API executor (`premium_executor.py`) | integration (fake runner + DB) | PZ-24..PZ-27 incl. unreadable `result_json` shapes | `api/tests/test_premium_executor.py` | API suite |
| API adapter (`premium_optimization.py`) | unit (fake engines) | PZ-28..PZ-31, PZ-16 stocks; every `status` value | `api/tests/test_premium_optimization.py` | API suite |
| API end-to-end (service + executor + adapter, fake engines) | integration | Two sequential v3 runs; Basic run untouched | `api/tests/test_premium_integration.py` | API suite |
| Measurement script (`scripts/protection_zone_turnover.py`) | unit (fake engines, fixture manifest) | PZ-32 output shape, PZ-33 exit 3 cases, PZ-34 byte-identity | `tests/test_protection_zone_turnover.py` | `$PY -m unittest tests.test_protection_zone_turnover` |
| Docs (`docs/validation/README.md`) | none | build gate only | - | build gate only |

`$PY` = `/home/caio/Documents/projetos/TCC/.venv/bin/python` (Python 3.14.4 with numpy and pandas; the venv lives in the main checkout and is shared by worktrees). Run every command from the worktree root `/home/caio/.t3/worktrees/TCC/t3-issue-46`.

## Gate Check Commands

> Generated from codebase - confirm before Execute. Baseline at `76a6d70`: root suite 472 tests OK (2 skipped) in 171 s; API suite 125 tests OK in 72 s.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | Phase 1 tasks and T15 | `$PY -m unittest tests.test_protection_zone tests.test_protection_zone_equivalence` (T15 adds `tests.test_protection_zone_turnover`) |
| Selector | Phase 2 tasks | `$PY -m unittest tests.test_protection_zone tests.test_protection_zone_equivalence tests.test_protection_zone_selectors tests.test_fii_selection tests.test_multi_run_backtest tests.test_missing_score_exclusion tests.test_convergence_analysis tests.test_multirun_convergence` |
| Full | Phase 3 tasks | `cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t . ; cd ..` |
| Build | Last task of each phase, docs-only tasks | `$PY -m compileall -q py api/app scripts && $PY -m unittest discover -s tests -p "test_*.py" && (cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .)` (no linter is configured; `compileall` is the syntax gate) |

Test counts never go down: each task's "Done when" states the expected minimum.

---

## Test Strategy

**Core function (T1-T3), property-like cases on synthetic dicts.** Each is a named test:

| Case | Asserts | Req |
| ---- | ------- | --- |
| No incumbents == consensus | For a grid of rankings, `apply_protection_zone([], ...)` equals the first `n_assets` of `rank_candidates` | PZ-10 |
| `K = 0` == consensus | Same with incumbents present; incumbents inside the consensus get `mantido` | PZ-10, PZ-04 |
| Protected stays | Incumbent at rank `K`, no candidate with `f >= f(p) + δ`: kept, `mantido` | PZ-03 |
| Rank `K + 1` leaves | Same incumbent at rank `K + 1`: `saiu: fora do top K` | PZ-07 |
| δ boundary inclusive | `f(c) - f(p) == δ` exactly in rationals (run_count 100, counts 21 vs 10, δ 0.11) swaps; one count less does not | PZ-05 |
| δ = 0 tie favors incumbent | Equal frequency, incumbent ahead by score: no swap; candidate ahead by score: swap | PZ-05 |
| Repeated swaps stop at first failure | Two strong outsiders and two weak incumbents: both swapped, weakest first; a third weaker outsider stops the loop | PZ-05 |
| Replaced never re-admitted | A swapped-out incumbent never re-enters in the same call | PZ-05 |
| Ineligible leaves | Incumbent absent from `eligible` but present in frequencies: never protected, `saiu: inelegível` | PZ-06 |
| Dedup, `POMO3` ranks above `POMO4` | Incumbent `POMO4` has no rank: it leaves with `saiu: fora do top K`; outsider `POMO3` enters as `novo`. A same-company swap never happens: the ranking holds one ticker per company (PZ-02, AD-022) | PZ-09 |
| Dedup, `POMO4` ranks above `POMO3` | Incumbent `POMO4` is protected (`mantido`); outsider `POMO3` is never selected, even with vacancies | PZ-09 |
| Capacity cap | 15 qualifying incumbents, `n_assets = 13`: 13 best-ranked kept, 2 `saiu: vagas reduzidas` | PZ-08 |
| Order independence | Shuffled incumbent lists give equal results; lower-case and duplicate tickers normalize | PZ-12 |
| Partition | Every incumbent in exactly one of `kept` or `departures`; `turnover == len(entered)`; `exits == len(departures)` | PZ-12, PZ-01 |
| Invalid input | `K = -1`, `K = 1.5`, `δ = nan`, `δ = -0.01`, `n_assets = 0`, frequency `1.2`: `ValueError` naming the field | PZ-11 |
| HHI fallback | `hhi_of` above `hhi_max + 1e-12`: pure consensus, `buffer_fallback == "hhi"`, `status == "fallback_hhi"`; exactly at `hhi_max + 1e-12`: no fallback | PZ-14 |
| No limit | `hhi_max=None`: `hhi_of` never called, `buffer_fallback is None` | PZ-15 |

**Oracle (T4).** `scripts/convergence_analysis.zone_portfolio` vs `apply_protection_zone` on a seeded random grid (incumbent sets of size `n`, all eligible, K in `{0, n..30}`, δ in the #9 grid, both selectors), then over the committed lots at prefix 150 and the recommended `K`, `δ`: mean / max turnover per cell equal 0.40 / 1, 0.10 / 1, 0.70 / 2, 0.00 / 0.

**Regression (T5, T7).** Fixed `results`/`portfolios` fixtures: the frame returned with `incumbents` omitted, `None` and `[]` is `pandas.testing.assert_frame_equal` to the frame from the pre-change function body (kept verbatim as a test helper copied from `76a6d70`) and has equal `attrs`. Existing suites (`test_fii_selection`, `test_multi_run_backtest`, `test_missing_score_exclusion`, `test_convergence_analysis`, `test_multirun_convergence`) run unchanged in the Selector gate; `test_convergence_analysis` calls `build_consensus_portfolio` and `_consensus` through `production_consensus`.

**API (T9-T14).** Existing fake-runner tests must pass untouched (no incumbent run exists in them, so the runner is called as today). New tests seed a completed run's `result_json` directly through the repository.

**Mutation targets for the Verifier's discrimination sensor**: the `- 1e-12` tolerance, `>=` vs `>` in the swap, `rank <= K` vs `<`, the dedup skip during swap, the eligibility check, the HHI `>` comparison, the `exclude_run_id` filter, the `plan == "premium"` guard, the Basic `protection_zone` omission.

---

## Rollback

- **Code**: revert the PR's merge commit on `develop`. No migration and no new table, so no DB rollback.
- **Persisted runs**: v3 runs stay readable after a revert. `policy_json.stocks.protection_zone` and `provenance.protection_zone` are extra keys in JSON columns typed `dict[str, Any]`; no reader validates them. Seeds return to the v2 derivation, so the next v2 run differs from a v3 run on the same inputs (expected).
- **Partial rollback**: setting `PROTECTION_ZONE` values to `K = 0` (via `k_offset = -n_assets`) is not supported; disabling means reverting. The zone has no runtime flag by design (`## Decisões`: always on for Premium).
- **Phase 2 artifacts**: `docs/validation/protection-zone-turnover.*` are generated files; delete and regenerate with `measure`.

---

## Execution Plan

Phases run in order; tasks within a phase run in order. Heavy tasks count as two toward a phase's size.

| Phase | Tasks | Size (heavy = 2) | Risk | Suggested tier |
| ----- | ----- | ---------------- | ---- | -------------- |
| 1. Core zone function | T1-T4 | 4 | Core domain, no persistence. Highest ambiguity: AD-022 fidelity and reason rules. | high reasoning |
| 2. Selector wiring | T5-T8 | 4 | Regression of the no-incumbent path; no persistence. | high reasoning (regression-sensitive) |
| 3. API policy, persistence, executor, adapter | T9-T14 | 6 | **High**: reads `result_json` of previous runs, bumps `POLICY_VERSION`, extends the provenance contract. | high reasoning |
| 4. Month-to-month measurement | T15-T17 | 4 (T17 heavy) | Medium for T15-T16; T17 is heavy and blocked on data. | mid tier for T15-T16; T17 operator-run |

### Phase 1: Core zone function

```
T1 → T2 → T3
T2 → T4
```

### Phase 2: Selector wiring

```
T5 → T6
T7 → T8
```

### Phase 3: API policy, persistence, executor, adapter

```
T9 → T11
T10 → T11
T9 → T12 → T13
T11 → T14
T13 → T14
```

### Phase 4: Month-to-month measurement

```
T15 → T16 → T17
```

---

## Task Breakdown

### Phase 1 tasks: Core zone function

#### T1: Ranking, base company and input validation

**Status**: Complete

**What**: Create `py/core/protection_zone.py` with `base_company`, `rank_candidates`, the `ZoneSelection` dataclass and the input validator used by `apply_protection_zone`.
**Where**: `py/core/protection_zone.py` (new)
**Depends on**: None
**Reuses**: sort key and suffix list of `py/pipelines/multi_run.py:225-239`
**Requirement**: PZ-01, PZ-02, PZ-11
**Risk area**: none (pure function)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] `rank_candidates` sorts by `(-frequency, -score, ticker)` over the mapping's order and dedups by base company only when asked; a NaN-score fixture reproduces the order `build_consensus_portfolio` gives on the same counts
- [x] `base_company` strips the first matching suffix of `("11", "3", "4", "5", "6")` (`TAEE11` → `TAEE`, `POMO4` → `POMO`, `ABC` → `ABC`)
- [x] Invalid `K`, `δ`, `n_assets`, frequency raise `ValueError` naming the field
- [x] Gate check passes: Quick
- [x] Test count: at least 8 new tests in `tests/test_protection_zone.py`, all pass

**Tests**: unit
**Gate**: quick
**Commit**: `feat(core): add protection-zone ranking and input checks (#46)`

---

#### T2: `apply_protection_zone`

**Status**: Complete

**What**: Implement the AD-022 zone with A-3/A-4 (capacity cap, reasons, departures, turnover, exits).
**Where**: `py/core/protection_zone.py` (modify)
**Depends on**: T1
**Reuses**: step-by-step structure of `scripts/convergence_analysis.py:514-568`
**Requirement**: PZ-01, PZ-03, PZ-04, PZ-05, PZ-06, PZ-07, PZ-08, PZ-09, PZ-10, PZ-12
**Risk area**: none (pure function)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Every core row of the Test Strategy table except the HHI rows has a passing test
- [x] The function imports nothing from `scripts/` and nothing from pandas
- [x] Gate check passes: Quick
- [x] Test count: at least 16 new tests, T1 tests still pass

**Tests**: unit
**Gate**: quick
**Commit**: `feat(core): apply the protection zone to the consensus ranking (#46)`

---

#### T3: `select_with_fallback` and the class record

**Status**: Complete

**What**: Add the HHI fallback wrapper and the JSON-ready class record builder.
**Where**: `py/core/protection_zone.py` (modify)
**Depends on**: T2
**Reuses**: tolerance `1e-12` of `py/fii_selection.py:735`
**Requirement**: PZ-14, PZ-15, PZ-16 (zone within the limit is used), PZ-28 (record fields produced by the core)
**Risk area**: provenance contract (defines the class record's field names)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] HHI rows of the Test Strategy pass, including the boundary at exactly `hhi_max + 1e-12`
- [x] The record has exactly the fields of design "Class record", with `replaced` as `[{"out", "in"}]`, sorted lists, `consensus_tickers` and `consensus_turnover`, and is `json.dumps(..., allow_nan=False)`-serializable
- [x] Gate check passes: Quick
- [x] Test count: at least 5 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(core): fall back to the pure consensus when the zone breaks HHI (#46)`

---

#### T4: AD-022 equivalence over the committed lots

**Status**: Complete

**What**: Prove the new function reproduces `zone_portfolio` and #9's recommended-point turnover.
**Where**: `tests/test_protection_zone_equivalence.py` (new)
**Depends on**: T2
**Reuses**: `scripts/convergence_analysis.py` `load_protocol`, `load_lots`, `SeedSeries`, `_ordered_pairs`, `zone_portfolio`; lots in `docs/validation/multirun-convergence/lots/`
**Requirement**: PZ-13
**Risk area**: none (reads committed data, writes nothing)
**Heavy**: no (reads about 4 MB of committed lots; expected well under a minute)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Seeded random grid: identical portfolio set and turnover to `zone_portfolio` for both selectors
- [x] Lots at prefix 150, recommended `K = n + k_offset` and `δ = 0.11`: mean / max turnover 0.40 / 1, 0.10 / 1, 0.70 / 2, 0.00 / 0 for the four cells
- [x] Lots and report files unchanged (`git status --porcelain docs/validation` empty after the run)
- [x] Gate check passes: Build (end of Phase 1)
- [x] Test count: root suite at least 472 + Phase 1 additions

**Tests**: unit
**Gate**: build
**Commit**: `test(core): check the protection zone against AD-022 on the #9 lots (#46)`

---

### Phase 2 tasks: Selector wiring

#### T5: Stock consensus applies the zone

**Status**: Complete

**What**: Add the keyword-only `incumbents` to `build_consensus_portfolio` and the zone branch after the pure consensus.
**Where**: `py/pipelines/multi_run.py` (modify `build_consensus_portfolio` only)
**Depends on**: T3
**Reuses**: existing frame construction and attrs; `core.metrics.hhi_sector`
**Requirement**: PZ-14, PZ-17, PZ-18
**Risk area**: none (regression of an existing path)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] With `incumbents` omitted, `None` or `[]`, the frame equals the `76a6d70` body's frame (`assert_frame_equal` + equal `attrs`)
- [x] With incumbents and a zone config: rows are the zone tickers, `attrs["protection_zone"]` holds the record, sort by `FREQUENCY` desc
- [x] HHI fallback from `selection_config["liquidity_and_size_filters"]["hhi_max"]` covered
- [x] Gate check passes: Selector
- [x] Test count: at least 5 new tests in `tests/test_protection_zone_selectors.py` (8 tests, all pass)

**Tests**: unit
**Gate**: selector
**Commit**: `feat(stocks): apply the protection zone in the consensus portfolio (#46)`

---

#### T6: Stock pipeline forwards incumbents

**Status**: Complete

**What**: Add `incumbents` to `run_multi_execution_profile` and expose `protection_zone` in the explicit-mode result.
**Where**: `py/pipelines/multi_run.py` (modify `run_multi_execution_profile` only)
**Depends on**: T5
**Reuses**: existing explicit-mode return dict
**Requirement**: PZ-20
**Risk area**: provenance contract (result key consumed by the adapter)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Explicit-mode result has `protection_zone` only when incumbents were applied; no key otherwise
- [x] Test runs 2 GA runs on a small fixture (the one of `tests/test_missing_score_exclusion.py`, which calls this function; `tests/test_multi_run_backtest.py` does not), not a long GA
- [x] Gate check passes: Selector
- [x] Test count: at least 2 new tests (3 new, all pass)

**Tests**: unit
**Gate**: selector
**Commit**: `feat(stocks): pass incumbents through the multi-run pipeline (#46)`

---

#### T7: FII consensus applies the zone

**Status**: Complete

**What**: Add `incumbents`, `protection_zone`, `hhi_max` keywords to `_consensus` and the zone branch without dedup.
**Where**: `py/fii_selection.py` (modify `_consensus` only)
**Depends on**: T3
**Reuses**: existing sort and frame construction
**Requirement**: PZ-14, PZ-17, PZ-19
**Risk area**: none (regression of an existing path)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] With no incumbents the frame equals the `76a6d70` body's frame
- [x] Zone path sorted by `FREQUENCY`, `SCORE` desc, `TICKER` asc; no base-company dedup (tested with `AAAA11` and `AAAA3`, which the stock rule would merge)
- [x] Gate check passes: Selector
- [x] Test count: at least 4 new tests (5 new, all pass)

**Tests**: unit
**Gate**: selector
**Commit**: `feat(fiis): apply the protection zone in the FII consensus (#46)`

---

#### T8: FII pipeline forwards incumbents

**Status**: Complete

**What**: Add `incumbents` to `run_fii_selection`, resolve `hhi_max` before `_consensus`, keep the post-check, expose `protection_zone`.
**Where**: `py/fii_selection.py` (modify `run_fii_selection` only)
**Depends on**: T7
**Reuses**: existing `hhi_max` lookup at `:730-736`
**Requirement**: PZ-16, PZ-20
**Risk area**: provenance contract (result key)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Fallen-back consensus that still violates HHI raises `FiiSelectionError` (PZ-16; the test stubs `optimize_fii_portfolio` so every run picks one segment)
- [x] Result has `protection_zone` only when applied
- [x] Existing `tests/test_fii_selection.py` passes unchanged (23 tests OK)
- [x] Gate check passes: Build (end of Phase 2): root 529 OK (2 skipped), API 125 OK
- [x] Test count: at least 3 new tests (3 new, all pass)

**Tests**: unit
**Gate**: build
**Commit**: `feat(fiis): pass incumbents through the FII selection (#46)`

---

### Phase 3 tasks: API policy, persistence, executor, adapter

#### T9: Premium policy v3 with the zone values

**What**: Bump `POLICY_VERSION`, add `SelectorPolicy.protection_zone` and `PROTECTION_ZONE`, set it for Premium, omit it for Basic.
**Where**: `api/app/services/premium_policy.py` (modify)
**Depends on**: None
**Reuses**: `dataclasses.replace` pattern of `resolve_basic_policy`
**Requirement**: PZ-21, PZ-22, PZ-23
**Risk area**: **policy version** (persisted `policy_version`, seed derivation)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] `api/tests/test_premium_policy.py:269`, `:280` pins updated to `premium-policy-v3`; new test: `premium-policy-v2` rejected
- [x] Premium `stocks.protection_zone == {"k_offset": 6, "K": n + 6, "delta": 0.11}`, FIIs `k_offset` 3, for conservador, moderado and arrojado anchors
- [x] Basic `to_json()` has no `protection_zone` and equals a golden string captured at `76a6d70` for one fixture profile
- [x] Gate check passes: Full
- [x] Test count: API suite at least 125 + new tests

**Tests**: unit
**Gate**: full
**Commit**: `feat(policy): add protection-zone values and bump to premium-policy-v3 (#46)`

---

#### T10: Latest completed Premium run of an account

**What**: Add `RecommendationRepository.get_latest_completed_for_account`.
**Where**: `api/app/repositories/recommendations.py` (modify)
**Depends on**: None
**Reuses**: ordering of `get_latest_for_profile`
**Requirement**: PZ-24
**Risk area**: **persistence read** (`recommendation_runs`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Tests: other account's run ignored; `basic` ignored; `failed`/`running` ignored; latest by `created_at` then `id`; `exclude_run_id` honored; across profile revisions of the same account
- [x] Gate check passes: Full
- [x] Test count: at least 5 new tests in `api/tests/test_recommendation_repository.py`

**Tests**: integration
**Gate**: full
**Commit**: `feat(api): look up the latest completed Premium run of an account (#46)`

---

#### T11: Executor loads and passes incumbents

**What**: In `execute_run`, read the incumbent run's tickers and call the runner with `incumbents=` only when one exists.
**Where**: `api/app/services/premium_executor.py` (modify)
**Depends on**: T9, T10
**Reuses**: first session block of `execute_run`
**Requirement**: PZ-24, PZ-25, PZ-26, PZ-27
**Risk area**: **`result_json` reads** of previous runs
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] No prior run: runner receives exactly the current arguments (existing fixed-signature fakes pass untouched)
- [x] Prior completed Premium run: `incumbents == {"run_id", "stocks", "fiis"}` with upper-cased, deduplicated tickers
- [x] Unreadable shapes (`None`, not a dict, class not a list, entry without string `ticker`) give `None` for that class; run completes
- [x] Basic run: repository lookup never called (spy)
- [x] Gate check passes: Full
- [x] Test count: at least 6 new tests

**Tests**: integration
**Gate**: full
**Commit**: `feat(api): pass the previous Premium portfolio to the optimizer (#46)`

---

#### T12: Adapter forwards zone config and incumbents to engines

**What**: Accept `incumbents` in `run_premium_optimization`, copy `protection_zone` into selector configs, forward `incumbents=` per class, serialize policies through `to_dict()`.
**Where**: `api/app/adapters/premium_optimization.py` (modify)
**Depends on**: T9, T6, T8
**Reuses**: `_selector_config`, `_default_stock_engine`, `_json_value`
**Requirement**: PZ-16, PZ-30
**Risk area**: provenance contract (`result["policy"]` serialization)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Fake engines record kwargs: `incumbents` present only for classes with a non-empty list; config has `protection_zone`
- [x] `_default_stock_engine` forwards `incumbents` to `run_multi_execution_profile` (patched)
- [x] Basic dataclass policy: `result["policy"]` has no `protection_zone`
- [x] Zone-fallback stock result that still breaks HHI yields `infeasible_constraints` (PZ-16)
- [x] Gate check passes: Full
- [x] Test count: at least 4 new tests

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): forward protection-zone settings and incumbents to the selectors (#46)`

---

#### T13: Provenance record and constituent reasons

**What**: Write `provenance.protection_zone` with per-class status and append zone reasons to constituents.
**Where**: `api/app/adapters/premium_optimization.py` (modify)
**Depends on**: T12
**Reuses**: `_constituents`
**Requirement**: PZ-26, PZ-28, PZ-29, PZ-31
**Risk area**: **provenance contract**
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Each status covered: `applied`, `fallback_hhi`, `no_incumbents`, `incumbents_unreadable`, `class_not_selected`
- [ ] `incumbent_run_id` equals the passed run id, `null` without incumbents
- [ ] Reasons: `mantido`/`novo`/`substituiu <t>` appended; unchanged when the class has no incumbents (A-14)
- [ ] No `protection_zone` key in provenance when the policy declares none
- [ ] Gate check passes: Full
- [ ] Test count: at least 6 new tests

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): record protection-zone provenance and reasons per constituent (#46)`

---

#### T14: End-to-end two-run check

**What**: Service + executor + adapter with fake engines: run 1 has no incumbents, run 2 records run 1 as incumbent; a Basic run is untouched.
**Where**: `api/tests/test_premium_integration.py` (modify, new test class)
**Depends on**: T11, T13
**Reuses**: existing integration harness (`runner`, fake `selector`/`fii` engines)
**Requirement**: PZ-24, PZ-28
**Risk area**: **`result_json` reads + provenance contract**
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Run 2 `provenance.protection_zone.incumbent_run_id == run1.id`, `policy_version == "premium-policy-v3"`, per-class `turnover` integers
- [ ] Basic run's `result_json` has no `protection_zone`
- [ ] Gate check passes: Build (end of Phase 3)
- [ ] Test count: API suite at least 125 + Phase 3 additions

**Tests**: integration
**Gate**: build
**Commit**: `test(api): cover two sequential Premium runs with the protection zone (#46)`

---

### Phase 4 tasks: Month-to-month measurement

#### T15: Turnover measurement tool

**What**: Create `scripts/protection_zone_turnover.py measure` with injectable engines, exit codes and deterministic output.
**Where**: `scripts/protection_zone_turnover.py` (new)
**Depends on**: T13
**Reuses**: `scripts/multirun_convergence.py` policy/snapshot helpers, `scripts/convergence_analysis.py` `dumps`, `write_files_atomically`
**Requirement**: PZ-32, PZ-33, PZ-34
**Risk area**: none (offline tool; reads a manifest, writes `docs/validation/`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Fixture manifest + fake engines: JSON and Markdown hold per-class `turnover`, `consensus_turnover`, kept, swapped, departures, K, δ, manifest id and sha256, commit, seed
- [ ] Exit 3 and no files for: missing manifest, cutoff before 2026-11-01, id `dev-2026-10-29-ibov`
- [ ] Two runs give byte-identical files
- [ ] Test file `tests/test_protection_zone_turnover.py`; gate check passes: Quick + `tests.test_protection_zone_turnover`
- [ ] Test count: at least 5 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(validation): add the month-to-month protection-zone turnover tool (#46)`

---

#### T16: Document the pending measurement

**What**: Add the protection-zone turnover entry to `docs/validation/README.md`: the command, the inputs it needs and that the measurement is pending a November snapshot.
**Where**: `docs/validation/README.md` (modify)
**Depends on**: T15
**Reuses**: README's existing entry format
**Requirement**: PZ-35
**Risk area**: none
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer`

**Done when**:

- [ ] Entry states "pendente: snapshot de novembro" and writes no number
- [ ] Gate check passes: Build
- [ ] Test count: unchanged from T15

**Tests**: none
**Gate**: build
**Commit**: `docs(validation): describe the pending month-to-month turnover measurement (#46)`

---

#### T17: Measure October-to-November turnover (HEAVY, PENDING DATA)

**What**: Once a November manifest is published in the registry, run `measure` and commit the generated report.
**Where**: `docs/validation/protection-zone-turnover.json` (generated, with its `.md`)
**Depends on**: T16
**Reuses**: T15 tool
**Requirement**: PZ-32, PZ-35
**Risk area**: data provenance (report cites manifest sha256 and commit)
**Heavy**: **yes** (one production Premium execution, about 2.5 h measured for a 13 + 13 profile; needs a November snapshot built, validated and published with `py/snapshot_registry.py`)

**Blocked until**: a published manifest with cutoff `>= 2026-11-01` exists in the registry (none on 2026-11-04). Do not start, and do not fill the report by hand, before that.

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `measure` exits 0 on the November manifest at a clean committed `HEAD`
- [ ] Generated files committed unmodified; README entry updated from "pendente" to a link
- [ ] Gate check passes: Build
- [ ] Test count: unchanged

**Tests**: none
**Gate**: build
**Commit**: `docs(validation): measure October-to-November turnover with and without the zone (#46)`

---

## Phase Execution Map

```
T1 → T2 → T3
T2 → T4
T5 → T6
T7 → T8
T9 → T11 → T14
T10 → T11
T9 → T12 → T13 → T14
T15 → T16 → T17
```

Phases run in order (1 → 2 → 3 → 4); execution is sequential within a phase in task-number order. Cross-phase dependencies point backward only: T5 and T7 need T3; T12 needs T6 and T8; T15 needs T13. T17 is blocked on data.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | 3 helpers + 1 dataclass in one new module | ✅ cohesive |
| T2 | 1 function | ✅ |
| T3 | 1 function + record builder, same file | ✅ cohesive |
| T4 | 1 test file | ✅ |
| T5 | 1 function | ✅ |
| T6 | 1 function | ✅ |
| T7 | 1 function | ✅ |
| T8 | 1 function | ✅ |
| T9 | 1 module: version constant + dataclass field + resolver | ✅ cohesive |
| T10 | 1 method | ✅ |
| T11 | 1 method + 1 helper | ✅ |
| T12 | adapter wiring (config, kwargs, serializer) in 1 file | ⚠️ 3 related things, same file, cohesive |
| T13 | provenance + reasons in 1 file | ✅ cohesive |
| T14 | 1 test class | ✅ |
| T15 | 1 new script | ✅ |
| T16 | 1 doc entry | ✅ |
| T17 | 1 generated report | ✅ |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | none | ✅ |
| T2 | T1 | T1 → T2 | ✅ |
| T3 | T2 | T2 → T3 | ✅ |
| T4 | T2 | T2 → T4 | ✅ |
| T5 | T3 (Phase 1) | cross-phase, backward | ✅ |
| T6 | T5 | T5 → T6 | ✅ |
| T7 | T3 (Phase 1) | cross-phase, backward | ✅ |
| T8 | T7 | T7 → T8 | ✅ |
| T9 | None | none | ✅ |
| T10 | None | none | ✅ |
| T11 | T9, T10 | T9 → T11, T10 → T11 | ✅ |
| T12 | T9, T6, T8 (Phase 2) | T9 → T12; T6, T8 cross-phase, backward | ✅ |
| T13 | T12 | T12 → T13 | ✅ |
| T14 | T11, T13 | T11 → T14, T13 → T14 | ✅ |
| T15 | T13 (Phase 3) | cross-phase, backward | ✅ |
| T16 | T15 | T15 → T16 | ✅ |
| T17 | T16 | T16 → T17 | ✅ |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Core domain | unit | unit | ✅ |
| T2 | Core domain | unit | unit | ✅ |
| T3 | Core domain | unit | unit | ✅ |
| T4 | Oracle test | unit | unit | ✅ |
| T5 | Selectors | unit | unit | ✅ |
| T6 | Selectors | unit | unit | ✅ |
| T7 | Selectors | unit | unit | ✅ |
| T8 | Selectors | unit | unit | ✅ |
| T9 | API policy | unit | unit | ✅ |
| T10 | API repository | integration | integration | ✅ |
| T11 | API executor | integration | integration | ✅ |
| T12 | API adapter | unit | unit | ✅ |
| T13 | API adapter | unit | unit | ✅ |
| T14 | API end-to-end | integration | integration | ✅ |
| T15 | Measurement script | unit | unit | ✅ |
| T16 | Docs | none | none | ✅ |
| T17 | Generated docs | none | none | ✅ (generated by the T15-tested tool) |

---

## Tools per Task

Skills used: `coding-guidelines` (T1-T3, T5, T7, T9, T11), `docs-writer` (T16). No MCP is needed. The orchestrator picks models per batch from the Execution Plan's risk column.
