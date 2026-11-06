# FII Segment Cap and Policy v4 Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/fii-segment-cap/spec.md`
**Design**: `.specs/features/fii-segment-cap/design.md`
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `REVIEW.md` (a new test must fail when the behaviour it names is removed; update `model_versions` when the selection method changes; check `data/processed/*.csv` caches), `.specs/codebase/TESTING.md` (deterministic functions on small synthetic frames, no network), `.specs/LESSONS.md` candidates L-001 (give each guard its own input that only that guard rejects) and L-003 (assert exact arguments of defensive filters). No CI workflow, coverage threshold, linter or formatter is configured. Runner: stdlib `unittest` (pytest is not installed). Strong defaults applied.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| GA (`py/core/optimizer.py`) | unit | SEG-09..SEG-14 1:1; boundary at exactly the cap; golden pre-change output for the uncapped path | `tests/test_segment_cap_optimizer.py` (new) | `$PY -m unittest tests.test_segment_cap_optimizer` |
| Zone (`py/core/protection_zone.py`) | unit | SEG-15, SEG-19..SEG-25 1:1; every spec edge case; existing `tests/test_protection_zone.py` and `tests/test_protection_zone_equivalence.py` unchanged | `tests/test_segment_cap_zone.py` (new) | `$PY -m unittest tests.test_segment_cap_zone tests.test_protection_zone tests.test_protection_zone_equivalence` |
| FII selector (`py/fii_selection.py`) | unit + selector-level integration on fixture CSV | SEG-15..SEG-18, SEG-26..SEG-29; pre-change frame equality without a cap | `tests/test_segment_cap_fii.py` (new) | `$PY -m unittest tests.test_segment_cap_fii` |
| API policy (`api/app/services/premium_policy.py`) | unit + real-data executability | SEG-01..SEG-08; Basic golden JSON unchanged; real-data anchors under v4 | `api/tests/test_premium_policy.py` | API suite |
| API adapter (`api/app/adapters/premium_optimization.py`) | unit (fake engines) | SEG-30..SEG-32: forwarded, rejected, absent | `api/tests/test_premium_optimization.py` | API suite |
| API end to end (service + executor + adapter, fake engines) | integration | SEG-33 and the stored-v3 edge case | `api/tests/test_premium_integration.py` | API suite |
| #9 research tooling tests | unit (existing harness) | SEG-34, SEG-35; frozen protocol untouched (SEG-36) | `tests/test_multirun_convergence.py`, `tests/test_protection_zone_turnover.py` | `$PY -m unittest tests.test_multirun_convergence tests.test_protection_zone_turnover` |

## Gate Check Commands

> Generated from codebase - confirm before Execute. `$PY` = `/home/caio/Documents/projetos/TCC/.venv/bin/python` (Python 3.14.4 with numpy and pandas; the venv lives in the main checkout and is shared by worktrees; the system `python3` has no pandas). Run every command from the worktree root `/home/caio/.t3/worktrees/TCC/t3-issue-56`. Baseline at `a61164a`: root suite 535 tests OK (2 skipped), about 175 s; API suite 160 tests OK, about 75 s.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Core | Phase 1 tasks | `$PY -m unittest tests.test_segment_cap_optimizer tests.test_segment_cap_zone tests.test_protection_zone tests.test_protection_zone_equivalence` (T1 runs only the modules that exist at that point) |
| Selector | Phase 2 tasks | `$PY -m unittest tests.test_segment_cap_optimizer tests.test_segment_cap_zone tests.test_segment_cap_fii tests.test_protection_zone tests.test_protection_zone_equivalence tests.test_protection_zone_selectors tests.test_fii_selection tests.test_multi_run_backtest tests.test_missing_score_exclusion tests.test_convergence_analysis` |
| Policy | Phase 3 tasks | `(cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .) && $PY -m unittest tests.test_multirun_convergence tests.test_protection_zone_turnover tests.test_benchmark_premium tests.test_premium_research_contracts` (about 4 min) |
| Full | Phase 4 tasks | `cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t . ; cd ..` |
| Build | Last task of each phase | `$PY -m compileall -q py api/app scripts && $PY -m unittest discover -s tests -p "test_*.py" && (cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .)` (about 5 min; no linter is configured, `compileall` is the syntax gate) |

Test counts never go down: each task's "Done when" states the expected minimum. Existing tests may change values only where spec A-12 lists them.

---

## Test Strategy (test-to-AC matrix)

Every named test fails when the behaviour it names is removed (REVIEW.md). The "Kills" column names the mutation each test exists to catch.

| Test (file :: name) | Asserts | Req | Kills |
| ------------------- | ------- | --- | ----- |
| `test_segment_cap_optimizer :: test_over_cap_candidate_is_infeasible` | 3 tickers of one `SETOR` with cap 2 → `fitness == -inf` | SEG-09 | cap check removed |
| `test_segment_cap_optimizer :: test_indefinido_is_capped_like_any_segment` | 3 "Indefinido" with cap 2 → `-inf` | SEG-09, A-4 | "Indefinido" exempted |
| `test_segment_cap_optimizer :: test_candidate_at_cap_scores_like_the_uncapped_ga` | exactly 2 per segment → equals uncapped GA fitness and `SCORE sum - λ·HHI·n` | SEG-10 | `>` → `>=` |
| `test_segment_cap_optimizer :: test_optimize_never_returns_more_than_cap_per_segment` | frame whose 6 best scores are "Papéis", n 5, λ 0, seeds 0-9: capped max count ≤ 2 for every seed; the same frame uncapped returns ≥ 3 "Papéis" for at least one seed (the frame binds) | SEG-11 | cap ignored in `optimize` |
| `test_segment_cap_optimizer :: test_invalid_cap_raises` | `0`, `-1`, `1.5`, `True`, `"2"` → `ValueError` naming `max_per_segment`; `None` accepted | SEG-12 | validation removed / `bool` accepted |
| `test_segment_cap_optimizer :: test_cap_without_setor_column_raises` | cap set, frame without `SETOR` → `ValueError` naming `SETOR` | SEG-13 | silent pass |
| `test_segment_cap_optimizer :: test_uncapped_ga_matches_the_pre_change_golden` | seeds 3 and 11 on a fixed synthetic frame, with the keyword omitted and with `None`: tickers, `fitness`, `hhi`, `generations_run`, `seed` equal values captured at `a61164a` | SEG-14 | any change on the `None` path |
| `test_segment_cap_zone :: test_take_without_cap_is_the_plain_prefix` | `take_within_segment_cap(ordered, n, segment_of=…, max_per_segment=None) == ordered[:n]` | SEG-15, SEG-18 | cap applied when `None` |
| `test_segment_cap_zone :: test_take_skips_a_full_segment_and_continues` | `[P1, P2, P3, Q1, P4]`, n 3, cap 2 → `[P1, P2, Q1]` | SEG-15 | skip → stop |
| `test_segment_cap_zone :: test_take_allows_exactly_the_cap` | two of P with cap 2 kept | SEG-15 | off-by-one |
| `test_segment_cap_zone :: test_invalid_segment_cap_inputs_raise` | cap `0`/`1.5`/`True`; cap without `segment_of`; `segment_of` missing a ranked ticker → `ValueError` naming the field (one input per guard, L-001) | SEG-25 | each guard |
| `test_segment_cap_zone :: test_third_incumbent_of_a_segment_leaves_with_the_segment_reason` | incumbents P1, P2, P3 at ranks 1-3 ≤ K, n 5 → kept P1, P2; `departures["P3"] == "saiu: teto do segmento"`; `exits` counts it | SEG-19 | segment cut removed / reason text |
| `test_segment_cap_zone :: test_two_incumbents_of_a_segment_both_stay` | P1, P2 qualify → both `mantido` | SEG-19 | `>` → `>=` |
| `test_segment_cap_zone :: test_segment_cut_runs_before_the_capacity_cut` | qualifying P1, P2, P3, Q1, n 3 → kept P1, P2, Q1, P3 `teto do segmento`; second case P1, Q1, R1, S1, n 3 → S1 `vagas reduzidas` | SEG-20, A-6 | order swapped / precedence |
| `test_segment_cap_zone :: test_fill_skips_a_full_segment` | two P protected, best outsider P3, next Q2 → Q2 enters | SEG-21 | fill ignores cap |
| `test_segment_cap_zone :: test_challenger_that_would_break_the_cap_is_skipped` | weakest protected in Q, best outsider in full P, next outsider in R with lead ≥ δ → R swaps in | SEG-22 | skip → stop / cap ignored |
| `test_segment_cap_zone :: test_challenger_of_the_weakest_segment_fits` | weakest in P (P full), challenger in P with lead ≥ δ → swap allowed | SEG-22 | counting `held` instead of `held - weakest` |
| `test_segment_cap_zone :: test_migration_from_13_incumbents_to_5_partitions_every_incumbent` | 13 incumbents over 4 segments, n 5, K 8 → every incumbent in exactly one of `kept`/`departures`, ≤ 2 per segment, `turnover == len(entered)` | SEG-19, SEG-20, edge | partition broken |
| `test_segment_cap_zone :: test_cap_at_or_above_n_assets_equals_uncapped` | cap ≥ n → same `ZoneSelection` as no cap | edge | spurious cut |
| `test_segment_cap_zone :: test_none_cap_equals_the_call_without_keywords` | `segment_of` given, cap `None` → equal to the pre-change call | SEG-24 | cap applied when `None` |
| `test_segment_cap_zone :: test_fallback_consensus_respects_the_cap` | `hhi_of` above `hhi_max` → selection = capped `K = 0`, `status == "fallback_hhi"` | SEG-23 | cap not forwarded to the K=0 call |
| `test_segment_cap_zone :: test_recorded_consensus_tickers_are_capped` | zone applied → `record["consensus_tickers"]` ≤ 2 per segment | SEG-23 | idem |
| existing `tests/test_protection_zone.py`, `tests/test_protection_zone_equivalence.py` | unchanged values | SEG-24 | regression |
| `test_segment_cap_fii :: test_config_cap_reaches_every_ga_run` | `run_fii_selection(selection_config={..., "max_per_segment": 2}, n_runs=3)` with a recording GA → `[2, 2, 2]` | SEG-27 | cap not read from config |
| `test_segment_cap_fii :: test_config_without_cap_constructs_the_ga_with_none` | no key → `[None]` | SEG-27, SEG-32 | default changed |
| `test_segment_cap_fii :: test_pure_consensus_skips_the_third_ticker_of_a_full_segment` | portfolios where the 3 most frequent are "Papéis" → result = 2 most frequent "Papéis" + best other-segment ticker (exact tickers) | SEG-15, SEG-16 | issue test (b) |
| `test_segment_cap_fii :: test_pure_consensus_caps_indefinido` | same with "Indefinido" → 2 | SEG-17 | issue criterion "Indefinido" |
| `test_segment_cap_fii :: test_pure_consensus_without_cap_equals_pre_change` | frame equal (rows, order, columns, values, `attrs`) to a verbatim copy of the `a61164a` `_consensus` body | SEG-18 | regression |
| `test_segment_cap_fii :: test_zone_consensus_passes_the_segment_map` | `_consensus` with incumbents + zone + cap → `attrs["protection_zone"]["departures"]` has `saiu: teto do segmento` | SEG-19 (selector level) | zone path drops the cap |
| `test_segment_cap_fii :: test_zone_run_records_the_segment_cap_departure` | `run_fii_selection` on a fixture CSV, 3 incumbents of one segment within K → `result["protection_zone"]["departures"][<lowest>] == "saiu: teto do segmento"` | SEG-26 | issue test (c) |
| `test_segment_cap_fii :: test_run_with_cap_returns_at_most_two_per_segment` | fixture CSV with a dominant segment → `max(segment counts) <= 2`, `n_selected == n_assets` | SEG-27 | issue test (a) end to end |
| `test_segment_cap_fii :: test_universe_below_segment_capacity_fails_before_the_ga` | capacity < n → `FiiSelectionError` naming `max_per_segment`; recording GA never constructed | SEG-28 | check removed / `<` → `<=` |
| `test_segment_cap_fii :: test_over_cap_final_selection_is_refused` | `_consensus` patched to return 3 of one segment → `FiiSelectionError` matching "segment cap" | SEG-29 | post-check removed |
| `test_premium_policy :: SegmentCapPolicyTests.test_premium_fiis_carry_max_per_segment_2_for_every_score` | scores 0.0, 0.25, 0.5, 0.7, 1.0 → `fiis.max_per_segment == 2` and `to_dict()["fiis"]["max_per_segment"] == 2` | SEG-04 | constant / field |
| `test_premium_policy :: SegmentCapPolicyTests.test_stocks_have_no_max_per_segment` | `stocks.max_per_segment is None`, key absent from `to_dict()["stocks"]` and `to_json()` stocks section | SEG-04 | cap on stocks / `None` serialised |
| existing `test_premium_policy :: test_basic_policy_has_no_zone_and_matches_the_pre_change_json` | Basic JSON == `BASIC_GOLDEN_JSON` (unchanged test) | SEG-08 | Basic drift |
| existing `test_premium_policy :: test_basic_generic_profile_bands_map_to_discrete_stock_and_fii_rules` | Basic FII 10/12/15 (unchanged test) | SEG-08 | Basic follows `FII_RULES` |
| updated `test_premium_policy :: test_conservador_fii_rules_pin_the_adr_0002_calibration` | `FII_RULES` n 4/5/6, filters unchanged | SEG-03 | constant |
| updated `test_premium_policy :: test_premium_sections_carry_the_issue_9_zone_values_for_every_anchor` | stocks n 10/12/15 with K n+6; FIIs n 4/5/6 with K n+3; δ 0.11 | SEG-03, SEG-06 | constant / K formula |
| `test_premium_policy :: SegmentCapPolicyTests.test_fii_count_interpolates_half_to_even` | score 0.25 → 4, 0.75 → 6 | SEG-03, A-14 | rounding change |
| `test_premium_policy :: SegmentCapPolicyTests.test_owner_profile_resolves_five_fiis_capped_at_two` | `profile(score=0.7, knowledge=0.7)` → fiis n 5, cap 2, zone `{3, 8, 0.11}`, stocks 13 | SEG-05 | issue test (d) |
| updated `test_premium_policy :: test_policy_versions_are_bumped_and_reject_the_v1_strings` / `test_persisted_provenance_reports_the_bumped_policy_versions` | `premium-policy-v4`, Basic `basic-policy-v2` | SEG-01, SEG-08 | version |
| `test_premium_policy :: SegmentCapPolicyTests.test_premium_policy_v3_is_rejected` | `rules_version="premium-policy-v3"` → `PremiumPolicyError` for both entry points | SEG-02 | v3 accepted |
| `test_premium_policy :: SegmentCapPolicyTests.test_model_versions_record_fii_selection_v2` | Premium `{"allocation-v1", "stock-selection-v1", "fii-selection-v2"}`; Basic `fii-selection-v1` | SEG-07, SEG-08 | REVIEW.md rule |
| updated `test_premium_policy :: ConservadorAnchorExecutabilityTests.test_unrestricted_anchor_differs_only_by_hhi_max_and_runs_above_the_cap` | config `n_assets` fixed at 10; `assertGreater(hhi, 0.25)` kept | A-12 | subject preserved |
| `test_premium_policy :: V4FiiExecutabilityTests.test_real_data_anchors_fill_n_assets_within_the_cap` | real `data/raw/status_invest_fii.csv`, GA budget 30 × 20, scores 0.0/0.5/0.7/1.0 → `n_selected == n_assets` (4/5/5/6), max segment count ≤ 2, recording GA saw cap 2 | SEG-05, SEG-27, A-9 | infeasible universe on real data |
| `test_premium_optimization :: SegmentCapForwardingTests.test_fii_engine_receives_the_cap` | fake FII engine's `selection_config["max_per_segment"] == 2` and is `int` | SEG-30 | copy removed |
| `test_premium_optimization :: SegmentCapForwardingTests.test_invalid_cap_fails_with_policy_invalid` | `0`, `True`, `1.5`, `"2"` → `PremiumOptimizationError.code == "policy_invalid"`; engines not called | SEG-31 | check removed |
| `test_premium_optimization :: SegmentCapForwardingTests.test_sections_without_cap_produce_the_pre_change_config` | stock section and a v3 FII section → config equals the literal pre-change dict, no `max_per_segment` key | SEG-32 | key always emitted |
| `test_premium_integration :: SegmentCapIntegrationTests.test_v4_run_records_and_forwards_the_cap` | create + execute → `policy_version == "premium-policy-v4"`, `policy_json["fiis"]["max_per_segment"] == 2`, FII engine got 2 | SEG-33 | wiring |
| `test_premium_integration :: SegmentCapIntegrationTests.test_stored_v3_policy_runs_without_the_cap` | run whose stored `policy_json` lacks the key → FII engine config has no `max_per_segment` | SEG-32, edge | stored v3 changed |
| updated `test_multirun_convergence :: PolicyTests.test_profiles_resolve_expected_n_assets` + `fixture_protocol` | `(13, 5)`, `(15, 6)`; `check_n_assets` vs declared `(13, 13)` raises `ChildError` with `EXIT_USAGE` and "expected_n_assets"; fixture protocol uses v4 counts | SEG-34 | guard |
| updated `test_protection_zone_turnover :: test_measure_writes_both_reports_with_the_turnover_per_class` | `report["policy_version"] == "premium-policy-v4"` | SEG-35 | version |
| Done-when check (T9, T10) | `git diff --exit-code a61164a -- docs/validation/` is empty | SEG-36 | frozen evidence edited |

**Mutation targets for the Verifier's discrimination sensor**: `>` vs `>=` in the GA cap; the `None` guard in `fitness`; segment cut before vs after the capacity cut; departure precedence order; `held - {weakest}` vs `held` in `fits`; skip vs stop for an over-cap challenger; `take_within_segment_cap` skip; cap forwarded to the `K = 0` call; `_consensus` passing `segment_of`; capacity check `<` vs `<=`; final post-check; `_selector_config` copy and its integer check; `_selector_dict` omission of `None`; Basic FII counts; `fii-selection-v2` vs `v1` for Basic.

---

## Rollback

- **Code**: revert the PR's merge commit on `develop`. No migration, no new table.
- **Persisted runs**: v4 runs stay readable after a revert: `policy_json.fiis.max_per_segment` is an extra key in a JSON column; `_selector_config` before this change ignores it, so re-executing a stored v4 run after a revert would run uncapped (acceptable; runs are not re-executed in place).
- **Partial**: setting `FII_MAX_PER_SEGMENT` high (≥ 6) disables the cap in practice without a version change, but changes the persisted policy, so it needs its own version bump; prefer reverting.

---

## Execution Plan

Phases run in order; tasks within a phase run in order.

| Phase | Tasks | Count | Sensitive path (REVIEW.md) | Risk | Suggested tier |
| ----- | ----- | ----- | -------------------------- | ---- | -------------- |
| 1. Core cap (GA and zone) | T1-T4 | 4 | **yes** - `py/core/optimizer.py`, `py/core/protection_zone.py` | High: GA fitness and zone rule that produce the delivered recommendation; byte-identity of the uncapped path. | high reasoning |
| 2. FII selector wiring | T5-T7 | 3 | no (`py/fii_selection.py` is not listed; it is the FII recommendation path) | Medium-high: regression of the uncapped consensus. | high reasoning |
| 3. Policy v4 and tooling tests | T8-T11 | 4 | no (`api/app/services/premium_policy.py`, tests) | Low-medium: constants, version bump, pinned tests; mechanical. | mid tier |
| 4. Adapter and end to end | T12-T13 | 2 | **yes** - `api/app/adapters/premium_optimization.py` | Medium: the bridge that decides what reaches the selector. | high reasoning |

No task is heavy in the migration/snapshot sense: no DB migration, no snapshot regeneration, no committed-data rewrite. The longest gates are the Policy gate (~4 min, T8-T11) and the Build gate at each phase end (~5 min). T11 runs the real FII selector on committed data with a 30 × 20 GA budget (seconds per anchor).

### Phase 1: Core cap (GA and zone)

```
T1
T2 → T3 → T4
```

### Phase 2: FII selector wiring

```
T5 → T7
T6 → T7
```

### Phase 3: Policy v4 and tooling tests

```
T8 → T9 → T10 → T11
```

### Phase 4: Adapter and end to end

```
T12 → T13
```

---

## Task Breakdown

### Phase 1 tasks: Core cap (GA and zone)

#### T1: Segment cap in the GA fitness

**Status**: Pending

**What**: Add `max_per_segment` to `GeneticAlgorithm` (validated in `__init__`, `-inf` in `fitness` above the cap, `ValueError` without `SETOR`), leaving the `None` path byte-identical.
**Where**: `py/core/optimizer.py`
**Depends on**: None
**Reuses**: `hhi_max` validation and infeasibility (`py/core/optimizer.py:92-95`, `:132-135`)
**Requirement**: SEG-09, SEG-10, SEG-11, SEG-12, SEG-13, SEG-14
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] FIRST, before editing `optimizer.py`: golden values (tickers, `fitness`, `hhi`, `generations_run`, `seed`) for seeds 3 and 11 on the test's synthetic frame are captured at `a61164a` and written as literals in `tests/test_segment_cap_optimizer.py`
- [ ] Every `test_segment_cap_optimizer` row of the Test Strategy passes
- [ ] The new keyword is appended after `early_stopping_min_delta`; `crossover`, `mutate`, `initialize_population`, `optimize`, `optimize_portfolio` unchanged (`git diff` shows edits only in `__init__` and `fitness`)
- [ ] Gate check passes: Core (modules present at this point)
- [ ] Test count: at least 7 new tests; `tests.test_protection_zone*` counts unchanged

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): make a GA candidate above the segment cap infeasible (#56)`

---

#### T2: Capped greedy pick and cap validation in the zone module

**Status**: Pending

**What**: Add `SEGMENT_CAP_DEPARTURE`, `validate_segment_cap` and `take_within_segment_cap` to the zone module.
**Where**: `py/core/protection_zone.py`
**Depends on**: None
**Reuses**: `_is_integer` (`py/core/protection_zone.py:71-72`)
**Requirement**: SEG-15, SEG-25
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `take_within_segment_cap` with `max_per_segment=None` returns `list(ordered[:n_assets])`
- [ ] The `take_*` and `test_invalid_segment_cap_inputs_raise` rows of the Test Strategy pass, one input per guard (L-001)
- [ ] The module still imports nothing from pandas or `scripts/`
- [ ] Gate check passes: Core
- [ ] Test count: at least 4 new tests in `tests/test_segment_cap_zone.py`

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): add the segment-capped pick and its input checks (#56)`

---

#### T3: Segment cap inside `apply_protection_zone`

**Status**: Pending

**What**: Add keyword-only `segment_of` and `max_per_segment` to `apply_protection_zone`: segment cut before the capacity cut, capped fill, capped challengers, `saiu: teto do segmento` with the A-6 precedence.
**Where**: `py/core/protection_zone.py`
**Depends on**: T2
**Reuses**: current protection/fill/swap loops; `validate_segment_cap`
**Requirement**: SEG-19, SEG-20, SEG-21, SEG-22, SEG-24
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Rows SEG-19..SEG-22 and the migration, cap ≥ n and `None`-cap rows of the Test Strategy pass
- [ ] `tests/test_protection_zone.py` and `tests/test_protection_zone_equivalence.py` pass with zero edits (`git diff --exit-code -- tests/test_protection_zone.py tests/test_protection_zone_equivalence.py`)
- [ ] The existing `SPEC_DEVIATION` comment at `py/core/protection_zone.py:116` is kept
- [ ] Gate check passes: Core
- [ ] Test count: at least 9 new tests in `tests/test_segment_cap_zone.py` (13+ in the file)

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): apply the segment cap in the protection zone (#56)`

---

#### T4: Forward the cap through `select_with_fallback`

**Status**: Pending

**What**: Add keyword-only `segment_of` and `max_per_segment` to `select_with_fallback` and pass them to both the zone call and the `K = 0` call.
**Where**: `py/core/protection_zone.py`
**Depends on**: T3
**Reuses**: existing fallback logic and record (no new record field, A-13)
**Requirement**: SEG-23, SEG-24
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `test_fallback_consensus_respects_the_cap` and `test_recorded_consensus_tickers_are_capped` pass
- [ ] Record keys are identical with and without a cap (asserted)
- [ ] Gate check passes: Build (last task of Phase 1)
- [ ] Test count: root suite ≥ 535 + new Phase 1 tests, 2 skipped; API suite 160

**Tests**: unit
**Gate**: build
**Commit**: `feat(core): forward the segment cap to the zone fallback (#56)`

---

### Phase 2 tasks: FII selector wiring

#### T5: Cap from the FII config reaches the GA

**Status**: Pending

**What**: `optimize_fii_portfolio` copies `config["max_per_segment"]` into its settings and constructs `GeneticAlgorithm(..., max_per_segment=...)`.
**Where**: `py/fii_selection.py`
**Depends on**: T1
**Reuses**: the `hhi_max` copy in `optimize_fii_portfolio` (`py/fii_selection.py:462-466`, `:486-495`)
**Requirement**: SEG-27
**Risk area**: FII recommendation path
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `test_config_cap_reaches_every_ga_run` and `test_config_without_cap_constructs_the_ga_with_none` pass (recording GA subclass, as `_RecordingGeneticAlgorithm` in `api/tests/test_premium_policy.py`)
- [ ] Gate check passes: Selector
- [ ] Test count: at least 2 new tests in `tests/test_segment_cap_fii.py`

**Tests**: unit
**Gate**: selector
**Commit**: `feat(fiis): pass the segment cap from the config to the GA (#56)`

---

#### T6: Capped FII consensus

**Status**: Pending

**What**: Add `max_per_segment` to `_consensus`: pure path through `take_within_segment_cap`, zone path with `segment_of` and the cap; uncapped path unchanged.
**Where**: `py/fii_selection.py`
**Depends on**: T2, T4
**Reuses**: `take_within_segment_cap`, `select_with_fallback`; a verbatim copy of the `a61164a` `_consensus` body as the regression oracle
**Requirement**: SEG-15, SEG-16, SEG-17, SEG-18
**Risk area**: FII recommendation path
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] The four `_consensus` rows of the Test Strategy pass (exact tickers asserted, not just counts)
- [ ] `tests/test_protection_zone_selectors.py`, `tests/test_fii_selection.py`, `tests/test_convergence_analysis.py` pass unchanged
- [ ] Gate check passes: Selector
- [ ] Test count: at least 4 new tests

**Tests**: unit
**Gate**: selector
**Commit**: `feat(fiis): skip a full segment in the FII consensus (#56)`

---

#### T7: `run_fii_selection` reads the cap, checks capacity and the result

**Status**: Pending

**What**: In explicit mode read `max_per_segment` from the config, fail before the GA when the universe cannot fill `n_assets` under it, pass it to `_consensus`, and refuse a final selection above it.
**Where**: `py/fii_selection.py`
**Depends on**: T5, T6
**Reuses**: `_require_feasible_universe` shape, HHI post-check (`py/fii_selection.py:768-770`), `_fixture_frame` style of `tests/test_fii_selection.py:28`
**Requirement**: SEG-26, SEG-27, SEG-28, SEG-29
**Risk area**: FII recommendation path
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `test_run_with_cap_returns_at_most_two_per_segment`, `test_zone_run_records_the_segment_cap_departure`, `test_universe_below_segment_capacity_fails_before_the_ga`, `test_over_cap_final_selection_is_refused` pass
- [ ] Named-profile mode never reads the key (no change to `outputs/` or `data/processed/` writers; `git status --porcelain data/ outputs/` empty after the gate)
- [ ] Gate check passes: Build (last task of Phase 2)
- [ ] Test count: at least 4 new tests; root suite ≥ previous + new

**Tests**: integration
**Gate**: build
**Commit**: `feat(fiis): enforce the segment cap in the explicit FII selection (#56)`

---

### Phase 3 tasks: Policy v4 and tooling tests

#### T8: `max_per_segment` on the Premium FII policy

**Status**: Pending

**What**: Add `SelectorPolicy.max_per_segment` (omitted from `to_dict()` when `None`) and set `FII_MAX_PER_SEGMENT = 2` on the Premium FII section only.
**Where**: `api/app/services/premium_policy.py`
**Depends on**: None
**Reuses**: `protection_zone` field and `_selector_dict` omission (`api/app/services/premium_policy.py:126`, `:173-177`)
**Requirement**: SEG-04
**Risk area**: none (policy constants)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `test_premium_fiis_carry_max_per_segment_2_for_every_score` and `test_stocks_have_no_max_per_segment` pass
- [ ] `BASIC_GOLDEN_JSON` test passes unchanged
- [ ] Gate check passes: Policy
- [ ] Test count: API suite ≥ 162

**Tests**: unit
**Gate**: policy
**Commit**: `feat(policy): record the FII segment cap in the Premium policy (#56)`

---

#### T9: FII anchors 4/5/6; Basic keeps 10/12/15

**Status**: Pending

**What**: Set `FII_RULES` n to 4/5/6, give Basic `BASIC_FII_ASSET_COUNTS` 10/12/15, and update the tests that pin the v3 FII counts (A-12): the two `test_premium_policy` pins, the conservador executability test, and the #9 tooling tests.
**Where**: `api/app/services/premium_policy.py` (tests: `api/tests/test_premium_policy.py`, `tests/test_multirun_convergence.py`)
**Depends on**: T8
**Reuses**: `_selector_policy` interpolation unchanged
**Requirement**: SEG-03, SEG-06, SEG-08, SEG-34, SEG-36
**Risk area**: none (policy constants, tests)
**Heavy**: no (Policy gate about 4 min: `tests.test_multirun_convergence` takes about 110 s)

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Updated `test_conservador_fii_rules_pin_the_adr_0002_calibration`, `test_premium_sections_carry_the_issue_9_zone_values_for_every_anchor` and new `test_fii_count_interpolates_half_to_even` pass
- [ ] `ConservadorAnchorExecutabilityTests.test_unrestricted_anchor_differs_only_by_hhi_max_and_runs_above_the_cap` fixes `n_assets` 10 in its config and keeps `assertGreater(result["hhi"], 0.25)`
- [ ] `tests/test_multirun_convergence.py`: `fixture_protocol` sets `expected_n_assets` to `{"stocks": 13, "fiis": 5}`; `PolicyTests.test_profiles_resolve_expected_n_assets` asserts `(13, 5)` / `(15, 6)` and that `check_n_assets` against the declared counts raises `ChildError` with `EXIT_USAGE` mentioning `expected_n_assets`; all 40 tests of the two tooling modules pass
- [ ] Both Basic tests pass unchanged
- [ ] `git diff --exit-code a61164a -- docs/validation/` is empty
- [ ] Gate check passes: Policy
- [ ] Test count: API suite ≥ 163; root tooling modules 40 tests

**Tests**: unit
**Gate**: policy
**Commit**: `feat(policy): reduce the Premium FII anchors to 4/5/6 (#56)`

---

#### T10: Bump to `premium-policy-v4` and `fii-selection-v2`

**Status**: Pending

**What**: Set `POLICY_VERSION = "premium-policy-v4"`, Premium `model_versions.fiis = "fii-selection-v2"` (Basic keeps `v1`), and update the tests that pin the v3 version string.
**Where**: `api/app/services/premium_policy.py` (tests: `api/tests/test_premium_policy.py`, `api/tests/test_premium_integration.py`, `tests/test_protection_zone_turnover.py`)
**Depends on**: T9
**Reuses**: version pin tests from #46 (`07f4c44`)
**Requirement**: SEG-01, SEG-02, SEG-05, SEG-07, SEG-08, SEG-35, SEG-36
**Risk area**: none (constants, tests)
**Heavy**: no (Policy gate about 4 min)

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Updated version tests, new `test_premium_policy_v3_is_rejected`, `test_model_versions_record_fii_selection_v2`, `test_owner_profile_resolves_five_fiis_capped_at_two` pass
- [ ] `test_premium_integration` and `test_protection_zone_turnover` assert `premium-policy-v4` (only the version literal changes)
- [ ] `BASIC_GOLDEN_JSON` test passes unchanged
- [ ] `git diff --exit-code a61164a -- docs/validation/` is empty
- [ ] Gate check passes: Policy
- [ ] Test count: API suite ≥ 166

**Tests**: unit
**Gate**: policy
**Commit**: `feat(policy): bump to premium-policy-v4 with fii-selection-v2 (#56)`

---

#### T11: Real-data v4 executability for the FII anchors

**Status**: Pending

**What**: Add `V4FiiExecutabilityTests` running the real FII selector on the committed dataset for selector scores 0.0, 0.5, 0.7, 1.0 with the resolved v4 FII section.
**Where**: `api/tests/test_premium_policy.py`
**Depends on**: T10, T7
**Reuses**: `ConservadorAnchorExecutabilityTests` harness (`_RecordingGeneticAlgorithm`, `GA_BUDGET`, `PINNED_SEED`)
**Requirement**: SEG-05, SEG-27
**Risk area**: none (test only)
**Heavy**: no (four small GA runs on real data)

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] For each score: `n_selected == n_assets` (4/5/5/6), max segment count ≤ 2, recording GA saw `max_per_segment` 2
- [ ] The commit body records the wall-clock of the owner-profile capped run and of the same run without the cap (design Risk 1)
- [ ] Gate check passes: Build (last task of Phase 3)
- [ ] Test count: API suite ≥ 167; root suite unchanged from T10

**Tests**: integration
**Gate**: build
**Commit**: `test(policy): run the v4 FII anchors on the committed dataset (#56)`

---

### Phase 4 tasks: Adapter and end to end

#### T12: Adapter forwards and validates the cap

**Status**: Pending

**What**: `_selector_config` copies an integer `max_per_segment >= 1` from the policy section to the selector config, rejects any other value with `policy_invalid`, and emits no key when the section has none.
**Where**: `api/app/adapters/premium_optimization.py`
**Depends on**: T8
**Reuses**: zone copy block (`api/app/adapters/premium_optimization.py:311-316`)
**Requirement**: SEG-30, SEG-31, SEG-32
**Risk area**: sensitive (`api/app/adapters/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] The three `SegmentCapForwardingTests` pass; the "without cap" test compares against a literal dict (L-003: exact arguments, not "key absent" alone)
- [ ] Existing `test_premium_optimization` tests pass unchanged
- [ ] Gate check passes: Full
- [ ] Test count: API suite ≥ 170

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): forward the FII segment cap to the selector (#56)`

---

#### T13: End-to-end v4 run and stored v3 run

**Status**: Pending

**What**: Add `SegmentCapIntegrationTests`: a created and executed Premium run records and forwards the cap; a run with a stored policy lacking the key executes uncapped.
**Where**: `api/tests/test_premium_integration.py`
**Depends on**: T12, T10
**Reuses**: `ProtectionZoneIntegrationTests` harness (`create_and_wait`, fake engines)
**Requirement**: SEG-32, SEG-33
**Risk area**: none (test only)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Both integration tests pass
- [ ] Gate check passes: Build (last task)
- [ ] Test count: root suite ≥ 535 + all new root tests (2 skipped); API suite ≥ 172

**Tests**: integration
**Gate**: build
**Commit**: `test(api): cover a v4 run and a stored v3 run end to end (#56)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1:  T1    T2 ──→ T3 ──→ T4
Phase 2:  T5 ──┐
          T6 ──┴→ T7
Phase 3:  T8 ──→ T9 ──→ T10 ──→ T11
Phase 4:  T12 ──→ T13
```

Execution is sequential (T1 … T13). Cross-phase dependencies point backward only: T5 ← T1, T6 ← T2/T4, T11 ← T7, T12 ← T8, T13 ← T10.

Suggested batches (whole phases, ~7 tasks): batch 1 = Phases 1-2 (T1-T7, 7 tasks, sensitive `py/core/**`); batch 2 = Phases 3-4 (T8-T13, 6 tasks; Phase 3 is low-risk, Phase 4 is sensitive `api/app/adapters/**`). Phase 3 can also run alone if the orchestrator wants the mechanical policy work separate.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: GA cap | 1 class, 2 methods, 1 file | ✅ Granular |
| T2: capped pick + validator | 2 related functions, 1 file | ✅ OK (cohesive) |
| T3: zone cap | 1 function | ✅ Granular |
| T4: fallback forwarding | 1 function | ✅ Granular |
| T5: GA config copy | 1 function | ✅ Granular |
| T6: capped consensus | 1 function | ✅ Granular |
| T7: run-level checks | 1 function + 1 helper, 1 file | ✅ OK (cohesive) |
| T8: policy field | 1 dataclass field + 1 constant | ✅ Granular |
| T9: FII anchors | 1 constant table + Basic counts; pinned tests in 2 files move with it | ⚠️ OK: one value change and the pins it breaks must land together or the gate fails |
| T10: version bump | 2 constants; pinned tests in 3 files move with it | ⚠️ OK: same reason as T9 |
| T11: real-data test | 1 test class | ✅ Granular |
| T12: adapter copy | 1 function | ✅ Granular |
| T13: integration tests | 1 test class | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | none (alone in Phase 1 line 1) | ✅ Match |
| T2 | None | none | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T1 (Phase 1) | none within Phase 2 | ✅ Match (cross-phase, backward) |
| T6 | T2, T4 (Phase 1) | none within Phase 2 | ✅ Match (cross-phase, backward) |
| T7 | T5, T6 | T5 → T7, T6 → T7 | ✅ Match |
| T8 | None | none | ✅ Match |
| T9 | T8 | T8 → T9 | ✅ Match |
| T10 | T9 | T9 → T10 | ✅ Match |
| T11 | T10, T7 (Phase 2) | T10 → T11 | ✅ Match (T7 cross-phase, backward) |
| T12 | T8 (Phase 3) | none within Phase 4 | ✅ Match (cross-phase, backward) |
| T13 | T12, T10 (Phase 3) | T12 → T13 | ✅ Match (T10 cross-phase, backward) |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | GA | unit | unit | ✅ OK |
| T2 | Zone | unit | unit | ✅ OK |
| T3 | Zone | unit | unit | ✅ OK |
| T4 | Zone | unit | unit | ✅ OK |
| T5 | FII selector | unit + integration | unit | ✅ OK (pure config plumbing; recording GA) |
| T6 | FII selector | unit + integration | unit | ✅ OK (`_consensus` on synthetic portfolios) |
| T7 | FII selector | unit + integration | integration | ✅ OK |
| T8 | API policy | unit | unit | ✅ OK |
| T9 | API policy + tooling tests | unit | unit | ✅ OK |
| T10 | API policy + tooling tests | unit | unit | ✅ OK |
| T11 | API policy (real data) | unit + real-data executability | integration | ✅ OK |
| T12 | API adapter | unit (fake engines) | unit | ✅ OK |
| T13 | API end to end | integration | integration | ✅ OK |

---

## Requirement → Task Traceability

| Requirement | Tasks | Requirement | Tasks |
| ----------- | ----- | ----------- | ----- |
| SEG-01 | T10 | SEG-19 | T3 (T6 selector level) |
| SEG-02 | T10 | SEG-20 | T3 |
| SEG-03 | T9 | SEG-21 | T3 |
| SEG-04 | T8 | SEG-22 | T3 |
| SEG-05 | T10, T11 | SEG-23 | T4 |
| SEG-06 | T9 | SEG-24 | T3, T4 |
| SEG-07 | T10 | SEG-25 | T2 |
| SEG-08 | T9, T10 | SEG-26 | T7 |
| SEG-09 | T1 | SEG-27 | T5, T7, T11 |
| SEG-10 | T1 | SEG-28 | T7 |
| SEG-11 | T1 | SEG-29 | T7 |
| SEG-12 | T1 | SEG-30 | T12 |
| SEG-13 | T1 | SEG-31 | T12 |
| SEG-14 | T1 | SEG-32 | T5, T12, T13 |
| SEG-15 | T2, T6 | SEG-33 | T13 |
| SEG-16 | T6 | SEG-34 | T9 |
| SEG-17 | T6 | SEG-35 | T10 |
| SEG-18 | T6 | SEG-36 | T9, T10 |

36 of 36 requirements mapped; every task maps to at least one requirement.

---

## Tools per Task

All tasks: no MCP; skill `coding-guidelines`. No external documentation is needed (stdlib `unittest`, pandas `value_counts`, dataclasses already in use).
