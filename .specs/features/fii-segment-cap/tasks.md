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
| Zone (`py/core/protection_zone.py`) | unit | SEG-15, SEG-19..SEG-25, SEG-37..SEG-41, SEG-43 1:1; every spec edge case; existing `tests/test_protection_zone.py` and `tests/test_protection_zone_equivalence.py` unchanged | `tests/test_segment_cap_zone.py` (new) | `$PY -m unittest tests.test_segment_cap_zone tests.test_protection_zone tests.test_protection_zone_equivalence` |
| FII selector (`py/fii_selection.py`) | unit + selector-level integration on fixture CSV | SEG-15..SEG-18, SEG-26..SEG-29, SEG-37, SEG-40, SEG-41; capacity boundary; pre-change frame equality without a cap | `tests/test_segment_cap_fii.py` (new) | `$PY -m unittest tests.test_segment_cap_fii` |
| API policy (`api/app/services/premium_policy.py`) | unit + real-data executability | SEG-01..SEG-08; Basic golden JSON unchanged; real-data anchors under v4; real-data multi-run with `limitar_concentracao` (SEG-42) | `api/tests/test_premium_policy.py` | API suite |
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
| `test_segment_cap_zone :: test_take_skips_a_ticker_that_would_break_hhi_max` | `[P1, P2, Q1, R1, S1, T1]`, n 5, cap 2, `hhi_max` 0.25 → `[P1, Q1, R1, S1, T1]` (P2 would give 2+1+1+1, HHI 0.28) | SEG-37 | HHI skip removed |
| `test_segment_cap_zone :: test_take_allows_one_pair_at_n6_under_hhi_max` | `[P1, P2, Q1, Q2, R1, S1, T1]`, n 6, cap 2, `hhi_max` 0.25 → `[P1, P2, Q1, R1, S1, T1]` (one pair fits, the second does not: `P = 1`) | SEG-37, A-16 | HHI checked on the partial sleeve (would reject P2 and fail) / pairs unlimited |
| `test_segment_cap_zone :: test_take_accepts_hhi_exactly_at_the_limit` | `[P1, Q1, R1, S1]`, n 4, `hhi_max` 0.25 → all four (HHI exactly 0.25) | SEG-37 | `<=` → `<` / tolerance dropped |
| `test_segment_cap_zone :: test_take_raises_instead_of_returning_a_short_list` | `[P1, P2, Q1, Q2, R1]`, n 4, cap 2, `hhi_max` 0.25 → `SegmentCapInfeasibleError` naming `hhi_max` | SEG-40 | silent short list |
| `test_segment_cap_zone :: test_hhi_max_without_cap_raises` | `take_within_segment_cap(..., max_per_segment=None, hhi_max=0.25)` and `apply_protection_zone(..., hhi_max=0.25)` without cap → `ValueError` naming `hhi_max`; `hhi_max` `0`, `nan` with cap → `ValueError` (one input per guard) | SEG-41 | HHI skip active without the cap / bad `hhi_max` accepted |
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
| `test_segment_cap_zone :: test_fill_skips_a_ticker_that_would_break_hhi_max` | incumbent P1 protected, n 5, cap 2, `hhi_max` 0.25, ranking P1, P2, Q1, R1, S1, T1 → zone tickers `{P1, Q1, R1, S1, T1}` | SEG-38 | fill ignores `hhi_max` |
| `test_segment_cap_zone :: test_challenger_that_would_break_hhi_max_is_skipped` | n 5, cap 2, `hhi_max` 0.25; weakest protected Q1; best outsider P2 (P1 held, a pair does not fit) leads by ≥ δ; next outsider R2 (R not held) leads by ≥ δ → R2 replaces Q1, P2 never enters | SEG-38 | challenger ignores `hhi_max` / stop instead of skip |
| `test_segment_cap_zone :: test_challenger_hhi_check_excludes_the_weakest` | n 5, `hhi_max` 0.25; weakest Q1, challenger Q2 (same segment) → swap allowed (the pair disappears with Q1) | SEG-38 | `held` instead of `held - weakest` |
| `test_segment_cap_zone :: test_protected_pair_over_hhi_max_keeps_the_higher_ranked_incumbent` | `apply_protection_zone`, n 5, K 8, cap 2, `hhi_max` 0.25; incumbents P1 (freq 0.9), Q1 (0.8), P2 (0.7), all rank ≤ K; ranking also holds R1, S1, T1 → no raise; `kept == (P1, Q1)`; `departures == {"P2": "saiu: teto do segmento"}`; tickers `{P1, Q1, R1, S1, T1}`. Second case, P1 and P2 at equal frequency, P2 with the higher `SCORE` → P2 kept, P1 `saiu: teto do segmento` (rank tie-break) | SEG-39, A-5 | protection-walk HHI term removed (both protected, fill cannot reach 5, the zone raises) / walk not in rank order |
| `test_segment_cap_zone :: test_incumbent_past_capacity_is_cut_by_capacity_not_by_hhi` | n 4, cap 2, `hhi_max` 0.25; qualifying P1, Q1, R1, S1, P2 in that rank → P1, Q1, R1, S1 kept; `departures["P2"] == "saiu: vagas reduzidas"` | SEG-20, SEG-39 | HHI term applied once `n_assets` are accepted |
| `test_segment_cap_zone :: test_zone_fill_short_raises_instead_of_a_short_sleeve` | `apply_protection_zone`, n 4, cap 2, `hhi_max` 0.25, ranking over 3 segments only (P1, P2, Q1, Q2, R1), incumbent P1 → `SegmentCapInfeasibleError` naming `hhi_max`, no `ZoneSelection` returned | SEG-43 | short zone returned |
| `test_segment_cap_zone :: test_protected_same_segment_pair_keeps_the_protection_without_fallback` | `select_with_fallback`, n 5, K 8, δ 0.11, cap 2, `hhi_max` 0.25, `hhi_of` = the real segment HHI over `segment_of`; incumbents P1 (freq 0.90) and P2 (0.85) of segment P plus U1 (0.45); outsiders Q1 0.80, R1 0.70, S1 0.60, T1 0.50 (ranks: P1, P2, Q1, R1, S1, T1, U1, all ≤ K) → no raise; record `status == "applied"` (not `"fallback_hhi"`), `buffer_fallback is None`; selection `(P1, Q1, R1, S1, U1)`, `kept == (P1, U1)`, `departures == {"P2": "saiu: teto do segmento"}`; selection HHI 0.20; `consensus_tickers == (P1, Q1, R1, S1, T1)`, so the selection differs from the `K = 0` consensus (U1's protection survives; T1 leads U1 by 0.05 < δ) | SEG-39, SEG-43 | protection-step HHI skip removed (the zone raises and the test errors) / `SegmentCapInfeasibleError` caught with a `K = 0` fallback (status `fallback_hhi`) |
| `test_segment_cap_zone :: test_segment_cap_infeasible_error_propagates_from_both_calls` | `apply_protection_zone` stubbed to raise `SegmentCapInfeasibleError` on the `K > 0` call only, then on the `K = 0` call only → `select_with_fallback` raises `SegmentCapInfeasibleError` both times, never returns a record | SEG-43 | `try`/fallback around either call |
| `test_segment_cap_zone :: test_stock_path_never_forwards_hhi_max` | `select_with_fallback` with `hhi_max` 0.25 and no cap, `apply_protection_zone` spied → every call has no `hhi_max` keyword; record equals the pre-change record | SEG-41, SEG-24 | HHI skip leaks to stocks |
| existing `tests/test_protection_zone.py`, `tests/test_protection_zone_equivalence.py` | unchanged values | SEG-24 | regression |
| `test_segment_cap_fii :: test_config_cap_reaches_every_ga_run` | `run_fii_selection(selection_config={..., "max_per_segment": 2}, n_runs=3)` with a recording GA → `[2, 2, 2]` | SEG-27 | cap not read from config |
| `test_segment_cap_fii :: test_config_without_cap_constructs_the_ga_with_none` | no key → `[None]` | SEG-27, SEG-32 | default changed |
| `test_segment_cap_fii :: test_pure_consensus_skips_the_third_ticker_of_a_full_segment` | portfolios where the 3 most frequent are "Papéis" → result = 2 most frequent "Papéis" + best other-segment ticker (exact tickers) | SEG-15, SEG-16 | issue test (b) |
| `test_segment_cap_fii :: test_pure_consensus_caps_indefinido` | same with "Indefinido" → 2 | SEG-17 | issue criterion "Indefinido" |
| `test_segment_cap_fii :: test_pure_consensus_without_cap_equals_pre_change` | frame equal (rows, order, columns, values, `attrs`) to a verbatim copy of the `a61164a` `_consensus` body | SEG-18 | regression |
| `test_segment_cap_fii :: test_pure_consensus_skips_a_pair_that_breaks_hhi_max` | n 5, cap 2, `hhi_max` 0.25, the two most frequent tickers in P → result holds one P plus the next four of other segments (exact tickers) | SEG-37 | `_consensus` does not pass `hhi_max` |
| `test_segment_cap_fii :: test_pure_consensus_with_cap_and_no_hhi_max_keeps_the_pair` | same portfolios, no `hhi_max` → both P tickers kept | SEG-41 | HHI applied without a limit |
| `test_segment_cap_fii :: test_infeasible_consensus_raises_fii_selection_error` | candidates in 3 segments, n 4, cap 2, `hhi_max` 0.25 → `FiiSelectionError` naming `max_per_segment` and `hhi_max` | SEG-40 | short sleeve returned / raw `ValueError` leaks |
| `test_segment_cap_fii :: test_zone_consensus_passes_the_segment_map` | `_consensus` with incumbents + zone + cap → `attrs["protection_zone"]["departures"]` has `saiu: teto do segmento` | SEG-19 (selector level) | zone path drops the cap |
| `test_segment_cap_fii :: test_zone_run_records_the_segment_cap_departure` | `run_fii_selection` on a fixture CSV, 3 incumbents of one segment within K → `result["protection_zone"]["departures"][<lowest>] == "saiu: teto do segmento"` | SEG-26 | issue test (c) |
| `test_segment_cap_fii :: test_run_with_cap_returns_at_most_two_per_segment` | fixture CSV with a dominant segment, `_consensus` wrapped by a spy → spy saw `max_per_segment=2` (and `hhi_max` from the config); `max(segment counts) <= 2`, `n_selected == n_assets`; fixture binds: the original `_consensus` called without the cap on the portfolios the spy recorded holds ≥ 3 of the dominant segment (review r1 S3) | SEG-27 | `run_fii_selection` not passing the cap to `_consensus` (GA cap alone would not satisfy the spy) |
| `test_segment_cap_fii :: test_universe_below_segment_capacity_fails_before_the_ga` | capacity < n → `FiiSelectionError` naming `max_per_segment`; recording GA never constructed | SEG-28 | check removed / `<` → `<=` |
| `test_segment_cap_fii :: test_universe_at_exact_segment_capacity_runs` | capacity == n (e.g. segments 3/1/1 with cap 2, n 4) → no error, `n_selected == 4`, ≤ 2 per segment (review r1 S4) | SEG-28 | `<` → `<=` |
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
| `test_premium_policy :: V4FiiExecutabilityTests.test_owner_profile_with_limitar_concentracao_runs_many_ga_runs` | real data, `resolve_premium_policy(profile(score=0.7, knowledge=0.7, restrictions=["limitar_concentracao"])).fiis` (n 5, cap 2, `hhi_max` 0.25), `n_runs=8`, GA 40 generations × 30 population, seeds 0-3 → no exception, `n_selected == 5`, HHI ≤ 0.25 + 1e-12, ≤ 2 per segment | SEG-42 | HHI skip removed (seed 0 raises "FII selection violates the configured HHI limit", reproduced in spec review r1 and in a scratch prototype on 2026-11-06) |
| `test_premium_policy :: V4FiiExecutabilityTests.test_owner_profile_with_limitar_concentracao_and_13_incumbents_runs` | same policy and budget, seeds 0-3, incumbents = the 13 best-`SCORE` tickers of the scored universe → no exception, 5 funds, HHI ≤ 0.25 + 1e-12, ≤ 2 per segment, zone `status != "fallback_hhi"`, every incumbent in exactly one of `kept`/`departures`, every departure reason in the A-6 set | SEG-42, SEG-39, SEG-43 | protection walk without the HHI term (the run raises) / `K = 0` fallback re-introduced |
| `test_premium_optimization :: SegmentCapForwardingTests.test_fii_engine_receives_the_cap` | fake FII engine's `selection_config["max_per_segment"] == 2` and is `int` | SEG-30 | copy removed |
| `test_premium_optimization :: SegmentCapForwardingTests.test_invalid_cap_fails_with_policy_invalid` | `0`, `True`, `1.5`, `"2"` → `PremiumOptimizationError.code == "policy_invalid"`; engines not called | SEG-31 | check removed |
| `test_premium_optimization :: SegmentCapForwardingTests.test_sections_without_cap_produce_the_pre_change_config` | stock section and a v3 FII section → config equals the literal pre-change dict, no `max_per_segment` key | SEG-32 | key always emitted |
| `test_premium_integration :: SegmentCapIntegrationTests.test_v4_run_records_and_forwards_the_cap` | create + execute → `policy_version == "premium-policy-v4"`, `policy_json["fiis"]["max_per_segment"] == 2`, FII engine got 2 | SEG-33 | wiring |
| `test_premium_integration :: SegmentCapIntegrationTests.test_stored_v3_policy_runs_without_the_cap` | run whose stored `policy_json` lacks the key → FII engine config has no `max_per_segment` | SEG-32, edge | stored v3 changed |
| updated `test_multirun_convergence :: PolicyTests.test_profiles_resolve_expected_n_assets` + `fixture_protocol` | `(13, 5)`, `(15, 6)`; `check_n_assets` vs declared `(13, 13)` raises `ChildError` with `EXIT_USAGE` and "expected_n_assets"; fixture protocol uses v4 counts | SEG-34 | guard |
| updated `test_protection_zone_turnover :: test_measure_writes_both_reports_with_the_turnover_per_class` | `report["policy_version"] == "premium-policy-v4"` | SEG-35 | version |
| Done-when check (T10, T11) | `git diff --exit-code a61164a -- docs/validation/` is empty | SEG-36 | frozen evidence edited |

**Mutation targets for the Verifier's discrimination sensor**: `>` vs `>=` in the GA cap; the `None` guard in `fitness`; segment cut before vs after the capacity cut; departure precedence order; `held - {weakest}` vs `held` in `fits`; skip vs stop for an over-cap challenger; `take_within_segment_cap` skip; cap forwarded to the `K = 0` call; `_consensus` passing `segment_of`; capacity check `<` vs `<=`; final post-check; `_selector_config` copy and its integer check; `_selector_dict` omission of `None`; Basic FII counts; `fii-selection-v2` vs `v1` for Basic; the HHI term of `_fits` removed; the HHI bound on the partial sleeve instead of the projection; `held - {weakest}` vs `held` in the challenger's HHI term; `select_with_fallback` catching `SegmentCapInfeasibleError` from either call (a `K = 0` fallback re-introduced, AD-029); the protection walk's HHI term removed (`_fits` replaced by the cap check); the walk's HHI term applied after `n_assets` are accepted; the walk iterating in a non-rank order; `hhi_max` forwarded without the cap; `_consensus` dropping `hhi_max` on the pure path; `SegmentCapInfeasibleError` swallowed (short list) instead of raised.

---

## Rollback

- **Code**: revert the PR's merge commit on `develop`. No migration, no new table.
- **Persisted runs**: v4 runs stay readable after a revert: `policy_json.fiis.max_per_segment` is an extra key in a JSON column; `_selector_config` before this change ignores it, so re-executing a stored v4 run after a revert would run uncapped (acceptable; runs are not re-executed in place).
- **Partial**: setting `FII_MAX_PER_SEGMENT` high (≥ 6) disables the cap in practice without a version change, but changes the persisted policy, so it needs its own version bump; prefer reverting.

---

## Execution Plan

Phases run in order; tasks within a phase run in order.

| Phase | Tasks | Count | Heavy | Sensitive path (REVIEW.md) | Risk | Suggested tier |
| ----- | ----- | ----- | ----- | -------------------------- | ---- | -------------- |
| 1. Core cap (GA and zone) | T1-T5 | 5 | none | **yes** - `py/core/optimizer.py`, `py/core/protection_zone.py` | High: GA fitness and zone rule that produce the delivered recommendation; byte-identity of the uncapped path; HHI skip (AD-028) including the protection walk, no `K = 0` fallback (AD-029). | high reasoning |
| 2. FII selector wiring | T6-T8 | 3 | none | no (`py/fii_selection.py` is not listed; it is the FII recommendation path) | Medium-high: regression of the uncapped consensus. | high reasoning |
| 3. Policy v4 and tooling tests | T9-T12 | 4 | none (T12 runs real data, under a minute) | no (`api/app/services/premium_policy.py`, tests) | Low-medium: constants, version bump, pinned tests; mechanical. | mid tier |
| 4. Adapter and end to end | T13-T14 | 2 | none | **yes** - `api/app/adapters/premium_optimization.py` | Medium: the bridge that decides what reaches the selector. | high reasoning |

No task is heavy in the migration/snapshot sense: no DB migration, no snapshot regeneration, no committed-data rewrite. The longest gates are the Policy gate (~4 min, T9-T12) and the Build gate at each phase end (~5 min). T12 runs the real FII selector on committed data: four anchors with a 30 × 20 GA budget plus the owner profile with `limitar_concentracao` at 8 runs × 40 × 30 for 4 seeds, with and without incumbents (seconds per seed; the scratch prototype ran 4 seeds in a few seconds).

### Phase 1: Core cap (GA and zone)

```
T1
T2 → T3 → T4 → T5
```

### Phase 2: FII selector wiring

```
T6 → T8
T7 → T8
```

### Phase 3: Policy v4 and tooling tests

```
T9 → T10 → T11 → T12
```

### Phase 4: Adapter and end to end

```
T13 → T14
```

---

## Task Breakdown

### Phase 1 tasks: Core cap (GA and zone)

#### T1: Segment cap in the GA fitness

**Status**: Done

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

- [x] FIRST, before editing `optimizer.py`: golden values (tickers, `fitness`, `hhi`, `generations_run`, `seed`) for seeds 3 and 11 on the test's synthetic frame are captured at `a61164a` and written as literals in `tests/test_segment_cap_optimizer.py`
- [x] Every `test_segment_cap_optimizer` row of the Test Strategy passes
- [x] The new keyword is appended after `early_stopping_min_delta`; `crossover`, `mutate`, `initialize_population`, `optimize`, `optimize_portfolio` unchanged (`git diff` shows edits only in `__init__` and `fitness`)
- [x] Gate check passes: Core (modules present at this point)
- [x] Test count: at least 7 new tests; `tests.test_protection_zone*` counts unchanged

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): make a GA candidate above the segment cap infeasible (#56)`

---

#### T2: Capped greedy pick and cap validation in the zone module

**Status**: Done

**What**: Add `SEGMENT_CAP_DEPARTURE`, `SegmentCapInfeasibleError`, `validate_segment_cap` (with the `hhi_max` checks), the shared `_fits` predicate (cap + projected HHI, spec A-16) and `take_within_segment_cap(..., hhi_max=None)` to the zone module.
**Where**: `py/core/protection_zone.py`
**Depends on**: None
**Reuses**: `_is_integer` (`py/core/protection_zone.py:71-72`), `TOLERANCE`
**Requirement**: SEG-15, SEG-25, SEG-37, SEG-40, SEG-41
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] `take_within_segment_cap` with `max_per_segment=None` returns `list(ordered[:n_assets])`
- [x] The `take_*`, `test_invalid_segment_cap_inputs_raise` and `test_hhi_max_without_cap_raises` (`take_within_segment_cap` half) rows of the Test Strategy pass, one input per guard (L-001)
- [x] `_fits` is the only place that decides the cap and the HHI term; `take_within_segment_cap` calls it (T4 reuses it for the fill and the challenger)
- [x] The module still imports nothing from pandas or `scripts/`
- [x] Gate check passes: Core
- [x] Test count: at least 9 new tests in `tests/test_segment_cap_zone.py`

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): add the segment-capped pick and its input checks (#56)`

---

#### T3: Segment cap inside `apply_protection_zone`

**Status**: Done

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

- [x] Rows SEG-19..SEG-22 and the migration, cap ≥ n and `None`-cap rows of the Test Strategy pass
- [x] `tests/test_protection_zone.py` and `tests/test_protection_zone_equivalence.py` pass with zero edits (`git diff --exit-code -- tests/test_protection_zone.py tests/test_protection_zone_equivalence.py`)
- [x] The existing `SPEC_DEVIATION` comment at `py/core/protection_zone.py:116` is kept
- [x] Gate check passes: Core
- [x] Test count: at least 9 new tests in `tests/test_segment_cap_zone.py` (18+ in the file)

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): apply the segment cap in the protection zone (#56)`

---

#### T4: HHI skip inside `apply_protection_zone`

**Status**: Done

**What**: Add keyword-only `hhi_max` to `apply_protection_zone` (accepted only with a cap): the protection walk, the fill and the challengers through `_fits` (an incumbent the walk rejects leaves with `saiu: teto do segmento`, AD-029), and `SegmentCapInfeasibleError` when the fill cannot reach `n_assets` (spec A-5, A-16, A-17).
**Where**: `py/core/protection_zone.py`
**Depends on**: T3
**Reuses**: `_fits` and `SegmentCapInfeasibleError` from T2; the T3 fill and swap loops
**Requirement**: SEG-38, SEG-39, SEG-41, SEG-43, SEG-24
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] `test_fill_skips_a_ticker_that_would_break_hhi_max`, `test_challenger_that_would_break_hhi_max_is_skipped`, `test_challenger_hhi_check_excludes_the_weakest`, `test_protected_pair_over_hhi_max_keeps_the_higher_ranked_incumbent`, `test_incumbent_past_capacity_is_cut_by_capacity_not_by_hhi`, `test_zone_fill_short_raises_instead_of_a_short_sleeve` and the `apply_protection_zone` half of `test_hhi_max_without_cap_raises` pass
- [x] The protection walk calls `_fits` while fewer than `n_assets` incumbents are accepted and the cap check alone after that (design step 2); no HHI gate raises on the protected set
- [x] `tests/test_protection_zone.py`, `tests/test_protection_zone_equivalence.py` and every T3 test pass with zero edits
- [x] Gate check passes: Core
- [x] Test count: at least 7 new tests in `tests/test_segment_cap_zone.py`

**Tests**: unit
**Gate**: core
**Commit**: `feat(core): apply the hhi_max skip to the zone walk, fill and challengers (#56)`

---

#### T5: Forward the cap through `select_with_fallback`

**Status**: Done

**What**: Add keyword-only `segment_of` and `max_per_segment` to `select_with_fallback` and pass them to both the zone call and the `K = 0` call; forward `hhi_max` to those calls only when the cap is set; let `SegmentCapInfeasibleError` from either call propagate (no `K = 0` fallback for it, AD-029).
**Where**: `py/core/protection_zone.py`
**Depends on**: T4
**Reuses**: existing fallback logic and record (no new record field, A-13)
**Requirement**: SEG-23, SEG-24, SEG-39, SEG-41, SEG-43
**Risk area**: sensitive (`py/core/**`)
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] `test_fallback_consensus_respects_the_cap`, `test_recorded_consensus_tickers_are_capped`, `test_protected_same_segment_pair_keeps_the_protection_without_fallback`, `test_segment_cap_infeasible_error_propagates_from_both_calls` and `test_stock_path_never_forwards_hhi_max` pass
- [x] Record keys are identical with and without a cap (asserted)
- [x] Discrimination evidence in the commit body: on a scratch copy with the protection walk's HHI term removed (cap check only, as in T3), `test_protected_same_segment_pair_keeps_the_protection_without_fallback` fails; the scratch copy is discarded and the real tree is unchanged
- [x] Gate check passes: Build (last task of Phase 1)
- [x] Test count: root suite ≥ 535 + new Phase 1 tests, 2 skipped; API suite 160

**Tests**: unit
**Gate**: build
**Commit**: `feat(core): forward the segment cap to the zone fallback (#56)`

---

### Phase 2 tasks: FII selector wiring

#### T6: Cap from the FII config reaches the GA

**Status**: Done

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

- [x] `test_config_cap_reaches_every_ga_run` and `test_config_without_cap_constructs_the_ga_with_none` pass (recording GA subclass, as `_RecordingGeneticAlgorithm` in `api/tests/test_premium_policy.py`)
- [x] Gate check passes: Selector
- [x] Test count: at least 2 new tests in `tests/test_segment_cap_fii.py`

**Tests**: unit
**Gate**: selector
**Commit**: `feat(fiis): pass the segment cap from the config to the GA (#56)`

---

#### T7: Capped FII consensus

**Status**: Done

**What**: Add `max_per_segment` to `_consensus`: pure path through `take_within_segment_cap` (with `hhi_max` only when the cap is set), zone path with `segment_of` and the cap; `SegmentCapInfeasibleError` becomes `FiiSelectionError`; uncapped path unchanged.
**Where**: `py/fii_selection.py`
**Depends on**: T2, T5
**Reuses**: `take_within_segment_cap`, `select_with_fallback`; a verbatim copy of the `a61164a` `_consensus` body as the regression oracle
**Requirement**: SEG-15, SEG-16, SEG-17, SEG-18, SEG-37, SEG-40, SEG-41
**Risk area**: FII recommendation path
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] The seven `_consensus` rows of the Test Strategy pass (four cap rows and three HHI rows; exact tickers asserted, not just counts)
- [x] `tests/test_protection_zone_selectors.py`, `tests/test_fii_selection.py`, `tests/test_convergence_analysis.py` pass unchanged
- [x] Gate check passes: Selector
- [x] Test count: at least 7 new tests

**Tests**: unit
**Gate**: selector
**Commit**: `feat(fiis): skip a full segment in the FII consensus (#56)`

---

#### T8: `run_fii_selection` reads the cap, checks capacity and the result

**Status**: Pending

**What**: In explicit mode read `max_per_segment` from the config, fail before the GA when the universe cannot fill `n_assets` under it, pass it to `_consensus`, and refuse a final selection above it.
**Where**: `py/fii_selection.py`
**Depends on**: T6, T7
**Reuses**: `_require_feasible_universe` shape, HHI post-check (`py/fii_selection.py:768-770`), `_fixture_frame` style of `tests/test_fii_selection.py:28`
**Requirement**: SEG-26, SEG-27, SEG-28, SEG-29
**Risk area**: FII recommendation path
**Heavy**: no

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] `test_run_with_cap_returns_at_most_two_per_segment` (with the `_consensus` spy and the fixture-binds assertion, review r1 S3), `test_zone_run_records_the_segment_cap_departure`, `test_universe_below_segment_capacity_fails_before_the_ga`, `test_universe_at_exact_segment_capacity_runs` (review r1 S4), `test_over_cap_final_selection_is_refused` pass
- [ ] Named-profile mode never reads the key (no change to `outputs/` or `data/processed/` writers; `git status --porcelain data/ outputs/` empty after the gate)
- [ ] Gate check passes: Build (last task of Phase 2)
- [ ] Test count: at least 5 new tests; root suite ≥ previous + new

**Tests**: integration
**Gate**: build
**Commit**: `feat(fiis): enforce the segment cap in the explicit FII selection (#56)`

---

### Phase 3 tasks: Policy v4 and tooling tests

#### T9: `max_per_segment` on the Premium FII policy

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

#### T10: FII anchors 4/5/6; Basic keeps 10/12/15

**Status**: Pending

**What**: Set `FII_RULES` n to 4/5/6, give Basic `BASIC_FII_ASSET_COUNTS` 10/12/15, and update the tests that pin the v3 FII counts (A-12): the two `test_premium_policy` pins, the conservador executability test, and the #9 tooling tests.
**Where**: `api/app/services/premium_policy.py` (tests: `api/tests/test_premium_policy.py`, `tests/test_multirun_convergence.py`)
**Depends on**: T9
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

#### T11: Bump to `premium-policy-v4` and `fii-selection-v2`

**Status**: Pending

**What**: Set `POLICY_VERSION = "premium-policy-v4"`, Premium `model_versions.fiis = "fii-selection-v2"` (Basic keeps `v1`), and update the tests that pin the v3 version string.
**Where**: `api/app/services/premium_policy.py` (tests: `api/tests/test_premium_policy.py`, `api/tests/test_premium_integration.py`, `tests/test_protection_zone_turnover.py`)
**Depends on**: T10
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

#### T12: Real-data v4 executability for the FII anchors

**Status**: Pending

**What**: Add `V4FiiExecutabilityTests` running the real FII selector on the committed dataset for selector scores 0.0, 0.5, 0.7, 1.0 with the resolved v4 FII section, plus the owner profile with `limitar_concentracao` over 8 runs × 40 × 30 for seeds 0-3, with and without 13 incumbents (SEG-42).
**Where**: `api/tests/test_premium_policy.py`
**Depends on**: T11, T8
**Reuses**: `ConservadorAnchorExecutabilityTests` harness (`_RecordingGeneticAlgorithm`, `GA_BUDGET`, `PINNED_SEED`)
**Requirement**: SEG-05, SEG-27, SEG-42, SEG-43
**Risk area**: none (test only)
**Heavy**: no (four small GA runs on real data)

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] For each score: `n_selected == n_assets` (4/5/5/6), max segment count ≤ 2, recording GA saw `max_per_segment` 2
- [ ] Both `limitar_concentracao` tests pass, the incumbent one with zone `status != "fallback_hhi"` for every seed; the commit body records that, on a scratch copy with the HHI term of `_fits` removed, the no-incumbent test raises "FII selection violates the configured HHI limit" for seed 0 (discrimination evidence; the scratch copy is discarded and the real tree is unchanged)
- [ ] The commit body records the wall-clock of the owner-profile capped run and of the same run without the cap (design Risk 1)
- [ ] Gate check passes: Build (last task of Phase 3)
- [ ] Test count: API suite ≥ 169; root suite unchanged from T11

**Tests**: integration
**Gate**: build
**Commit**: `test(policy): run the v4 FII anchors on the committed dataset (#56)`

---

### Phase 4 tasks: Adapter and end to end

#### T13: Adapter forwards and validates the cap

**Status**: Pending

**What**: `_selector_config` copies an integer `max_per_segment >= 1` from the policy section to the selector config, rejects any other value with `policy_invalid`, and emits no key when the section has none.
**Where**: `api/app/adapters/premium_optimization.py`
**Depends on**: T9
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
- [ ] Test count: API suite ≥ 172

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): forward the FII segment cap to the selector (#56)`

---

#### T14: End-to-end v4 run and stored v3 run

**Status**: Pending

**What**: Add `SegmentCapIntegrationTests`: a created and executed Premium run records and forwards the cap; a run with a stored policy lacking the key executes uncapped.
**Where**: `api/tests/test_premium_integration.py`
**Depends on**: T13, T11
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
- [ ] Test count: root suite ≥ 535 + all new root tests (2 skipped); API suite ≥ 174

**Tests**: integration
**Gate**: build
**Commit**: `test(api): cover a v4 run and a stored v3 run end to end (#56)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1:  T1    T2 ──→ T3 ──→ T4 ──→ T5
Phase 2:  T6 ──┐
          T7 ──┴→ T8
Phase 3:  T9 ──→ T10 ──→ T11 ──→ T12
Phase 4:  T13 ──→ T14
```

Execution is sequential (T1 … T14). Cross-phase dependencies point backward only: T6 ← T1, T7 ← T2/T5, T12 ← T8, T13 ← T9, T14 ← T11.

Suggested batches (whole phases, ~7 tasks, cap ~10): batch 1 = Phases 1-2 (T1-T8, 8 tasks, sensitive `py/core/**` in Phase 1); batch 2 = Phases 3-4 (T9-T14, 6 tasks; Phase 3 is low-risk, Phase 4 is sensitive `api/app/adapters/**`). Phase 3 can also run alone if the orchestrator wants the mechanical policy work separate from the adapter.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: GA cap | 1 class, 2 methods, 1 file | ✅ Granular |
| T2: capped pick + validator + `_fits` | 3 related helpers and 1 exception, 1 file | ✅ OK (cohesive: one predicate and its first caller) |
| T3: zone cap | 1 function | ✅ Granular |
| T4: zone HHI skip | 1 function (protection walk, fill, swap) | ✅ Granular |
| T5: fallback forwarding, no catch | 1 function | ✅ Granular |
| T6: GA config copy | 1 function | ✅ Granular |
| T7: capped consensus | 1 function | ✅ Granular |
| T8: run-level checks | 1 function + 1 helper, 1 file | ✅ OK (cohesive) |
| T9: policy field | 1 dataclass field + 1 constant | ✅ Granular |
| T10: FII anchors | 1 constant table + Basic counts; pinned tests in 2 files move with it | ⚠️ OK: one value change and the pins it breaks must land together or the gate fails |
| T11: version bump | 2 constants; pinned tests in 3 files move with it | ⚠️ OK: same reason as T10 |
| T12: real-data test | 1 test class | ✅ Granular |
| T13: adapter copy | 1 function | ✅ Granular |
| T14: integration tests | 1 test class | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | none (alone in Phase 1 line 1) | ✅ Match |
| T2 | None | none | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T4 | T4 → T5 | ✅ Match |
| T6 | T1 (Phase 1) | none within Phase 2 | ✅ Match (cross-phase, backward) |
| T7 | T2, T5 (Phase 1) | none within Phase 2 | ✅ Match (cross-phase, backward) |
| T8 | T6, T7 | T6 → T8, T7 → T8 | ✅ Match |
| T9 | None | none | ✅ Match |
| T10 | T9 | T9 → T10 | ✅ Match |
| T11 | T10 | T10 → T11 | ✅ Match |
| T12 | T11, T8 (Phase 2) | T11 → T12 | ✅ Match (T8 cross-phase, backward) |
| T13 | T9 (Phase 3) | none within Phase 4 | ✅ Match (cross-phase, backward) |
| T14 | T13, T11 (Phase 3) | T13 → T14 | ✅ Match (T11 cross-phase, backward) |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | GA | unit | unit | ✅ OK |
| T2 | Zone | unit | unit | ✅ OK |
| T3 | Zone | unit | unit | ✅ OK |
| T4 | Zone | unit | unit | ✅ OK |
| T5 | Zone | unit | unit | ✅ OK |
| T6 | FII selector | unit + integration | unit | ✅ OK (pure config plumbing; recording GA) |
| T7 | FII selector | unit + integration | unit | ✅ OK (`_consensus` on synthetic portfolios) |
| T8 | FII selector | unit + integration | integration | ✅ OK |
| T9 | API policy | unit | unit | ✅ OK |
| T10 | API policy + tooling tests | unit | unit | ✅ OK |
| T11 | API policy + tooling tests | unit | unit | ✅ OK |
| T12 | API policy (real data) | unit + real-data executability | integration | ✅ OK |
| T13 | API adapter | unit (fake engines) | unit | ✅ OK |
| T14 | API end to end | integration | integration | ✅ OK |

---

## Requirement → Task Traceability

| Requirement | Tasks | Requirement | Tasks |
| ----------- | ----- | ----------- | ----- |
| SEG-01 | T11 | SEG-19 | T3 (T7 selector level) |
| SEG-02 | T11 | SEG-20 | T3 |
| SEG-03 | T10 | SEG-21 | T3 |
| SEG-04 | T9 | SEG-22 | T3 |
| SEG-05 | T11, T12 | SEG-23 | T5 |
| SEG-06 | T10 | SEG-24 | T3, T4, T5 |
| SEG-07 | T11 | SEG-25 | T2 |
| SEG-08 | T10, T11 | SEG-26 | T8 |
| SEG-09 | T1 | SEG-27 | T6, T8, T12 |
| SEG-10 | T1 | SEG-28 | T8 |
| SEG-11 | T1 | SEG-29 | T8 |
| SEG-12 | T1 | SEG-30 | T13 |
| SEG-13 | T1 | SEG-31 | T13 |
| SEG-14 | T1 | SEG-32 | T6, T13, T14 |
| SEG-15 | T2, T7 | SEG-33 | T14 |
| SEG-16 | T7 | SEG-34 | T10 |
| SEG-17 | T7 | SEG-35 | T11 |
| SEG-18 | T7 | SEG-36 | T10, T11 |
| SEG-37 | T2, T7 | SEG-40 | T2, T7 |
| SEG-38 | T4 | SEG-41 | T2, T4, T5, T7 |
| SEG-39 | T4, T5 | SEG-42 | T12 |
| SEG-43 | T4, T5, T12 | | |

43 of 43 requirements mapped; every task maps to at least one requirement.

---

## Tools per Task

All tasks: no MCP; skill `coding-guidelines`. No external documentation is needed (stdlib `unittest`, pandas `value_counts`, dataclasses already in use).
