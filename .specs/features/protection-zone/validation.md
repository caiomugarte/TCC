# Protection Zone Validation

**Date**: 2026-11-04
**Spec**: `.specs/features/protection-zone/spec.md`
**Diff range**: `6df46f1..7f54329` (18 commits: `c48e396`..`7f54329`)
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-46-verify-r1`

## Validation Verdict

**Result**: PASS ✅. Every requirement that can be tested without November data (PZ-01 to PZ-31, PZ-33, PZ-34) has a test that asserts the outcome the spec defines. The build gate is green. Of 67 mutants, 64 were killed. Two of the three survivors are equivalent mutants (no observable behavior changes), and the third is in validation tooling, outside the critical path. PZ-32 and PZ-35 are **pending data** (T17 deferred, no November snapshot). They are not failures: the tool is tested with fake engines, and the README says the measurement is pending without giving any number.

Gaps: 0 blocking, 3 nonblocking (G-1 to G-3 below).

---

## Task Completion

| Task | Status | Notes |
| ---- | ------ | ----- |
| T1-T16 | ✅ Done | Commits `c48e396`..`c633918` (per `.specs/STATE.md` Handoff, matched against `git log`) |
| T17 | ⏸️ Deferred by design | No published manifest with cutoff ≥ 2026-11-01. `docs/validation/protection-zone-turnover.{json,md}` do not exist, and `docs/validation/README.md` says "Estado: pendente: snapshot de novembro … não há número aqui" |

---

## Spec-Anchored Acceptance Criteria

| Req | Spec-defined outcome | `file:line` + assertion | Result |
| --- | -------------------- | ----------------------- | ------ |
| PZ-01 | One function returns tickers, reasons, departures, kept, swaps, turnover, exits | `tests/test_protection_zone.py:246` asserts all of `tickers`, `reasons`, `departures`, `replaced`, `kept`, `entered`, `(turnover, exits) == (2, 2)` | ✅ |
| PZ-02 | Order freq desc, score desc, ticker asc; stable; dedup by base company | `tests/test_protection_zone.py:33,41,49,60,68,100`; `assertEqual(rank_candidates(...), [...])`; NaN fixture matches `build_consensus_portfolio` | ✅ |
| PZ-03 | Eligible incumbent with rank ≤ K is kept as `mantido` | `tests/test_protection_zone.py:192`: `reasons == {"ALFA3": "mantido", "BETA3": "mantido"}` with BETA3 at rank 3 = K | ✅ |
| PZ-04 | Fill in rank order; `mantido` for incumbent, `novo` otherwise | `tests/test_protection_zone.py:182,372` | ✅ |
| PZ-05 | Swap iff rank above and `f(c) ≥ f(p)+δ−1e-12`; repeat; stop at the first failure | `tests/test_protection_zone.py:211` (21/100 vs 10/100 swaps, 20/100 does not), `:222` (exact tolerance boundary), `:231` (δ=0 tie), `:246` (two swaps, then stop), `:265` | ✅ |
| PZ-06 | Ineligible incumbent never protected; `saiu: inelegível` | `tests/test_protection_zone.py:272`: `departures == {"BETA3": "saiu: inelegível"}`, BETA3 at rank 2 ≤ K excluded; selector level `tests/test_protection_zone_selectors.py:325` | ✅ |
| PZ-07 | Rank > K or unranked → `saiu: fora do top K` | `tests/test_protection_zone.py:201,302` | ✅ |
| PZ-08 | More qualifiers than `n_assets` → best ranked kept, rest `saiu: vagas reduzidas` | `tests/test_protection_zone.py:308` | ✅ |
| PZ-09 | Never two tickers of one base company with dedup | `tests/test_protection_zone.py:286,293`; `tests/test_protection_zone_selectors.py:279,295` | ✅ |
| PZ-10 | No incumbents or K=0 → first `n_assets` of ranking | `tests/test_protection_zone.py:162` (40 random cases), `:182` | ✅ |
| PZ-11 | Invalid K/δ/n_assets/frequency → `ValueError` naming the field | `tests/test_protection_zone.py:120`: `assertRaisesRegex(ValueError, field)` per field; `:116` valid boundaries | ✅ |
| PZ-12 | Order-independent; every incumbent appears once in kept or departures | `tests/test_protection_zone.py:323` (order/case/dup variants equal), `:339` (60 random cases, partition asserted) | ✅ |
| PZ-13 | Equals `zone_portfolio`; lots reproduce 0.40/1, 0.10/1, 0.70/2, 0.00/0 | `tests/test_protection_zone_equivalence.py:58` (random grid, K∈{0, n..30} × δ grid); `:87`: `assertAlmostEqual(mean, …, places=12)`, `assertEqual(max, …)` per cell | ✅ |
| PZ-14 | Zone HHI > hhi_max+1e-12 → consensus, `buffer_fallback: "hhi"`, reasons recomputed | `tests/test_protection_zone.py:418,440` (boundary 1e-12 vs 2e-12), `:490`; selectors `:266,493`; adapter `api/tests/test_premium_optimization.py` `test_hhi_fallback_is_recorded_with_reasons_against_the_consensus` | ✅ |
| PZ-15 | No limit → never falls back, `buffer_fallback: null` | `tests/test_protection_zone.py:453` (`hhi_of` raises if called) | ✅ |
| PZ-16 | Fallback still over → `FiiSelectionError` / `infeasible_constraints`; zone within limit is used even if consensus is over | `tests/test_protection_zone_selectors.py:590,312,510`; `api/tests/test_premium_optimization.py:800` (`code == "infeasible_constraints"`), `:814` | ✅ |
| PZ-17 | No incumbents → frame equals the pre-change frame (rows, order, columns, values, attrs) | `tests/test_protection_zone_selectors.py:209,450`: `assert_frame_equal` + attrs equality against a frozen copy of the pre-change function, with omitted/None/[] and a config that has a zone and hhi_max | ✅ |
| PZ-18 | Stock zone rows + FREQUENCY + attrs incl. `protection_zone` | `tests/test_protection_zone_selectors.py:236` | ✅ |
| PZ-19 | FII zone rows sorted FREQUENCY/SCORE/TICKER, no dedup | `tests/test_protection_zone_selectors.py:466,481` | ✅ |
| PZ-20 | Explicit-mode result has `protection_zone` with incumbents, none without | `tests/test_protection_zone_selectors.py:401,409,415,601,609` | ✅ |
| PZ-21 | `premium-policy-v3`, Basic `basic-policy-v2`, v2 rejected | `api/tests/test_premium_policy.py:316`; integration `api/tests/test_premium_integration.py:368` asserts `premium-policy-v3` | ✅ |
| PZ-22 | Stocks `{6, n+6, 0.11}`, FIIs `{3, n+3, 0.11}` | `api/tests/test_premium_policy.py:324,355`; `api/tests/test_premium_optimization.py:725` | ✅ |
| PZ-23 | Basic JSON has no `protection_zone`, equal to the pre-change JSON | `api/tests/test_premium_policy.py:360`; `api/tests/test_premium_optimization.py:784` | ✅ |
| PZ-24 | Latest completed Premium run of the account, not this run; normalized incumbents | `api/tests/test_premium_executor.py:453` (upper-cased, deduped), `:467`, `:535`; repo `api/tests/test_recommendation_repository.py:193-248` | ✅ (see G-2) |
| PZ-25 | No prior run → runner called without `incumbents` | `api/tests/test_premium_executor.py:441` (fixed-signature runner) | ✅ |
| PZ-26 | Unreadable class → None, `incumbents_unreadable`, run completes | `api/tests/test_premium_executor.py:484`; `api/tests/test_premium_optimization.py:948` | ✅ |
| PZ-27 | Basic plan never queries | `api/tests/test_premium_executor.py:511` (`assert_not_called`), `:524` | ✅ |
| PZ-28 | Provenance record with every listed field and status | `tests/test_protection_zone.py:465` (full dict equality); `api/tests/test_premium_optimization.py` `test_applied_records_the_core_record_and_appends_reasons`, `:927,936,948`, `test_class_with_zero_weight_records_class_not_selected`; `api/tests/test_premium_integration.py:368` (`incumbent_run_id == first.id`) | ✅ |
| PZ-29 | Zone reason appended to `reasons` | `api/tests/test_premium_optimization.py` `test_applied_records_…`: `{"DDD3": ["substituiu CCC3"], "AAA3": ["mantido"]}`; integration `:368` | ✅ |
| PZ-30 | `incumbents` only for a class with ≥1 incumbent; zone copied into the config | `api/tests/test_premium_optimization.py:700,725,745,760` | ✅ |
| PZ-31 | No zone in the policy → no `provenance.protection_zone` | `api/tests/test_premium_optimization.py:980`; `api/tests/test_premium_integration.py:406` | ✅ |
| PZ-32 | Measured October→November report | `tests/test_protection_zone_turnover.py:154` (fake engines: file contents, both turnovers, kept/swapped/departed, K, δ, manifest id/sha, commit, seed) | ⏸️ Pending data (tool verified; measurement not run) |
| PZ-33 | Cutoff < 2026-11-01 or the October id → exit 3, nothing written | `tests/test_protection_zone_turnover.py:242`: `code == EXIT_SNAPSHOT`, `files() == []` | ✅ (see G-1) |
| PZ-34 | Byte-identical rerun | `tests/test_protection_zone_turnover.py:231`: `assertEqual(first, second)` on bytes | ✅ |
| PZ-35 | README links the report once measured; until then says pending | `docs/validation/README.md` section "Giro mês a mês com a zona de proteção": "Estado: pendente … não há número aqui"; no generated file exists | ⏸️ Pending data (pending statement correct) |

**Status**: ✅ All testable ACs covered with spec-anchored assertions; PZ-32/PZ-35 pending data.

---

## Discrimination Sensor

Run in a temporary `git worktree` (scratchpad `mut/`). Each mutant was reverted with `git checkout -- <file>`. The scratch worktree was removed afterwards, and the real tree's porcelain matched the pre-sensor baseline (empty). A first batch was thrown out: `-f` was placed before `discover`, so unittest exited with a usage error and every "kill" was false. After the fix, a control run with no mutant passed, and every mutant below was re-run. Each kill below names the assertion that failed.

**Sensor depth**: P0-full (manual, 67 mutants: data/persistence, policy version, provenance, wiring and the zone rule).

| # | File | Mutation | Result |
| - | ---- | -------- | ------ |
| M01 | `py/core/protection_zone.py:151` | δ test `>=` → `>` | ✅ Killed (`test_swap_happens_exactly_at_the_tolerance_boundary`) |
| M02 | same | drop `- TOLERANCE` | ✅ Killed |
| M03 | same | drop `+ delta` | ✅ Killed |
| M04 | `:119` | rank ineligible picks too | ✅ Killed (`test_ineligible_incumbent_is_never_protected…`) |
| M05 | `:177` | ineligible reason → `fora do top K` | ✅ Killed |
| M06 | `:125` | top-K `<=` → `<` | ✅ Killed |
| M07 | `:125` | ignore K | ✅ Killed |
| M08 | `:127` | no cap at `n_assets` | ✅ Killed |
| M09 / M10 | `:45-46` | dedup ignored / always on | ✅ Killed / ✅ Killed |
| M11 | `:150` | drop the rank condition of the swap | ✅ Killed (δ=0 tie test) |
| M12 | `:146` | drop the `not in replaced` check on the challenger | ⚪ Survived: **equivalent** (see below) |
| M13 | `:145` | weakest = min rank | ✅ Killed |
| M14 | `:158` | single swap only | ✅ Killed |
| M15 / M16 / M17 | `:233-234` | HHI `>`→`>=` / never fall back / fall back but keep the zone selection | ✅ Killed ×3 |
| M18 | `:251` | `consensus_turnover` from the selection | ✅ Killed |
| M19 | `:171` | `mantido` → `novo` | ✅ Killed |
| M20 | `:44` | score tie-break ascending | ✅ Killed |
| M21 | `:115` | no upper/strip of incumbents | ✅ Killed |
| M22 | `py/pipelines/multi_run.py:286` | zone applied without incumbents | ✅ Killed (PZ-17 test) |
| M23 / M24 / M25 | `multi_run.py:298,300,302` | stocks dedup off / hhi_max ignored / zone tickers discarded | ✅ Killed ×3 |
| M26 / M27 | `multi_run.py:1182,1241` | pipeline drops incumbents / omits record | ✅ Killed ×2 |
| M28-M32 | `py/fii_selection.py` | zone without incumbents / dedup on / hhi_max ignored / zone not forwarded / record omitted | ✅ Killed ×5 |
| M33 | `api/app/adapters/premium_optimization.py:315` | `_selector_config` does not copy `protection_zone` | ✅ Killed |
| M34 / M35 | adapter engine calls | stocks / FII incumbents not forwarded | ✅ Killed ×2 |
| M36 | adapter `engine_incumbents` | pass `[]` to the engine | ✅ Killed |
| M37 | `_default_stock_engine` | drop `incumbents=` | ✅ Killed |
| M38 | adapter provenance | `incumbent_run_id` → None | ✅ Killed |
| M39 / M40 | `_constituents` / `_zone_reasons` | reason not appended / `substituiu` → `mantido` | ✅ Killed ×2 |
| M41 / M42 | `_zone_record` | `incumbents_unreadable` / `class_not_selected` → `no_incumbents` | ✅ Killed ×2 |
| M43 | `_zone_record` | missing record no longer raises `engine_error` | ✅ Killed |
| M44 | adapter | provenance key always written | ✅ Killed |
| M45 | adapter | `policy_dict` via `asdict` (Basic emits `protection_zone: null`) | ✅ Killed |
| M46 | `api/app/services/premium_executor.py:176` | drop `exclude_run_id=run.id` | ⚪ Survived: **equivalent** (see below) |
| M47 / M48 / M49 | executor | Basic plan queries / ignores zone declaration / always passes `incumbents` | ✅ Killed ×3 |
| M50 / M51 / M52 / M53 | `_incumbent_tickers` | no upper / no dedup / bad entry skipped instead of unreadable / FIIs read from `stocks` | ✅ Killed ×4 |
| M54 / M55 / M56 | `api/app/repositories/recommendations.py:171-173` | drop the status / plan / account filter | ✅ Killed ×3 |
| M57b / M58b | `recommendations.py:179-180` | `created_at` ascending / `id` ascending | ✅ Killed ×2 |
| M59 | `api/app/services/premium_policy.py:12` | version back to v2 | ✅ Killed |
| M60 / M61 / M62 | `PROTECTION_ZONE` / `_protection_zone` | stocks offset 5 / FII δ 0.1 / K = offset | ✅ Killed ×3 |
| M63 | `_selector_dict` | Basic emits `protection_zone: null` | ✅ Killed |
| M64 | `scripts/protection_zone_turnover.py:49` | minimum cutoff → October | ✅ Killed |
| M65 | `scripts/protection_zone_turnover.py:48` | October-id refusal disabled | ❌ Survived (G-1) |

**Equivalent-mutant arguments**

- **M12**: a replaced `p` was the protected ticker with the highest rank still held. Any not-held ticker ranked above `p` comes before it in `ranking`. So if the next challenger can be `p`, no not-held ticker ranks above `p`. Without the exclusion the challenger is `p`; with it, the challenger ranks below `p`. Either way `rank[c] < rank[weakest']` fails, because every remaining protected ticker ranks above `p`, and the loop stops with the same result. `scripts/convergence_analysis.py:zone_portfolio` keeps the same check, and the equivalence grid passes either way.
- **M46**: `execute_run` commits `mark_running(run_id)` (`premium_executor.py:115-116`) before `_incumbents` runs (`:123`). The query also filters `status == "completed"`, so the run being executed can never be returned. Observable behavior is the same. The repository parameter itself is covered by `api/tests/test_recommendation_repository.py:238`.

**Result**: 64 killed, 2 equivalent, 1 survived outside the critical path. PASS ✅.

---

## Deviations reviewed

1. **SPEC_DEVIATION `py/core/protection_zone.py:116`** (the ranking uses only eligible picks; the design says "every pick"). **Not a gap.** In production every pick is eligible. Stocks look up each pick's score with `df_ranked[...]["SCORE"].iloc[0]` (`py/pipelines/multi_run.py:239`), which raises if the pick is not in `df_ranked`. FIIs index `scores = df_ranked.set_index("TICKER")["SCORE"]` (`py/fii_selection.py:569`). Both selectors pass `eligible = set(df_ranked["TICKER"])`. So the two rankings are identical for every reachable input. For unreachable inputs, the implemented rule is the one consistent with PZ-06 ("never protected", and the fill draws from R) and PZ-10. Under the design's literal reading, an ineligible non-incumbent pick could be filled. The spec text A-1 still reads "every picked ticker" (G-3).
2. **The adapter fails with `engine_error` when a selector receives incumbents but returns no zone record** (`api/app/adapters/premium_optimization.py:446-451`). **Not a gap.** It is recorded in design "Error Handling". It can only trigger on a selector bug: the adapter forwards incumbents only when it also copies `protection_zone` into that selector's config (`_class_incumbents`, `_selector_config`). Both real selectors then always return `applied` or `fallback_hhi` (M26/M27/M31/M32 prove the record path), and a fallback that is still over the limit raises before reaching the adapter (PZ-16). Failing loudly protects the provenance contract better than recording a wrong `status`. It does not contradict A-8: unreadable data reaches the adapter as `None`, so the engine gets no incumbents and the run completes (`api/tests/test_premium_optimization.py:948`). Killed by M43.

---

## Gaps

| ID | Severity | Path | Description | Fix task |
| -- | -------- | ---- | ----------- | -------- |
| G-1 | Nonblocking | `tests/test_protection_zone_turnover.py:246`, `scripts/protection_zone_turnover.py:121,128` | The "October manifest id" case passes `dev-2026-10-29-ibov`, which has no fixture file, so it exits 3 through the "missing" path. Neither id guard is discriminated (M65 survives). The code is correct by inspection, and the tool is outside the critical path | Add a fixture published as `dev-2026-10-29-ibov` with cutoff ≥ 2026-11-01 (and one whose file holds that id under another name), and assert exit 3 with "October manifest" in stderr |
| G-2 | Nonblocking | `api/tests/test_premium_executor.py:453` | The executor's `exclude_run_id=run.id` argument is not asserted. The mutant is equivalent today (status filter plus `mark_running` first), but the guard would go silently missing if the call moved before `mark_running` | Optional: assert `get_latest_completed_for_account.assert_called_once_with(account_id, plan="premium", exclude_run_id=run.id)` |
| G-3 | Nonblocking (spec text) | `.specs/features/protection-zone/spec.md` A-1, `design.md` algorithm step 2 | The text says R = "every picked ticker", while the implementation (and PZ-06/PZ-10) rank eligible picks only | Reword A-1 and design step 2 to "every eligible picked ticker" and drop the SPEC_DEVIATION marker |

---

## Code Quality

| Principle | Status |
| --------- | ------ |
| Minimum code / surgical changes | ✅ The no-incumbent paths in both selectors are unchanged (PZ-17 frame equality) |
| No scope creep | ✅ Basic unchanged; no UI; no recalibration |
| Matches patterns | ✅ Same sort key, suffix list and tolerance as the existing code and `convergence_analysis` |
| Spec-anchored outcome check | ✅ Exact reasons, record dicts, codes and statuses asserted |
| Per-layer coverage | ✅ Core 1:1 per AC; selectors, adapter, executor, repository and integration each covered |
| Every test maps to a requirement | ✅ Tests carry `# PZ-NN` tags or sit in AC-named classes |
| Guidelines | `.specs/STATE.md` decisions AD-022..AD-026 followed |

---

## Edge Cases

- [x] Class weight 0 → `class_not_selected`, no turnover (`api/tests/test_premium_optimization.py` `test_class_with_zero_weight_records_class_not_selected`)
- [x] Fewer candidates than `n_assets` (`tests/test_protection_zone.py:372`)
- [x] Duplicate or lower-case incumbents normalized (`api/tests/test_premium_executor.py:453`, `tests/test_protection_zone.py:323`)
- [x] Two legacy incumbents of one company (`tests/test_protection_zone.py:302`)
- [x] δ=0 tie broken in favor of the incumbent → no swap (`tests/test_protection_zone.py:231`)
- [x] Float boundary 0.21 vs 0.10+0.11 swaps (`tests/test_protection_zone.py:211,222`)

---

## Gate Check

- **Gate command**: root `.venv/bin/python -m unittest discover -s tests -p "test_*.py"`; API `cd api && PYTHONPATH=.:../py .venv/bin/python -m unittest discover -s tests -t .`
- **Result**: root 535 passed, 0 failed, 2 skipped (both skips are in suites this feature did not touch); API 160 passed, 0 failed
- **Test count before feature**: root 472 (tasks.md T4 baseline); API not recorded
- **Delta**: root +63 (the `test_protection_zone*.py` files hold 63 tests); no test was deleted or weakened (the v2 pin updated to v3 in T9 is a change of requirement)

---

## Requirement Traceability Update

| Requirement | Previous Status | New Status |
| ----------- | --------------- | ---------- |
| PZ-01 … PZ-31, PZ-33, PZ-34 | Implemented | ✅ Verified |
| PZ-32, PZ-35 | In Tasks (T17 deferred) | ⏸️ Pending data (tool and pending notice verified) |

---

## Summary

**Overall**: ✅ Ready (P2 measurement pending a November snapshot)

**Spec-anchored check**: 33/33 testable ACs matched the spec outcome; 2 pending data
**Sensor**: 64/67 killed, 2 equivalent, 1 nonblocking survivor
**Gate**: root 535 OK (2 skipped), API 160 OK

**Next steps**: Optional fixes for G-1 to G-3. Run T17 as an operator task once a November manifest is published.
