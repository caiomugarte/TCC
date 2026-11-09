# Premium Whole-Unit Buy List Validation

**Date**: 2026-11-09
**Spec**: `.specs/features/premium-whole-units/spec.md`
**Issue**: #57
**Diff range**: `af1080e..70b051c` (`origin/develop..t3/issue-57`, 16 commits)
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-57-verify-r1`

## Validation: PASS

**Verdict**: PASS. 29/29 requirements and 7/7 issue acceptance criteria have code and a passing test. 26 of 27 API mutants were killed; the one survivor (M12) is equivalent. Four nonblocking gaps remain, all outside the critical paths.

---

## Task Completion

| Task | Status | Commit |
| ---- | ------ | ------ |
| T1 schema fields | ✅ Done | `c649dcf` |
| T2 router forwards dropped lists | ✅ Done | `6eef239` |
| T3 hashed CSV price loader | ✅ Done | `1cf4ed3` |
| T4 floor + remainder loop | ✅ Done | `f732afd` |
| T5 drop + 3-FII minimum | ✅ Done | `495834e` |
| T6 `whole-units-v1` model version | ✅ Done | `c91752e` |
| T7 adapter buy list | ✅ Done | `800125a` |
| T8 gate + price failures | ✅ Done | `297e023` |
| T9 executor gate | ✅ Done | `fd7defc` |
| T10 incumbents = bought only | ✅ Done | `55eebea` |
| T11 end to end | ✅ Done | `0d3eba7` |
| T12 web types | ✅ Done | `c60c07b` |
| T13 web view | ✅ Done | `413c744` |

---

## Spec-Anchored Acceptance Criteria

Code references are at HEAD `70b051c`. Test references point to the test method.

### P1 Buy list (LOT-01..LOT-17)

| Req | Spec-defined outcome | Code | Test + assertion | Result |
| --- | -------------------- | ---- | ---------------- | ------ |
| LOT-01 | Each bought row has `price_brl`, int `quantity` ≥ 1, `amount_brl = q × price`, `effective_portfolio_weight = round(amount/capital, 6)` | `api/app/adapters/premium_optimization.py:544-595` | `api/tests/test_premium_optimization.py:1214` asserts `price_brl == float(price)`, `quantity == LOT_TEST_1 qty`, `cents(amount) == q × cents(price)`, `KNRI11 effective_portfolio_weight == 0.014789`, `MXRF11 amount_brl == 103.68` | ✅ PASS |
| LOT-02 | `PRECO` of the stripped, upper-cased `TICKER` row; Decimal, cents, half up | `api/app/services/status_invest_inputs.py:125-168` | `api/tests/test_status_invest_inputs.py:198` (`"  knri11 "` row → `KNRI11: 14789`, BOM file, 13 exact cent values); `:277` (`10.005 → 1001`, `10.004 → 1000`) | ✅ PASS |
| LOT-03 | Hash recomputed and equal to `selector_inputs.<class>.sha256`; no network | `status_invest_inputs.py:149`; adapter passes `selector_inputs.stocks` / `.fiis` at `premium_optimization.py:804-821` | `test_status_invest_inputs.py:226` (changed bytes → `SelectorSnapshotChangedError`); `:287` (`socket.socket` patched to raise → prices still `{KNRI11: 14789}`); `test_premium_optimization.py:1532` (`assertIs(calls[0][0], inputs.stocks)`, `assertIs(calls[1][0], inputs.fiis)`) | ✅ PASS |
| LOT-04 | Hash mismatch → `failure_code == "snapshot_changed"`, no completed result | `premium_optimization.py:616-617`; executor `_fail` path `premium_executor.py:177-180` | `test_premium_optimization.py:1551` asserts `raised.exception.code == "snapshot_changed"` | ✅ PASS |
| LOT-05 | Missing row / column, or empty, non-numeric, non-finite or ≤ 0 `PRECO` → `engine_error` | `status_invest_inputs.py:125-136,156-167`; `premium_optimization.py:618-619,562-563` | `test_status_invest_inputs.py:236` (9 subcases + unreadable file, message names `KNRI11`); `test_premium_optimization.py:1568` asserts `code == "engine_error"` (absent ticker, loader without the price) | ✅ PASS |
| LOT-06 | Start at `floor(target_cents / price_cents)` | `api/app/services/whole_units.py:79` | `api/tests/test_whole_units.py:48` asserts `(4, 1)` and residual `2000` | ✅ PASS |
| LOT-07 | While a unit fits: +1 unit to the largest deficit among those that fit, `≤` remaining, ties to the earlier ticker | `whole_units.py:81-90` | `test_whole_units.py:55` `(10, 15, 2)`; `:65` exact fill `(10, 10, 1)`, residual 0 (kills `<`); `:76` tie `(2, 1)` | ✅ PASS |
| LOT-08 | The loop stops when nothing fits | `whole_units.py:82-84` | `test_whole_units.py:108` (200 random cases: `all(price > residual)`) | ✅ PASS |
| LOT-09 | Class Σ amount ≤ target, except LOT-13 | `whole_units.py:80-90,107-113` | `test_whole_units.py:108` (`spent ≤ class_target`, `over_target == 0`); `:246` (minimum case: `over_target == spent − target` only when forced); adapter `test_premium_optimization.py:1342` (Σ amount + residual − excess == class target, in cents) | ✅ PASS |
| LOT-10 | Quantity 0 → leaves the list, goes to `*_dropped` in selector order with `ticker`, weights, target, `price_brl`, reasons + `não coube no capital` | `premium_optimization.py:568-579` | `test_premium_optimization.py:1240` asserts the exact dict for `DDD11` (`reasons == ["mantido", "não coube no capital"]`) and for `ZZZ3` | ✅ PASS |
| LOT-11 | Bought rows keep selector order and the existing fields | `premium_optimization.py:581-590` (`**row`) | `test_premium_optimization.py:1214` (order == LOT_TEST_1 order, `target_amount_brl == 100.0`, `reasons == []`); `:1510` (the shared fields are equal on/off for every ticker) | ✅ PASS |
| LOT-12 | No re-normalisation after a drop | `whole_units.py:78-90` (single pass) | `test_whole_units.py:170` `(1, 0, 8)`, residual 0; `:157` `(16, 7, 0)`, residual 0 | ✅ PASS |
| LOT-13 | Fewer than `min(3, n)` FIIs bought → 1 quota of the best-placed dropped, reason `mínimo de 3 FIIs` | `whole_units.py:92-102`; `premium_optimization.py:584,819` | `test_whole_units.py:180` `(2, 2, 1, 0)`, forced `[F,F,T,F]`, excess 12000; `:207` no FII fits → `(1, 1, 1, 0)`; `:221` n=2 cap; `:194` not applied with 3 bought; adapter `test_premium_optimization.py:1291` (`FFC11` reasons `["mínimo de 3 FIIs"]`, `over_target_brl == 120.0`) | ✅ PASS |
| LOT-14 | Stocks and FIIs classes carry `cash_residual_brl` / `over_target_brl` in cents; `over > 0` only under LOT-13 | `premium_optimization.py:591-595,609-627` | `test_premium_optimization.py:1214` (`cash_residual_brl == 4.04`, `over_target_brl == 0.0`); `:1291` (`120.0` / `0.0`); `:1309` (zero-weight class `cash_residual == target`, `over == 0.0`) | ✅ PASS |
| LOT-15 | No minimum for stocks | `premium_optimization.py:811` (`minimum_count=0`) | `test_whole_units.py:233` `(0, 0, 0)`, residual 10000; adapter `:1240` (`ZZZ3` dropped, not forced) | ✅ PASS |
| LOT-16 | `assumptions` contains the exact sentence | `premium_optimization.py:538-541,856-857` | `test_premium_optimization.py:1337` asserts `assumptions == [*PRE_CHANGE_ASSUMPTIONS, WHOLE_UNIT_ASSUMPTION]`; the test's literal at `:1105` | ✅ PASS |
| LOT-17 | Class weights, targets, tickers, seed and `policy_version` identical on/off | the buy list runs after the selectors (`premium_optimization.py:803-821`) | `test_premium_optimization.py:1510` asserts class tuples, shared ticker fields, `random_seed` and `policy` equal on/off | ✅ PASS |

### P1 Incumbents (LOT-18..LOT-20)

| Req | Spec-defined outcome | Code | Test + assertion | Result |
| --- | -------------------- | ---- | ---------------- | ------ |
| LOT-18 | No `quantity` / `null` / int ≥ 1 kept; int 0 skipped | `api/app/services/premium_executor.py:42-50` | `api/tests/test_premium_executor.py:562` asserts incumbents `stocks == ["ITSA4"]`, `fiis == ["HGLG11", "KNRI11", "XPML11"]` | ✅ PASS |
| LOT-19 | Negative, float, bool or string → class `None` | `premium_executor.py:46-47` | `test_premium_executor.py:583` (subcases `-1`, `1.5`, `True`, `"2"` → `stocks: None`) | ✅ PASS |
| LOT-20 | `*_dropped` never incumbents | `premium_executor.py:36-38` (reads only `stocks` / `fiis`) | `test_premium_executor.py:605`; end to end `api/tests/test_premium_integration.py:685` (run 2 FII incumbents `== ["WUA11", "WUB11", "WUC11"]`, `WUD11` dropped in run 1) | ✅ PASS |

### P1 Old results (LOT-21..LOT-24)

| Req | Spec-defined outcome | Code | Test + assertion | Result |
| --- | -------------------- | ---- | ---------------- | ------ |
| LOT-21 | Pre-change `result_json` → 200, new fields `null`, dropped lists `[]` | `api/app/routers/recommendations.py:36-37`; schema defaults `api/app/schemas/recommendation.py:44-47,81-82` | `api/tests/test_product_routes.py:636` (by id and latest-completed: `quantity`, `priceBrl`, `amountBrl`, `effectivePortfolioWeight`, `cashResidualBrl`, `overTargetBrl` are `None`; `stocksDropped == []`); `test_premium_integration.py:710` (a real stored pre-change run reads and feeds incumbents) | ✅ PASS (see G3: `GET ""` latest is not exercised directly) |
| LOT-22 | New fields exposed camel-cased | `recommendations.py:59-60`; schema `:34-35,44-56` | `test_product_routes.py:658` (`priceBrl 100.0`, `quantity 200`, `cashResidualBrl 12.5`, `overTargetBrl 120.0`, `fiisDropped[0].priceBrl 147.89`); `api/tests/test_recommendation_schema.py:81` | ✅ PASS |
| LOT-23 | `quantity` int ≥ 1, `price_brl` > 0, the others ≥ 0, all optional | `recommendation.py:34-35,44-47` | `test_recommendation_schema.py:119` (each bound rejects its own input); `:67` (legacy entries validate) | ✅ PASS |
| LOT-24 | Basic `result_json` identical to before (no new keys) | `premium_policy.py:113-117` (Basic without `whole_units`); `premium_executor.py:144` (plan gate); adapter default `whole_units=False` | `test_premium_optimization.py:1502` (`== PRE_CHANGE_GOLDEN` captured from `af1080e`, loader raises if called); `test_premium_executor.py` `test_basic_or_undeclared_policy_never_passes_the_flag` (Basic with the declaration → no flag); `api/tests/test_premium_policy.py:423` (Basic model versions exact) | ✅ PASS |

### P1 Screen (LOT-25..LOT-29)

| Req | Spec-defined outcome | Code | Test + assertion | Result |
| --- | -------------------- | ---- | ---------------- | ------ |
| LOT-25 | Ticker, quantity, unit price and `amountBrl` in BRL per stock and FII | `web/components/recommendation/recommendation-view.tsx:66-74,285,295` | `web/tests/premium-recommendation.test.mjs:139` (BRL formatter, `${item.quantity} × ${money(item.priceBrl)} = ${money(item.amountBrl)}`) | ✅ PASS |
| LOT-26 | Class `cashResidualBrl` shown; `overTargetBrl > 0` shown as above target | `recommendation-view.tsx:92-99,289,299` | `premium-recommendation.test.mjs:139` (regex on both lines) | ✅ PASS (weak: see G1) |
| LOT-27 | Dropped tickers listed with unit price and the API `reasons`, no reason text in `web/` | `recommendation-view.tsx:76-90,288,298` | `premium-recommendation.test.mjs:146` (both `DroppedList` call sites, `reasons.join`, walk of `web/**/*.ts(x)` rejects the two reason strings) | ✅ PASS (unit price not asserted: see G2) |
| LOT-28 | Price note with `retrieved_at`, guidance, not a live quote | `recommendation-view.tsx:303-307` | `premium-recommendation.test.mjs:159` (gate `some(hasQuantity)`, note text with `statusInvestRetrievedAt(completed)`) | ✅ PASS |
| LOT-29 | No `quantity` → `ticker: weight`, no note | `recommendation-view.tsx:41-43,69-71` | `premium-recommendation.test.mjs:159` (legacy branch regex, `typeof item.quantity === "number"`) | ✅ PASS |

### Issue #57 acceptance criteria

| Issue AC | Evidence | Result |
| -------- | -------- | ------ |
| Fields `price_brl`, `quantity`, `amount_brl`, `effective_portfolio_weight`; class `cash_residual_brl` | LOT-01, LOT-14 | ✅ |
| Price from `PRECO` of the run's hashed selector CSV, no network | LOT-02, LOT-03 | ✅ |
| Floor, then 1 unit at a time to the ticker furthest below target; class ≤ target except the minimum | LOT-06..LOT-09 | ✅ |
| Quantity 0 leaves with `não coube no capital`; its value funds the others; 3-FII minimum with the excess recorded | LOT-10, LOT-12, LOT-13, LOT-14 | ✅ |
| Only quantity > 0 are next month's incumbents | LOT-18..LOT-20 | ✅ |
| Screen shows quantity, amount, class cash and the snapshot-date note | LOT-25, LOT-26, LOT-28 | ✅ |
| Tests: KNRI11/KNCR11 (`test_whole_units.py:81`, adapter `:1214`: `cash_residual_brl == 4.04`, spent 1295.96); ticker that drops (`:157`, adapter `:1240`); 3-FII minimum (`:180`, adapter `:1291`); class ≤ target (`:108`, `:246`); old `result_json` read (`test_product_routes.py:636`, `test_premium_integration.py:710`) | listed | ✅ |

**Status**: ✅ All ACs covered. No spec-precision gap: every criterion names an exact outcome, and the tests assert those values.

---

## Edge Cases

- [x] Exact fill accepts the last unit (`≤`): `test_whole_units.py:65`, mutant M1 killed.
- [x] Every stock drops: allocator `test_whole_units.py:233` (`(0,0,0)`, residual = class target).
- [x] No FII fits: `test_whole_units.py:207`.
- [x] Class weight ≤ 1e-9: `test_premium_optimization.py:1309` (no price read, `cash_residual == target`, empty lists).
- [x] Duplicate CSV rows: equal values accepted (`test_status_invest_inputs.py:270`), conflicting values rejected (`:236` subcase).
- [x] `effective_portfolio_weight` > 1 kept: the schema has no upper bound (`recommendation.py:47`); the forced-FII adapter test gets 0.12.

---

## Discrimination Sensor

Scratch: a detached `git worktree` at HEAD under the session scratchpad. Each mutant was applied, the targeted test classes were run, and the file was restored byte for byte. The scratch was removed at the end, and the real tree's porcelain still matches the baseline (`?? web/node_modules`). Depth: **P0** (`api/app/adapters/**` and `api/app/routers/**` are sensitive paths in REVIEW.md), so 27 API mutants plus 5 web mutants.

| # | File | Mutation | Result |
| - | ---- | -------- | ------ |
| M1 | `api/app/services/whole_units.py:82` | `price ≤ remaining` → `<` | ✅ Killed (8 failures) |
| M2 | `whole_units.py:87` | tie goes to the later ticker | ✅ Killed (8) |
| M3 | `whole_units.py:79` | no floor (start at 0) | ✅ Killed (11) |
| M4 | `whole_units.py:96` | minimum takes the worst-placed (`reversed(order)`) | ✅ Killed (11) |
| M5 | `whole_units.py:87` | cheapest unit instead of largest deficit | ✅ Killed (9) |
| M6 | `api/app/adapters/premium_optimization.py:576` | dropped reason without `não coube no capital` | ✅ Killed |
| M7 | `premium_optimization.py:617` | `snapshot_changed` mapped to `engine_error` | ✅ Killed |
| M8 | `premium_optimization.py:819` | FII minimum 0 | ✅ Killed |
| M9 | `premium_optimization.py:588` | effective weight from the target, not the amount | ✅ Killed |
| M10 | `premium_optimization.py:857` | assumption sentence not added | ✅ Killed |
| M11 | `premium_optimization.py:627` | class residual/excess not written | ✅ Killed (7 errors) |
| M12 | `premium_optimization.py:611` | zero-weight class `cash_residual_brl = 0.0` instead of the target | ⚪ Survived, **equivalent** (see below) |
| M13 | `premium_optimization.py:584` | forced FII loses `mínimo de 3 FIIs` | ✅ Killed |
| M14 | `premium_optimization.py:568` | quantity-0 rows kept in the buy list | ✅ Killed |
| M15 | `api/app/services/premium_executor.py:48` | quantity 0 kept as incumbent | ✅ Killed |
| M16 | `premium_executor.py:46` | bool accepted as a quantity | ✅ Killed |
| M17 | `premium_executor.py:144` | plan check dropped (Basic gets whole units) | ✅ Killed |
| M18 | `premium_executor.py:78` | any `whole_units` value enables it | ✅ Killed |
| M19 | `api/app/services/status_invest_inputs.py:149` | hash check removed | ✅ Killed |
| M20 | `status_invest_inputs.py:132` | default rounding (half even) | ✅ Killed |
| M21 | `status_invest_inputs.py:162` | conflicting duplicate rows accepted | ✅ Killed |
| M22 | `status_invest_inputs.py:133` | price of 0 cents accepted | ✅ Killed |
| M23 | `api/app/routers/recommendations.py:37` | `fiis_dropped` not forwarded | ✅ Killed |
| M24 | `recommendations.py:36` | `stocks_dropped` not forwarded | ✅ Killed |
| M25 | `api/app/services/premium_policy.py:111` | `whole_units` dropped from Premium model versions | ✅ Killed |
| M26 | `api/app/schemas/recommendation.py:45` | `quantity ge=0` | ✅ Killed |
| M27 | `premium_optimization.py:804` | whole-unit wiring off | ✅ Killed (28) |
| W1 | `web/components/recommendation/recommendation-view.tsx:299` | FII `ClassCash` removed | ❌ Survived (G1) |
| W2 | `recommendation-view.tsx:298` | FII `DroppedList` removed | ✅ Killed |
| W3 | `recommendation-view.tsx:96` | "above target" gate `> 0` → `>= 0` | ❌ Survived (G1) |
| W4 | `recommendation-view.tsx:289` | stocks `ClassCash` reads the FII class | ❌ Survived (G1) |
| W5 | `recommendation-view.tsx:84` | dropped row without the unit price | ❌ Survived (G2) |

**M12 is equivalent, not a weak test.** The `if not rows` branch runs only when the class weight is ≤ 1e-9. Class weights are rounded to 6 decimals first (`premium_optimization.py:690`, `_rounded_weights`), so in that branch the weight is 0.0 and `_rounded_amounts` gives the class 0 cents. `float(target)` and `0.0` are then the same value. A fuzz of about 200k random allocations, with capital from R$ 1 to R$ 999M, gave a gated-off class a non-zero target 0 times.

**Result**: PASS - API 26/26 non-equivalent mutants killed (critical paths fully discriminated). Web 1/5 killed: nonblocking (outside the critical paths, the code is correct, and the criteria have proofs; see G1 and G2).

---

## Gate Check

| Gate | Command | Result |
| ---- | ------- | ------ |
| API | `(cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .)` | **218 OK** (207.8 s) |
| Root | `$PY -m unittest discover -s tests -p "test_*.py"` | **587 OK, 2 skipped** (183.7 s) |
| Web static | `node --test web/tests/premium-recommendation.test.mjs` | **14 pass**, 0 fail |
| Web types | `(cd web && npm run typecheck)` | `tsc --noEmit` exit 0 |

- **API test count**: 174 before (`af1080e`), 218 after: +44. No test was deleted or weakened. The existing tests the diff changes are `test_premium_policy.py :: test_model_versions_record_fii_selection_v2` (named by T6) and four integration helpers (T9 deviation, assertions unchanged).
- **Web**: 9 before, 14 after: +5.

---

## REVIEW.md checks

| Rule | Finding |
| ---- | ------- |
| `model_versions` updated when the method changes | ✅ `PREMIUM_MODEL_VERSIONS["whole_units"] = "whole-units-v1"` (`premium_policy.py:111`). `POLICY_VERSION` stays `premium-policy-v4` by A-4, because the seed derives from it. Pinned by `test_premium_policy.py:423`; mutant M25 killed. |
| Strings / enums copied between `api/` and `web/lib/api-types.ts` | ✅ Field names are compared with the API schema source by `premium-recommendation.test.mjs:167`, using the same `to_camel` rule. The reason strings are not copied: `:146` walks `web/**` and rejects them. |
| `api/app/routers/**` scoping and Premium gate | ✅ Only two `result.get` reads were added (`recommendations.py:36-37`) under the existing `_response_with_profile`. `get_owned_run` scoping (`:133`) and the plan gates are untouched. |
| `api/app/adapters/**` decides the recommendation | ✅ Every mutant in the adapter rounding, price and incumbent logic was killed, except the equivalent M12. |
| A new test fails when its behavior is removed | ✅ API. ⚠️ Web class cash and dropped price (G1, G2). |
| Wrong number with no visible error | ✅ A missing or bad price or a changed hash fails the run (LOT-04, LOT-05); nothing is guessed. |

---

## Gaps

| # | Severity | Gap | Evidence | Fix task |
| - | -------- | --- | -------- | -------- |
| G1 | Nonblocking (web, outside the critical paths; the code is correct) | LOT-26 render is not discriminated per class: removing the FII class cash (W1), wiring stocks to the FII class (W4) or showing "Acima do alvo" at 0 (W3) all pass. | `premium-recommendation.test.mjs:139-144` matches the `ClassCash` body once, not the two call sites or the `> 0` gate | Assert both call sites (`key === "brazilian_stocks"`, `key === "fiis"`) and the `> 0` gate. |
| G2 | Nonblocking (web) | LOT-27 "with their unit price": no assertion covers `money(item.priceBrl)` in `DroppedList` (W5). | `premium-recommendation.test.mjs:146-157` | Add a regex for `{item.ticker} · {money(item.priceBrl)} · {item.reasons.join(" · ")}`. |
| G3 | Nonblocking | LOT-21 names the latest route; tests exercise by-id and latest-completed, not `GET /v1/recommendations` (latest). It goes through the same `_response_with_profile` (`recommendations.py:87`). | `test_product_routes.py:636` | Add `read_latest_recommendation` to the route loop. |
| G4 | Nonblocking | No test chains the hash recorded by `refresh_inputs` (`file_sha256` → `sha256_path`) into `load_selector_prices` (`hashlib.sha256(...).hexdigest()`). Both are lowercase hex SHA-256 of the whole file (`py/snapshot_manifest.py:55-65`, `status_invest_inputs.py:149`), so production matches today; a format change on one side would only show up at run time as `snapshot_changed`. | `test_status_invest_inputs.py` and `test_premium_integration.py` build `sha256` with `hashlib` directly | One test: `refresh_inputs` with fake collectors, then `load_selector_prices` on its snapshot. |

**Observation (not counted, out of scope by spec)**: `PremiumRecommendationService.create_run` returns the latest terminal Premium run when `force` is false (`api/app/services/premium_recommendation.py:149,162`). After deploy, a user keeps seeing their pre-change result, without quantities, until they force a new run. The spec puts migrating stored runs out of scope, and LOT-21 requires that result to read as it is.

---

## Code Quality

| Principle | Status |
| --------- | ------ |
| Minimum code | ✅ Pure allocator in one 113-line module; the adapter adds two helpers. |
| Surgical changes | ✅ Basic is byte-identical (golden). Research tools are unchanged (adapter default off; the root suite is green). |
| No scope creep | ✅ |
| Matches patterns | ✅ The flag passes like `incumbents`. Failure codes reuse `snapshot_changed` / `engine_error`. |
| Spec-anchored outcome check | ✅ |
| Per-layer coverage | ✅ Allocator 1:1 with LOT-06..15. Adapter, executor, route and end to end cover the happy, drop, minimum, failure and legacy paths. |
| Every test maps to a requirement | ✅ |
| Documented guidelines followed | `REVIEW.md` (above) |

---

## Requirement Traceability Update

LOT-01..LOT-29: Implemented → ✅ Verified. LOT-26 and LOT-27 are verified with weak web proofs (G1, G2).

---

## Summary

**Overall**: ✅ Ready (PASS, 0 blocking, 4 nonblocking)

**Spec-anchored check**: 29/29 requirements and 7/7 issue ACs match the spec outcome; 0 spec-precision gaps.
**Sensor**: API 26/26 non-equivalent killed (1 equivalent); web 1/5 killed.
**Gate**: API 218 OK; root 587 OK (2 skipped); web 14 pass; typecheck exit 0.

**Next steps**: optional follow-ups G1–G4; then the PR to `develop`.
