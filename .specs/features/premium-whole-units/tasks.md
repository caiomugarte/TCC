# Premium Whole-Unit Buy List Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/premium-whole-units/spec.md`
**Design**: `.specs/features/premium-whole-units/design.md`
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `REVIEW.md` (a new test must fail when the behaviour it names is removed; `model_versions` must change with the allocation method; strings copied between `api/` and `web/lib/api-types.ts` need a comparing test or a single source; `api/app/adapters/**` and `api/app/routers/**` are sensitive), `.specs/codebase/TESTING.md` (deterministic functions, no network), `.specs/LESSONS.md` candidates L-001 (one input per guard) and L-003 (assert exact arguments of defensive calls). Runner: stdlib `unittest` for Python (no pytest), `node --test` for web static tests. No CI, coverage threshold, linter or formatter. Strong defaults applied.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Pure rule (`api/app/services/whole_units.py`) | unit | LOT-06..LOT-15 1:1; every spec edge case; each validation guard with its own input; deterministic enumeration for LOT-09 | `api/tests/test_whole_units.py` (new) | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_whole_units)` |
| Price loader (`api/app/services/status_invest_inputs.py`) | unit | LOT-02..LOT-05; one input per bad-price guard; hash checked before parsing | `api/tests/test_status_invest_inputs.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_status_invest_inputs)` |
| Response schema (`api/app/schemas/recommendation.py`) | unit | LOT-21..LOT-23: legacy dict, new dict, each bound | `api/tests/test_recommendation_schema.py` (new) | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_recommendation_schema)` |
| Router (`api/app/routers/recommendations.py`) | integration (TestClient) | LOT-21, LOT-22 through `GET /v1/recommendations/{id}` and latest-completed | `api/tests/test_product_routes.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_product_routes)` |
| Policy (`api/app/services/premium_policy.py`) | unit | `whole-units-v1` in Premium only; Basic golden JSON unchanged; `premium-policy-v4` unchanged | `api/tests/test_premium_policy.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_premium_policy)` (about 130 s: real-data v4 tests) |
| Adapter (`api/app/adapters/premium_optimization.py`) | unit (fake engines, temp CSV) | LOT-01, LOT-03..LOT-05 mapping, LOT-10, LOT-11, LOT-13, LOT-14, LOT-16, LOT-17, LOT-24; default-off golden | `api/tests/test_premium_optimization.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_premium_optimization)` |
| Executor (`api/app/services/premium_executor.py`) | unit (SQLite + fake runner) | LOT-18..LOT-20, flag passed only when declared | `api/tests/test_premium_executor.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_premium_executor)` |
| End to end (service + executor + adapter, fake engines) | integration | LOT-20, LOT-21 across two sequential runs | `api/tests/test_premium_integration.py` | `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_premium_integration)` |
| Web types and view | static source tests (repo convention) + typecheck | LOT-25..LOT-29 render paths, legacy fallback, no reason text hard-coded | `web/tests/premium-recommendation.test.mjs` | `node --test web/tests/premium-recommendation.test.mjs && (cd web && npm run typecheck)` |

## Gate Check Commands

> Generated from codebase - confirm before Execute. `$PY` = `/home/caio/Documents/projetos/TCC/.venv/bin/python` (Python 3.14.4 with numpy, pandas, FastAPI; the venv lives in the main checkout and is shared by worktrees; the system `python3` has no pandas). Run from the worktree root `/home/caio/.t3/worktrees/TCC/t3-issue-57`. Baseline at `af1080e`: API suite **174 tests OK in 212 s** (measured 2026-11-09); root suite 587 OK, 2 skipped, about 175 s (STATE handoff of #56, no root change since); web static tests **9 pass** (measured). `web/node_modules` is absent in this worktree: before the first web typecheck run `ln -s /home/caio/Documents/projetos/TCC/web/node_modules web/node_modules` (ignored by `web/.gitignore`). Commit with `git commit -F <file>`: the shell hook rewrites a `-m` message containing the word "make".

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | Tasks with unit tests only (T1, T3, T4, T6) | the task's module command from the matrix |
| Full | Tasks touching the adapter, executor, router or end to end (T2, T7-T11) | `(cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .)` (about 4 min) |
| Web | T12 | `node --test web/tests/premium-recommendation.test.mjs && (cd web && npm run typecheck)` |
| Build | Last task of each phase | `$PY -m compileall -q py api/app scripts && $PY -m unittest discover -s tests -p "test_*.py" && (cd api && PYTHONPATH=.:../py $PY -m unittest discover -s tests -t .) && node --test web/tests/premium-recommendation.test.mjs` (about 7 min; `compileall` is the syntax gate; the root suite proves research tools that patch the adapter are unchanged) |

Test counts never go down. Existing tests may change only where a task names them (T6: `test_model_versions_record_fii_selection_v2`).

---

## Test Strategy (test-to-AC matrix)

Every named test fails when the behaviour it names is removed (REVIEW.md). Prices and targets are in BRL; the pure function takes cents.

**LOT-TEST-1 fixture (run `3289af79`, `fiis.csv` SHA-256 `11da8a99…ba52`, `retrieved_at` 2026-11-06T12:29:37Z)**: selector order and `PRECO`: MXRF11 8.64, KNCR11 102.16, BTLG11 97.51, BPML11 80.63, GGRC11 8.35, ALZR11 9.54, XPML11 96.00, KNRI11 147.89, RECT11 31.82, BRCR11 38.28, KNUQ11 94.98, RBRF11 6.53, CPTS11 7.00. Targets R$ 100.00 each, class R$ 1,300.00. Expected quantities: 12, 1, 1, 1, 12, 11, 1, 1, 3, 2, 1, 15, 14; spent R$ 1,295.96; `cash_residual_brl` 4.04; `over_target_brl` 0; nothing dropped. Plain floor would give KNCR11 0, KNRI11 0 and R$ 280.62 left.

| Test (file :: name) | Asserts | Req | Kills |
| ------------------- | ------- | --- | ----- |
| `test_recommendation_schema :: test_legacy_entries_validate_with_null_new_fields` | pre-change constituent and class dicts validate; new fields `None`; response `stocks_dropped == []` | LOT-21, LOT-23 | new field made required |
| `test_recommendation_schema :: test_new_fields_serialise_camel_case` | `model_dump(by_alias=True)` has `priceBrl`, `quantity`, `amountBrl`, `effectivePortfolioWeight`, `cashResidualBrl`, `overTargetBrl`, `stocksDropped`, `fiisDropped` | LOT-22 | field missing |
| `test_recommendation_schema :: test_each_bound_rejects_its_own_input` | `quantity` 0; `price_brl` 0; `amount_brl` −0.01; `effective_portfolio_weight` −0.01; `cash_residual_brl` −0.01; `over_target_brl` −0.01; dropped entry with an extra key → `ValidationError` each; `effective_portfolio_weight` 1.5 accepted | LOT-23, edge | each bound / upper bound added |
| `test_product_routes :: test_pre_change_result_json_reads_with_null_whole_unit_fields` | completed run with a pre-change `result_json` → `GET /v1/recommendations/{id}` and latest-completed 200; `quantity` null; `stocksDropped == []` | LOT-21 | router/schema regression |
| `test_product_routes :: test_whole_unit_result_json_exposes_new_fields` | `result_json` with the new keys → body has `quantity`, `cashResidualBrl`, `fiisDropped[0].reasons[-1] == "não coube no capital"` | LOT-22 | router not forwarding `*_dropped` |
| `test_status_invest_inputs :: test_prices_come_from_the_hashed_csv_in_cents` | temp CSV (BOM, lower-case ticker row) with the LOT-TEST-1 rows → `{"KNRI11": 14789, "KNCR11": 10216, ...}` only for the asked tickers | LOT-02 | wrong column / no upper-case |
| `test_status_invest_inputs :: test_changed_bytes_raise_before_parsing` | recorded sha256 of the original bytes, file rewritten with one price changed → `SelectorSnapshotChangedError` | LOT-03, LOT-04 | hash check removed / after parse |
| `test_status_invest_inputs :: test_each_bad_price_raises_naming_the_ticker` | one input per guard: ticker row missing; no `PRECO` column; `""`; `"abc"`; `"NaN"`; `"Infinity"`; `"0"`; `"-1.00"`; two rows with 10.00 and 11.00 → `SelectorPriceError` whose message names the ticker | LOT-05, edge | each guard |
| `test_status_invest_inputs :: test_equal_duplicate_rows_are_accepted` | two rows 10.00 → 1000 | edge | duplicate always rejected |
| `test_status_invest_inputs :: test_price_is_quantised_half_up` | `10.005` → 1001, `10.004` → 1000 | A-12 | truncation / float |
| `test_status_invest_inputs :: test_reading_prices_opens_no_socket` | `socket.socket` patched to raise → call succeeds | LOT-03 | network fetch |
| `test_whole_units :: test_floor_when_no_unit_fits_the_remainder` | targets 100/100, prices 30.00/45.00, class 200 → (3, 2), residual 20.00 | LOT-06, LOT-08 | floor → round |
| `test_whole_units :: test_remainder_goes_to_the_largest_deficit_that_fits` | A 7.00 / B 8.00 / C 53.00, targets 50/100/150, class 300 → (10, 15, 2), residual 4.00 | LOT-07 | first-fit or cheapest-first → (14, 12, 2); relative deficit → (9, 16, 2) |
| `test_whole_units :: test_a_unit_that_fills_the_class_exactly_is_bought` | A 11.00 / B 12.00 / C 70.00, targets 100 each, class 300 → (10, 10, 1), residual 0 | LOT-07, edge | `price < remaining` |
| `test_whole_units :: test_equal_deficits_go_to_the_earlier_ticker` | A and B price 10.00, target 15.00 each, class 30.00 → (2, 1) | LOT-07 | tie to the later ticker |
| `test_whole_units :: test_knri11_kncr11_real_prices` | LOT-TEST-1 → the 13 quantities, spent 129596, residual 404, over 0 | LOT-06..LOT-09, issue test (a) | stop-at-first-misfit (residual 3057) / floor only (residual 28062) |
| `test_whole_units :: test_class_total_never_exceeds_target_without_minimum` | seeds 0-199: 1-8 tickers, prices 1.00-300.00, class 0-5,000.00, targets from an integer split, `minimum_count` 0 → `spent <= class`, no price ≤ residual, `over == 0` | LOT-08, LOT-09, issue test (d) | any over-target unit / early stop |
| `test_whole_units :: test_each_invalid_input_raises` | one input per guard (design.md validation list) → `ValueError` | design | each guard |
| `test_whole_units :: test_ticker_that_never_fits_ends_at_zero_and_its_target_funds_the_others` | AAA3 10.00 / BBB3 20.00 / CCC3 250.00, targets 100 each, class 300 → (16, 7, 0), residual 0 | LOT-10, LOT-12, issue test (b) | target of the dropped ticker withheld |
| `test_whole_units :: test_no_renormalisation_after_a_drop` | A 60.00 / B 70.00 / C 5.00, targets 33.34/33.33/33.33, class 100 → (1, 0, 8) | LOT-12, A-7 | recompute after drop → (0, —, 20) |
| `test_whole_units :: test_fii_minimum_buys_one_quota_of_the_best_placed_dropped` | F1 20 / F2 30 / F3 120 / F4 150, targets 25 each, class 100, minimum 3 → (2, 2, 1, 0); F3 `forced`; spent 220.00; over 120.00; residual 0 | LOT-13, LOT-14, issue test (c) | minimum removed / worst-placed picked / more than 1 quota |
| `test_whole_units :: test_minimum_is_not_applied_when_three_are_bought` | 3 bought and 1 at zero, minimum 3 → no `forced`, over 0 | LOT-13 | minimum applied always |
| `test_whole_units :: test_minimum_when_no_fii_fits` | 4 FIIs all priced above the class → first 3 at 1 `forced`, 4th 0 | LOT-13, edge | — |
| `test_whole_units :: test_minimum_is_capped_by_the_selected_count` | 2 FIIs, minimum 3, none fits → both at 1, no error | A-16 | index error / raise |
| `test_whole_units :: test_stocks_get_no_minimum` | all stocks priced above the class, `minimum_count` 0 → all 0, residual = class | LOT-15, edge | minimum applied to stocks |
| `test_whole_units :: test_total_within_target_except_the_minimum` | seeds 0-199 with `minimum_count` 3: without a `forced` line `spent <= class` and over 0; with one, `over == spent − class` and the forced lines are exactly those at 0 under `minimum_count` 0, each at 1; `spent + residual − over == class` always | LOT-09, LOT-13, LOT-14, issue test (d) | excess unrecorded / extra lots |
| `test_premium_policy :: test_model_versions_record_fii_selection_v2` (updated) | Premium = `{allocation-v1, stock-selection-v1, fii-selection-v2, whole_units: whole-units-v1}`; Basic = the three keys without `whole_units`; `BASIC_GOLDEN_JSON` test unchanged and green; `POLICY_VERSION == "premium-policy-v4"` | A-4, LOT-24 | REVIEW.md rule / Basic gets whole units / version bump |
| `test_premium_optimization :: WholeUnitAdapterTests.test_knri11_kncr11_run_from_the_hashed_csv` | fake engines, temp CSVs with real SHA-256, capital 10,000, FII weight 0.13, LOT-TEST-1 order → `fiis` quantities, `price_brl`, `amount_brl` (KNRI11 147.89), `effective_portfolio_weight` (KNRI11 0.014789); FII class `cash_residual_brl` 4.04, `over_target_brl` 0.0; every entry validates through the schema models | LOT-01, LOT-11, LOT-14 | adapter not wired / wrong capital |
| `test_premium_optimization :: WholeUnitAdapterTests.test_dropped_ticker_leaves_the_list_with_its_reasons` | a FII with a zone reason drops while 3 others are bought → absent from `fiis`; `fiis_dropped[0].reasons == ["mantido", "não coube no capital"]`; stock drop likewise in `stocks_dropped` | LOT-10, LOT-15 | dropped kept in list / reason lost |
| `test_premium_optimization :: WholeUnitAdapterTests.test_forced_fii_has_the_minimum_reason_and_the_class_records_the_excess` | adapter version of the minimum case → reason ends `mínimo de 3 FIIs`, `over_target_brl` 120.0, `cash_residual_brl` 0.0 | LOT-13, LOT-14 | reason / excess not written |
| `test_premium_optimization :: WholeUnitAdapterTests.test_zero_weight_class_reads_no_prices` | stock weight 0 → `cash_residual_brl == target_amount_brl`, `stocks == []`, spy loader called only with `selector_inputs.fiis` | edge, L-003 | loader called for an empty class |
| `test_premium_optimization :: WholeUnitAdapterTests.test_assumption_sentence_is_added` | the LOT-16 sentence is in `assumptions`, after the three existing ones | LOT-16 | — |
| `test_premium_optimization :: WholeUnitAdapterTests.test_amounts_residual_and_excess_add_up_to_the_target` | both classes: `Σ amount + residual − over == target_amount_brl` (cents) | LOT-14 | rounding drift |
| `test_premium_optimization :: WholeUnitGateTests.test_default_off_equals_the_pre_change_output` | `whole_units` omitted and `False` → result equal to a golden dict captured at `af1080e` for the same fixture; spy loader never called | LOT-17, LOT-24 | step always on |
| `test_premium_optimization :: WholeUnitGateTests.test_on_and_off_share_weights_targets_tickers_and_seed` | `classes[*].target_weight/target_amount_brl`, selected tickers ∪ dropped, `random_seed`, `policy` equal between on and off | LOT-17 | targets re-normalised / selection touched |
| `test_premium_optimization :: WholeUnitGateTests.test_loader_receives_the_run_selector_snapshots` | spy loader called with `selector_inputs.stocks` then `selector_inputs.fiis` (same objects) and the exact ticker lists in selector order | LOT-03, L-003 | another price source |
| `test_premium_optimization :: WholeUnitGateTests.test_changed_csv_fails_with_snapshot_changed` | CSV rewritten after the snapshot → `PremiumOptimizationError.code == "snapshot_changed"` | LOT-04 | mapping lost |
| `test_premium_optimization :: WholeUnitGateTests.test_missing_price_fails_with_engine_error` | selected ticker absent from the CSV → `code == "engine_error"` | LOT-05 | silent default price |
| `test_premium_executor :: test_declared_whole_units_reaches_the_runner` | Premium run whose `policy_json.provenance.model_versions.whole_units == "whole-units-v1"` → runner kwargs `whole_units is True` | A-3 | flag not passed |
| `test_premium_executor :: test_basic_or_undeclared_policy_never_passes_the_flag` | Basic run; Premium policy without the key; key with another value → no `whole_units` kwarg (fixed-signature runner completes) | A-2, LOT-24 | flag always passed |
| `test_premium_executor :: test_incumbents_skip_zero_quantity_and_keep_legacy_entries` | previous `fiis`: no `quantity`, `quantity: null`, `1`, `0` → incumbents are the first three | LOT-18 | quantity ignored |
| `test_premium_executor :: test_malformed_quantity_makes_the_class_unreadable` | `-1`, `1.5`, `True`, `"2"` (one subTest each) → that class `None`, the other class read | LOT-19, L-001 | bool accepted as 1 |
| `test_premium_executor :: test_dropped_lists_are_not_incumbents` | previous result with `fiis_dropped: [{"ticker": "KNRI11", ...}]` → `KNRI11` not an incumbent | LOT-20 | executor reads `*_dropped` |
| `test_premium_integration :: WholeUnitIntegrationTests.test_ticker_dropped_in_run_one_is_not_an_incumbent_in_run_two` | create + execute two Premium runs with a declaring policy, real temp CSVs, fake engines; run 1 drops one FII → run 2's FII engine `incumbents` lacks it; run 1 response has `quantity` and `fiisDropped` | LOT-18, LOT-20, issue AC | end-to-end wiring |
| `test_premium_integration :: WholeUnitIntegrationTests.test_stored_pre_change_result_reads_and_feeds_incumbents` | stored completed Premium run with a pre-change `result_json` → route 200; next run's incumbents are all its tickers | LOT-21, LOT-18, issue test (e) | legacy result unreadable |
| `premium-recommendation.test.mjs :: whole-unit fields are typed for both shapes` | types: `quantity: number \| null`, `priceBrl`, `amountBrl`, `effectivePortfolioWeight`, `cashResidualBrl`, `overTargetBrl`, `DroppedConstituent`, `stocksDropped: DroppedConstituent[]`, `fiisDropped: DroppedConstituent[]` | LOT-22, LOT-29 | type missing |
| `premium-recommendation.test.mjs :: buy list renders quantity, price, amount and class cash` | view source uses `item.quantity`, `item.priceBrl`, `item.amountBrl`, `cashResidualBrl`, `overTargetBrl` and a BRL currency formatter | LOT-25, LOT-26 | field not rendered |
| `premium-recommendation.test.mjs :: dropped tickers render the API reasons` | view maps `stocksDropped` and `fiisDropped`, renders `reasons`; `doesNotMatch(/não coube no capital\|mínimo de 3 FIIs/)` in `web/` sources | LOT-27, REVIEW.md copy rule | hard-coded reason |
| `premium-recommendation.test.mjs :: snapshot price note and legacy fallback` | note gated on a constituent having `quantity`, reuses `statusInvestRetrievedAt(completed)`; `percentage.format(item.portfolioWeight)` path kept for entries without `quantity` | LOT-28, LOT-29 | note always shown / legacy branch removed |

**Mutation targets for the Verifier's discrimination sensor**: `price <= remaining` vs `<`; largest absolute deficit vs first-fit, cheapest, relative deficit; tie to earlier vs later; skip vs stop on a misfit; re-normalising after a drop; minimum applied to stocks; minimum picking the worst-placed or buying 2 quotas; `over_target_brl` not written; hash check removed or moved after parsing; a default price for a missing row; `whole_units` default `True`; executor passing the flag for Basic; `_incumbent_tickers` ignoring `quantity` or accepting `True`; router dropping `*_dropped`; schema field made required; `BASIC_MODEL_VERSIONS` inheriting `whole_units`.

---

## Execution Plan

Phases run in order; tasks within a phase run in order.

| Phase | Tasks | Count (heavy = 2) | Touches `api/app/adapters/**` | Other sensitive path | Risk | Suggested tier |
| ----- | ----- | ----------------- | ----------------------------- | -------------------- | ---- | -------------- |
| 1. Contract and pure rule | T1-T5 | 5 | no | T2 `api/app/routers/**` (two forwarded keys) | Medium: the rule that decides every quantity; schema must land before the adapter. | high reasoning (T4, T5), mid (T1-T3) |
| 2. Policy and adapter | T6-T8 | 3 | **yes: T7, T8** | `premium_policy.py` (REVIEW.md `model_versions` rule) | High: decides what reaches the user; byte-identity of the off path. | high reasoning |
| 3. Executor and end to end | T9-T11 | 3 | no | — | Medium: incumbents for next month's zone. | mid-high |
| 4. Web | T12-T13 | 2 | no | — | Low: rendering, shared Basic/Premium view. | mid |

No task is heavy: no migration, no snapshot regeneration, no real-data GA run, no browser test. Total 13 tasks.

### Phase 1: Contract and pure rule

```
T1 → T2
T3
T4 → T5
```

### Phase 2: Policy and adapter

```
T6 → T7 → T8
```

### Phase 3: Executor and end to end

```
T9 → T10 → T11
```

### Phase 4: Web

```
T12 → T13
```

---

## Task Breakdown

### Phase 1 tasks: Contract and pure rule

### T1: Add the whole-unit fields to the response schema

**What**: Optional `price_brl`, `quantity`, `amount_brl`, `effective_portfolio_weight` on `RecommendationConstituent`; `cash_residual_brl`, `over_target_brl` on `AllocationClass`; new `DroppedConstituent`; `stocks_dropped`, `fiis_dropped` on `RecommendationResponse` (bounds per design.md).
**Where**: `api/app/schemas/recommendation.py`
**Depends on**: None
**Reuses**: `ProfileSchema` (camelCase aliases, `extra="forbid"`)
**Requirement**: LOT-21, LOT-22, LOT-23
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] New `api/tests/test_recommendation_schema.py` with the three schema tests of the Test Strategy
- [x] Gate check passes: `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_recommendation_schema)`
- [x] Test count: 3 new tests pass

**Tests**: unit
**Gate**: quick

**Commit**: `feat(api): accept whole-unit fields in the recommendation response (#57)`

**Status**: ✅ Done - `tests.test_recommendation_schema` 3 OK.

---

### T2: Forward the dropped lists from the stored result

**What**: `_response_with_profile` passes `stocks_dropped` / `fiis_dropped` from a completed `result_json` (`[]` otherwise).
**Where**: `api/app/routers/recommendations.py`
**Depends on**: T1
**Reuses**: existing `result.get("stocks", [])` pattern (`:34-35`)
**Requirement**: LOT-21, LOT-22
**High risk (`api/app/adapters/**`)**: no (sensitive `api/app/routers/**`: no scoping change)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Two route tests of the Test Strategy added to `api/tests/test_product_routes.py`
- [ ] Gate check passes: Full gate, 174 + 3 + 2 = 179 tests OK

**Tests**: integration
**Gate**: full

**Commit**: `feat(api): return the dropped tickers of a whole-unit result (#57)`

---

### T3: Read selected-ticker prices from the hashed selector CSV

**What**: `load_selector_prices(snapshot, tickers) -> dict[str, int]` plus `SelectorSnapshotChangedError` and `SelectorPriceError` (design.md behaviour).
**Where**: `api/app/services/status_invest_inputs.py`
**Depends on**: None
**Reuses**: `file_sha256`, the `utf-8-sig` `csv.DictReader` of `_validate_csv`
**Requirement**: LOT-02, LOT-03, LOT-04, LOT-05
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Six loader tests of the Test Strategy added to `api/tests/test_status_invest_inputs.py`, the LOT-TEST-1 rows used for the happy path
- [ ] Gate check passes: `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_status_invest_inputs)`
- [ ] Test count: existing module count + 6

**Tests**: unit
**Gate**: quick

**Commit**: `feat(api): read selector prices from the hashed Status Invest CSV (#57)`

---

### T4: Floor and remainder loop in integer cents

**What**: `api/app/services/whole_units.py` with `UnitLine`, `WholeUnitAllocation`, the constants, input validation and steps 1-2 of the design algorithm (`minimum_count` accepted, step 3 still a no-op).
**Where**: `api/app/services/whole_units.py` (new)
**Depends on**: None
**Reuses**: none (pure)
**Requirement**: LOT-06, LOT-07, LOT-08, LOT-09
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] New `api/tests/test_whole_units.py` with the seven T4 tests (floor, largest deficit, exact fill, tie, KNRI11/KNCR11, no-minimum enumeration, invalid inputs)
- [ ] KNRI11/KNCR11 test asserts the LOT-TEST-1 quantities exactly
- [ ] Gate check passes: `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_whole_units)`
- [ ] Test count: 7 new tests pass

**Tests**: unit
**Gate**: quick

**Commit**: `feat(api): round class targets to whole units by price (#57)`

---

### T5: Drop at zero and the 3-FII minimum

**What**: Step 3 of the algorithm (`forced` lines) and the residual/excess fields; drops stay as quantity-0 lines in input order.
**Where**: `api/app/services/whole_units.py`
**Depends on**: T4
**Reuses**: T4's state
**Requirement**: LOT-09, LOT-10, LOT-12, LOT-13, LOT-14, LOT-15
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Eight T5 tests of the Test Strategy added to `api/tests/test_whole_units.py`
- [ ] Gate check passes: Build gate (last task of Phase 1)
- [ ] Test count: API 179 + 6 (T3) + 7 (T4) + 8 = 200 OK; root 587 OK

**Tests**: unit
**Gate**: build

**Commit**: `feat(api): drop tickers that do not fit and keep at least 3 FIIs (#57)`

---

### Phase 2 tasks: Policy and adapter

### T6: Record whole-units-v1 in the Premium model versions

**What**: `WHOLE_UNITS_MODEL_VERSION`, `PREMIUM_MODEL_VERSIONS["whole_units"]`, `BASIC_MODEL_VERSIONS` written out without it; `POLICY_VERSION` unchanged.
**Where**: `api/app/services/premium_policy.py`
**Depends on**: None
**Reuses**: existing constants (`:102-107`)
**Requirement**: A-4, LOT-24
**High risk (`api/app/adapters/**`)**: no (REVIEW.md `model_versions` rule)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `test_model_versions_record_fii_selection_v2` updated as in the Test Strategy; `BASIC_GOLDEN_JSON` test untouched and green
- [ ] Gate check passes: `(cd api && PYTHONPATH=.:../py $PY -m unittest tests.test_premium_policy)`
- [ ] Test count: module count unchanged

**Tests**: unit
**Gate**: quick

**Commit**: `feat(policy): record whole-units-v1 in the Premium model versions (#57)`

---

### T7: Build the whole-unit buy list in the adapter

**What**: `whole_units` and `price_loader` keywords; `_whole_unit_rows`; bought/dropped rows, class `cash_residual_brl` / `over_target_brl`, `*_dropped` keys, LOT-16 assumption; zero-weight class handling.
**Where**: `api/app/adapters/premium_optimization.py`
**Depends on**: T6, T1, T3, T5
**Reuses**: `_constituents`, `allocate_whole_units` (T4-T5), `load_selector_prices` (T3), schema models (T1) in tests
**Requirement**: LOT-01, LOT-10, LOT-11, LOT-13, LOT-14, LOT-16
**High risk (`api/app/adapters/**`)**: **yes**

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `WholeUnitAdapterTests` (six tests of the Test Strategy) added to `api/tests/test_premium_optimization.py`, using temp CSVs whose SHA-256 is computed from the written bytes
- [ ] Every emitted constituent, dropped entry and class validates through the T1 schema models
- [ ] Gate check passes: Full gate
- [ ] Test count: API 200 + 6 = 206 OK

**Tests**: unit
**Gate**: full

**Commit**: `feat(api): deliver whole-unit quantities in the Premium result (#57)`

---

### T8: Keep the off path identical and map price failures

**What**: Golden test of the default-off output, on/off equality of weights/targets/tickers/seed, loader-argument spy, `snapshot_changed` / `engine_error` mapping of the loader errors.
**Where**: `api/app/adapters/premium_optimization.py`
**Depends on**: T7
**Reuses**: existing fixture `policy()`, `allocation_result()`, `selector_result()`
**Requirement**: LOT-03, LOT-04, LOT-05, LOT-17, LOT-24
**High risk (`api/app/adapters/**`)**: **yes**

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `WholeUnitGateTests` (five tests) added to `api/tests/test_premium_optimization.py`; the golden dict is the fixture's output under the `af1080e` adapter (load `git show af1080e:api/app/adapters/premium_optimization.py` from a scratch copy, never by switching branches) pasted as a literal
- [ ] Gate check passes: Build gate (last task of Phase 2); root suite still 587 OK (research tools unchanged)
- [ ] Test count: API 206 + 5 = 211 OK

**Tests**: unit
**Gate**: build

**Commit**: `test(api): pin the whole-unit gate and its price failures (#57)`

---

### Phase 3 tasks: Executor and end to end

### T9: Turn whole units on from the persisted policy

**What**: `_declares_whole_units(policy_json)` and `runner_kwargs["whole_units"] = True` only when declared.
**Where**: `api/app/services/premium_executor.py`
**Depends on**: T6, T8
**Reuses**: `_declares_protection_zone` pattern; `WHOLE_UNITS_MODEL_VERSION` from `premium_policy.py`
**Requirement**: A-2, A-3, LOT-24
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Two executor flag tests of the Test Strategy added
- [ ] Gate check passes: Full gate
- [ ] Test count: API 211 + 2 = 213 OK

**Tests**: unit
**Gate**: full

**Commit**: `feat(api): run Premium whole units when the policy declares them (#57)`

---

### T10: Read only bought tickers as incumbents

**What**: `_incumbent_tickers` applies LOT-18/LOT-19 per entry.
**Where**: `api/app/services/premium_executor.py`
**Depends on**: T9
**Reuses**: existing unreadable-class convention (`:30-43`)
**Requirement**: LOT-18, LOT-19, LOT-20
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Three incumbent tests of the Test Strategy added (one subTest per malformed value)
- [ ] Gate check passes: Full gate
- [ ] Test count: API 213 + 3 = 216 OK

**Tests**: unit
**Gate**: full

**Commit**: `feat(api): protect only bought tickers in the next Premium run (#57)`

---

### T11: Cover two sequential whole-unit runs end to end

**What**: `WholeUnitIntegrationTests` (two tests) with a policy declaring `whole-units-v1`, real temp CSVs and fake engines.
**Where**: `api/tests/test_premium_integration.py`
**Depends on**: T10
**Reuses**: the module's `policy()` fixture (with `model_versions.whole_units` added in the new tests only), `_default_selector_inputs` replaced by real files for these tests
**Requirement**: LOT-18, LOT-20, LOT-21
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Both tests pass, including issue test (e) through the route
- [ ] Gate check passes: Build gate (last task of Phase 3)
- [ ] Test count: API 216 + 2 = 218 OK; root 587 OK

**Tests**: integration
**Gate**: build

**Commit**: `test(api): cover whole-unit runs and stored results end to end (#57)`

---

### Phase 4 tasks: Web

### T12: Type the whole-unit fields in the web client

**What**: Fields and `DroppedConstituent` in `web/lib/api-types.ts` per design.md.
**Where**: `web/lib/api-types.ts`
**Depends on**: T11
**Reuses**: existing `RecommendationConstituent`, `AllocationClass`
**Requirement**: LOT-22, LOT-29
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Static test "whole-unit fields are typed for both shapes" added to `web/tests/premium-recommendation.test.mjs`
- [ ] Gate check passes: Web gate (typecheck green before T13: the new fields are nullable)
- [ ] Test count: 9 + 1 = 10 pass

**Tests**: unit (static source)
**Gate**: web

**Commit**: `feat(web): type the whole-unit buy list fields (#57)`

---

### T13: Show quantities, cash left over and the price note

**What**: Render bought rows, class cash/excess, dropped lists with API reasons, and the snapshot price note; keep today's render for entries without `quantity`.
**Where**: `web/components/recommendation/recommendation-view.tsx`
**Depends on**: T12
**Reuses**: `statusInvestRetrievedAt`, `percentage`, existing `split-card` layout and `globals.css` classes
**Requirement**: LOT-25, LOT-26, LOT-27, LOT-28, LOT-29
**High risk (`api/app/adapters/**`)**: no

**Tools**:

- MCP: NONE
- Skill: `impeccable` (optional, only for layout polish of the new rows at 320 px)

**Done when**:

- [ ] Three view static tests of the Test Strategy added
- [ ] Gate check passes: Build gate (last task)
- [ ] Test count: web 10 + 3 = 13 pass; API 218 OK; root 587 OK

**Tests**: unit (static source)
**Gate**: build

**Commit**: `feat(web): show the Premium buy list with quantities and cash (#57)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1:  T1 ──→ T2
          T3
          T4 ──→ T5
Phase 2:  T6 ──→ T7 ──→ T8
Phase 3:  T9 ──→ T10 ──→ T11
Phase 4:  T12 ──→ T13
```

Execution is sequential (T1 … T13). Cross-phase dependencies point backward only: T7 ← T1/T3/T5, T9 ← T6/T8, T12 ← T11. Suggested batches (~7 tasks, whole phases): b1 = Phase 1 (5); b2 = Phases 2 + 3 (6, contains both high-risk tasks); b3 = Phase 4 (2).

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | 1 schema file, 4 models | ✅ cohesive |
| T2 | 1 function | ✅ |
| T3 | 1 function + 2 error types | ✅ cohesive |
| T4 | 1 function (steps 1-2) | ✅ |
| T5 | same function (step 3-4) | ✅ |
| T6 | 3 constants | ✅ |
| T7 | adapter wiring + 1 helper | ✅ cohesive |
| T8 | error mapping + gate tests | ✅ |
| T9 | 1 predicate + 1 kwarg | ✅ |
| T10 | 1 function | ✅ |
| T11 | tests only | ✅ |
| T12 | 1 types file | ✅ |
| T13 | 1 component | ✅ |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | — | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | None | — | ✅ Match |
| T4 | None | — | ✅ Match |
| T5 | T4 | T4 → T5 | ✅ Match |
| T6 | None | — | ✅ Match |
| T7 | T6, T1, T3, T5 | T6 → T7; T1/T3/T5 earlier phase | ✅ Match |
| T8 | T7 | T7 → T8 | ✅ Match |
| T9 | T6, T8 | earlier phase | ✅ Match |
| T10 | T9 | T9 → T10 | ✅ Match |
| T11 | T10 | T10 → T11 | ✅ Match |
| T12 | T11 | earlier phase | ✅ Match |
| T13 | T12 | T12 → T13 | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Response schema | unit | unit | ✅ OK |
| T2 | Router | integration | integration | ✅ OK |
| T3 | Price loader | unit | unit | ✅ OK |
| T4 | Pure rule | unit | unit | ✅ OK |
| T5 | Pure rule | unit | unit | ✅ OK |
| T6 | Policy | unit | unit | ✅ OK |
| T7 | Adapter | unit | unit | ✅ OK |
| T8 | Adapter | unit | unit | ✅ OK |
| T9 | Executor | unit | unit | ✅ OK |
| T10 | Executor | unit | unit | ✅ OK |
| T11 | End to end | integration | integration | ✅ OK |
| T12 | Web types | static + typecheck | unit (static source) | ✅ OK |
| T13 | Web view | static + typecheck | unit (static source) | ✅ OK |

---

## Checkpoint: defaults to confirm

None blocks execution; each has a default in spec.md. A human may want to confirm:

1. **A-5/A-6/A-7 (remainder rule)**: the loop keeps buying the largest-deficit unit that fits even past a ticker's own target, and a floor-0 ticker can still get a unit. Consequence for the issue's own example: KNRI11 and KNCR11 are each bought once (R$ 4.04 left), not dropped.
2. **A-2 (Basic)**: Basic results get no buy list.
3. **A-4 (versions)**: `model_versions` gains `whole_units: whole-units-v1`; `premium-policy-v4` and the GA seed stay.
4. **A-8 (shape)**: dropped tickers go to `stocks_dropped` / `fiis_dropped` instead of staying in `stocks` / `fiis` with quantity 0.
