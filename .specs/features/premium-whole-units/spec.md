# Premium Whole-Unit Buy List Specification

Issue: [#57](https://github.com/caiomugarte/TCC/issues/57) "Entrega a lista de compra com quantidades inteiras no resultado Premium". Depends on #56 (`premium-policy-v4`, FII segment cap, 4/5/6 FIIs), merged in `af1080e`.

## Problem Statement

The Premium result gives a BRL target per ticker (`target_amount_brl`) but not how many shares or quotas to buy. Stocks (fractional market) and FIIs trade in whole units, and with small capital the rounding matters: in run `3289af79` (R$ 10,000, 13 FIIs at R$ 100 each) a plain floor leaves the FII sleeve R$ 280.62 below its R$ 1,300.00 target because KNRI11 (R$ 147.89) and KNCR11 (R$ 102.16) cost more than R$ 100. The user does this arithmetic by hand today.

## Goals

- [ ] Every Premium stock and FII the user is told to buy carries a price from the run's own hashed selector CSV, an integer quantity and the resulting BRL amount.
- [ ] Each class total stays at or below its target (except the 3-FII minimum), and the leftover appears as `cash_residual_brl`.
- [ ] On the run-`3289af79` prices the FII sleeve ends R$ 4.04 below target instead of R$ 280.62.
- [ ] Only bought tickers become next month's protection-zone incumbents.
- [ ] Stored results without the new fields stay readable by the API and the web view.

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Live quotes, broker orders | Issue `## Fora de escopo`; breaks reproducibility (issue `## Decisões`). |
| Instruments (ETFs) for fixed income, international, crypto | Issue `## Fora de escopo`. |
| Per-class repeated metrics (#11) | Issue `## Fora de escopo`. |
| Whole units for the Basic plan | The issue scopes the buy list to the Premium result; Basic results stay byte-identical (A-2). |
| Whole units in research tools (`scripts/benchmark_premium.py`, `scripts/multirun_convergence.py`, `scripts/protection_zone_turnover.py`) | They measure selection, not purchase; the adapter default stays off, so their outputs do not change (A-3). |
| Re-running or migrating stored runs | Stored results are read as they are (LOT-21). |
| Changing class weights, selection, GA seed or `policy_version` | Issue `## Decisões`: the validated class allocation must not change; `premium-policy-v4` and its derived seed stay. |

---

## Assumptions & Open Questions

Decisions from the issue's `## Decisões` are settled and not repeated as assumptions: price-based rounding (no fixed BRL floor, no expensive-quota filter); minimum of 3 FIIs; the class total never exceeds its target except for that minimum; price source `PRECO` from the run's snapshot.

| ID | Assumption / decision | Chosen default | Rationale | Confirmed? |
| -- | --------------------- | -------------- | --------- | ---------- |
| A-1 | "Snapshot used in the run" for prices | The Status Invest selector CSV of the run (`selector_inputs.stocks.path` / `.fiis.path`, written by `refresh_inputs` into the run workspace), whose SHA-256 is already recorded in `provenance.selector_sources.<class>.sha256`. Before reading `PRECO`, the file's SHA-256 is recomputed and must equal that recorded value. | It is the only per-run file that carries `PRECO` and the one the issue calls "already hashed in provenance" (`api/app/services/status_invest_inputs.py:63-69`, `api/app/adapters/premium_optimization.py:712`). Recomputing the hash proves the prices come from the hashed bytes. | n (default) |
| A-2 | Plan scope | Whole units apply only to Premium runs whose persisted policy declares `provenance.model_versions.whole_units == "whole-units-v1"`. Basic policies never declare it; Basic `result_json` is byte-identical to today. | Issue title and AC say "resultado Premium"; Basic and Premium share the executor and adapter (`api/app/routers/recommendations.py` `create_recommendation` → `create_run(plan="basic")`), so an ungated change would hand a Premium feature to Basic. | n (default) |
| A-3 | Who turns whole units on | `run_premium_optimization` gains `whole_units: bool = False`; `PremiumExecutor` passes `whole_units=True` only when the run's `policy_json` declares A-2's model version (same pattern as `incumbents`, passed only when needed). | Keeps research tools, existing adapter tests and fixed-signature runners unchanged; the persisted policy records which method produced each run. | n (default) |
| A-4 | `model_versions` and `policy_version` | `PREMIUM_MODEL_VERSIONS` gains `"whole_units": "whole-units-v1"`; `BASIC_MODEL_VERSIONS` is written out without it. `POLICY_VERSION` stays `premium-policy-v4`. | REVIEW.md requires `model_versions` to change when the allocation method changes (the delivered per-ticker amount does). The GA seed is derived from `policy_version` (`api/app/services/premium_recommendation.py:111-113`), so bumping it would change every selection although selection did not change. | n (default) |
| A-5 | Meaning of "the ticker most below its own target" | Deficit in BRL cents = per-ticker target − `quantity × price` (the per-ticker target is today's `target_amount_brl`). Among tickers whose unit price fits the remaining class cash, the largest deficit wins; a tie goes to the earlier ticker in selector order. A deficit can be negative: the loop keeps buying while any unit fits the remaining cash. | Literal reading of "a sobra compra 1 unidade por vez, sempre do papel mais abaixo do próprio alvo, enquanto o total da classe não passar do alvo": the only stop condition is the class total. Absolute BRL matches "valor" in the issue. | n (default) |
| A-6 | A ticker whose unit does not fit the remaining cash at some step | Skipped for that step; the loop takes the next-largest deficit that fits (skip, not stop). It stops when no unit fits. | "Enquanto o total da classe não passar do alvo" bounds the class, not the order; stopping at the first misfit would leave cash a cheaper unit could use (run `3289af79`: R$ 30.57 instead of R$ 4.04). Same skip-not-stop convention as AD-027. | n (default) |
| A-7 | When "quantity 0" is decided | After the floor and the remainder loop. A ticker with floor 0 can still get a unit from the remainder (on the run-`3289af79` prices KNCR11 and KNRI11 each get 1). One pass: targets are not re-normalised after a drop. | "O valor dele fica para os outros" holds because the dropped ticker's target is part of the remainder cash the loop spends on the others. Re-normalising and recomputing can drop a ticker that fitted before (counter-example in design.md). | n (default) |
| A-8 | Where a dropped ticker goes | It leaves `stocks` / `fiis` and is recorded in new top-level lists `stocks_dropped` / `fiis_dropped` with its price, target, weights and reasons ending in `não coube no capital`. | "Sai da lista com o motivo": the buy list keeps only bought tickers, so everything that reads `stocks` / `fiis` as the run's portfolio (protection-zone incumbents, web) sees only quantity > 0 without extra filters. | n (default) |
| A-9 | "Best placed" for the 3-FII minimum | Selector order of `selected_tickers`: for FIIs it is frequency desc, score desc, ticker asc (`py/fii_selection.py:643`, `:845`). | The FII selector already returns its ranking in this order. | y (code) |
| A-10 | Recording the minimum's excess | Each class with instruments gets `cash_residual_brl = max(0, target − bought)` and `over_target_brl = max(0, bought − target)`; at most one is non-zero. A FII bought by the minimum gets the reason `mínimo de 3 FIIs`. | Two non-negative fields read clearly in the UI ("sobra em caixa" / "acima do alvo") and keep the schema's `ge=0` bounds; the reason ties the excess to the rule. | n (default) |
| A-11 | Missing or bad price for a selected ticker | The run fails with `engine_error`, message naming the ticker; nothing is persisted as completed. Hash mismatch fails with `snapshot_changed`. | A buy list with a guessed price is "a wrong number without a visible error" (REVIEW.md). Status Invest rows carry a 2-decimal `PRECO` today (checked on the committed and run-`3289af79` CSVs), so this is a guard, not a common path. | n (default) |
| A-12 | Price precision | `PRECO` parsed as `Decimal` and quantised to cents, half up; all arithmetic in integer cents. | Avoids float drift in `quantity × price` and in the `<=` target check. | n (default) |
| A-13 | `effective_portfolio_weight` | `round(amount_brl / investable_capital_brl, 6)`; bounded `>= 0` only (a forced FII on tiny capital can exceed 1). | Same rounding as `portfolio_weight` (`api/app/adapters/premium_optimization.py:517-518`). | n (default) |
| A-14 | Classes without instruments | `international`, `fixed_income`, `crypto` carry no `cash_residual_brl` / `over_target_brl` (absent in `result_json`, `null` in the response). A stock or FII class with weight 0 gets `cash_residual_brl = target_amount_brl` and `over_target_brl = 0`. | The issue's "cada classe" refers to the classes that receive a buy list; ETF instruments are out of scope. | n (default) |
| A-15 | Where the 3 comes from | Constant `FII_MINIMUM_HOLDINGS = 3` in the whole-units module, versioned by `whole-units-v1`; not added to the policy JSON. | Changing it changes the method and therefore `whole-units-v1`; keeping it out of the policy keeps `policy_json` sections untouched. | n (default) |
| A-16 | Fewer than 3 FIIs selected | The minimum becomes `min(3, number of FIIs selected)`. | Cannot buy a FII the selector did not pick; under v4 `n_assets` is 4/5/6, so this is a guard. | n (default) |

**Open questions:** none blocking. A-2, A-4, A-5/A-6/A-7 (the expected outcome of the KNRI11/KNCR11 test depends on them) are the defaults a human is most likely to want to confirm; they are repeated at the end of `tasks.md` for the checkpoint.

---

## User Stories

### P1: Buy list with whole units ⭐ MVP

**User Story**: As a Premium user with small capital, I want each recommended stock and FII to come with a price, a whole quantity and its cost, so that I can place the orders without doing the rounding myself.

**Why P1**: It is the issue.

**Acceptance Criteria**:

1. LOT-01: WHERE the run's persisted policy declares `model_versions.whole_units == "whole-units-v1"` THEN each entry of `stocks` and `fiis` in `result_json` SHALL carry `price_brl` (float, cents), `quantity` (int ≥ 1), `amount_brl` (= `quantity × price_brl`, cents) and `effective_portfolio_weight` (`round(amount_brl / investable_capital_brl, 6)`), in addition to the fields it has today.
2. LOT-02: The price of a ticker SHALL be the `PRECO` value of its `TICKER` row (ticker compared stripped and upper-cased) in the run's selector CSV of its class, `Decimal`-parsed and quantised to cents half up.
3. LOT-03: WHEN prices are read THEN the system SHALL recompute the CSV's SHA-256 and SHALL use it only if it equals `selector_inputs.<class>.sha256` (the value recorded in `provenance.selector_sources.<class>.sha256`); no network call SHALL be made to obtain a price.
4. LOT-04: IF the recomputed SHA-256 differs from the recorded one THEN the run SHALL fail with `failure_code = "snapshot_changed"` and no completed result.
5. LOT-05: IF a selected ticker has no row, no `PRECO` column, or a `PRECO` that is empty, non-numeric, non-finite or ≤ 0 THEN the run SHALL fail with `failure_code = "engine_error"` and no completed result.
6. LOT-06: For each class the system SHALL start with `quantity = floor(target_cents / price_cents)` per ticker, where `target_cents` is the ticker's `target_amount_brl` in cents.
7. LOT-07: WHILE some ticker's unit price is ≤ the class's remaining cash (`class target − Σ quantity × price`) the system SHALL add 1 unit to the ticker with the largest deficit (`target_cents − quantity × price_cents`) among those that fit, ties going to the earlier ticker in selector order.
8. LOT-08: WHEN no ticker's unit price fits the remaining cash THEN the remainder loop SHALL stop.
9. LOT-09: The total `Σ amount_brl` of a class SHALL be ≤ its `target_amount_brl`, except under LOT-13.
10. LOT-10: WHEN a ticker ends LOT-06..LOT-08 with quantity 0 THEN it SHALL be removed from `stocks` / `fiis` and appended, in selector order, to `stocks_dropped` / `fiis_dropped` with `ticker`, `sleeve_weight`, `portfolio_weight`, `target_amount_brl`, `price_brl` and `reasons` = its selector/zone reasons followed by `não coube no capital`.
11. LOT-11: Bought entries SHALL keep the selector order and their existing `sleeve_weight`, `portfolio_weight`, `target_amount_brl` and `reasons`.
12. LOT-12: Targets SHALL NOT be re-normalised after a drop; a dropped ticker's target stays in the remainder cash of LOT-07.
13. LOT-13: IF after LOT-10 fewer than `min(3, number of FIIs selected)` FIIs have quantity ≥ 1 THEN the system SHALL buy exactly 1 quota of the best-placed dropped FIIs (selector order) until that count is reached, even above the class target; each such FII SHALL move back to `fiis` with quantity 1 and the reason `mínimo de 3 FIIs` appended.
14. LOT-14: WHERE whole units are on, the classes that get a buy list (stocks and FIIs, that is the `brazilian_stocks` and `fiis` entries of `classes`) SHALL carry `cash_residual_brl = max(0, target − Σ amount_brl)` and `over_target_brl = max(0, Σ amount_brl − target)`, both in cents; `over_target_brl > 0` SHALL occur only under LOT-13.
15. LOT-15: The rule SHALL NOT apply a minimum count to stocks.
16. LOT-16: WHERE whole units are on THEN `assumptions` SHALL contain the sentence `Quantidades calculadas com o preço da coleta Status Invest desta execução; a cotação na hora da compra pode ser diferente.`
17. LOT-17: Class weights, `target_amount_brl` of classes and tickers, selected tickers, the GA seed and `policy_version` SHALL be identical with and without whole units for the same inputs.

**Independent Test**: run the adapter with fake engines that return the 13 FIIs of run `3289af79`, a temp CSV with their real prices, capital R$ 10,000 and FII weight 0.13 → the FII list has the quantities of the LOT-TEST-1 table and `cash_residual_brl == 4.04`.

---

### P1: Only bought tickers are next month's incumbents ⭐ MVP

**User Story**: As a Premium user, I want next month's protection zone to protect only what I could actually buy, so that a ticker I never bought is not kept for me.

**Why P1**: Issue AC; otherwise a dropped ticker blocks a better replacement.

**Acceptance Criteria**:

1. LOT-18: WHEN `PremiumExecutor` reads a previous run's incumbents THEN it SHALL take the `stocks` / `fiis` tickers whose entry has no `quantity` key, a `null` quantity, or an integer (not bool) quantity ≥ 1, and SHALL skip an entry whose quantity is the integer 0.
2. LOT-19: IF an entry's `quantity` is any other value (negative, float, bool, string) THEN that class's incumbents SHALL be unreadable (`None`, zone status `incumbents_unreadable`), as for a malformed ticker today.
3. LOT-20: Tickers listed only in `stocks_dropped` / `fiis_dropped` SHALL NOT be incumbents.

**Independent Test**: two sequential Premium runs where run 1 drops a FII → run 2's FII engine receives incumbents without it.

---

### P1: Old results stay readable ⭐ MVP

**User Story**: As a user with results saved before this change, I want them to keep opening, so that the history does not break.

**Acceptance Criteria**:

1. LOT-21: WHEN a completed run's `result_json` has none of the new fields THEN `GET /v1/recommendations/{id}` and the latest/latest-completed routes SHALL return 200 with the new constituent and class fields `null` and `stocksDropped` / `fiisDropped` empty.
2. LOT-22: WHEN a completed run's `result_json` has the new fields THEN the response SHALL expose them camel-cased (`priceBrl`, `quantity`, `amountBrl`, `effectivePortfolioWeight`, `cashResidualBrl`, `overTargetBrl`, `stocksDropped`, `fiisDropped`).
3. LOT-23: The response schema SHALL validate `quantity` as int ≥ 1, `price_brl` > 0, `amount_brl`, `effective_portfolio_weight`, `cash_residual_brl`, `over_target_brl` ≥ 0, all optional (default `null`).
4. LOT-24: A Basic run's `result_json` SHALL be identical to the pre-change output for the same inputs (no new keys).

---

### P1: The Premium screen shows the buy list ⭐ MVP

**User Story**: As a Premium user, I want to see the quantity and cost of each ticker and what is left in cash per class, with a note that prices are from the snapshot date.

**Acceptance Criteria**:

1. LOT-25: WHEN a completed result has `quantity` on a constituent THEN the recommendation view SHALL show, per stock and FII, the ticker, the quantity, the unit price and `amountBrl` formatted as BRL.
2. LOT-26: WHEN a class has `cashResidualBrl` THEN the view SHALL show it as the class's cash left over; WHEN `overTargetBrl > 0` THEN it SHALL show that amount as above target.
3. LOT-27: WHEN `stocksDropped` / `fiisDropped` is non-empty THEN the view SHALL list those tickers with their unit price and their `reasons` as sent by the API (no reason text hard-coded in `web/`).
4. LOT-28: WHEN any constituent has `quantity` THEN the view SHALL show a note that prices come from the Status Invest collection of the run, with its `retrieved_at`, and are guidance, not live quotes.
5. LOT-29: IF a constituent has no `quantity` (old or Basic result) THEN the view SHALL render it as today (`ticker: portfolio weight`) and SHALL NOT show the price note.

---

## Edge Cases

- WHEN `Σ amount_brl` equals the class target exactly THEN LOT-07 SHALL accept the last unit (`price ≤ remaining`, not `<`) and `cash_residual_brl` SHALL be 0.
- WHEN every stock drops THEN `stocks` SHALL be empty, `stocks_dropped` SHALL hold them all and `cash_residual_brl` SHALL equal the class target.
- WHEN no FII fits at all THEN LOT-13 SHALL buy 1 quota of each of the first `min(3, n)` FIIs.
- WHEN the stock or FII class weight is ≤ 1e-9 (no selector call) THEN the class SHALL get `cash_residual_brl = target_amount_brl`, `over_target_brl = 0`, empty lists.
- WHEN the same ticker appears in two CSV rows THEN the first row SHALL be used only if every row has the same `PRECO`; otherwise LOT-05 applies.
- IF a forced FII makes `effective_portfolio_weight` > 1 (capital far below one quota) THEN the value SHALL be kept (no upper bound).

---

## Implicit-Requirement Dimensions

| Dimension | Coverage |
| --------- | -------- |
| Input validation & bounds | LOT-02, LOT-05, LOT-23, A-12; duplicate rows edge case. |
| Failure / partial-failure states | LOT-04, LOT-05: the run fails before `mark_completed`; no partial result (existing `_fail` path). |
| Idempotency / retry / duplicates | Pure integer computation over the run's own files: same inputs, same quantities. Run reuse is unchanged (not keyed by policy). |
| Auth boundaries & rate limits | N/A because no route, entitlement or scoping changes; the router only forwards two more `result_json` keys under the existing `get_owned_run` scoping. |
| Concurrency / ordering | N/A because the computation runs inside the single Premium worker after the selectors return. |
| Data lifecycle / expiry | LOT-21 (old results), rollback note in design.md (new results vs. the old schema's `extra="forbid"`). |
| Observability | The result records the method in `policy.provenance.model_versions.whole_units`; the price source hash is already in `provenance.selector_sources`. No new logging. |
| External-dependency failure | N/A because prices come from a local file already validated by `refresh_inputs`; no new network call (LOT-03). |
| State-transition integrity | N/A because run states do not change; failures reuse `snapshot_changed` / `engine_error`. |

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| LOT-01 | P1 buy list | Tasks (T7) | Implemented |
| LOT-02 | P1 buy list | Tasks (T3) | Implemented |
| LOT-03 | P1 buy list | Tasks (T3, T8) | Implemented |
| LOT-04 | P1 buy list | Tasks (T3, T8) | Implemented |
| LOT-05 | P1 buy list | Tasks (T3, T8) | Implemented |
| LOT-06 | P1 buy list | Tasks (T4) | Implemented |
| LOT-07 | P1 buy list | Tasks (T4) | Implemented |
| LOT-08 | P1 buy list | Tasks (T4) | Implemented |
| LOT-09 | P1 buy list | Tasks (T4, T5) | Implemented |
| LOT-10 | P1 buy list | Tasks (T5, T7) | Implemented |
| LOT-11 | P1 buy list | Tasks (T7) | Implemented |
| LOT-12 | P1 buy list | Tasks (T5) | Implemented |
| LOT-13 | P1 buy list | Tasks (T5, T7) | Implemented |
| LOT-14 | P1 buy list | Tasks (T5, T7) | Implemented |
| LOT-15 | P1 buy list | Tasks (T5) | Implemented |
| LOT-16 | P1 buy list | Tasks (T7) | Implemented |
| LOT-17 | P1 buy list | Tasks (T7) | Implemented |
| LOT-18 | P1 incumbents | Tasks (T10) | Pending |
| LOT-19 | P1 incumbents | Tasks (T10) | Pending |
| LOT-20 | P1 incumbents | Tasks (T10, T11) | Pending |
| LOT-21 | P1 old results | Tasks (T1, T2) | In progress (T1, T2; T11 end to end) |
| LOT-22 | P1 old results | Tasks (T2) | Implemented |
| LOT-23 | P1 old results | Tasks (T1) | Implemented |
| LOT-24 | P1 old results | Tasks (T6, T7, T9) | In progress (T6, T7, T8; T9) |
| LOT-25 | P1 screen | Tasks (T13) | Pending |
| LOT-26 | P1 screen | Tasks (T13) | Pending |
| LOT-27 | P1 screen | Tasks (T13) | Pending |
| LOT-28 | P1 screen | Tasks (T13) | Pending |
| LOT-29 | P1 screen | Tasks (T12, T13) | Pending |

**Coverage:** 29 total, 29 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] On the run-`3289af79` FII prices (13 tickers, R$ 100 targets, R$ 1,300 class) the FII sleeve buys R$ 1,295.96 and keeps R$ 4.04 in cash (plain floor: R$ 1,019.38 / R$ 280.62).
- [ ] Every test the issue lists exists and fails when its rule is removed (REVIEW.md).
- [ ] A stored pre-change `result_json` opens in the API and the web view without error.
