# Protection Zone (zona de proteção) Specification

Issue: [#46](https://github.com/caiomugarte/TCC/issues/46) "Mantém ativos da carteira anterior com zona de proteção por frequência no consenso". Route DRIVEN. The design decisions are closed in the issue's `## Decisões`; the zone semantics are fixed by AD-022 (issue #9 door 3) and the values by `docs/validation/multirun-convergence.md`, section "Zona de proteção (K, δ)".

## Problem Statement

The Premium stock and FII sleeves are the frequency consensus of up to 150 GA runs, and those runs agree little with each other (Jaccard 0.14 for stocks). Re-running on every new snapshot swaps a large part of the 26 assets without a real reason, at a tax cost (FII 20% on gains; stocks above R$ 20k/month in sales) and a credibility cost. The protection zone keeps an asset of the previous Premium portfolio while it stays well placed in the new consensus.

## Goals

- [ ] A held asset that is eligible and inside the top K of the new consensus by frequency stays, unless an outside candidate beats it by at least δ in frequency.
- [ ] With no incumbents, every selector output is identical to the current output (regression).
- [ ] Every Premium run records which run it treated as incumbent, K, δ, what was kept, what was swapped and the turnover per class.
- [ ] The month-to-month turnover with and without the zone is measured on real October and November runs and recorded in `docs/validation/` (pending a November snapshot; see Assumptions).

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Scheduling re-runs | Issue #47. |
| Incumbent bonus in the GA fitness | Rejected in `## Decisões`: changes the objective, needs a calibrated weight, hurts reproducibility. |
| The user's real holdings per ticker | Rejected in `## Decisões`: `portfolio_snapshots` holds class weights only; needs a per-ticker contract and new UI. |
| Protection zone on the class weights | ADR 0003's 1/N split is already stable. |
| Protection zone for the Basic plan | The issue targets Premium only (`O executor Premium ...`, `política Premium`). Basic policy stays `basic-policy-v2`, byte-identical. |
| Recalibrating K and δ | `## Decisões`: the #9 values are used as recommended. |
| Showing the zone reasons in the web UI | The API already returns `reasons: list[str]` per constituent; rendering is a separate change. |
| Fixing #9's NaN-score calibration caveat | See Assumptions A-9: recorded, not reopened. |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| A-1 Zone semantics | Exactly AD-022: ranking `R` = every picked ticker by frequency desc, score desc, ticker asc, stocks skipping a ticker whose base company (suffix `11`, `3`, `4`, `5`, `6` stripped) is already ranked; (i) protect incumbents with rank `<= K`; (ii) fill from `R`; (iii) repeated swap of the lowest-ranked protected incumbent still held by the best-ranked outside candidate iff it ranks above it and `f(c) >= f(p) + δ - 1e-12`. | AD-022 is active and accepted; K and δ were calibrated against it, so any other rule would invalidate the #9 values. | y (AD-022, AD-024) |
| A-2 Values | Stocks `k_offset = 6`, FIIs `k_offset = 3`, `δ = 0.11` for both, `K = n_assets + k_offset`. | `docs/validation/multirun-convergence.md` "Zona de proteção (K, δ)" table; AD-022 defines `k_offset = K - n_assets`. | y (issue `## Decisões`) |
| A-3 Incumbent count differs from `n_assets` (profile changed) | `n = n_assets` of the current run. If more incumbents qualify for protection than `n_assets`, the `n_assets` best-ranked are protected and the rest leave with `saiu: vagas reduzidas`. Fewer incumbents leave more vacancies to fill. | AD-022 assumes `n = len(incumbents)` because the simulation never changes the profile; production can. Capping by rank is the only rule that keeps "best placed stays". When `len(incumbents) == n_assets` the result equals AD-022's. | n (author default) |
| A-4 Departure reasons | Incumbents not in the new portfolio carry exactly one of `saiu: inelegível`, `saiu: fora do top K`, `saiu: substituído por <ticker>`, `saiu: vagas reduzidas`. Selected tickers carry exactly one of `mantido`, `novo`, `substituiu <ticker>`. | The issue lists `mantido`, `novo`, `substituiu <ticker>`, `saiu: inelegível`; an incumbent that is eligible but outside the top K, swapped, or capped needs its own reason or the record cannot explain every exit. | n (author default) |
| A-5 Eligible universe | An incumbent is eligible iff its ticker (upper-cased, stripped) is in the ranked frame the GA runs on (`df_ranked` for stocks, `ranked` for FIIs): that frame already applies liquidity/size filters, the restrictions, NaN-score exclusion (#52) and snapshot presence. | One test covers every listed cause; no new eligibility logic. | n (author default) |
| A-6 Turnover | `turnover` = number of tickers in the new class portfolio that are not incumbents (entries). `exits` = incumbents not in the new portfolio. Both recorded. | Equals AD-022's `n - |held ∩ incumbents|` when sizes match; stays well defined when `n_assets` changes. | n (author default) |
| A-7 Which run is the incumbent | The latest `completed` run with `plan = "premium"` of the same account (any profile revision, any policy version, ordered by `created_at` desc then `id` desc as `get_latest_for_profile` does), read at execution time, never the run being executed. Incumbent tickers are `result_json["stocks"][*]["ticker"]` and `result_json["fiis"][*]["ticker"]`. | `## Decisões` fixes "último run Premium `completed` da conta" and says the tickers are already in `result_json`. Reading at execution time lets the single worker see a run that completed while this one was queued. A v2 run is a valid incumbent: the first v3 run must protect what users already hold. | y (issue `## Decisões`) |
| A-8 Unreadable incumbent data | If the incumbent run's `result_json` is missing, not an object, or a class entry is not a list of objects with a non-empty string `ticker`, that class runs with no incumbents and provenance records `status: "incumbents_unreadable"` for it. The run never fails because of this. | Same principle as the HHI decision ("o cliente ficaria sem recomendação por causa de uma regra de estabilidade"). | n (author default) |
| A-9 #9 NaN caveat | K and δ stay as recommended. #52 now drops NaN-score stocks before the GA, while #9's lots include them (AD-025 says #46 must re-simulate if production's NaN handling changes). The equivalence check (PZ-13) re-simulates over the committed lots with the new function; recalibrating on a NaN-free lot is not in scope. | `## Decisões`: "Valores de K e δ: os recomendados pela #9"; reopening needs a new GA experiment (~24 CPU-hours). Logged as an open item for the orchestrator. | n (flagged, see Open questions) |
| A-10 Seed change | `derive_premium_seed` hashes the policy version, so a v3 run draws different GA seeds than a v2 run on the same account, profile and manifest. "Identical without incumbents" (PZ-17) is asserted at the selector level for the same GA runs, not across policy versions. | Bumping the version is mandated by the issue; the seed derivation is existing behavior. | n (author default) |
| A-11 HHI tolerance | The zone falls back when `hhi_sector(zone portfolio) > hhi_max + 1e-12` (the FII selector's tolerance), for both classes. The adapter keeps its own `1e-9` check afterwards. | The stricter check never lets a zone portfolio reach the adapter's check and fail it. | n (author default) |
| A-12 Phase 2 data | No November snapshot exists (only `dev-2026-10-29` and `dev-2026-10-29-ibov` in the main checkout's untracked `snapshots/`; no `docs/validation/` November artifact). The measurement tooling ships with this feature; the measurement itself is PENDING DATA and its numbers are never written by hand. | Orchestrator instruction: do not fabricate; plan a reproducible measurement. | y (orchestrator) |
| A-13 October baseline for Phase 2 | October incumbents = the tickers users received in run `9e22cc16-cfda-495e-a8c9-c0d202ef32b6` (`docs/validation/multirun-convergence-protocol.json` `production_run`, profile `premium-2026-11-02`, seed 46641103). The November run uses the same profile record, policy v3 and seed 46641103. One November execution yields both numbers: the pure consensus (without zone) and the zone portfolio (with zone). | The zone runs after the consensus, so the same GA runs give both portfolios; no second 2.5 h execution. | n (author default) |
| A-14 Reasons without incumbents | When a class has no incumbents, constituent `reasons` are unchanged (no `novo` is added). | Keeps the first-run output identical to today. | n (author default) |

**Open questions** (logged, none blocks Specify/Design/Tasks):

1. A-9: should K and δ be re-checked on a NaN-free lot before the first v3 run reaches users? Default: no (issue decision). The equivalence check (PZ-13) proves the function reproduces #9's simulated turnover on the committed lots.
2. A-12: who registers the November snapshot (`py/snapshot_registry.py build` + `validate` + `publish`) and when? T17 cannot start until a published manifest with a November cutoff exists.

---

## User Stories

### P1: Zone selection function ⭐ MVP

**User Story**: As the Premium selector, I want one pure function that turns the incumbents and the new consensus frequencies into a portfolio plus a reason per ticker, so that both selectors apply the same rule.

**Why P1**: Everything else wires this function.

**Acceptance Criteria** (`py/core/protection_zone.py`):

1. PZ-01: The system SHALL expose one function that takes the incumbent tickers, the frequency of every candidate picked by at least one run (`count / run_count`), the score of every candidate, `n_assets`, `K`, `δ`, the eligible ticker set and whether base-company dedup applies, and returns the ordered portfolio tickers, one reason per selected ticker, one reason per departed incumbent, the kept list, the `(out, in)` swap pairs, `turnover` and `exits`. <!-- ubiquitous -->
2. PZ-02: The system SHALL rank candidates by frequency desc, score desc, ticker asc, using one stable sort over the candidates in the order the frequencies mapping yields them, and, when dedup applies, drop a ticker whose base company is already ranked. <!-- ubiquitous -->
3. PZ-03: WHEN an incumbent is eligible and its rank is `<= K` THEN the system SHALL protect it and keep it with reason `mantido` unless PZ-05 swaps it. <!-- event-driven -->
4. PZ-04: WHEN vacancies remain after protection THEN the system SHALL fill them in rank order, skipping tickers already held and, with dedup, tickers whose base company is held, giving reason `mantido` to a filled ticker that is an incumbent and `novo` to any other. <!-- event-driven -->
5. PZ-05: WHEN the best-ranked outside candidate `c` (not held, never previously replaced, with dedup sharing a base company only with `p`) ranks above the lowest-ranked protected incumbent still held `p` and `f(c) >= f(p) + δ - 1e-12` THEN the system SHALL replace `p` by `c`, give `c` reason `substituiu <p>` and `p` reason `saiu: substituído por <c>`, and repeat; it SHALL stop at the first `c` that fails either condition. <!-- event-driven -->
6. PZ-06: IF an incumbent is not in the eligible set THEN the system SHALL never protect it and SHALL give it reason `saiu: inelegível`. <!-- unwanted-behavior -->
7. PZ-07: WHEN an eligible incumbent has no rank or a rank `> K` and is not selected by the fill THEN the system SHALL give it reason `saiu: fora do top K`. <!-- event-driven -->
8. PZ-08: IF more incumbents qualify for protection than `n_assets` THEN the system SHALL protect only the `n_assets` best-ranked and give the others reason `saiu: vagas reduzidas`. <!-- unwanted-behavior -->
9. PZ-09: WHERE dedup applies the system SHALL never return two tickers with the same base company. <!-- optional-feature -->
10. PZ-10: WHEN the incumbents are empty or `K = 0` THEN the system SHALL return exactly the first `n_assets` tickers of the ranking. <!-- event-driven -->
11. PZ-11: IF `K` is not an integer `>= 0`, `δ` is not finite or is `< 0`, `n_assets < 1`, or a frequency is outside `[0, 1]` THEN the system SHALL raise `ValueError` naming the field. <!-- unwanted-behavior -->
12. PZ-12: The system SHALL return the same result for any order of the incumbent list, and every incumbent SHALL appear exactly once in either the kept list or the departures. <!-- ubiquitous -->
13. PZ-13: WHEN `len(incumbents) == n_assets` and every incumbent is eligible THEN the system SHALL return the same portfolio and turnover as `scripts/convergence_analysis.py:zone_portfolio` (AD-022), and over the committed #9 lots at prefix 150 with the recommended `K`, `δ` it SHALL reproduce the report's turnover at the recommended point: arrojado-anchor/stocks 0.40 / 1, premium-2026-11-02/stocks 0.10 / 1, arrojado-anchor/fiis 0.70 / 2, premium-2026-11-02/fiis 0.00 / 0 (mean / max over ordered seed pairs). <!-- event-driven -->

**Independent Test**: `tests/test_protection_zone.py` on synthetic frequencies; `tests/test_protection_zone_equivalence.py` over `docs/validation/multirun-convergence/lots/`.

---

### P1: HHI fallback ⭐ MVP

**User Story**: As the Premium client, I want a stability rule never to break my concentration limit, so that the zone yields to the pure consensus instead of failing my run.

**Why P1**: Decided in `## Decisões`; without it a zone portfolio could fail the adapter's HHI check.

**Acceptance Criteria**:

1. PZ-14: IF a sector HHI limit is configured and the zone portfolio's `hhi_sector` exceeds `hhi_max + 1e-12` THEN the system SHALL use the pure consensus for that class and record `buffer_fallback: "hhi"`, with reasons recomputed against the consensus (`mantido` for incumbents in it, `novo` otherwise, departures as PZ-06/PZ-07). <!-- unwanted-behavior -->
2. PZ-15: WHERE no HHI limit is configured the system SHALL never fall back and SHALL record `buffer_fallback: null`. <!-- optional-feature -->
3. PZ-16: IF the zone falls back (PZ-14) AND the pure consensus also exceeds `hhi_max + 1e-12` THEN the system SHALL behave as today: the FII selector raises `FiiSelectionError`, and the adapter raises `PremiumOptimizationError` with code `infeasible_constraints` for stocks. IF the zone portfolio is within the limit THEN the system SHALL use it, even when the pure consensus exceeds the limit. <!-- unwanted-behavior -->

**Independent Test**: selector tests with a two-sector fixture where the zone portfolio's HHI exceeds `hhi_max` and the consensus's does not.

---

### P1: Selectors apply the zone ⭐ MVP

**User Story**: As the multi-run pipeline, I want `build_consensus_portfolio` and `_consensus` to apply the zone when given incumbents, and to behave exactly as today otherwise.

**Why P1**: The issue's criterion "sem incumbentes, o resultado é idêntico ao atual".

**Acceptance Criteria**:

1. PZ-17: WHEN `build_consensus_portfolio` or `_consensus` receives no incumbents (argument omitted, `None` or empty) THEN the system SHALL return a frame equal to the current one: same rows, row order, columns, values and `attrs`. <!-- event-driven -->
2. PZ-18: WHEN `build_consensus_portfolio` receives incumbents and a zone config `{K, delta}` THEN the system SHALL return the zone portfolio's rows from `df_ranked` with `FREQUENCY`, `attrs["hhi"]`, `attrs["avg_score"]`, `attrs["method"] = "consensus_frequency"` and `attrs["protection_zone"]` holding the class record (PZ-25), sorted by `FREQUENCY` desc as today. <!-- event-driven -->
3. PZ-19: WHEN `_consensus` receives incumbents and a zone config THEN the system SHALL return the zone portfolio's rows from `df_ranked` with `FREQUENCY`, sorted by `FREQUENCY`, `SCORE` desc and `TICKER` asc as today, and `attrs["protection_zone"]` holding the class record, with no base-company dedup (FIIs). <!-- event-driven -->
4. PZ-20: WHEN `run_multi_execution_profile` (explicit mode) or `run_fii_selection` (explicit mode) receives `incumbents` and its config carries `protection_zone` THEN the system SHALL return the class record under the result key `protection_zone`; WHEN either receives no incumbents THEN the result SHALL carry no `protection_zone` key. <!-- event-driven -->

**Independent Test**: frame-equality tests against the pre-change functions on fixed fixtures with and without incumbents.

---

### P1: Premium policy v3 ⭐ MVP

**User Story**: As an auditor, I want K and δ in the versioned Premium policy, so that a persisted run says which zone produced it.

**Acceptance Criteria** (`api/app/services/premium_policy.py`):

1. PZ-21: The system SHALL set `POLICY_VERSION = "premium-policy-v3"`, keep `BASIC_POLICY_VERSION = "basic-policy-v2"`, and reject `premium-policy-v2` in `resolve_premium_policy` with `PremiumPolicyError`. <!-- ubiquitous -->
2. PZ-22: WHEN a Premium policy is resolved THEN the system SHALL set `stocks.protection_zone = {"k_offset": 6, "K": stocks.n_assets + 6, "delta": 0.11}` and `fiis.protection_zone = {"k_offset": 3, "K": fiis.n_assets + 3, "delta": 0.11}`. <!-- event-driven -->
3. PZ-23: WHEN a Basic policy is resolved THEN the system SHALL emit a `to_dict()`/`to_json()` with no `protection_zone` key, equal to the pre-change output for the same profile. <!-- event-driven -->

---

### P1: Executor passes the incumbents ⭐ MVP

**User Story**: As the Premium worker, I want to read the account's last completed Premium portfolio and hand it to the selectors.

**Acceptance Criteria** (`api/app/services/premium_executor.py`, `api/app/repositories/recommendations.py`):

1. PZ-24: WHEN the executor runs a `premium` run whose policy declares `protection_zone` for a class THEN the system SHALL load the latest `completed` `premium` run of the same account other than this run (A-7) and pass `incumbents = {"run_id": <id>, "stocks": [...], "fiis": [...]}` to the optimization runner. <!-- event-driven -->
2. PZ-25: IF the account has no completed Premium run THEN the system SHALL call the optimization runner without the `incumbents` keyword, exactly as today. <!-- unwanted-behavior -->
3. PZ-26: IF the incumbent run's `result_json` is unreadable for a class (A-8) THEN the system SHALL pass no incumbents for that class, record `status: "incumbents_unreadable"` for it, and complete the run. <!-- unwanted-behavior -->
4. PZ-27: WHEN the run's plan is `basic` THEN the system SHALL never query for incumbents. <!-- event-driven -->

---

### P1: Provenance and reasons ⭐ MVP

**User Story**: As the client and as an auditor, I want each run to say what it kept and why.

**Acceptance Criteria** (`api/app/adapters/premium_optimization.py`):

1. PZ-28: WHEN a Premium run's policy declares a zone THEN the system SHALL record `provenance.protection_zone = {"incumbent_run_id": <id or null>, "stocks": <class record>, "fiis": <class record>}`, where a class record holds `status` (`applied`, `fallback_hhi`, `no_incumbents`, `incumbents_unreadable` or `class_not_selected`), `K`, `k_offset`, `delta`, `n_assets`, `incumbents`, `kept`, `replaced` (list of `{"out", "in"}`), `entered`, `departures` (`{ticker: reason}`), `turnover`, `exits`, `buffer_fallback` (`"hhi"` or `null`), `consensus_tickers` (pure consensus, sorted) and `consensus_turnover` (entries of the pure consensus versus the incumbents). <!-- event-driven -->
2. PZ-29: WHEN a class has incumbents THEN the system SHALL append the ticker's zone reason (`mantido`, `novo` or `substituiu <ticker>`) to that constituent's `reasons` list. <!-- event-driven -->
3. PZ-30: The adapter SHALL pass the `incumbents` keyword to a selector engine only for a class that has at least one incumbent, and SHALL copy the policy section's `protection_zone` into that selector's config. <!-- ubiquitous -->
4. PZ-31: WHERE the policy declares no zone (Basic, legacy policies) the system SHALL produce a result with no `provenance.protection_zone` key, unchanged from today. <!-- optional-feature -->

---

### P2: Month-to-month turnover (fase 2 da #9)

**User Story**: As the product owner, I want the real October-to-November turnover with and without the zone, so that the zone's effect is measured beyond seed noise.

**Why P2**: Needs a November snapshot that does not exist yet (A-12).

**Acceptance Criteria** (`scripts/protection_zone_turnover.py`):

1. PZ-32: WHEN `measure` runs with a published November manifest THEN the system SHALL run the production Premium pipeline once with the profile record `premium-2026-11-02`, policy v3, seed 46641103 and the October incumbents of A-13, and write `docs/validation/protection-zone-turnover.json` and `.md` with, per class, the turnover with zone (`turnover`), without zone (`consensus_turnover`), the kept, swapped and departed tickers, K, δ, the manifest id and sha256, the code commit and the seed. <!-- event-driven -->
2. PZ-33: IF the manifest's cutoff date is before 2026-11-01 or its id is the October manifest's (`dev-2026-10-29-ibov`) THEN the system SHALL exit with code 3 and write nothing. <!-- unwanted-behavior -->
3. PZ-34: The system SHALL produce byte-identical report files when `measure` reruns with the same manifest, code and seed. <!-- ubiquitous -->
4. PZ-35: WHEN a November snapshot is published THEN the measured report SHALL be committed to `docs/validation/` and linked from `docs/validation/README.md`; until then the README SHALL state the measurement is pending data. <!-- event-driven -->

**Independent Test**: `measure` with injected fake engines and a fixture manifest (no GA), checking exit codes, file contents and byte-identity.

---

## Edge Cases

- WHEN the class weight is 0 (the selector did not run) THEN the system SHALL record that class with `status: "class_not_selected"` and no turnover. (PZ-28)
- WHEN fewer candidates are ranked than `n_assets` THEN the system SHALL return every ranked ticker the rules admit, as the current consensus does. (PZ-04)
- WHEN an incumbent appears twice or in lower case in `result_json` THEN the system SHALL normalize to upper-case and deduplicate before calling the function. (PZ-24)
- WHEN two incumbents share a base company (legacy data) THEN only the one present in the deduped ranking can be protected; the other leaves with `saiu: fora do top K`. (PZ-02, PZ-07)
- WHEN a frequency tie at `δ = 0` is broken by score or ticker in favor of the incumbent THEN the system SHALL not swap (AD-022 rank condition). (PZ-05)
- WHEN `f(c) - f(p)` equals `δ` exactly in exact arithmetic but not in floats (run_count 100, counts 21 and 10, δ 0.11: `0.21 < 0.10 + 0.11 = 0.21000000000000002`) THEN the `- 1e-12` tolerance SHALL make the swap happen (`>=` is inclusive). (PZ-05)

---

## Implicit-Requirement Dimensions Sweep

| Dimension | Resolution |
| --------- | ---------- |
| Input validation & bounds | PZ-11 (function inputs); PZ-24/PZ-26 and the edge cases (incumbent normalization, unreadable data). |
| Failure / partial-failure states | PZ-14, PZ-16 (HHI), PZ-26 (unreadable incumbents never fail the run). |
| Idempotency / retry / duplicate handling | PZ-12 (order-independent, deterministic); PZ-34 (byte-identical report). A retried run reads the same latest completed run, so it gets the same incumbents. |
| Auth boundaries & rate limits | PZ-24: incumbents come only from runs of the same account (`account_id` filter). No new endpoint. |
| Concurrency / ordering | The executor has one worker (`max_workers == 1`); incumbents are read at execution time (A-7), so a run queued behind another sees it once completed. No DB lock needed. |
| Data lifecycle / expiry | N/A because the feature reads existing `result_json` rows and writes no new table; no migration. |
| Observability | PZ-28 provenance record per class; PZ-29 reasons per constituent. |
| External-dependency failure | N/A because no new external call: Status Invest refresh and manifest loading keep their existing error codes. |
| State-transition integrity | Only `completed` runs are incumbents (PZ-24); the run state machine is untouched. |

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| PZ-01 | P1: Zone selection function | Tasks T1-T2 | Verified |
| PZ-02 | P1: Zone selection function | Tasks T1 | Verified |
| PZ-03 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-04 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-05 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-06 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-07 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-08 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-09 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-10 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-11 | P1: Zone selection function | Tasks T1 | Verified |
| PZ-12 | P1: Zone selection function | Tasks T2 | Verified |
| PZ-13 | P1: Zone selection function | Tasks T4 | Verified |
| PZ-14 | P1: HHI fallback | Tasks T3, T5, T7 | Verified |
| PZ-15 | P1: HHI fallback | Tasks T3 | Verified |
| PZ-16 | P1: HHI fallback | Tasks T3, T8, T12 | Verified |
| PZ-17 | P1: Selectors apply the zone | Tasks T5, T7 | Verified |
| PZ-18 | P1: Selectors apply the zone | Tasks T5 | Verified |
| PZ-19 | P1: Selectors apply the zone | Tasks T7 | Verified |
| PZ-20 | P1: Selectors apply the zone | Tasks T6, T8 | Verified |
| PZ-21 | P1: Premium policy v3 | Tasks T9 | Verified |
| PZ-22 | P1: Premium policy v3 | Tasks T9 | Verified |
| PZ-23 | P1: Premium policy v3 | Tasks T9 | Verified |
| PZ-24 | P1: Executor passes the incumbents | Tasks T10, T11 | Verified |
| PZ-25 | P1: Executor passes the incumbents | Tasks T11 | Verified |
| PZ-26 | P1: Executor passes the incumbents | Tasks T11, T13 | Verified |
| PZ-27 | P1: Executor passes the incumbents | Tasks T11 | Verified |
| PZ-28 | P1: Provenance and reasons | Tasks T3, T13, T14 | Verified |
| PZ-29 | P1: Provenance and reasons | Tasks T13 | Verified |
| PZ-30 | P1: Provenance and reasons | Tasks T12 | Verified |
| PZ-31 | P1: Provenance and reasons | Tasks T13 | Verified |
| PZ-32 | P2: Month-to-month turnover | Tasks T15, T17 | Pending data (T15 verified; T17 deferred) |
| PZ-33 | P2: Month-to-month turnover | Tasks T15 | Verified |
| PZ-34 | P2: Month-to-month turnover | Tasks T15 | Verified |
| PZ-35 | P2: Month-to-month turnover | Tasks T16, T17 | Pending data (T16 verified; T17 deferred) |

**ID format:** `PZ-NN`.

**Coverage:** 35 total, 35 mapped to tasks, 0 unmapped. PZ-32 and PZ-35 cannot reach Verified until a November snapshot exists (T17).

---

## Success Criteria

- [ ] Over the committed #9 lots, the new function reproduces #9's turnover at the recommended point in all four cells (PZ-13).
- [ ] Zero diffs in selector output for runs without incumbents (PZ-17), checked by frame equality.
- [ ] A second v3 Premium run of an account records `incumbent_run_id` equal to the first run's id and a per-class `turnover` (PZ-28).
- [ ] `docs/validation/protection-zone-turnover.md` reports October-to-November turnover with and without the zone (pending data, PZ-35).
