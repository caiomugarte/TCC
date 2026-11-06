# FII Segment Cap and Policy v4 Specification

Issue: [#56](https://github.com/caiomugarte/TCC/issues/56) "Limita os FIIs a 2 por segmento e reduz para 4/5/6 por perfil (política v4)". Route DRIVEN. Every design decision is closed in the issue's `## Decisões`; this spec encodes them and does not reopen them. The protection-zone contract it extends is AD-026 (issue #46); the cap's place in the zone is recorded as AD-027.

Baseline: `a61164a` (branch `t3/issue-56`, base `origin/develop`). Root suite 535 tests (2 skipped); API suite 160 tests.

## Problem Statement

The `premium-policy-v3` FII sleeve uses the stock asset counts (10/12/15 per anchor, 13 for the owner profile). With whole quotas at ~R$ 100 or more, that is too many FIIs to buy with small capital. Concentration by segment is only penalised through `λ × HHI × n` (`py/core/optimizer.py:138`), so with few slots the GA can still put most of the sleeve in one segment: at the owner's λ 0.19, four "Papéis" funds out of five cost only ~0.46 of fitness. The 2026-11-06 measurement (92 FIIs) shows the fund count matters more than the segment, but a cap stops the sleeve from landing entirely in one bad segment.

## Goals

- [ ] Premium FII sleeves hold 4 (conservador), 5 (moderado) and 6 (arrojado) funds at the anchors, 5 for the owner profile, interpolated as today.
- [ ] No Premium FII sleeve holds more than 2 funds of one `SETOR`, at every stage that decides membership (GA, pure consensus, protection zone).
- [ ] Every caller that does not pass a cap (stocks, Basic, offline named profiles, a stored `premium-policy-v3` run) gets byte-identical output.
- [ ] The resolved policy records the cap (`policy.fiis.max_per_segment = 2`), the policy version (`premium-policy-v4`) and the new FII method version (`fii-selection-v2`).

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Buy list with whole-quota quantities | Next issue (`## Fora de escopo`). |
| Recalibrating the FII zone K and δ for n = 5 | Needs its own convergence measurement (`## Decisões`). `k_offset` 3 and δ 0.11 stay. |
| Segment cap on stocks | With 10 to 15 stocks HHI is enough (9 sectors today). |
| Repeated metrics per class in the result | Issue #11. |
| Re-running the accounts under v4 | Issue #47. |
| Changing the Basic plan | The issue names the Premium policy only. Basic stays `basic-policy-v2`, byte-identical (A-2, open question Q1). |
| Editing the frozen #9 protocol `docs/validation/multirun-convergence-protocol.json` or its committed lots | Committed research evidence (REVIEW.md). Its declared FII count (13) no longer matches v4; that mismatch is the tool's guard working (A-10, Q2). |
| Showing the cap or the new departure reason in the web UI | The API already records reasons; rendering is separate. |
| A repair operator in GA mutation | Rejected in `## Decisões` (more code; infeasible rate is low with 5 of ~70 candidates). |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| A-1 Name and place of the cap | `max_per_segment` (int, `None` = no cap). It is a field of `SelectorPolicy` (omitted from `to_dict()`/`to_json()` when `None`, as `protection_zone` is), a top-level key of the FII `selection_config`, a keyword of `GeneticAlgorithm`, `apply_protection_zone` and `select_with_fallback`. | One name end to end; omission when `None` keeps every pre-change JSON byte-identical. | n (author default) |
| A-2 Basic plan | Basic keeps FII anchors 10/12/15, no cap, `basic-policy-v2` and `model_versions.fiis = "fii-selection-v1"`; `resolve_basic_policy(...).to_json()` equals the pinned `BASIC_GOLDEN_JSON`. | The issue scopes the change to "a política Premium"; Basic already uses its own discrete bands and #46 kept Basic byte-identical; changing Basic's output without bumping `basic-policy-v2` would persist a wrong version (REVIEW.md). | n (flagged, Q1) |
| A-3 Method version | Premium `provenance.model_versions` becomes `{"allocation": "allocation-v1", "stocks": "stock-selection-v1", "fiis": "fii-selection-v2"}` when the caller passes none. Basic keeps `fii-selection-v1`. | REVIEW.md: update `model_versions` when the selection method changes; the FII method changes (cap), the stock method does not. | n (author default) |
| A-4 Segment identity | A segment is the exact `SETOR` string of the ranked frame (the same column `hhi_sector` counts). "Indefinido" is an ordinary label and is capped like any other. | Issue `## Decisões` ("Indefinido" counts as its own segment); `load_fii_data` already rejects a blank `SETOR` (`py/fii_selection.py:278`). | y (issue) |
| A-5 Order inside the zone | Qualifying incumbents (rank `<= K`) are walked in rank order (frequency desc, score desc, ticker asc); one whose segment already has `max_per_segment` protected incumbents leaves with `saiu: teto do segmento`. The `n_assets` capacity cut (`saiu: vagas reduzidas`) applies after the segment cut. | "mantendo os de maior frequência" = rank order. Cutting by segment first keeps the most incumbents: with qualifying `A1, A2, A3` (segment P) and `B1`, n = 3, segment-first keeps `A1, A2, B1`; capacity-first would drop `B1` for nothing. | n (author default) |
| A-6 Departure reason precedence | Exactly one per departing incumbent: `saiu: inelegível` > `saiu: substituído por <t>` > `saiu: teto do segmento` > `saiu: vagas reduzidas` > `saiu: fora do top K`. | Extends AD-026's four reasons with the issue's new one; a ticker cut by segment never reaches the capacity cut (A-5), so the two never compete. | n (author default) |
| A-7 Fill and challengers under the cap | Fill takes ranked tickers in order, skipping one whose segment is full. A challenger is the best-ranked ticker not held and not replaced whose segment has fewer than `max_per_segment` held tickers once the challenged incumbent leaves; a candidate that would break the cap is skipped and the next one is tried. The swap test itself (rank above, `f(c) >= f(p) + δ - 1e-12`) is unchanged. | Issue (c): "os desafiantes também respeitam o teto". Skipping (not stopping) mirrors the pure consensus rule (b), so `K = 0` still equals the capped pure consensus. | n (author default) |
| A-8 The cap is a partition-matroid constraint | Greedy rank-order fill under the cap always reaches `n_assets` when the candidate set can (any GA run is a feasible `n_assets` portfolio, so the union of runs always can). No new "could not fill" error in `_consensus` or the zone. | Partition-matroid greedy property; every GA output is cap-feasible (SEG-12). | n (author default) |
| A-9 Universe too small for the cap | Before any GA run, explicit-mode `run_fii_selection` raises `FiiSelectionError` (message names `max_per_segment` and the capacity) when `Σ_segments min(count, max_per_segment) < n_assets` on the scored universe. The adapter reports it through its existing generic path (`engine_error`); no new error code. | Otherwise the GA would score every chromosome `-inf` and raise "GA não convergiu". Unreachable on the committed data: capacity with cap 2 is 9 (conservador, n 4), 18 (moderado, n 5), 28 (arrojado, n 6), 20 (owner, n 5), measured 2026-11-06 on `data/raw/status_invest_fii.csv`. | n (author default) |
| A-10 #9 research tooling | `scripts/multirun_convergence.py` and `scripts/protection_zone_turnover.py` resolve the live `POLICY_VERSION`. Under v4 the frozen protocol's owner profile resolves `(stocks 13, fiis 5)` against a declared `(13, 13)`, so `check_n_assets` exits 2. The protocol file and its lots are not edited. Tests that only use the protocol as a fixture base set `expected_n_assets` to the v4 counts; the test of the real protocol asserts the v4 counts and the exit-2 guard. | Measured: the v4 bump alone fails 16 + 5 tests in `tests/test_multirun_convergence.py` (all `expected_n_assets`) and 1 in `tests/test_protection_zone_turnover.py` (version string). Re-running #9 under v4 is #47 territory. | n (flagged, Q2) |
| A-11 Seed change | `derive_premium_seed` hashes the policy version, so a v4 run draws different GA seeds than a v3 run on the same inputs. "Identical without a cap" is asserted at the GA, zone and selector level for the same seeds, not across policy versions. | Existing behaviour (AD-026 trade-off). | n (author default) |
| A-12 Which existing tests change | Only tests that pin v3 policy values: `PremiumPolicyTests.test_conservador_fii_rules_pin_the_adr_0002_calibration`, `test_policy_versions_are_bumped_and_reject_the_v1_strings`, `test_persisted_provenance_reports_the_bumped_policy_versions`, `ProtectionZonePolicyTests.test_premium_sections_carry_the_issue_9_zone_values_for_every_anchor`, `ConservadorAnchorExecutabilityTests.test_unrestricted_anchor_differs_only_by_hhi_max_and_runs_above_the_cap` (fixes `n_assets` 10 so its subject, an uncapped HHI above 0.25, survives n = 4), `ProtectionZoneIntegrationTests.test_second_premium_run_records_the_first_as_incumbent` (version string), `tests/test_protection_zone_turnover.py` (version string) and `tests/test_multirun_convergence.py` (A-10). Every other existing test passes with no value change. | Measured on a scratch copy of `a61164a` with only `POLICY_VERSION` and the shared `FII_RULES` counts changed: the API tests above failed, plus the two Basic tests (`test_basic_generic_profile_bands_map_to_discrete_stock_and_fii_rules`, `test_basic_policy_has_no_zone_and_matches_the_pre_change_json`) because Basic shares `FII_RULES` today; A-2 keeps those two green unchanged. Root: 21 tests in `tests/test_multirun_convergence.py` (every one through `expected_n_assets`) and 1 in `tests/test_protection_zone_turnover.py`. `tests/test_fii_selection.py`, `tests/test_protection_zone*.py`, `tests/test_benchmark_premium.py`, `tests/test_premium_research_contracts.py` stayed green. | n (author default) |
| A-13 Zone record | The per-class zone record gains no field. The cap lives in `policy.fiis`; a cap departure appears in `departures` with `saiu: teto do segmento` and counts in `exits`. | Keeps the stock record and the record shape of AD-026 unchanged; the policy is already persisted next to it. | n (author default) |
| A-14 Interpolation | `n_assets = int(round(_interpolate(left, right, fraction)))` as today, Python half-to-even: selector score 0.25 → 4, 0.70 → 5, 0.75 → 6. | "interpolados como hoje". | y (issue) |
| A-15 Caches | No `data/processed/*.csv` or `outputs/` file changes: the cap acts on selection, not on preprocessing, and the offline named-profile path passes no cap. | REVIEW.md cache check. | n (author default) |

**Open questions** (logged with a default, none blocks Design or Tasks):

1. **Q1 (A-2)**: should Basic also move to 4/5/6 FIIs with the cap (a `basic-policy-v3`)? Default: no, the issue names Premium only. Changing it later is one constant and a version bump.
2. **Q2 (A-10)**: the frozen #9 protocol cannot be replayed under v4 (exit 2 on `expected_n_assets`). Default: accept it; #47 or a new protocol owns any v4 re-measurement.

---

## User Stories

### P1: Premium policy v4 ⭐ MVP

**User Story**: As a Premium user with small capital, I want a FII sleeve of 4 to 6 funds with at most 2 per segment so that I can buy it with whole quotas and it never sits in one segment.

**Why P1**: The issue's first two acceptance criteria; every other story reads the cap from this policy.

**Acceptance Criteria**:

1. The system SHALL expose `POLICY_VERSION == "premium-policy-v4"` and record it in `provenance.policy_version` of every resolved Premium policy. <!-- SEG-01 -->
2. IF `resolve_premium_policy` or `resolve_recommendation_policy(plan="premium")` receives `rules_version="premium-policy-v3"` THEN the system SHALL raise `PremiumPolicyError` matching "unsupported policy version". <!-- SEG-02 -->
3. WHEN the Premium selector score is 0.0, 0.5 or 1.0 THEN the system SHALL resolve `fiis.n_assets` 4, 5 and 6 respectively, and `stocks.n_assets` 10, 12 and 15. <!-- SEG-03 -->
4. WHEN a Premium policy is resolved THEN the system SHALL set `fiis.max_per_segment = 2` for every selector score and SHALL leave `stocks.max_per_segment` as `None`, absent from `to_dict()["stocks"]`. <!-- SEG-04 -->
5. WHEN the owner profile (selector score 0.70) is resolved THEN the system SHALL give `fiis.n_assets == 5`, `fiis.max_per_segment == 2`, `fiis.protection_zone == {"k_offset": 3, "K": 8, "delta": 0.11}` and `stocks.n_assets == 13`. <!-- SEG-05 -->
6. The system SHALL keep the FII zone at `k_offset` 3 and δ 0.11 with `K = fiis.n_assets + 3` for every selector score, and the stock zone at `k_offset` 6 and δ 0.11. <!-- SEG-06 -->
7. WHEN a Premium policy is resolved without explicit `model_versions` THEN the system SHALL record `{"allocation": "allocation-v1", "stocks": "stock-selection-v1", "fiis": "fii-selection-v2"}`. <!-- SEG-07 -->
8. WHEN a Basic policy is resolved THEN the system SHALL keep `basic-policy-v2`, FII `n_assets` 10/12/15 per band, no `max_per_segment`, `model_versions.fiis == "fii-selection-v1"`, and `to_json()` equal to `BASIC_GOLDEN_JSON`. <!-- SEG-08 -->

**Independent Test**: resolve the three anchors, the owner score and a Basic band in `api/tests/test_premium_policy.py` and compare fields and JSON.

---

### P1: GA treats an over-cap candidate as infeasible ⭐ MVP

**User Story**: As the optimizer, I want a candidate above the segment cap to score `-inf`, like `hhi_max`, so that no GA run ever returns it.

**Why P1**: Issue criterion (a); every downstream consensus draws from GA runs.

**Acceptance Criteria**:

1. WHERE `GeneticAlgorithm` has `max_per_segment` set, WHEN a candidate holds more than `max_per_segment` tickers with one `SETOR` value THEN `fitness` SHALL return `-inf`. <!-- SEG-09 -->
2. WHERE `max_per_segment` is set, WHEN a candidate holds at most `max_per_segment` per segment THEN `fitness` SHALL return the same value as a GA without the cap (`SCORE sum - λ × HHI × n`). <!-- SEG-10 -->
3. WHERE `max_per_segment` is set, WHEN `optimize` returns THEN the portfolio SHALL hold at most `max_per_segment` tickers per `SETOR`, on a frame whose top scores sit in one segment, for every seed tested. <!-- SEG-11 -->
4. IF `max_per_segment` is not `None` and is not an integer `>= 1` (`0`, `-1`, `1.5`, `True`, `"2"`) THEN the `GeneticAlgorithm` constructor SHALL raise `ValueError` naming `max_per_segment`. <!-- SEG-12 -->
5. IF `max_per_segment` is set and the frame has no `SETOR` column THEN `fitness` SHALL raise `ValueError` naming `SETOR`. <!-- SEG-13 -->
6. WHILE `max_per_segment` is `None` the GA SHALL return the same tickers, `fitness`, `hhi`, `generations_run` and `seed` as the pre-change GA for the same frame and seed (golden values captured at `a61164a`). <!-- SEG-14 -->

**Independent Test**: `tests/test_segment_cap_optimizer.py` on a synthetic frame with a dominant segment.

---

### P1: Pure consensus skips a full segment ⭐ MVP

**User Story**: As the FII selector, I want the frequency consensus to skip a ticker whose segment is already full so that a sleeve without incumbents respects the cap.

**Why P1**: Issue criterion (b) and its first test.

**Acceptance Criteria**:

1. WHERE the FII consensus runs with `max_per_segment` and without a protection zone, WHEN it walks tickers by frequency desc, score desc, ticker asc THEN it SHALL skip a ticker whose segment already holds `max_per_segment` selected tickers and continue with the next one until `n_assets` are selected. <!-- SEG-15 -->
2. WHEN the three most frequent tickers share one segment and `max_per_segment` is 2 THEN the consensus SHALL select the two most frequent of them and the next-ranked ticker of another segment. <!-- SEG-16 -->
3. WHEN the three most frequent tickers have `SETOR == "Indefinido"` and `max_per_segment` is 2 THEN the consensus SHALL select only two of them. <!-- SEG-17 -->
4. WHILE no `max_per_segment` is given the consensus frame (rows, order, columns, values, `attrs`) SHALL equal the pre-change `_consensus` output. <!-- SEG-18 -->

**Independent Test**: `_consensus` on synthetic portfolios in `tests/test_segment_cap_fii.py`.

---

### P1: Protection zone respects the cap ⭐ MVP

**User Story**: As a Premium user moving from 13 to 5 FIIs, I want the zone to keep my best-placed funds while never exceeding 2 per segment, with a reason for each fund that leaves.

**Why P1**: Issue criterion (c), its "titular removido por teto" test, and the `## Decisões` rule that the cap is a zone exit reason, not a fallback.

**Acceptance Criteria**:

1. WHEN qualifying incumbents (rank `<= K`) hold more than `max_per_segment` tickers of one segment THEN `apply_protection_zone` SHALL protect the `max_per_segment` best-ranked of that segment and SHALL record every other one in `departures` as `saiu: teto do segmento`. <!-- SEG-19 -->
2. WHEN the incumbents left after the segment cut still exceed `n_assets` THEN the zone SHALL protect the `n_assets` best-ranked of them and record the rest as `saiu: vagas reduzidas` (segment cut first, A-5). <!-- SEG-20 -->
3. WHEN the zone fills vacancies THEN it SHALL skip a ranked ticker whose segment already holds `max_per_segment` held tickers. <!-- SEG-21 -->
4. WHEN the zone looks for a challenger THEN it SHALL take the best-ranked ticker not held and not replaced whose segment holds fewer than `max_per_segment` tickers once the challenged incumbent leaves, and SHALL apply the unchanged rank and δ test to it. <!-- SEG-22 -->
5. WHERE `max_per_segment` is set, `select_with_fallback` SHALL apply the cap to both the zone portfolio and the `K = 0` consensus it records (`consensus_tickers`) and falls back to on `hhi_max`. <!-- SEG-23 -->
6. WHILE `max_per_segment` is `None` `apply_protection_zone` and `select_with_fallback` SHALL return the same selection and record as before the change (every existing test in `tests/test_protection_zone.py` and `tests/test_protection_zone_equivalence.py` passes unchanged). <!-- SEG-24 -->
7. IF `max_per_segment` is set and `segment_of` is missing or lacks a ranked ticker, or `max_per_segment` is not an integer `>= 1`, THEN the zone SHALL raise `ValueError` naming the field. <!-- SEG-25 -->
8. WHEN `run_fii_selection` runs with incumbents, a zone and `max_per_segment` 2, and three incumbents of one segment rank within K THEN `result["protection_zone"]["departures"]` SHALL map the lowest-ranked of them to `saiu: teto do segmento`. <!-- SEG-26 -->

**Independent Test**: `tests/test_segment_cap_zone.py` on plain dicts; SEG-26 in `tests/test_segment_cap_fii.py`.

---

### P1: FII selector and adapter carry the cap ⭐ MVP

**User Story**: As the Premium pipeline, I want the cap from `policy.fiis` to reach every GA run and the consensus, and nothing else to change for callers without it.

**Why P1**: Without the wiring the policy value has no effect.

**Acceptance Criteria**:

1. WHEN the explicit FII `selection_config` carries `max_per_segment` THEN `run_fii_selection` SHALL construct every `GeneticAlgorithm` with that value and SHALL pass it to the consensus. <!-- SEG-27 -->
2. IF `max_per_segment` is set and `Σ_segments min(count, max_per_segment) < n_assets` over the scored universe THEN `run_fii_selection` SHALL raise `FiiSelectionError` naming `max_per_segment` before any GA run. <!-- SEG-28 -->
3. IF the final FII selection holds more than `max_per_segment` tickers of one segment THEN `run_fii_selection` SHALL raise `FiiSelectionError` matching "segment cap". <!-- SEG-29 -->
4. WHEN the adapter builds the FII selector config from a policy section with `max_per_segment` THEN `_selector_config` SHALL copy it as an `int` to the top level of the config passed to the FII engine. <!-- SEG-30 -->
5. IF the policy section's `max_per_segment` is present and is not an integer `>= 1` THEN the adapter SHALL raise `PremiumOptimizationError` with code `policy_invalid`. <!-- SEG-31 -->
6. WHEN a policy section has no `max_per_segment` (stocks, Basic, a stored `premium-policy-v3` run) THEN `_selector_config` SHALL return a config without that key, equal to the pre-change config. <!-- SEG-32 -->
7. WHEN a Premium run is created and executed end to end THEN `policy_json["fiis"]["max_per_segment"]` SHALL be 2, `policy_version` SHALL be `premium-policy-v4`, and the FII engine SHALL receive `selection_config["max_per_segment"] == 2`. <!-- SEG-33 -->

**Independent Test**: `api/tests/test_premium_optimization.py` and `api/tests/test_premium_integration.py` with fake engines.

---

### P2: Research tooling follows v4

**User Story**: As the maintainer, I want the #9 tooling tests to keep exercising the tools under v4 without editing the frozen protocol.

**Why P2**: Not user-facing; required only so the build gate passes honestly.

**Acceptance Criteria**:

1. WHEN the frozen protocol's `premium-2026-11-02` profile is resolved under v4 THEN the test SHALL assert `(stocks, fiis) == (13, 5)` and that `check_n_assets` against the declared counts raises the `expected_n_assets` error with exit code 2. <!-- SEG-34 -->
2. WHEN the turnover tool writes its report THEN `policy_version` SHALL be `premium-policy-v4`. <!-- SEG-35 -->
3. The system SHALL leave `docs/validation/multirun-convergence-protocol.json` and `docs/validation/multirun-convergence/**` byte-identical. <!-- SEG-36 -->

**Independent Test**: `tests/test_multirun_convergence.py`, `tests/test_protection_zone_turnover.py`, `git diff --stat` on `docs/validation/`.

---

## Edge Cases

- IF the scored universe cannot fill `n_assets` under the cap THEN the selector SHALL fail before the GA (SEG-28).
- WHEN fewer than `max_per_segment` incumbents of a segment qualify THEN all of them SHALL stay protected (SEG-19 boundary: exactly `max_per_segment` is allowed).
- WHEN incumbents come from a v3 run with 13 FIIs and the v4 run has `n_assets` 5 THEN every incumbent SHALL end in exactly one of `kept` or `departures` (partition invariant of AD-026 holds with the new reason).
- WHEN `max_per_segment >= n_assets` THEN the result SHALL equal the uncapped result (the cap never binds).
- IF a v3 run queued before the deploy is executed after it THEN it SHALL run from its stored `policy_json` without `max_per_segment` and behave as v3 (SEG-32).

## Implicit-requirement dimensions

| Dimension | Coverage |
| --------- | -------- |
| Input validation & bounds | SEG-12, SEG-13, SEG-25, SEG-31. |
| Failure / partial-failure states | SEG-28 (universe too small), SEG-29 (defensive post-check). |
| Idempotency / retry / duplicates | N/A because selection is a pure seeded computation; the run cache keyed by policy version is unchanged. |
| Auth boundaries & rate limits | N/A because no route, entitlement or account scoping changes. |
| Concurrency / ordering | N/A because the cap travels inside the pickled config to `mp.Pool` workers; no shared state is added. |
| Data lifecycle / expiry | Stored v3 runs execute from their stored policy (SEG-32, edge case). |
| Observability | Cap recorded in `policy.fiis` (SEG-04), method version (SEG-07), departure reason (SEG-19, SEG-26). |
| External-dependency failure | N/A because no external call is added. |
| State-transition integrity | N/A because run states and transitions do not change. |

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| SEG-01 | P1: Premium policy v4 | Tasks T10 | In Tasks |
| SEG-02 | P1: Premium policy v4 | Tasks T10 | In Tasks |
| SEG-03 | P1: Premium policy v4 | Tasks T9 | In Tasks |
| SEG-04 | P1: Premium policy v4 | Tasks T8 | In Tasks |
| SEG-05 | P1: Premium policy v4 | Tasks T10, T11 | In Tasks |
| SEG-06 | P1: Premium policy v4 | Tasks T9 | In Tasks |
| SEG-07 | P1: Premium policy v4 | Tasks T10 | In Tasks |
| SEG-08 | P1: Premium policy v4 | Tasks T9, T10 | In Tasks |
| SEG-09 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-10 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-11 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-12 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-13 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-14 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | In Tasks |
| SEG-15 | P1: Pure consensus skips a full segment | Tasks T2, T6 | In Tasks |
| SEG-16 | P1: Pure consensus skips a full segment | Tasks T6 | In Tasks |
| SEG-17 | P1: Pure consensus skips a full segment | Tasks T6 | In Tasks |
| SEG-18 | P1: Pure consensus skips a full segment | Tasks T6 | In Tasks |
| SEG-19 | P1: Protection zone respects the cap | Tasks T3, T6 | In Tasks |
| SEG-20 | P1: Protection zone respects the cap | Tasks T3 | In Tasks |
| SEG-21 | P1: Protection zone respects the cap | Tasks T3 | In Tasks |
| SEG-22 | P1: Protection zone respects the cap | Tasks T3 | In Tasks |
| SEG-23 | P1: Protection zone respects the cap | Tasks T4 | In Tasks |
| SEG-24 | P1: Protection zone respects the cap | Tasks T3, T4 | In Tasks |
| SEG-25 | P1: Protection zone respects the cap | Tasks T2 | In Tasks |
| SEG-26 | P1: Protection zone respects the cap | Tasks T7 | In Tasks |
| SEG-27 | P1: FII selector and adapter carry the cap | Tasks T5, T7, T11 | In Tasks |
| SEG-28 | P1: FII selector and adapter carry the cap | Tasks T7 | In Tasks |
| SEG-29 | P1: FII selector and adapter carry the cap | Tasks T7 | In Tasks |
| SEG-30 | P1: FII selector and adapter carry the cap | Tasks T12 | In Tasks |
| SEG-31 | P1: FII selector and adapter carry the cap | Tasks T12 | In Tasks |
| SEG-32 | P1: FII selector and adapter carry the cap | Tasks T5, T12, T13 | In Tasks |
| SEG-33 | P1: FII selector and adapter carry the cap | Tasks T13 | In Tasks |
| SEG-34 | P2: Research tooling follows v4 | Tasks T9 | In Tasks |
| SEG-35 | P2: Research tooling follows v4 | Tasks T10 | In Tasks |
| SEG-36 | P2: Research tooling follows v4 | Tasks T9, T10 | In Tasks |

**Coverage:** 36 total, 36 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] The owner profile's Premium policy resolves 5 FIIs with `max_per_segment` 2 and K 8 (SEG-05).
- [ ] No GA run, pure consensus or zone portfolio in the new tests holds 3 FIIs of one segment.
- [ ] Root suite (535 + new) and API suite (160 + new) pass; the only existing tests whose values change are the ones listed in A-12.
