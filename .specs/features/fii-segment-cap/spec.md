# FII Segment Cap and Policy v4 Specification

Issue: [#56](https://github.com/caiomugarte/TCC/issues/56) "Limita os FIIs a 2 por segmento e reduz para 4/5/6 por perfil (política v4)". Route DRIVEN. Every design decision is closed in the issue's `## Decisões`; this spec encodes them and does not reopen them. The protection-zone contract it extends is AD-026 (issue #46); the cap's place in the zone is recorded as AD-027. The HHI skip that keeps the capped consensus within `hhi_max` (spec review r1, finding S1) is a human decision recorded as AD-028; its extension to the protected incumbents (closing Q3) is a human decision recorded as AD-029.

Baseline: `a61164a` (branch `t3/issue-56`, base `origin/develop`). Root suite 535 tests (2 skipped); API suite 160 tests.

## Problem Statement

The `premium-policy-v3` FII sleeve uses the stock asset counts (10/12/15 per anchor, 13 for the owner profile). With whole quotas at ~R$ 100 or more, that is too many FIIs to buy with small capital. Concentration by segment is only penalised through `λ × HHI × n` (`py/core/optimizer.py:138`), so with few slots the GA can still put most of the sleeve in one segment: at the owner's λ 0.19, four "Papéis" funds out of five cost only ~0.46 of fitness. The 2026-11-06 measurement (92 FIIs) shows the fund count matters more than the segment, but a cap stops the sleeve from landing entirely in one bad segment.

## Goals

- [ ] Premium FII sleeves hold 4 (conservador), 5 (moderado) and 6 (arrojado) funds at the anchors, 5 for the owner profile, interpolated as today.
- [ ] No Premium FII sleeve holds more than 2 funds of one `SETOR`, at every stage that decides membership (GA, pure consensus, protection zone).
- [ ] Under the `limitar_concentracao` restriction (`hhi_max` 0.25) a capped Premium FII run never fails on its own HHI post-check: the consensus, the protection walk, the fill and the challengers skip a ticker that would push the sleeve above `hhi_max` (AD-028, AD-029).
- [ ] Under the same restriction the zone keeps its protection: an incumbent the HHI rule rejects leaves with `saiu: teto do segmento` and the others stay protected; the zone never falls back to the `K = 0` consensus for it (AD-029).
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
| A-5 Order inside the zone | Qualifying incumbents (rank `<= K`) are walked in rank order (frequency desc, score desc, ticker asc); one whose segment already has `max_per_segment` accepted incumbents leaves with `saiu: teto do segmento`. WHERE `hhi_max` is also set, while fewer than `n_assets` incumbents are accepted, one that does not satisfy `fits` (A-16) against the incumbents already accepted also leaves with `saiu: teto do segmento` and the walk continues (AD-029). The `n_assets` capacity cut (`saiu: vagas reduzidas`) applies after the segment cut to the accepted list. | "mantendo os de maior frequência" = rank order. Cutting by segment first keeps the most incumbents: with qualifying `A1, A2, A3` (segment P) and `B1`, n = 3, segment-first keeps `A1, A2, B1`; capacity-first would drop `B1` for nothing. | n (author default) |
| A-6 Departure reason precedence | Exactly one per departing incumbent: `saiu: inelegível` > `saiu: substituído por <t>` > `saiu: teto do segmento` > `saiu: vagas reduzidas` > `saiu: fora do top K`. `saiu: teto do segmento` covers both the cap and the HHI rule of the protection walk (A-5); the record does not distinguish them. | Extends AD-026's four reasons with the issue's new one; a ticker cut by segment never reaches the capacity cut (A-5), so the two never compete. An HHI rejection always involves a segment already accepted: adding a ticker of a new segment leaves the A-16 bound unchanged, so it fits whenever the accepted set fits. | n (author default) |
| A-7 Fill and challengers under the cap | Fill takes ranked tickers in order, skipping one that does not `fit` (A-16). A challenger is the best-ranked ticker not held and not replaced that fits the held set minus the challenged incumbent; a candidate that does not fit is skipped and the next one is tried. The swap test itself (rank above, `f(c) >= f(p) + δ - 1e-12`) is unchanged. | Issue (c): "os desafiantes também respeitam o teto". Skipping (not stopping) mirrors the pure consensus rule (b), so `K = 0` still equals the capped pure consensus. | n (author default) |
| A-8 Greedy exactness and the infeasible case | With `max_per_segment <= 2` the rank-order greedy under `fits` is exact: it reaches `n_assets` if and only if some `n_assets`-subset of the candidates satisfies the cap and `hhi_max`. Proof sketch: with cap 2 a sleeve of `k` tickers holding `p` pairs has `Σc² = k + 2p`, so the projected bound of A-16 is `n + 2p` and the HHI rule reduces to "at most `P = floor((hhi_max·n² + tol − n) / 2)` pairs"; a ticker of a new segment always fits, so the greedy fails only after walking every candidate, holding `d + min(P, q)` tickers (`d` distinct segments, `q` segments with 2 or more candidates), which is the maximum any feasible subset can hold. Every GA run is a feasible `n_assets` subset of the union of runs (GA infeasibility for both the cap and `hhi_max`), so the pure consensus and the `K = 0` consensus never fail in production. The zone fill starts from protected incumbents accepted under `fits` (A-5), and the count above does not depend on the starting set (protected pairs count toward `P`, protected segments toward `d`), so with cap 2 the zone fill reaches `n_assets` whenever the `K = 0` consensus does. For `max_per_segment >= 3` the greedy can fail on a feasible set; it then raises (SEG-40), never returns a shorter sleeve. With `hhi_max` 0.25: `P = 0` at n 4 and 5, `P = 1` at n 6. | Combinatorics above; Premium sets cap 2 (AD-027). | n (author default) |
| A-9 Universe too small for the cap | Before any GA run, explicit-mode `run_fii_selection` raises `FiiSelectionError` (message names `max_per_segment` and the capacity) when `Σ_segments min(count, max_per_segment) < n_assets` on the scored universe. The adapter reports it through its existing generic path (`engine_error`); no new error code. | Otherwise the GA would score every chromosome `-inf` and raise "GA não convergiu". Unreachable on the committed data. Capacity with cap 2 on the scored universe (the `build_fii_scores` frame the check runs on, after eligibility filters and scoring): conservador 8 of 12 funds (segment multiplicities 5/3/2/1/1, as pinned in `api/tests/test_premium_policy.py:508-512`) for n 4; moderado 18 of 57 (11 segments) for n 5; owner (selector score 0.70) 18 of 71 (11 segments) for n 5; arrojado 25 of 137 (15 segments) for n 6. The `limitar_concentracao` restriction does not change these frames. Measured 2026-11-06 on `data/raw/status_invest_fii.csv` (spec review r1 S2 corrected the earlier 9/18/28/20, which counted the eligible frame before scoring). The distinct-segment counts (5/11/11/15) also cover the `hhi_max` 0.25 need of at least n distinct segments at n 4 and 5 and 5 at n 6. | n (author default) |
| A-10 #9 research tooling | `scripts/multirun_convergence.py` and `scripts/protection_zone_turnover.py` resolve the live `POLICY_VERSION`. Under v4 the frozen protocol's owner profile resolves `(stocks 13, fiis 5)` against a declared `(13, 13)`, so `check_n_assets` exits 2. The protocol file and its lots are not edited. Tests that only use the protocol as a fixture base set `expected_n_assets` to the v4 counts; the test of the real protocol asserts the v4 counts and the exit-2 guard. | Measured: the v4 bump alone fails 16 + 5 tests in `tests/test_multirun_convergence.py` (all `expected_n_assets`) and 1 in `tests/test_protection_zone_turnover.py` (version string). Re-running #9 under v4 is #47 territory. | n (flagged, Q2) |
| A-11 Seed change | `derive_premium_seed` hashes the policy version, so a v4 run draws different GA seeds than a v3 run on the same inputs. "Identical without a cap" is asserted at the GA, zone and selector level for the same seeds, not across policy versions. | Existing behaviour (AD-026 trade-off). | n (author default) |
| A-12 Which existing tests change | Only tests that pin v3 policy values: `PremiumPolicyTests.test_conservador_fii_rules_pin_the_adr_0002_calibration`, `test_policy_versions_are_bumped_and_reject_the_v1_strings`, `test_persisted_provenance_reports_the_bumped_policy_versions`, `ProtectionZonePolicyTests.test_premium_sections_carry_the_issue_9_zone_values_for_every_anchor`, `ConservadorAnchorExecutabilityTests.test_unrestricted_anchor_differs_only_by_hhi_max_and_runs_above_the_cap` (fixes `n_assets` 10 so its subject, an uncapped HHI above 0.25, survives n = 4), `ProtectionZoneIntegrationTests.test_second_premium_run_records_the_first_as_incumbent` (version string), `tests/test_protection_zone_turnover.py` (version string) and `tests/test_multirun_convergence.py` (A-10). Every other existing test passes with no value change. | Measured on a scratch copy of `a61164a` with only `POLICY_VERSION` and the shared `FII_RULES` counts changed: the API tests above failed, plus the two Basic tests (`test_basic_generic_profile_bands_map_to_discrete_stock_and_fii_rules`, `test_basic_policy_has_no_zone_and_matches_the_pre_change_json`) because Basic shares `FII_RULES` today; A-2 keeps those two green unchanged. Root: 21 tests in `tests/test_multirun_convergence.py` (every one through `expected_n_assets`) and 1 in `tests/test_protection_zone_turnover.py`. `tests/test_fii_selection.py`, `tests/test_protection_zone*.py`, `tests/test_benchmark_premium.py`, `tests/test_premium_research_contracts.py` stayed green. | n (author default) |
| A-13 Zone record | The per-class zone record gains no field. The cap lives in `policy.fiis`; a cap departure appears in `departures` with `saiu: teto do segmento` and counts in `exits`. | Keeps the stock record and the record shape of AD-026 unchanged; the policy is already persisted next to it. | n (author default) |
| A-14 Interpolation | `n_assets = int(round(_interpolate(left, right, fraction)))` as today, Python half-to-even: selector score 0.25 → 4, 0.70 → 5, 0.75 → 6. | "interpolados como hoje". | y (issue) |
| A-16 HHI skip under the cap (AD-028, AD-029) | WHERE `max_per_segment` and `hhi_max` are both set, one predicate `fits(t, S)` decides membership at every greedy step (pure consensus, zone protection walk while fewer than `n_assets` are accepted, zone fill, zone challenger): `count_S(segment_of[t]) < max_per_segment` AND `(Σ_s c_s(S ∪ {t})² + (n_assets − |S ∪ {t}|)) / n_assets² <= hhi_max + 1e-12`. The second term is the lowest HHI any completion of `S ∪ {t}` to `n_assets` can reach (each remaining slot in a new segment adds 1 to `Σc²`); it is evaluated at the target size, never on the partial sleeve. Tolerance `1e-12` is the one the GA and the post-check use. For a challenger, `S` is the held set minus the challenged incumbent; for the protection walk, `S` is the incumbents already accepted. Without a cap, `hhi_max` never reaches the greedy: stocks and uncapped FIIs keep the pre-change code path (the GA `hhi_max` infeasibility and the zone's post-hoc `hhi_of` check are unchanged). | Human decision after spec review r1 (S1): at n 4 and 5 a 2+1+1(+1) split gives HHI 0.375 and 0.28 > 0.25, so the cap-2 consensus could pick a pair that every GA run avoided and fail the post-check at `py/fii_selection.py:769-770`; reproduced at owner score 0.70, n 5, 8 runs 40×30, seed 0. One predicate in one walk order makes the cap skip and the HHI skip order-consistent: whichever term rejects a ticker, the walk continues with the next one. | y (human, AD-028) |
| A-17 Zone under `hhi_max` (AD-029) | WHERE both are set, the protection walk applies `fits` (A-5), so the protected incumbents always satisfy the A-16 bound: of two qualifying incumbents of one segment that cannot both stay (at n 4 and 5 with `hhi_max` 0.25 no pair fits), the lower-ranked leaves with `saiu: teto do segmento` and every other protection is kept. `select_with_fallback` never falls back to the `K = 0` consensus because of `SegmentCapInfeasibleError`: it catches it from neither call. IF the zone fill still cannot reach `n_assets` under `fits` (only with `max_per_segment >= 3` or a candidate set with no feasible `n_assets` subset, A-8) THEN `apply_protection_zone` raises `SegmentCapInfeasibleError` (a `ValueError` subclass in `py/core/protection_zone.py`), `select_with_fallback` propagates it, and `_consensus` turns it into `FiiSelectionError` (SEG-40), never a shorter sleeve. The post-hoc `hhi_of(zone) > hhi_max` fallback of AD-026 is unchanged; under the cap it cannot fire, because the zone is built within `hhi_max`. | Human decision closing Q3: the issue's `## Decisões` rejected falling back to the pure consensus because it loses protection exactly in the 13 → 5 FII migration. Rejected: accept `fallback_hhi` and lose every protection (the previous revision's default). Propagating the residual error, instead of falling back, is the author's default: it is unreachable with cap 2 (A-8) and keeps one failure path. | y (human, AD-029) |
| A-15 Caches | No `data/processed/*.csv` or `outputs/` file changes: the cap acts on selection, not on preprocessing, and the offline named-profile path passes no cap. | REVIEW.md cache check. | n (author default) |

**Open questions** (logged with a default, none blocks Design or Tasks):

1. **Q1 (A-2)**: should Basic also move to 4/5/6 FIIs with the cap (a `basic-policy-v3`)? Default: no, the issue names Premium only. Changing it later is one constant and a version bump.
2. **Q2 (A-10)**: the frozen #9 protocol cannot be replayed under v4 (exit 2 on `expected_n_assets`). Default: accept it; #47 or a new protocol owns any v4 re-measurement.

**Closed questions**:

- **Q3 (A-17)**, closed by the human on 2026-11-06 (AD-029): the HHI rule also applies to the protected incumbents. Under `limitar_concentracao` at n 4 or 5, of two qualifying incumbents of one segment the lower-ranked leaves with `saiu: teto do segmento` and the zone keeps the rest; it does not fall back to the `K = 0` consensus.

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

1. WHEN qualifying incumbents (rank `<= K`) hold more than `max_per_segment` tickers of one segment THEN `apply_protection_zone` SHALL protect the `max_per_segment` best-ranked of that segment and SHALL record every other one in `departures` as `saiu: teto do segmento` (the HHI rule of SEG-39 can cut further). <!-- SEG-19 -->
2. WHEN the incumbents left after the segment cut still exceed `n_assets` THEN the zone SHALL protect the `n_assets` best-ranked of them and record the rest as `saiu: vagas reduzidas` (segment cut first, A-5). <!-- SEG-20 -->
3. WHEN the zone fills vacancies THEN it SHALL skip a ranked ticker whose segment already holds `max_per_segment` held tickers. <!-- SEG-21 -->
4. WHEN the zone looks for a challenger THEN it SHALL take the best-ranked ticker not held and not replaced whose segment holds fewer than `max_per_segment` tickers once the challenged incumbent leaves, and SHALL apply the unchanged rank and δ test to it. <!-- SEG-22 -->
5. WHERE `max_per_segment` is set, `select_with_fallback` SHALL apply the cap to both the zone portfolio and the `K = 0` consensus it records (`consensus_tickers`) and falls back to. <!-- SEG-23 -->
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
2. IF `max_per_segment` is set and `Σ_segments min(count, max_per_segment) < n_assets` over the scored universe THEN `run_fii_selection` SHALL raise `FiiSelectionError` naming `max_per_segment` before any GA run; WHEN the capacity equals `n_assets` it SHALL NOT raise and SHALL return `n_assets` funds. <!-- SEG-28 -->
3. IF the final FII selection holds more than `max_per_segment` tickers of one segment THEN `run_fii_selection` SHALL raise `FiiSelectionError` matching "segment cap". <!-- SEG-29 -->
4. WHEN the adapter builds the FII selector config from a policy section with `max_per_segment` THEN `_selector_config` SHALL copy it as an `int` to the top level of the config passed to the FII engine. <!-- SEG-30 -->
5. IF the policy section's `max_per_segment` is present and is not an integer `>= 1` THEN the adapter SHALL raise `PremiumOptimizationError` with code `policy_invalid`. <!-- SEG-31 -->
6. WHEN a policy section has no `max_per_segment` (stocks, Basic, a stored `premium-policy-v3` run) THEN `_selector_config` SHALL return a config without that key, equal to the pre-change config. <!-- SEG-32 -->
7. WHEN a Premium run is created and executed end to end THEN `policy_json["fiis"]["max_per_segment"]` SHALL be 2, `policy_version` SHALL be `premium-policy-v4`, and the FII engine SHALL receive `selection_config["max_per_segment"] == 2`. <!-- SEG-33 -->

**Independent Test**: `api/tests/test_premium_optimization.py` and `api/tests/test_premium_integration.py` with fake engines.

---

### P1: HHI skip under the cap ⭐ MVP

**User Story**: As a Premium user with the `limitar_concentracao` restriction, I want the capped FII consensus and zone to stay within `hhi_max` so that my run never fails on its own HHI post-check.

**Why P1**: Spec review r1 S1: without it, a capped run at n 4 or 5 with `hhi_max` 0.25 can raise "FII selection violates the configured HHI limit" (reproduced on committed data). Human decision AD-028.

**Acceptance Criteria**:

1. WHERE `max_per_segment` and `hhi_max` are both set, WHEN the pure consensus walks tickers in rank order THEN it SHALL skip a ticker that does not satisfy `fits` (A-16) and continue with the next one, using the same walk and the same predicate as the segment-cap skip. <!-- SEG-37 -->
2. WHERE `max_per_segment` and `hhi_max` are both set, WHEN the zone fills vacancies or looks for a challenger THEN it SHALL take only tickers that satisfy `fits`, evaluated on the held set for the fill and on the held set minus the challenged incumbent for a challenger, skipping the others. <!-- SEG-38 -->
3. WHERE `max_per_segment` and `hhi_max` are both set, WHEN the protection walk reaches a qualifying incumbent that does not satisfy `fits` against the incumbents already accepted, while fewer than `n_assets` are accepted, THEN `apply_protection_zone` SHALL leave it out with `saiu: teto do segmento` and SHALL keep walking the remaining qualifying incumbents in rank order (frequency desc, score desc, ticker asc), so the protected incumbents never break the A-16 bound. <!-- SEG-39 -->
4. IF the capped pure consensus, the `K = 0` consensus or the zone cannot reach `n_assets` under `fits` THEN `_consensus` SHALL raise `FiiSelectionError` naming `max_per_segment` and `hhi_max`, and SHALL NOT return fewer than `n_assets` funds. <!-- SEG-40 -->
5. WHILE `max_per_segment` is `None` the HHI skip SHALL NOT run: `_consensus` and `select_with_fallback` SHALL NOT pass `hhi_max` to `take_within_segment_cap` or `apply_protection_zone`, and IF either receives `hhi_max` without `max_per_segment`, or a `hhi_max` that is not a finite number `> 0`, THEN it SHALL raise `ValueError` naming `hhi_max`. <!-- SEG-41 -->
6. WHEN `run_fii_selection` runs on the committed `data/raw/status_invest_fii.csv` with the v4 FII section resolved for selector score 0.70 and `limitar_concentracao` (n 5, cap 2, `hhi_max` 0.25), 8 runs, GA 40 generations × 30 population, seeds 0 to 3, with and without 13 incumbents and the zone THEN it SHALL return 5 funds with HHI `<= 0.25 + 1e-12` and at most 2 per segment, without raising, and with incumbents the zone `status` SHALL NOT be `fallback_hhi`. <!-- SEG-42 -->
7. WHERE `max_per_segment` and `hhi_max` are both set, IF the zone fill cannot reach `n_assets` under `fits` THEN `apply_protection_zone` SHALL raise `SegmentCapInfeasibleError`, and `select_with_fallback` SHALL propagate a `SegmentCapInfeasibleError` raised by either the zone call or the `K = 0` call, and SHALL NOT return the `K = 0` consensus or `status "fallback_hhi"` because of it. <!-- SEG-43 -->

**Independent Test**: `tests/test_segment_cap_zone.py` (greedy, zone and protection walk), `tests/test_segment_cap_fii.py` (consensus), `api/tests/test_premium_policy.py` (real data).

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

- IF the scored universe cannot fill `n_assets` under the cap THEN the selector SHALL fail before the GA (SEG-28); a capacity of exactly `n_assets` runs.
- WHEN `n_assets` is 4 and the 4 picks sit in 4 distinct segments THEN HHI is exactly 0.25 and the sleeve fits `hhi_max` 0.25 (tolerance `1e-12`, SEG-37 boundary).
- IF the scored universe has fewer distinct segments than `hhi_max` needs THEN the GA scores every chromosome `-inf` as it does today without the cap; this pre-existing `hhi_max` behaviour is unchanged and not part of the SEG-28 check.
- IF no candidate fits at a greedy step THEN the selector SHALL raise, never deliver a smaller sleeve (SEG-43 for the zone, SEG-40 for the consensus).
- WHEN two qualifying incumbents of one segment have equal frequency under `limitar_concentracao` at n 4 or 5 THEN the one with the higher `SCORE` (then the lower ticker) SHALL stay and the other SHALL leave with `saiu: teto do segmento` (SEG-39, rank tie-break).
- WHEN fewer than `max_per_segment` incumbents of a segment qualify THEN all of them SHALL stay protected (SEG-19 boundary: exactly `max_per_segment` is allowed).
- WHEN incumbents come from a v3 run with 13 FIIs and the v4 run has `n_assets` 5 THEN every incumbent SHALL end in exactly one of `kept` or `departures` (partition invariant of AD-026 holds with the new reason).
- WHEN `max_per_segment >= n_assets` and `hhi_max` is `None` THEN the selected tickers, `kept`, `replaced` and `entered` SHALL equal the uncapped result (the cap never binds). WHERE `hhi_max` is set the HHI skip (A-16) still applies, so the equality need not hold. Departure reasons also match unless more than `max_per_segment` qualifying incumbents share one segment: those beyond the cap leave with `saiu: teto do segmento` instead of `saiu: vagas reduzidas` (SEG-19, A-6).
- IF a v3 run queued before the deploy is executed after it THEN it SHALL run from its stored `policy_json` without `max_per_segment` and behave as v3 (SEG-32).

## Implicit-requirement dimensions

| Dimension | Coverage |
| --------- | -------- |
| Input validation & bounds | SEG-12, SEG-13, SEG-25, SEG-31. |
| Failure / partial-failure states | SEG-28 (universe too small), SEG-29 (defensive post-check), SEG-39 (protected incumbents over `hhi_max` → the lower-ranked leaves, no fallback), SEG-40 and SEG-43 (consensus or zone cannot fill → error, never a shorter sleeve or a silent fallback). |
| Idempotency / retry / duplicates | N/A because selection is a pure seeded computation; the run cache keyed by policy version is unchanged. |
| Auth boundaries & rate limits | N/A because no route, entitlement or account scoping changes. |
| Concurrency / ordering | N/A because the cap travels inside the pickled config to `mp.Pool` workers; no shared state is added. |
| Data lifecycle / expiry | Stored v3 runs execute from their stored policy (SEG-32, edge case). |
| Observability | Cap recorded in `policy.fiis` (SEG-04), method version (SEG-07), departure reason (SEG-19, SEG-26, SEG-39), zone `status` never `fallback_hhi` from the cap (SEG-42, SEG-43). |
| External-dependency failure | N/A because no external call is added. |
| State-transition integrity | N/A because run states and transitions do not change. |

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| SEG-01 | P1: Premium policy v4 | Tasks T11 | In Tasks |
| SEG-02 | P1: Premium policy v4 | Tasks T11 | In Tasks |
| SEG-03 | P1: Premium policy v4 | Tasks T10 | In Tasks |
| SEG-04 | P1: Premium policy v4 | Tasks T9 | Implemented (T9) |
| SEG-05 | P1: Premium policy v4 | Tasks T11, T12 | In Tasks |
| SEG-06 | P1: Premium policy v4 | Tasks T10 | In Tasks |
| SEG-07 | P1: Premium policy v4 | Tasks T11 | In Tasks |
| SEG-08 | P1: Premium policy v4 | Tasks T10, T11 | In Tasks |
| SEG-09 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-10 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-11 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-12 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-13 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-14 | P1: GA treats an over-cap candidate as infeasible | Tasks T1 | Implemented (T1) |
| SEG-15 | P1: Pure consensus skips a full segment | Tasks T2, T7 | Implemented (T2, T7) |
| SEG-16 | P1: Pure consensus skips a full segment | Tasks T7 | Implemented (T7) |
| SEG-17 | P1: Pure consensus skips a full segment | Tasks T7 | Implemented (T7) |
| SEG-18 | P1: Pure consensus skips a full segment | Tasks T7 | Implemented (T7) |
| SEG-19 | P1: Protection zone respects the cap | Tasks T3, T7 | Implemented (T3, T7) |
| SEG-20 | P1: Protection zone respects the cap | Tasks T3 | Implemented (T3) |
| SEG-21 | P1: Protection zone respects the cap | Tasks T3 | Implemented (T3) |
| SEG-22 | P1: Protection zone respects the cap | Tasks T3 | Implemented (T3) |
| SEG-23 | P1: Protection zone respects the cap | Tasks T5 | Implemented (T5) |
| SEG-24 | P1: Protection zone respects the cap | Tasks T3, T5 | Implemented (T3, T4, T5) |
| SEG-25 | P1: Protection zone respects the cap | Tasks T2 | Implemented (T2) |
| SEG-26 | P1: Protection zone respects the cap | Tasks T8 | Implemented (T8) |
| SEG-27 | P1: FII selector and adapter carry the cap | Tasks T6, T8, T12 | In progress (T6, T8 done) |
| SEG-28 | P1: FII selector and adapter carry the cap | Tasks T8 | Implemented (T8) |
| SEG-29 | P1: FII selector and adapter carry the cap | Tasks T8 | Implemented (T8) |
| SEG-30 | P1: FII selector and adapter carry the cap | Tasks T13 | In Tasks |
| SEG-31 | P1: FII selector and adapter carry the cap | Tasks T13 | In Tasks |
| SEG-32 | P1: FII selector and adapter carry the cap | Tasks T6, T13, T14 | In progress (T6 done) |
| SEG-33 | P1: FII selector and adapter carry the cap | Tasks T14 | In Tasks |
| SEG-34 | P2: Research tooling follows v4 | Tasks T10 | In Tasks |
| SEG-35 | P2: Research tooling follows v4 | Tasks T11 | In Tasks |
| SEG-36 | P2: Research tooling follows v4 | Tasks T10, T11 | In Tasks |
| SEG-37 | P1: HHI skip under the cap | Tasks T2, T7 | Implemented (T2, T7) |
| SEG-38 | P1: HHI skip under the cap | Tasks T4 | Implemented (T4) |
| SEG-39 | P1: HHI skip under the cap | Tasks T4, T5 | Implemented (T4, T5) |
| SEG-40 | P1: HHI skip under the cap | Tasks T2, T7 | Implemented (T2, T7) |
| SEG-41 | P1: HHI skip under the cap | Tasks T2, T4, T5, T7 | Implemented (T2, T4, T5, T7) |
| SEG-42 | P1: HHI skip under the cap | Tasks T12 | In Tasks |
| SEG-43 | P1: HHI skip under the cap | Tasks T4, T5, T12 | In progress (T4, T5 done) |

**Coverage:** 43 total, 43 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] The owner profile's Premium policy resolves 5 FIIs with `max_per_segment` 2 and K 8 (SEG-05).
- [ ] No GA run, pure consensus or zone portfolio in the new tests holds 3 FIIs of one segment.
- [ ] The owner profile with `limitar_concentracao` runs 8 × 40 × 30 on committed data for seeds 0 to 3 without raising (SEG-42); seed 0 raises with the HHI skip removed.
- [ ] Two protected incumbents of one segment under `limitar_concentracao` at n 5 keep the higher-ranked one, record the other as `saiu: teto do segmento`, and the zone status is not `fallback_hhi` (SEG-39, SEG-43).
- [ ] Root suite (535 + new) and API suite (160 + new) pass; the only existing tests whose values change are the ones listed in A-12.
