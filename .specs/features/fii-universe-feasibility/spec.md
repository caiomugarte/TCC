# FII Universe Feasibility Policy Specification

**Status:** Approved (human checkpoint passed 2026-10-27: ADR-0002 option (a) + policy-version bump). Implementation parked until issue #7 (P0-02) lands.
**Issue:** caiomugarte/TCC#12 — P0-05: Política de factibilidade do universo FII
**Route:** DRIVEN (spec → design → tasks, then IMPLEMENT)
**Parent feature:** `.specs/features/profile-v1-premium-optimization/`
**Decision record:** `docs/adr/0002-fii-universe-feasibility-policy.md`

## Problem Statement

The conservative FII policy asks for 10 assets (`n_assets=10`) but its eligibility
thresholds (`size_min=5e9`, `liq_min=2e6`) leave only 7 eligible FIIs out of 588
in `data/raw/status_invest_fii.csv`, so every FII selection run dies with
`FiiSelectionError("FII universe has N assets; M required")`
(`py/fii_selection.py:392-396`) — a failure with no thresholds, no profile, and
no counts that survives to the user. Downstream, the adapter collapses it to
`engine_error` and the executor replaces the message with a fixed generic string,
so the recorded failure explains nothing. Silently reducing `n_assets` would hide
the problem and violate the declared policy, so the shortage must become an
explicit, recorded policy decision plus a hard, self-explaining precondition.

## Goals

- [ ] No FII selector execution proceeds with `n_assets` greater than its
      pickable universe; a shortage fails fast as `infeasible_universe`.
- [ ] The failure records universe count, required count, thresholds, and
      profile — sanitized (no paths, no raw exception text) and within the
      `String(64)`/`String(512)` column limits.
- [ ] The policy calibration is chosen between the two issue options with
      quantified sensitivity evidence and an accepted ADR.
- [ ] A failed Basic publication stays a closed failure and keeps the previous
      completed batch published.

## Out of Scope

| Feature / change | Reason |
| --- | --- |
| Relaxing eligibility filters beyond the recorded calibration | Issue forbids "afrouxar filtros sem validação"; only the ADR-approved calibration lands |
| Silently reducing `n_assets` at runtime | Explicitly forbidden by the issue; the precondition fails instead |
| Inventing or substituting instruments | Forbidden by `.specs/features/profile-v1-premium-optimization/spec.md:281-283` |
| Changing stock (`STOCK_RULES`) or moderado/arrojado FII anchors | Issue scope is the conservador FII shortage; other profiles change only with their own analysis |
| Named-profile offline runs (`py/run_fii.py` and committed artifacts) | Guard stays in the explicit/premium path; `data/processed/fii_clean_*.csv`, `outputs/carteira_fii_*_consensus.json`, `data/allocation/caio_fiis.csv` must not be rewritten |
| Behavior of `filters["enforce_liquidity_and_size"]` | Computed but unread by the engines today; changing it is a side effect this feature must not take. The feature SHALL NOT alter how that key is computed or consumed — review checklist item (T7), not observable behavior (decision recorded in design) |
| New persistence tables or columns | `RecommendationRun.failure_code`/`failure_message` already exist and suffice |
| Web/UI redesign | `web/components/recommendation/recommendation-view.tsx` already renders closed failures and keeps the last completed batch; this feature freezes that contract with tests |

## Assumptions & Open Questions

Every ambiguity is resolved or recorded here — nothing is left silently unclear.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Which universe counts as "elegível" for the precondition | The scored/ranked universe (`build_fii_scores` output — the rows the GA can actually pick from); the eligibility count is also reported in diagnostics | This is the exact base the existing check at `py/fii_selection.py:393` uses; scoring drops rows with unusable fundamentals (13 eligible → 12 ranked at `size_min=3e9`), so a precondition on eligibility alone would let runs reach the GA and die late | y (evidence: counts recomputed with production functions) |
| Normative matrix definition (issue lists 7/13/28/50, triage found anchors 61/162) | Both series are normative: the size sweep at constant `liq_min=2e6` (7/13/28/50 eligible; 7/12/26/47 ranked) **plus** the three profile anchors (61/57 moderado, 162/137 arrojado) **plus** the premium interpolation point `3.4e9/1.4e6` (10/9) | The issue's numbers are a size sweep, not the profile anchors; freezing one table with explicit provenance removes the ambiguity permanently | y |
| Policy choice between issue options (a) and (b) | Option (a): `conservador` FII `size_min` calibrated `5e9 → 3e9`; `liq_min=2e6` and `n_assets=10` unchanged | Quantified in ADR 0002: option (b) (`n_assets=7`) is hard-infeasible under `limitar_concentracao` (sector HHI fixed 0.3061 > 0.25), option (a) makes the anchor (12 ranked ≥ 10, min sector HHI 0.24 ≤ 0.25) and all 21 sampled band points feasible | y — approved 2026-10-27 (ADR 0002 `Accepted`) |
| Policy change registration | Bump `POLICY_VERSION` → `premium-policy-v2` and `BASIC_POLICY_VERSION` → `basic-policy-v2` alongside the calibration | "Nunca reduzir silenciosamente" applies to widening too: persisted provenance must distinguish runs created before/after the calibration (AC3 "decisão registrada") | y — approved 2026-10-27 (ADR 0002 `Accepted`) |
| Definition of "published batch" for AC4 | The response of `GET /v1/recommendations/latest-completed` (latest `completed` run for account+profile+plan); `GET /v1/recommendations` may show the latest run in `failed` state as a closed failure and is not the published batch | Matches current API semantics (`api/app/routers/recommendations.py:62-100`) and current web rendering (`recommendation-view.tsx:149`); makes AC4 testable without a new table | y |
| `suitability.html` drift | Out of scope for value changes (its `BASE_PROFILES` mirrors `STOCK_RULES`, the stock policy, which this feature does not touch); covered by a drift-guard test instead | The FII calibration must not be mirrored into the stock calculator; a test pins the page to `STOCK_RULES` so future drift is caught | y |
| Failure message language and shape | Portuguese, built from structured fields only, template frozen in `design.md`; includes ranked count, eligible count, required count, both thresholds, and profile label; no paths, no raw exception text | Consistent with existing `_safe_failure_message` strings; satisfies the existing sanitisation tests (`api/tests/test_premium_executor.py:129-131`, `api/tests/test_product_routes.py:343-352`) | y |
| Offline named-profile path | The precondition runs only in the explicit/premium path; named-profile offline runs keep `filters=None` behavior and rewrite nothing; on shortage the legacy check in `optimize_fii_portfolio` (`py/fii_selection.py:392-396`) stays byte-identical (same `FiiSelectionError`, same message) and is the only shortage guard on that path | Preserves committed artifacts byte-for-byte and keeps this feature's blast radius on the API path | y |

**Open questions:** none - all resolved or logged above (the ADR-0002 rows above
are the feature's human checkpoint and must be approved before IMPLEMENT).

---

## User Stories

### P1: Feasible FII universe with an honest failure ⭐ MVP

**User Story**: As an investor using the API, I want an FII run that cannot be
satisfied by the eligible universe to fail fast with a diagnostic that names the
count, the thresholds, and the profile, so that I understand the shortage instead
of seeing a generic engine error.

**Why P1**: This is the issue's core defect: today's failure loses every number
(`api/app/adapters/premium_optimization.py:539-546`,
`api/app/services/premium_executor.py:203-212`) and the profile-v1 spec already
requires the shortage to be reported instead of papered over
(`.specs/features/profile-v1-premium-optimization/spec.md:281-283`, `:293-295`).

**Acceptance Criteria**:

1. WHERE an FII selection run uses an explicit/premium selection config, WHEN the run has resolved its scored universe THEN the system SHALL proceed to selector execution only if the ranked universe count is greater than or equal to `n_assets`. <!-- compound WHERE+WHEN: optional-feature + event-driven -->
2. IF the ranked universe count is less than `n_assets` THEN the system SHALL fail before selector execution and SHALL NOT reduce `n_assets`, substitute instruments, or write result artifacts. <!-- unwanted-behavior -->
3. WHEN the universe-shortage precondition fails THEN the system SHALL record failure code `infeasible_universe` on the run record and SHALL NOT map it to `engine_error` or `infeasible_constraints`. <!-- event-driven -->
4. WHEN the system records `infeasible_universe` THEN the failure message SHALL contain the ranked universe count, the eligible count, the required `n_assets`, the numeric `size_min` and `liq_min` thresholds, and the profile label. <!-- event-driven -->
5. WHILE a failure message is persisted on a run record the system SHALL keep it free of file paths and raw exception text and within 512 characters, and SHALL keep the failure code within 64 characters. <!-- state-driven -->
6. WHEN a run fails with `infeasible_universe` THEN no partial recommendation SHALL be visible through the recommendation read path. <!-- event-driven -->

**Independent Test**: Feed the API path a policy whose `n_assets` exceeds the
ranked universe for its thresholds; observe `RecommendationRun.status="failed"`,
`failure_code="infeasible_universe"`, and a `failure_message` equal to the
frozen template rendering in `design.md` for the injected counts (exact-string
equality — the verbatim assertion frozen in T5 supersedes substring checks like
`contains "7"`, which would also match `"17"`/`"70"`), e.g.
`Universo FII insuficiente: 7 ativos utilizáveis (7 elegíveis) para 10 exigidos; limiares size_min=5000000000, liq_min=2000000; perfil conservador.`
for the 5e9/2e6 shortage; no `/`-path fragments; read paths return no
class/stock/FII payload.

### P1: Recorded policy calibration with sensitivity evidence

**User Story**: As the policy owner, I want the conservador FII calibration
chosen between the two proposed options with quantified sensitivity analysis, so
that the change is auditable and never happens silently.

**Why P1**: The issue's acceptance criterion 3 requires a registered decision
with justification and sensitivity; the numbers must survive in tests, not only
in prose.

**Acceptance Criteria**:

1. WHERE the FII policy rules resolve the conservador anchor, the rules SHALL use `size_min=3_000_000_000`, `liq_min=2_000_000`, and `n_assets=10`. <!-- optional-feature -->
2. WHEN the calibration lands THEN the system SHALL identify it with bumped policy versions (`premium-policy-v2`, `basic-policy-v2`) so persisted provenance distinguishes pre- and post-calibration runs. <!-- event-driven -->
3. WHEN the normative eligibility matrix is evaluated against the committed snapshot `data/raw/status_invest_fii.csv` with the production functions THEN the system SHALL yield eligible counts 7, 13, 28, 50 for `size_min` 5e9, 3e9, 2e9, 1e9 at `liq_min=2e6`, and eligible counts 10 (3.4e9/1.4e6), 61 (1e9/5e5), 162 (2e8/5e4) at the stated thresholds. <!-- event-driven -->
4. WHEN the premium interpolation band is sampled at segment-fraction steps of 0.05 (21 points per segment, i.e. score steps of 0.025 on the 0.00–0.50 axis) from the conservador anchor to the moderado anchor THEN every sampled point SHALL satisfy ranked universe count greater than or equal to the interpolated `n_assets` AND SHALL have a minimum achievable sector HHI at that `n_assets` of at most 0.25. <!-- event-driven -->
5. WHEN `n_assets` or an FII threshold value is changed THEN the change SHALL require a policy-version bump and an update to the ADR-pinned policy test in the same change; no other code path SHALL alter those values. <!-- process-rule (change control): enforced by the ADR-pinned policy test (T1), not runtime behavior -->

**Independent Test**: Run the pinned matrix test, the band-grid test (21 points
per segment, both checks per point), and the anchor-wiring test (restricted vs
unrestricted `limitar_concentracao`); delete one line of `FII_RULES`
calibration and observe the pin tests fail.

### P1: Basic publication stays closed on failure

**User Story**: As a Basic user, I want a failed run to end as a visible, closed
failure while my previous completed recommendation stays published, so that a
broken run never replaces or corrupts my current batch.

**Why P1**: Issue acceptance criterion 4; the semantics exist today but are
untested and undefined — this story freezes them.

**Acceptance Criteria**:

1. WHEN a Basic run fails THEN the run record SHALL be terminal with `status="failed"`, cleared result payload, and populated `failure_code` and `failure_message`, and the read path SHALL expose no class, stock, or FII payload for that run. <!-- event-driven -->
2. WHEN a Basic run fails THEN `GET /v1/recommendations/latest-completed` SHALL keep returning the previous completed run with its result payload unchanged. <!-- event-driven -->
3. WHEN a new Basic run is created THEN previous completed runs SHALL remain readable and unchanged. <!-- event-driven -->

**Independent Test**: Complete a Basic run, trigger an `infeasible_universe`
failure on a second run, and compare the `latest-completed` response before and
after (identical payload) while the latest response shows the closed failure.

### P2: Drift guards for duplicated policy values

**User Story**: As a maintainer, I want the duplicated policy constants in
`suitability.html` pinned to their source of truth, so that this FII calibration
cannot silently leak into the stock calculator or vice versa.

**Why P2**: Preventive; no user-visible behavior.

**Acceptance Criteria**:

1. WHERE `suitability.html` embeds `BASE_PROFILES` filter anchors, a guard test SHALL assert they equal `STOCK_RULES` filters, which this feature SHALL NOT change. <!-- optional-feature -->
2. WHEN eligibility filters are applied THEN the presence of `filters["enforce_liquidity_and_size"]` SHALL NOT change the eligible set. <!-- event-driven -->

The second clause this AC carried in round 1 ("the feature SHALL NOT alter how
that key is computed or consumed") is a review constraint, not observable
behavior — it lives in Out of Scope and the T7 review checklist instead.

**Independent Test**: Guard test fails if `suitability.html` and `STOCK_RULES`
diverge; eligibility test fails if the flag starts filtering.

---

## Edge Cases

- IF the eligibility filter passes rows that `build_fii_scores` later drops (missing fundamentals) THEN the precondition SHALL judge the ranked universe and the diagnostic SHALL report both counts (measured: 13 eligible → 12 ranked at `size_min=3e9`).
- WHEN `fii_weight == 0` in an allocation result THEN the FII engine SHALL not be invoked and the precondition SHALL not apply (existing behavior at `api/app/adapters/premium_optimization.py:525-526`).
- WHERE a named-profile offline run passes `filters=None` (`py/fii_selection.py:558-567`) THEN the new precondition SHALL NOT run: the legacy check in `optimize_fii_portfolio` (`py/fii_selection.py:392-396`) stays byte-identical (same `FiiSelectionError`, same message) and remains the only shortage guard on that path, and committed artifacts SHALL remain byte-identical.
- IF the run record already has a failure code longer than 64 characters or a message longer than 512 THEN the repository truncation at `api/app/repositories/recommendations.py:205-206` SHALL remain the only bound and the new message SHALL fit without truncation.
- WHEN the HHI hard check at `py/fii_selection.py:639-640` fails after selection THEN it SHALL keep its current `FiiSelectionError` behavior and SHALL NOT be reclassified as `infeasible_universe` (it is a different failure: selection quality, not universe size).

---

## Requirement Traceability

| Requirement ID | Story | Task(s) | Phase | Status |
| --- | --- | --- | --- | --- |
| UNIV-01 | P1: Feasible FII universe | T3 | Tasks | In Tasks |
| UNIV-02 | P1: Feasible FII universe | T3 | Tasks | In Tasks |
| ERR-01 | P1: Feasible FII universe | T4, T5 | Tasks | In Tasks |
| ERR-02 | P1: Feasible FII universe | T5 | Tasks | In Tasks |
| ERR-03 | P1: Feasible FII universe | T4, T6 | Tasks | In Tasks |
| POL-01 | P1: Recorded policy calibration | T1 | Tasks | In Tasks |
| POL-02 | P1: Recorded policy calibration | T2 | Tasks | In Tasks |
| POL-03 | P1: Recorded policy calibration | T1, T8, T9 | Tasks | In Tasks |
| PUB-01 | P1: Basic publication closed failure | T6 | Tasks | In Tasks |
| PUB-02 | P1: Basic publication closed failure | T6 | Tasks | In Tasks |
| DFT-01 | P2: Drift guards | T7 | Tasks | In Tasks |

**ID format:** `[CATEGORY]-[NUMBER]` (e.g., `UNIV-01`)

**Coverage:** 11 total, 11 mapped to tasks, 0 unmapped (per-row task ownership in the `Task(s)` column above; POL-03 is fully owned: band grid → T8, anchor executability → T9, change control → T1)

### Requirement definitions

- **UNIV-01** — AC P1.1: explicit-path precondition `ranked_count >= n_assets`.
- **UNIV-02** — AC P1.2: fail instead of shrinking; no silent `n_assets` reduction.
- **ERR-01** — AC P1.3 + P1.4: code `infeasible_universe` with counts, thresholds, profile.
- **ERR-02** — AC P1.5: sanitized message within column limits.
- **ERR-03** — AC P1.6 + edge case: no partial results; wrong-code mapping forbidden.
- **POL-01** — AC P2.1 + P2.2: calibrated values + version bump (ADR-0002).
- **POL-02** — AC P2.3: normative matrix pinned to the committed snapshot.
- **POL-03** — AC P2.4 + P2.5: band feasibility (21-point-per-segment band grid with ranked ≥ `n_assets` and min sector HHI ≤ 0.25 per point, pinned by T8; conservador-anchor executability under `limitar_concentracao`, pinned by T9) and change control for policy values (pinned by T1's policy test).
- **PUB-01** — AC P3.1: closed failure semantics.
- **PUB-02** — AC P3.2 + P3.3: previous completed batch kept and unchanged.
- **DFT-01** — AC P4.1 + P4.2: drift guard and `enforce_liquidity_and_size` non-change.

---

## Issue Traceability (caiomugarte/TCC#12)

Every issue acceptance criterion and test bullet mapped to a requirement ID and
to the task(s) that satisfy it.

### Acceptance criteria

| Issue AC | Requirement(s) | Task(s) | Testable as written? |
| --- | --- | --- | --- |
| "Nenhuma execução usa n_assets maior que o universo elegível." | UNIV-01, UNIV-02 | T2, T3 | Yes, after the universe is defined (Assumption 1): every matrix row either passes the precondition or raises `infeasible_universe`; no execution proceeds with `n_assets` above the ranked universe |
| "Falha explica contagem, limiar e perfil." | ERR-01, ERR-02 | T4, T5 | Partially — "explica" is vague. Testable restatement (frozen as AC P1.4/P1.5): the persisted message contains ranked count, eligible count, required count, `size_min`, `liq_min`, and profile label, and contains no paths/raw exception text |
| "Decisão de política registrada com justificativa e sensibilidade." | POL-01, POL-03 | T1, T2, T8, T9 | Partially — "registrada" é vago. Testable restatement (frozen as AC P2.1/P2.2/P2.4/P2.5): `docs/adr/0002-*.md` holds the quantified sensitivity tables; `FII_RULES` values and policy versions are pinned by `api/tests/test_premium_policy.py`; changing them requires version bump + pin update in the same change; the ADR's band claim (0/21 failing points per segment) is pinned by T8 and its "anchor becomes executable" claim by T9's restricted-vs-unrestricted wiring test |
| "Publicação Basic falha fechada e mantém o lote anterior." | PUB-01, PUB-02 | T6 | Not as written — "lote anterior" and "falha fechada" are undefined. Testable restatement (frozen as Assumption 5 and AC P3.1/P3.2): failed run is terminal with cleared result payload and failure fields; `GET /v1/recommendations/latest-completed` keeps returning the previous completed run unchanged |

### Test bullets

| Issue test bullet | Requirement(s) | Task(s) | Notes |
| --- | --- | --- | --- |
| "Matriz de perfis × limiares confirma contagens (7, 13, 28, 50) e comportamento esperado." | POL-02, POL-03, UNIV-01 | T2, T3, T8 | Normative matrix per Assumption 2: the four sweep counts (eligible column 7/13/28/50; ranked column 7/12/26/47) plus anchor rows 61/57 and 162/137 plus band point 10/9 (T2 pins counts); "comportamento esperado" = precondition passes or raises per row (T3); the interpolation band adds the profiles × thresholds grid — 21 points per segment with ranked ≥ `n_assets` and min sector HHI ≤ 0.25 per point (T8) |
| "Teste garante que n_assets nunca é reduzido sem registro explícito." | UNIV-02, POL-01 | T1, T3 | Split into two testable halves: (a) runtime — a shortage raises instead of shrinking (T3); (b) record — `n_assets` values are pinned to ADR-0002 and can only change with version bump + pin update (T1). "Sem registro explícito" alone is untestable; the pin test makes the record mechanical |
| "Teste de mensagem de erro inclui números do universo." | ERR-01, ERR-02 | T5 | Asserts the exact structured fields in the persisted message and the sanitisation constraints |

---

## Success Criteria

- [ ] `/tmp/opencode/tcc-venv/bin/python3 -m unittest` matrix test passes with the normative counts (7, 13, 28, 50 / 7, 12, 26, 47 and anchor rows) over the committed snapshot (same interpreter as the Gate Check Commands — system `python3` has no pandas).
- [ ] The band-grid test passes with 0/21 failing points per segment (ranked ≥ `n_assets` and min sector HHI ≤ 0.25 at every sampled point), and the anchor-wiring test proves the `limitar_concentracao` run completes at HHI ≤ 0.25 while the unrestricted run differs exactly by that wiring.
- [ ] Forcing an undersized universe through the API path yields `failure_code="infeasible_universe"` and a message containing all six mandated fields, with zero path/raw-text leakage.
- [ ] `docs/adr/0002-fii-universe-feasibility-policy.md` is approved (human checkpoint) with quantified sensitivity tables.
- [ ] A failed Basic run leaves `GET /v1/recommendations/latest-completed` byte-identical to its pre-failure response.
- [ ] Full quick+full gates pass with no test deletions.
