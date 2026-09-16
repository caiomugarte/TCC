# Profile v1 to Premium Optimization Tasks

**Design:** `.specs/features/profile-v1-premium-optimization/design.md`
**Status:** Draft

## Verification Commands

Run API tests from the repository root:

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q
```

Run research-engine tests from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Run web checks from `web/`:

```bash
npm run typecheck
npm run build
node --test tests/vertical-slice-smoke.test.mjs
```

## Execution Plan

### Phase 1: Foundation

Sequential data and policy contracts:

```text
T1 -> T2 -> T3 -> T4 -> T5
T6
T7
```

`T6` and `T7` can run in parallel with `T1` through `T5`.

### Phase 2: Engine Integration

After `T5` and `T6`, the independent engine seams can be developed in parallel.
Workspace isolation follows the selector seams:

```text
T5,T6 -> T8 -> T9 --┐
T5    -> T10 -------┼-> T12 --┐
T5    -> T11 -------┘         ├-> T13
                              │
T9 ---------------------------┘
```

### Phase 3: API Run Lifecycle

```text
T1,T2 -> T14
T7,T13,T14 -> T15 -> T16 -> T17 -> T18,T19
```

### Phase 4: Client Integration

```text
T17 -> T20 -> T21 -> T22
```

### Phase 5: End-to-End Verification

```text
T18,T19,T22 -> T23
```

## Task Breakdown

### T1: Extend profile and recommendation persistence models

**What:** Add the persisted profile provenance fields and Premium run lifecycle,
policy, result, failure, and provenance fields to the existing SQLAlchemy
models.

**Where:** `api/app/db/models.py`

**Depends on:** None

**Reuses:** Existing `ProfileRecord`, `RecommendationRun`, account foreign
keys, and existing Basic summary columns.

**Requirement:** `POL-02`, `PROV-01`, `PROV-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] `ProfileRecord` can preserve raw score, applied rules, warnings,
      normalized restrictions, and profile schema version.
- [ ] `RecommendationRun` supports `queued`, `running`, `completed`, and
      `failed` states plus timestamps and failure diagnostics.
- [ ] `RecommendationRun` can store policy JSON, result JSON, provenance JSON,
      policy version, and output hash.
- [ ] Existing Basic fields and account relationships remain available.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_db_schema -q
```

Expected: existing schema tests pass and the new fields are importable.

**Commit:** `feat(premium): add run persistence fields`

---

### T2: Add the database migration for Premium run fields

**What:** Create one Alembic revision that adds the model fields from T1 and
backfills compatible existing recommendation rows as completed Basic runs.

**Where:** `api/migrations/versions/`

**Depends on:** T1

**Reuses:** The initial migration and existing Alembic metadata configuration.

**Requirement:** `PROV-01`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] The revision upgrades a fresh database and an existing database.
- [ ] Existing recommendation rows remain readable as completed Basic results.
- [ ] The downgrade removes only fields introduced by this feature.
- [ ] Alembic reports no model drift after the revision.

**Verify:** Run from `api/`:

```bash
alembic upgrade head
alembic check
```

Expected: upgrade succeeds and `alembic check` reports no pending operations.

**Commit:** `feat(premium): migrate recommendation run state`

---

### T3: Persist the complete computed Profile v1 snapshot

**What:** Store raw score, applied rules, warnings, normalized restrictions,
and schema version when the profile route creates a revision.

**Where:** `api/app/services/profile.py`, `api/app/routers/profile.py`

**Depends on:** T1

**Reuses:** Existing `compute_profile()` result and profile versioning flow.

**Requirement:** `POL-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] A new profile revision stores all computed values without recomputing
      them during a later Premium request.
- [ ] Stored restrictions are canonical and preserve the six supported values.
- [ ] Existing profile response behavior remains compatible.
- [ ] Profile tests cover raw score, rules, warnings, and restrictions.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_profile_service -q
```

Expected: profile computation and persistence tests pass.

**Commit:** `feat(profile): persist computed provenance`

---

### T4: Define the versioned Premium policy contract and rules

**What:** Add frozen internal policy value objects and a plain versioned rule
table that describes all profile-dependent allocation, stock, and FII values.

**Where:** `api/app/services/premium_policy.py`

**Depends on:** T1, T3

**Reuses:** Existing allocation profile validation, allocation anchors, stock
factor keys, and FII factor keys.

**Requirement:** `POL-01`, `POL-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] The policy contains separate allocation, stock, FII, profile, and
      provenance sections.
- [ ] The rule table emits `selection_preset`, `n_assets`, factor weights,
      filters, HHI penalty, and system GA configuration for both selectors.
- [ ] The rule table has an explicit version identifier.
- [ ] System GA knobs are not derived from profile answers.
- [ ] Policy objects serialize with stable key ordering.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_policy -q
```

Expected: contract validation and serialization tests pass.

**Commit:** `feat(premium): define resolved policy contract`

---

### T5: Implement the deterministic Profile v1 policy resolver

**What:** Resolve one owned persisted profile into complete explicit engine
inputs, including the locked restriction policy.

**Where:** `api/app/services/premium_policy.py`,
`api/tests/test_premium_policy.py`

**Depends on:** T4

**Reuses:** `interpolate_profile()` and existing profile dimensions and rules.

**Requirement:** `POL-01`, `POL-02`, `REST-01`, `REST-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Normalized score drives allocation interpolation; raw score is retained.
- [ ] Score, dimensions, and restrictions generate every profile-dependent
      stock/FII field; generic profile is only the baseline label.
- [ ] `evitar_cripto`, `evitar_exterior`, `priorizar_renda` at 40%,
      `limitar_concentracao` at HHI 0.25, and `evitar_illiquidez` produce the
      locked constraints.
- [ ] Multiple restrictions intersect and `nenhuma` cannot coexist with
      another restriction.
- [ ] Invalid or infeasible policy inputs fail before engine execution.
- [ ] Deterministic fixtures cover all generic profiles and restrictions.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_policy -q
```

Expected: all policy fixture tests pass with byte-stable serialized output.

**Commit:** `feat(premium): resolve profile optimizer policy`

---

### T6: Add immutable snapshot manifests and compatibility validation [P]

**What:** Define the manifest schema and registry lookup that selects the
latest compatible allocation, stock, and FII source set and verifies hashes.

**Where:** `py/snapshot_manifest.py`, `tests/test_snapshot_manifest.py`

**Depends on:** None

**Reuses:** Existing snapshot metadata and common-date validation.

**Requirement:** `EXEC-02`, `PROV-01`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] A manifest records source paths, SHA-256 hashes, providers, cutoff,
      common dates, supported classes, and manifest version.
- [ ] Registry lookup selects the latest compatible manifest by market cutoff,
      not by the computer clock.
- [ ] Hash changes, missing sources, unsupported classes, and incompatible dates
      are rejected.
- [ ] No live fetch, IFIX fallback, or fixed-artifact substitution occurs.

**Verify:**

```bash
python3 -m unittest tests.test_snapshot_manifest -v
```

Expected: selection, hash mismatch, and incomplete-manifest tests pass.

**Commit:** `feat(premium): add immutable snapshot manifests`

---

### T7: Add the private-pilot Premium authorization dependency [P]

**What:** Compose the existing Premium entitlement check with a fail-closed
approved-account allowlist.

**Where:** `api/app/entitlements/dependencies.py` and application settings

**Depends on:** None

**Reuses:** `require_premium()` and account identity resolution.

**Requirement:** `AUTH-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Active and grace-period Premium accounts still pass entitlement checks.
- [ ] Non-Premium accounts are rejected before any run record or engine work.
- [ ] Premium accounts outside the configured pilot allowlist are rejected.
- [ ] An empty production allowlist fails closed.
- [ ] Tests can inject pilot account IDs without environment leakage.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_entitlements -q
```

Expected: entitlement and pilot denial/grant cases pass.

**Commit:** `feat(premium): gate recommendations to pilot`

---

### T8: Add strict manifest-backed allocation input loading [P]

**What:** Load Premium allocation inputs from the validated manifest without
the current IFIX or fixed-artifact fallback.

**Where:** `py/allocation_data.py`, `py/snapshot_manifest.py`,
`tests/test_allocation_data.py`

**Depends on:** T6

**Reuses:** Existing `SnapshotBundle`, common-date intersection, PTAX, and
missing-data checks.

**Requirement:** `EXEC-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Premium loading requires all five class inputs named by the manifest.
- [ ] Common dates and BRL conversion use the existing documented behavior.
- [ ] Missing FII portfolio data fails instead of selecting IFIX.
- [ ] Basic loading behavior remains unchanged.
- [ ] Source metadata and manifest ID reach the allocation result.

**Verify:**

```bash
python3 -m unittest tests.test_allocation_data -v
```

Expected: strict Premium fixtures fail on missing data and pass on valid data.

**Commit:** `feat(allocation): add strict premium snapshot loading`

---

### T9: Apply explicit Premium class constraints to allocation [P]

**What:** Allow Premium allocation to receive class constraints and a zero
minimum class weight while preserving Basic defaults.

**Where:** `py/pipelines/asset_allocation.py`, `py/allocation_config.py`,
`tests/test_allocation.py`

**Depends on:** T5, T8

**Reuses:** Existing `AllocationProfile`, candidate validation, HHI metrics,
and Basic five-percent floor.

**Requirement:** `REST-01`, `REST-02`, `EXEC-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Premium can set excluded class maximum weight to zero.
- [ ] Premium can enforce fixed-income minimum 0.40 and class HHI <= 0.25.
- [ ] Constraint failures return an infeasibility diagnostic.
- [ ] Basic callers retain the current five-percent minimum and outputs.
- [ ] Allocation parameters remain separate from selector configurations.

**Verify:**

```bash
python3 -m unittest tests.test_allocation -v
```

Expected: Premium constraint fixtures pass and Basic regression fixtures remain
green.

**Commit:** `feat(allocation): enforce premium class constraints`

---

### T10: Add explicit configuration to the stock selector [P]

**What:** Make stock preprocessing, scoring, and GA execution accept the
resolved stock configuration without looking up a static named profile.

**Where:** `py/core/preprocessing.py`, `py/core/scoring.py`,
`py/core/optimizer.py`, `py/pipelines/single_run.py`,
`py/pipelines/multi_run.py`

**Depends on:** T5

**Reuses:** Existing filter, score, optimizer, multi-run, and consensus logic.

**Requirement:** `POL-01`, `EXEC-02`, `REST-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Premium passes explicit filters, factor weights, HHI penalty, asset count,
      and system GA settings through the full stock path.
- [ ] Existing named-profile offline callers continue to work through wrappers.
- [ ] The result exposes selected tickers, sleeve weights, metrics, and
      exclusion reasons.
- [ ] Premium execution does not write to shared profile-named output paths.

**Verify:**

```bash
python3 -m unittest discover -s tests -v
```

Expected: stock fixture and existing research tests pass.

**Commit:** `feat(stock): accept explicit selection policy`

---

### T11: Add explicit configuration to the FII selector [P]

**What:** Make FII scoring, eligibility, and GA execution accept the resolved
FII configuration and return a structured selection result.

**Where:** `py/fii_selection.py`, `tests/test_fii_selection.py`

**Depends on:** T5

**Reuses:** Existing lower-level custom weights/GA seams and FII consensus
logic.

**Requirement:** `POL-01`, `EXEC-02`, `REST-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Premium passes explicit factor weights, asset count, HHI penalty,
      eligibility thresholds, and system GA settings.
- [ ] The top-level runner no longer requires a static named profile for
      Premium execution.
- [ ] The result exposes selected tickers, sleeve weights, metrics, and
      exclusion reasons.
- [ ] IFIX is never used as a Premium selector fallback.

**Verify:**

```bash
python3 -m unittest discover -s tests -v
```

Expected: FII custom-policy and existing selector tests pass.

**Commit:** `feat(fii): accept explicit selection policy`

---

### T12: Make selector execution deterministic and workspace-scoped [P]

**What:** Seed every random source, derive stable per-run seeds, disable
interactive checkpoint behavior, and isolate generated files by run ID.

**Where:** `py/core/optimizer.py`, `py/pipelines/multi_run.py`,
`py/fii_selection.py`, and the Premium runner workspace helper

**Depends on:** T6, T10, T11

**Reuses:** Existing random-seed parameters, checkpoint logic, and output
writers.

**Requirement:** `EXEC-02`, `PROV-01`, `PROV-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] NumPy and Python random sources use the persisted seed.
- [ ] Replaying the same policy, manifest, and seed produces equivalent
      selection output within the declared tolerance.
- [ ] Selector jobs do not prompt for checkpoints.
- [ ] Two run workspaces cannot overwrite each other's files.
- [ ] The seed and workspace metadata are returned to the orchestrator.

**Verify:**

```bash
python3 -m unittest discover -s tests -v
```

Expected: deterministic replay and workspace-isolation fixtures pass.

**Commit:** `fix(optimization): isolate premium selector runs`

---

### T13: Compose engine outputs into a Premium result [P]

**What:** Run the explicit allocation, stock, and FII adapters and convert
selected sleeve weights into total-portfolio weights and BRL amounts.

**Where:** `api/app/adapters/premium_optimization.py`,
`api/tests/test_premium_optimization.py`

**Depends on:** T9, T10, T11, T12

**Reuses:** Existing allocation adapter injection and class amount mapping.

**Requirement:** `EXEC-01`, `EXEC-02`, `REST-01`, `REST-02`, `PROV-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Fake engines receive only their own resolved policy section.
- [ ] Class weights are non-negative and sum to 1.
- [ ] Non-empty stock/FII sleeve weights sum to 1.
- [ ] Total security weight equals class weight multiplied by sleeve weight.
- [ ] BRL amounts use investable capital and are included for classes and
      constituents.
- [ ] Missing or infeasible engine output raises a terminal run error without
      returning partial result data.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_optimization -q
```

Expected: injected-engine composition and failure tests pass.

**Commit:** `feat(premium): compose personalized optimization result`

---

### T14: Implement the run repository and state transitions [P]

**What:** Add repository methods for owned run creation, lifecycle transitions,
terminal immutability, and status/result serialization.

**Where:** `api/app/repositories/recommendations.py`,
`api/app/schemas/recommendation.py`,
`api/tests/test_recommendation_repository.py`

**Depends on:** T1, T2

**Reuses:** Existing recommendation account-scoped queries and response models.

**Requirement:** `AUTH-02`, `PROV-01`, `PROV-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Repository creates queued runs with policy and provenance before work.
- [ ] Only valid state transitions are accepted.
- [ ] Completed runs cannot be overwritten by later runs or failure updates.
- [ ] Failed/non-terminal responses omit result JSON.
- [ ] Owned reads filter by both authenticated account and run ID.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_recommendation_repository -q
```

Expected: state-machine, immutability, and ownership tests pass.

**Commit:** `feat(api): add recommendation run repository`

---

### T15: Implement the Premium run orchestration service [P]

**What:** Validate profile and manifest, resolve the policy, create the queued
run, derive the deterministic seed, and submit execution only after commit.

**Where:** `api/app/services/premium_recommendation.py`,
`api/tests/test_premium_recommendation_service.py`

**Depends on:** T5, T6, T7, T13, T14

**Reuses:** Existing profile lookup and recommendation adapter dependency
injection.

**Requirement:** `EXEC-01`, `AUTH-01`, `AUTH-02`, `PROV-01`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] The service loads only the authenticated account's latest or requested
      owned profile revision.
- [ ] It rejects missing profile or manifest before enqueue.
- [ ] It persists policy, manifest reference, and seed before submission.
- [ ] It submits no engine work when authorization or validation fails.
- [ ] Submission failure marks the queued run failed.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_recommendation_service -q
```

Expected: service validation, transaction ordering, and submission tests pass.

**Commit:** `feat(api): orchestrate premium recommendation runs`

---

### T16: Add the bounded asynchronous executor and recovery [P]

**What:** Execute queued runs in a single bounded application worker, update
statuses through a fresh database session, and mark stale work failed after a
process restart.

**Where:** `api/app/services/premium_executor.py`,
`api/app/main.py`, `api/tests/test_premium_executor.py`

**Depends on:** T13, T14, T15

**Reuses:** Standard-library executor and SQLAlchemy session factory.

**Requirement:** `EXEC-01`, `PROV-01`, `PROV-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Queued runs transition to running and then completed or failed.
- [ ] The worker never shares a request-bound SQLAlchemy session.
- [ ] Complete result JSON and `completed` status commit atomically.
- [ ] Engine errors store safe diagnostics and no result JSON.
- [ ] Stale queued/running runs are marked failed during recovery.
- [ ] The private-pilot deployment is restricted to one API worker.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_executor -q
```

Expected: lifecycle, restart recovery, and failure atomicity tests pass.

**Commit:** `feat(api): run premium jobs asynchronously`

---

### T17: Add Premium API routes and status responses

**What:** Add the Premium start route and extend account-scoped recommendation
reads to expose queued, running, failed, and completed Premium results.

**Where:** `api/app/routers/premium.py`, `api/app/routers/recommendations.py`,
`api/app/schemas/recommendation.py`,
`api/tests/test_premium_routes.py`

**Depends on:** T7, T14, T15, T16

**Reuses:** Existing auth dependencies, route error codes, and Basic response
fields.

**Requirement:** `EXEC-01`, `AUTH-01`, `AUTH-02`, `PROV-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] `POST /v1/premium/recommendations` returns `202` and a queued run ID.
- [ ] Authorization occurs before run creation and engine submission.
- [ ] `GET /v1/recommendations/{id}` returns lifecycle status and hides partial
      results.
- [ ] Completed Premium results include classes, stock/FII outputs, amounts,
      policy, and provenance.
- [ ] Basic routes preserve their existing completed response contract.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_routes -q
```

Expected: HTTP contract, status, entitlement, and result serialization tests
pass.

**Commit:** `feat(api): expose premium recommendation routes`

---

### T18: Test Premium authorization, isolation, and no-partial guarantees [P]

**What:** Add route-level tests covering two accounts, entitlement/pilot
denials, profile ownership, run ownership, Basic regression, and failed-run
visibility.

**Where:** `api/tests/test_premium_routes.py`,
`api/tests/test_product_routes.py`

**Depends on:** T17

**Reuses:** Existing in-memory SQLite fixtures and fake account dependencies.

**Requirement:** `AUTH-01`, `AUTH-02`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] Basic accounts receive denial before any optimization work.
- [ ] Non-pilot Premium accounts receive denial.
- [ ] Account A cannot read account B's run or profile.
- [ ] Failed runs expose diagnostics but no partial recommendation.
- [ ] Existing Basic tests continue to pass.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q
```

Expected: the complete API suite passes.

**Commit:** `test(premium): cover access and isolation`

---

### T19: Add full deterministic Premium integration fixtures [P]

**What:** Exercise the resolver, manifest, three engine adapters, persistence,
and replay using synthetic data and fake or dependency-light engines.

**Where:** `api/tests/test_premium_integration.py`,
`tests/test_premium_integration.py`

**Depends on:** T6, T9, T10, T11, T13, T16, T17

**Reuses:** Existing allocation and FII fixture patterns.

**Requirement:** `POL-01`, `POL-02`, `REST-01`, `REST-02`, `EXEC-01`,
`EXEC-02`, `PROV-01`, `PROV-02`

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`

**Done when:**

- [ ] A valid synthetic Premium profile creates a queued run and reaches
      completed status.
- [ ] All three engines receive explicit profile-derived parameters.
- [ ] Output amounts and sleeve-to-total weights are valid.
- [ ] Stored provenance identifies policy, seed, manifest, hashes, cutoff, and
      model versions.
- [ ] Replaying the stored inputs reproduces the result within tolerance.

**Verify:**

```bash
PYTHONPATH=api:py api/.venv/bin/python -m unittest api.tests.test_premium_integration -q
python3 -m unittest tests.test_premium_integration -v
```

Expected: end-to-end deterministic fixtures pass.

**Commit:** `test(premium): verify deterministic end to end flow`

---

### T20: Add Premium recommendation client contracts [P]

**What:** Add typed start/status/result contracts and API client methods for
starting a Premium run and reading its status.

**Where:** `web/lib/api-types.ts`, `web/lib/api-client.ts`

**Depends on:** T17

**Reuses:** Existing authenticated fetch helper and recommendation types.

**Requirement:** `EXEC-01`, `PROV-02`

**Tools:**

- MCP: NONE
- Skill: `react-best-practices`

**Done when:**

- [ ] Client can start a Premium run and parse `202` queued response.
- [ ] Client can read queued, running, failed, and completed states.
- [ ] Premium output types include stock/FII constituents and provenance.
- [ ] Basic response typing remains compatible.

**Verify:** Run from `web/`:

```bash
npm run typecheck
```

Expected: no TypeScript errors.

**Commit:** `feat(web): add premium recommendation client`

---

### T21: Add bounded Premium polling to the recommendation view [P]

**What:** Branch the existing recommendation loader by account plan, start a
Premium run for eligible users, poll its ID, and render terminal states using
the existing recommendation shell.

**Where:** `web/components/recommendation/recommendation-view.tsx`

**Depends on:** T20

**Reuses:** Existing one-shot loader, cancellation flag, retry action, and
authenticated account data.

**Requirement:** `EXEC-01`, `AUTH-01`

**Tools:**

- MCP: NONE
- Skill: `react-best-practices`

**Done when:**

- [ ] Premium pilot users start and poll a run without invoking the Basic POST.
- [ ] Polling stops on completed or failed status and on component unmount.
- [ ] The UI shows queued/running, failure, and completed states.
- [ ] Basic users retain the existing one-shot recommendation behavior.
- [ ] The client never treats UI plan state as an authorization boundary.

**Verify:** Run from `web/`:

```bash
npm run typecheck
npm run build
```

Expected: typecheck and production build pass.

**Commit:** `feat(web): poll premium recommendation runs`

---

### T22: Add web regression coverage for Premium polling [P]

**What:** Test Premium start, polling, terminal failure, unmount cancellation,
and Basic non-regression with mocked API responses.

**Where:** `web/tests/`

**Depends on:** T20, T21

**Reuses:** Existing Node smoke-test conventions and API fixture patterns where
available.

**Requirement:** `EXEC-01`, `AUTH-01`, `FAIL-01`

**Tools:**

- MCP: NONE
- Skill: `react-best-practices`

**Done when:**

- [ ] Premium polling tests cover queued -> running -> completed.
- [ ] Failed runs render no partial result.
- [ ] Unmount cancels future polling updates.
- [ ] Basic loader tests prove no Premium route call.

**Verify:** Run from `web/`:

```bash
node --test tests/vertical-slice-smoke.test.mjs
```

Expected: web smoke and regression tests pass.

**Commit:** `test(web): cover premium polling states`

---

### T23: Run the release verification matrix

**What:** Run migration, API, research, and web checks together and confirm
that every requirement has an automated verification path.

**Where:** Repository root; no production code changes unless a verification
failure requires a follow-up task.

**Depends on:** T18, T19, T22

**Reuses:** Commands documented at the top of this file.

**Requirement:** All requirements

**Tools:**

- MCP: NONE
- Skill: `coding-guidelines`, `react-best-practices`

**Done when:**

- [ ] `alembic upgrade head` and `alembic check` pass.
- [ ] The complete API suite passes.
- [ ] The complete research-engine suite passes.
- [ ] Web typecheck, build, and smoke tests pass.
- [ ] No requirement remains without a test or explicit manual verification.

**Verify:** Execute all commands in the Verification Commands section and the
Alembic commands from `api/`.

Expected: all commands exit successfully.

**Commit:** `test(premium): verify release matrix`

## Parallel Execution Map

```text
Phase 1:
  T1 -> T2 -> T3 -> T4 -> T5
  T6 [P]
  T7 [P]

Phase 2 after T5/T6:
  T8 [P] -> T9 [P]
  T10 [P]
  T11 [P]
  T9,T10,T11 -> T12
  T9,T10,T11,T12 -> T13

Phase 3:
  T1,T2 -> T14
  T7,T13,T14 -> T15 -> T16 -> T17
  T17 -> T18 [P]
  T17 -> T19 [P]

Phase 4:
  T17 -> T20 -> T21 -> T22

Phase 5:
  T18,T19,T22 -> T23
```

## Requirement Traceability

| Requirement | Tasks | Coverage |
| --- | --- | --- |
| `POL-01` | T4, T5, T10, T11, T19 | Resolver and explicit selector inputs. |
| `POL-02` | T1, T3, T4, T5, T19 | Persisted profile and deterministic policy. |
| `REST-01` | T5, T9, T13, T19 | Class exclusion, fixed-income floor, and HHI cap. |
| `REST-02` | T5, T9, T10, T11, T13, T19 | Selector filters and intersection behavior. |
| `EXEC-01` | T13, T15, T16, T17, T20, T21, T22 | Async start, execution, result, and client polling. |
| `EXEC-02` | T6, T8, T9, T10, T11, T12, T13, T19 | Common snapshots and explicit engine configuration. |
| `AUTH-01` | T7, T15, T17, T18, T21, T22 | Entitlement and private-pilot gate. |
| `AUTH-02` | T14, T15, T17, T18 | Account-owned profile and run reads. |
| `PROV-01` | T1, T2, T6, T12, T14, T15, T16, T19 | Immutable policy, seed, snapshot, and model data. |
| `PROV-02` | T1, T6, T12, T13, T14, T16, T17, T19, T20 | Complete stored and returned result provenance. |
| `FAIL-01` | T1, T2, T5, T6, T8, T9, T12, T13, T14, T15, T16, T17, T18, T22 | Infeasible, stale, failed, and partial-result handling. |

**Coverage:** 11 requirements, 11 mapped to tasks, 0 unmapped.

## Task Granularity Check

| Task range | Scope | Status |
| --- | --- | --- |
| T1-T7 | One model, migration, persistence, policy, manifest, or auth concern per task. | Granular |
| T8-T13 | One engine or orchestration seam per task. | Granular |
| T14-T17 | One persistence, service, executor, or route concern per task. | Granular |
| T18-T19 | One API test layer per task. | Granular |
| T20-T22 | One client, UI, or web-test concern per task. | Granular |
| T23 | One release verification deliverable. | Granular |

## Execution Checkpoint

Before implementation, confirm the tool choice for each task. Default choices
in this plan are repository file tools, `coding-guidelines` for Python/API
changes, and `react-best-practices` for web changes. No external MCP is needed
unless implementation discovers an undocumented library API.
