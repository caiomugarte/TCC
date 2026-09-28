# Status Invest-backed Recommendations Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: activate it by name and follow its Execute flow and Critical Rules. Do not search for skill files by filesystem path. Do not execute until user approves tasks.

Existing working-tree changes already cover Premium max-quality/adaptive settings, active-run reuse, forced reruns, and long-run UI state. Preserve those changes and their tests; they are groundwork, not completion of Status Invest requirements. Leave `.specs/features/premium-max-quality-rerun/` and unrelated dirty files untouched.

---

**Design**: `.specs/features/status-invest-backed-recommendations/design.md`  
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `web/tests/README.md`; repository uses Python `unittest` and Node built-in `node:test`. `.specs/codebase/TESTING.md` is stale. No lint command is configured.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Snapshot manifest and Status Invest input service | unit | Every SIR-05–SIR-09 branch: provider/role selection, hash, invalid/empty files, stock/FII partial failure, per-run isolation, no canonical-file overwrite | `tests/test_snapshot_manifest.py`; `api/tests/test_status_invest_inputs.py` | `python3 -m unittest tests.test_snapshot_manifest -v`; `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q` |
| Repository, profile policy, optimization adapter, executor | unit | Every SIR-01–SIR-09, SIR-11, and SIR-15–SIR-17 outcome: plan/profile policy, success, persisted provenance, account scoping, no fallback, failure preservation, deterministic Premium profile mapping | `api/tests/test_*recommendation*.py`; `api/tests/test_premium_policy.py` | `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q` |
| API routes | integration | Basic/Premium happy paths, profile ownership, 202/poll, latest-completed scoping, GET without refresh, and external-source failure | `api/tests/test_*routes*.py` | `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q` |
| Web API client and recommendation UI | unit | SIR-03 and SIR-10–SIR-14: both plans, explicit run trigger, status polling, last-success retention, constituents/provenance rendering | `web/tests/premium-recommendation.test.mjs` | From `web/`: `node --test tests/premium-recommendation.test.mjs` |
| Web type/build/smoke | build | TypeScript passes; production bundle builds; built-route smoke passes | `web/` | From `web/`: `npm run typecheck`; `npm run build`; `npm run smoke` |

## Gate Check Commands

> Generated from codebase - confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After isolated Python source/policy tasks | `python3 -m unittest tests.test_snapshot_manifest -v`; `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q` |
| Full | After API integration tasks | `python3 -m unittest discover -s tests -v`; `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -q`; from `web/`: `node --test tests/premium-recommendation.test.mjs` |
| Build | After web integration / phase completion | From `web/`: `npm run typecheck`; `npm run build`; `npm run smoke` (smoke requires successful build; build requires Clerk configuration per `web/tests/README.md`) |

**Lint**: no configured lint command; no lint gate.

---

## Execution Plan

Phases run sequentially. Tasks within each phase run in listed order.

### Phase 1: Source foundation

```text
T1 → T2
```

### Phase 2: Run state and profile policy

```text
T2 → T3 → T4
```

### Phase 3: Combined execution

```text
T2 → T5 → T6
T4 ─────→ T5
T3 ─────────→ T6
```

### Phase 4: API and app integration

```text
T6 → T7 → T8
```

---

## Task Breakdown

### T1: Resolve selector sources by role

**What**: Make manifest selector lookup choose explicit stock/FII selection sources without confusing them with allocation class benchmarks.
**Where**: `py/snapshot_manifest.py`
**Depends on**: None
**Reuses**: Existing manifest parser, source aliases, and `tests/test_snapshot_manifest.py`.
**Requirement**: SIR-07, SIR-09

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Explicit selector keys resolve independently from class-level sources.
- [x] Missing or ambiguous selector input fails validation before optimization.
- [x] Add at least 3 tests covering role collision, explicit selector resolution, and missing selector input.
- [x] Gate passes with all existing and new manifest tests; no existing assertion is removed.

**Tests**: unit
**Gate**: quick
**Commit**: `fix(snapshot): resolve selector sources by role`

---

### T2: Capture run-scoped Status Invest inputs

**What**: Add a helper that refreshes stock and FII CSVs into unique run paths, validates each source, and returns immutable paths plus provenance.
**Where**: `api/app/services/status_invest_inputs.py`
**Depends on**: T1
**Reuses**: `py/fetch_status_invest.py`, `py/fetch_status_invest_fii.py`, existing schema checks, atomic `output_path` writes, and SHA-256 utilities.
**Requirement**: SIR-05, SIR-06, SIR-07, SIR-08, SIR-09

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Each collector receives a path under the run-specific workspace and existing per-request timeout.
- [x] Provider, UTC retrieval timestamp, and SHA-256 are returned for each validated file.
- [x] Empty/malformed data and either collector's failure abort the combined input result.
- [x] Canonical raw CSVs remain unchanged by API runs.
- [x] Add at least 5 tests covering success, each partial-source failure, malformed/empty data, and path isolation; collectors are mocked.
- [x] Gate passes with all existing and new API tests.

**Tests**: unit
**Gate**: quick
**Commit**: `feat(recommendations): capture statusinvest run inputs`

---

### T3: Generalize recommendation run lifecycle

**What**: Generalize the existing coordinator and repository for Basic/Premium runs, active-run reuse, and account/profile/plan-scoped latest-completed lookup.
**Where**: `api/app/services/premium_recommendation.py`
**Depends on**: T2
**Reuses**: `api/app/repositories/recommendations.py`, `RecommendationRun`, existing status transitions, and result/provenance JSON fields.
**Requirement**: SIR-01, SIR-10, SIR-11, SIR-12, SIR-15

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Queue creation and coordinator accept the existing `basic` and `premium` plans without changing account/profile ownership checks.
- [x] Latest-completed query filters by account, selected profile, plan, and `completed` status.
- [x] Exact resolved Premium profile policy is persisted on its recommendation run.
- [x] Queued/running requests for the same account/profile/plan reuse the active run; explicit new-run actions after terminal state start fresh, with Premium sending `force=true`.
- [x] Premium `force=false` retains existing cached-result behavior for API compatibility.
- [x] Add at least 5 tests covering plan persistence, active-run reuse, forced fresh reruns, latest-completed selection, and account/profile isolation.
- [x] Gate passes with all existing and new API tests; no migration is added unless existing fields prove insufficient.

**Tests**: unit
**Gate**: quick
**Commit**: `feat(recommendations): persist plan-aware runs`

---

### T4: Resolve profile-specific selector and GA policy

**What**: Convert Basic profile bands and Premium questionnaire fields into deterministic explicit stock/FII selector and supported GA policy while preserving shared max-quality bounds.
**Where**: `api/app/services/premium_policy.py`
**Depends on**: T3
**Reuses**: Existing Premium policy anchors and profile interpolation.
**Requirement**: SIR-02, SIR-15, SIR-16, SIR-17

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Each supported Basic profile maps deterministically to stock/FII selection settings.
- [x] Premium suitability score, dimensions, and restrictions map to their configured selector/GA settings.
- [x] The policy does not substitute fixed legacy named profile `caio` for the authenticated app profile.
- [x] Existing Premium mappings remain unchanged.
- [x] Existing Premium max-quality/adaptive values and their tests remain unchanged.
- [x] Add at least 5 tests covering Basic bands, mapped Premium profile differences, same-input determinism, and max-quality/adaptive policy preservation.
- [x] Gate passes with all existing and new policy tests.

**Tests**: unit
**Gate**: quick
**Commit**: `feat(recommendations): map basic profiles to selectors`

---

### T5: Combine selectors and class allocation

**What**: Generalize the optimization adapter to consume explicit run-scoped Status Invest sources plus unchanged plan-specific class-allocation inputs and return one combined result.
**Where**: `api/app/adapters/premium_optimization.py`
**Depends on**: T2, T4
**Reuses**: `multi_run`, existing Premium selector configuration, and class-allocation adapters.
**Requirement**: SIR-02, SIR-03, SIR-04

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Stock/FII selector receives only the validated run-specific Status Invest paths.
- [x] Basic and Premium policies produce the same response contract for classes, stocks, FIIs, and BRL targets.
- [x] Class history continues to use existing reference inputs; current SI selections do not leak into historical weights.
- [x] Result provenance distinguishes selector inputs from allocation-history inputs.
- [x] Existing Premium max-quality/adaptive controls are passed through unchanged.
- [x] Add at least 5 tests covering both plans, source-path use, combined output, reference-history separation, and adaptive controls.
- [x] Gate passes with all existing and new API tests.

**Tests**: unit
**Gate**: quick
**Commit**: `feat(recommendations): combine allocation and selectors`

---

### T6: Execute shared asynchronous recommendation runs

**What**: Generalize the Premium executor to refresh validated run inputs, call the combined adapter for either plan, and persist result/provenance or sanitized failure.
**Where**: `api/app/services/premium_executor.py`
**Depends on**: T3, T5
**Reuses**: Existing queue, run-specific workspace, lifecycle transitions, restart recovery, and failure persistence.
**Requirement**: SIR-05, SIR-06, SIR-07, SIR-08, SIR-09, SIR-11

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [x] Both plans use the same queued → running → completed/failed lifecycle.
- [x] Successful runs persist combined result, source provenance, and output hash.
- [x] Either source failure marks run failed without invoking the selector or replacing a completed result.
- [x] Failed runs store sanitized diagnostics and preserve per-run snapshots for reproduction.
- [x] Add at least 4 integration tests covering Basic/Premium success, partial refresh failure, provenance persistence, and previous-success preservation.
- [x] Gate passes with all API tests.

**Tests**: integration
**Gate**: full
**Commit**: `feat(recommendations): execute shared async runs`

---

### T7: Route all API recommendations through shared runs

**What**: Convert Basic POST to queued execution, preserve Premium policy flow, and add account/profile/plan-scoped latest-completed lookup.
**Where**: `api/app/routers/recommendations.py`
**Depends on**: T6
**Reuses**: Shared coordinator, account-scoped run polling, and existing response schemas.
**Requirement**: SIR-01, SIR-03, SIR-08, SIR-10, SIR-11, SIR-12

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Basic POST returns `202` and a pollable run status; profile ownership is checked before dispatch.
- [ ] Premium and Basic return the common completed-result shape.
- [ ] `GET /v1/recommendations/latest-completed` returns only the authenticated account's current profile/plan completed result and is registered before `/{recommendation_id}`.
- [ ] GET/status/latest-completed requests do not call Status Invest or execute optimization.
- [ ] Add at least 5 route tests covering Basic, Premium, failed source, latest-completed filters, and GET without refresh.
- [ ] Gate passes with all API tests.

**Tests**: integration
**Gate**: full
**Commit**: `feat(api): unify recommendation routes`

---

### T8: Show combined runs in the app

**What**: Update web API types/client and recommendation view to start explicit runs, poll both plans, show latest completed result beside current status, and render source provenance.
**Where**: `web/components/recommendation/recommendation-view.tsx`
**Depends on**: T7
**Reuses**: Existing Premium polling, class/constituent rendering, and Node source-contract tests.
**Requirement**: SIR-03, SIR-10, SIR-12, SIR-13, SIR-14

**Tools**:

- MCP: NONE
- Skill: `coding-guidelines`

**Done when**:

- [ ] Basic and Premium use the same async run/status contract.
- [ ] Page load reads stored runs; only explicit new-run action triggers refresh.
- [ ] Polling continues with capped backoff until terminal state and resumes after reload.
- [ ] Reaching the current 30-second polling window never changes a queued/running result to failed.
- [ ] Current failed/pending status appears separately from the latest completed result.
- [ ] Both plans display class weights, BRL targets, stock/FII constituents, and Status Invest retrieval time.
- [ ] Add at least 5 Node contract tests; `npm run typecheck`, build, and smoke pass.
- [ ] Existing Premium retry/forced-run and truthful in-progress-state assertions remain passing.

**Tests**: unit
**Gate**: build
**Commit**: `feat(web): show statusinvest recommendations`

---

## Phase Execution Map

```text
Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1: T1 ------→ T2 ------→ T3
Phase 2: T3 ------→ T4
Phase 3: T2 --------------→ T5 ------→ T6
         T4 --------------→ T5        ↑
         T3 --------------------------┘
Phase 4: T6 ------→ T7 ------→ T8
```

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| --- | --- | --- | --- |
| T1 | None | None | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T2, T4 | T2 → T5; T4 → T5 | ✅ Match |
| T6 | T3, T5 | T3 → T6; T5 → T6 | ✅ Match |
| T7 | T6 | T6 → T7 | ✅ Match |
| T8 | T7 | T7 → T8 | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| --- | --- | --- | --- | --- |
| T1 | Snapshot manifest | unit | unit | ✅ OK |
| T2 | Status Invest input service | unit | unit | ✅ OK |
| T3 | Recommendation repository | unit | unit | ✅ OK |
| T4 | Profile policy | unit | unit | ✅ OK |
| T5 | Optimization adapter | unit | unit | ✅ OK |
| T6 | Async executor | integration | integration | ✅ OK |
| T7 | API routes | integration | integration | ✅ OK |
| T8 | Web API/UI | unit | unit | ✅ OK |

## Task Granularity Check

| Task | Scope | Status |
| --- | --- | --- |
| T1: Resolve selector sources by role | One manifest source-resolution behavior | ✅ Granular |
| T2: Capture run-scoped Status Invest inputs | One input snapshot helper | ✅ Granular |
| T3: Generalize recommendation run lifecycle | One run lifecycle contract across service/repository | ✅ Granular |
| T4: Resolve profile-specific selector and GA policy | One profile-policy mapping | ✅ Granular |
| T5: Combine selectors and class allocation | One optimization adapter contract | ✅ Granular |
| T6: Execute shared asynchronous recommendation runs | One worker execution lifecycle | ✅ Granular |
| T7: Route all API recommendations through shared runs | One API recommendation contract | ✅ Granular |
| T8: Show combined runs in the app | One recommendation view flow | ✅ Granular |

---

## MCP and Skill Selection

No MCP is required for local implementation. Use `tlc-spec-driven` for execution gates and `coding-guidelines` for code tasks. Confirm tool/skill preferences before Execute.
