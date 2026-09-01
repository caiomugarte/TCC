# Prumo Frontend Shell and SaaS Vertical Slice Tasks

**Design:** `.specs/features/frontend-shell-mvp/design.md`  
**Status:** In Progress

## Execution Plan

### Phase 1: Foundation

```text
T1 (web manifest) ─┐
T2 (api manifest) ──┼──> T3 (route shell) ──> T4 (API client/contract)
T5 (API health) ────┘
```

T1, T2, and T5 can run in parallel. T3 and T4 depend on their respective
foundations. No database or auth provider is added during shell-only
scaffolding.

### Phase 2: Product vertical slice

```text
T4 ──> T6 profile schema/service ──> T7 onboarding UI
  └──> T8 recommendation adapter ──┐
T5 ──> T8.5 persistence foundation ─┼──> T9 recommendation endpoint ──> T10 recommendation UI
                                   └──> T11 portfolio endpoint ──> T12 portfolio UI
T9 + T11 ──> T13 review service/endpoint ──> T14 review UI
```

T6, T8, and T8.5 can be parallel after the foundation. Endpoint/UI tasks
depend on persistence and their API contracts.

### Phase 3: Access, quality, and integration

```text
T7 + T10 + T12 + T14 ──> T15 route prerequisites
T15 ──> T16 entitlement boundary
T16 ──> T17 end-to-end/accessibility verification
```

## Task Breakdown

### T1: Create web application manifest [P]

**What:** Add the minimal Next.js/TypeScript package and scripts under `web/`.
**Where:** `web/package.json`, `web/tsconfig.json`, `web/next.config.*`,
`web/app/layout.tsx`
**Depends on:** None  
**Requirement:** `SHELL-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] `npm run build` can run from `web/` after dependencies are installed.
- [x] Root layout declares `lang="pt-BR"`, metadata, and global styles entry.
- [x] No UI framework or state library is added.

**Verify:** `cd web && npm run build` — passed.

### T2: Create API package manifest [P]

**What:** Add isolated FastAPI dependencies and application package under
`api/`.
**Where:** `api/pyproject.toml` or `api/requirements.txt`, `api/app/__init__.py`
**Depends on:** None  
**Requirement:** `API-01`

**Tools:** local filesystem; Python standard tooling.

**Done when:**

- [x] API dependencies are isolated from research dependencies.
- [x] Python runtime constraint is compatible with existing modern annotations.
- [x] Package imports without touching research modules.

**Verify:** `PYTHONPATH=api api/.venv/bin/python -m compileall -q api/app` — passed.

### T3: Create public/auth/protected route shell

**What:** Add route pages, shared layouts, navigation, and placeholder states
for all MVP routes.
**Where:** `web/app/`, `web/components/`
**Depends on:** T1  
**Requirement:** `SHELL-01`, `UX-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] All eight browser routes resolve.
- [x] Public, auth, and protected layout boundaries are visible in source.
- [x] Placeholder pages have semantic headings and keyboard-usable links.
- [x] Protected placeholders do not claim real authentication yet.

**Verify:** `cd web && npm run build`; inspect route manifest — passed.

### T4: Define frontend API client boundary

**What:** Add typed request/error helpers for the agreed `/v1` resources.
**Where:** `web/lib/api-client.ts`, `web/lib/api-types.ts`
**Depends on:** T1, T2  
**Requirement:** `API-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] Client exposes typed functions for account, profile, recommendation,
  portfolio, and review.
- [x] Non-2xx responses normalize to a stable client error shape.
- [x] No page component contains duplicated URL or fetch logic.

**Verify:** `cd web && npm run typecheck` — passed.

### T5: Add API health and OpenAPI shell [P]

**What:** Create FastAPI app startup, `/health`, and empty versioned router
registration.
**Where:** `api/app/main.py`, `api/app/routers/health.py`
**Depends on:** T2  
**Requirement:** `API-01`

**Tools:** local filesystem; Python standard tooling.

**Done when:**

- [x] API starts with the documented command.
- [x] `/health` returns a small JSON success response through the FastAPI app contract.
- [x] OpenAPI includes `/health` and the router registration without secrets.

**Verify:** app import, OpenAPI path assertion, and Python compilation passed. Direct localhost binding is sandbox-blocked.

### T6: Define profile schemas and service [P]

**What:** Add validated profile input/output schemas and account-scoped
profile service interfaces without persistence coupling.
**Where:** `api/app/schemas/profile.py`, `api/app/services/profile.py`
**Depends on:** T2, T4  
**Requirement:** `PROF-01`

**Tools:** local filesystem; Python tests.

**Done when:**

- [x] Required answer and capital validation rejects invalid input.
- [x] Score, dimensions, generic anchor, consent, and version are represented.
- [x] Service contract is deterministic for fixed inputs.

**Verify:** `PYTHONPATH=api api/.venv/bin/python -m unittest discover -s api/tests -v` — passed.

### T7: Port suitability onboarding UI [P]

**What:** Convert the existing questionnaire behavior into an accessible
`/app/onboarding` form shell.
**Where:** `web/app/(protected)/app/onboarding/page.tsx`,
`web/components/onboarding/`
**Depends on:** T3, T4, T6  
**Requirement:** `PROF-01`, `UX-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] Existing question intent and limitation copy are preserved.
- [x] Required fields, labels, focus, and field-level errors work.
- [x] Submit calls the API client and handles loading/error states.

**Verify:** `cd web && npm run typecheck && npm run build` — passed.

### T8: Create recommendation engine adapter [P]

**What:** Add one adapter translating Basic profile/capital/snapshot inputs to
the existing allocation modules and normalized API output.
**Where:** `api/app/adapters/allocation_engine.py`, import seam tests
**Depends on:** T2, T6  
**Requirement:** `REC-01`

**Tools:** local filesystem; existing Python tests; `codenavi` findings.

**Done when:**

- [x] Adapter calls the allocation path, not the legacy stock backtest.
- [x] Output has five class keys, valid weights, amounts, and provenance.
- [x] Missing/infeasible input raises a typed domain error.
- [x] Existing research CLI behavior remains unchanged.

**Verify:** `PYTHONPATH=api:py api/.venv/bin/python -m unittest discover -s api/tests -v` — passed with injected synthetic engine result.

### T8.5: Add persistence foundation [P]

**What:** Add PostgreSQL configuration, SQLAlchemy base/session, initial
account-owned models, and an Alembic migration boundary without wiring routes.
**Where:** `api/app/db/`, `api/migrations/`, `api/alembic.ini`,
`api/requirements.txt`
**Depends on:** T2, T5  
**Requirement:** `AUTH-01`, `PROF-01`, `REC-01`, `PORT-01`, `ENT-01`

**Tools:** local filesystem; Python tests.

**Done when:**

- [x] Account, profile version, recommendation run, portfolio snapshot, and
  entitlement tables/models have account ownership and timestamps.
- [x] Local test configuration can create schema without a production
  database connection.
- [x] Alembic can generate/apply the initial migration.
- [x] API dependency list keeps persistence dependencies isolated from research
  dependencies.

**Verify:** SQLAlchemy metadata/schema test, `alembic upgrade head`, and
`alembic check` pass against SQLite.

### T9: Add recommendation endpoint [P]

**What:** Expose profile-scoped Basic recommendation creation and immutable
read endpoints.
**Where:** `api/app/routers/recommendations.py`,
`api/app/schemas/recommendation.py`
**Depends on:** T5, T8, T8.5  
**Requirement:** `REC-01`, `API-01`

**Tools:** local filesystem; API tests.

**Done when:**

- [x] Endpoint validates prerequisites and returns structured domain errors.
- [x] Response includes model, profile, snapshot, cutoff, assumptions, and
  class targets.
- [x] Resource ownership is account-scoped.

**Verify:** API route tests with valid, missing-snapshot, and cross-account
cases; real snapshot adapter smoke check passed.

### T10: Create recommendation page [P]

**What:** Render Basic class allocation, target amounts, provenance, risks,
and unavailable states.
**Where:** `web/app/(protected)/app/recommendation/page.tsx`,
`web/components/recommendation/`
**Depends on:** T3, T4, T9  
**Requirement:** `REC-01`, `UX-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] Five classes render with percentages and amounts.
- [x] Cutoff/model/assumption information is visible.
- [x] Loading, unavailable, and retry states are explicit.

**Verify:** `cd web && npm run typecheck && npm run build` pass.

### T11: Add manual portfolio endpoint [P]

**What:** Validate and persist one account-scoped class-level portfolio
snapshot.
**Where:** `api/app/routers/portfolio.py`, `api/app/schemas/portfolio.py`,
`api/app/services/portfolio.py`
**Depends on:** T5, T4, T8.5  
**Requirement:** `PORT-01`, `API-01`

**Tools:** local filesystem; API tests.

**Done when:**

- [x] Five class values are finite, non-negative, and normalized.
- [x] Zero-total snapshots are rejected for review.
- [x] Prior valid snapshots remain readable.
- [x] Cross-account access is rejected without ownership disclosure.

**Verify:** API route tests for valid, invalid, history, and ownership cases.

### T12: Create portfolio input page [P]

**What:** Render class-level BRL input form and save/reload states.
**Where:** `web/app/(protected)/app/portfolio/page.tsx`,
`web/components/portfolio/`
**Depends on:** T3, T4, T11  
**Requirement:** `PORT-01`, `UX-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] Each supported class has labeled numeric input.
- [x] Invalid/zero-total values show useful errors.
- [x] Saved snapshot reloads through the API client.

**Verify:** `cd web && npm run typecheck && npm run build` pass; responsive CSS is included.

### T13: Add drift review service and endpoint [P]

**What:** Calculate class-level drift and contribution-first actions from the
latest valid recommendation and portfolio snapshot.
**Where:** `api/app/services/review.py`, `api/app/routers/review.py`,
`api/app/schemas/review.py`
**Depends on:** T9, T11  
**Requirement:** `REVIEW-01`

**Tools:** local filesystem; dependency-light Python tests.

**Done when:**

- [x] Drift, BRL gap, five-point band, and status are deterministic.
- [x] Underweight contributions precede sale review actions.
- [x] Missing prerequisite returns unavailable state.
- [x] No order object is emitted.

**Verify:** fixed-target/fixed-portfolio unit tests pass.

### T14: Create review page [P]

**What:** Render current-versus-target class rows and review action list.
**Where:** `web/app/(protected)/app/review/page.tsx`,
`web/components/review/`
**Depends on:** T3, T4, T13  
**Requirement:** `REVIEW-01`, `UX-01`

**Tools:** local filesystem; `react-best-practices` skill.

**Done when:**

- [x] Current, target, signed drift, gap, and status are visible.
- [x] Contribution-first action wording is clear and no trade action exists.
- [x] Unavailable and stale states are explicit.

**Verify:** `cd web && npm run typecheck && npm run build` pass.

### T15: Add route prerequisites and session boundary

**What:** Enforce login/profile/recommendation/portfolio prerequisites in web
navigation and API dependencies.
**Where:** `web/middleware.ts`, `api/app/auth/`, affected router dependency
files
**Depends on:** T3, T6, T9, T11, T14  
**Requirement:** `SHELL-01`, `AUTH-01`, `API-01`

**Tools:** local filesystem; security review of ownership paths.

**Done when:**

- [x] Protected routes redirect unauthenticated users safely.
- [x] API derives account identity from verified session.
- [x] Prerequisite redirects do not leak protected data.
- [x] Auth provider implementation remains behind one boundary.

**Verify:** authenticated/unauthenticated route and cross-account API tests.

### T16: Add entitlement checks and locked Premium state [P]

**What:** Expose plan state and reject Premium-only API behavior while showing
a stable frontend lock state.
**Where:** `api/app/entitlements/`, `web/components/entitlement/`,
`web/app/(protected)/account/page.tsx`
**Depends on:** T15  
**Requirement:** `ENT-01`

**Tools:** local filesystem; API tests.

**Done when:**

- [x] Basic cannot access Premium endpoint behavior.
- [x] Premium state comes from server entitlement, not client flags.
- [x] Account page shows plan and access state.
- [x] Billing webhook/provider implementation remains in its own feature.

**Verify:** Basic/Premium authorization tests.

### T17: Run vertical-slice verification [P]

**What:** Add one automated browser smoke flow and document manual mobile,
keyboard, and unavailable-state checks.
**Where:** `web/tests/`, `api/tests/`, feature verification notes
**Depends on:** T7, T10, T12, T14, T15, T16  
**Requirement:** `SHELL-01`, `AUTH-01`, `PROF-01`, `REC-01`, `PORT-01`,
`REVIEW-01`, `UX-01`

**Tools:** local filesystem; Playwright if already available; existing Python
test runner.

**Done when:**

- [x] Signup fixture reaches review without manual database edits.
- [x] Reload preserves profile, recommendation, and portfolio state.
- [x] Cross-account access and Premium lock checks pass.
- [x] Keyboard/mobile/unavailable-state checklist is recorded.

**Verify:** documented web/API test commands pass.

## Planned Commit Messages

Use one atomic commit per completed implementation task when the user requests
commit history:

- `build(shell): add web and api foundations`
- `feat(shell): add protected route structure`
- `feat(profile): persist suitability onboarding`
- `feat(recommendation): expose basic allocation`
- `feat(portfolio): add manual snapshot input`
- `feat(review): add drift review`
- `feat(auth): enforce account boundaries`
- `feat(entitlements): protect premium surface`
- `test(shell): verify onboarding to drift flow`

## Granularity Check

Each task has one primary deliverable: one manifest, endpoint, adapter,
service, page, boundary, or verification slice. Cross-file tasks are kept only
when the files form one cohesive endpoint/component boundary.
