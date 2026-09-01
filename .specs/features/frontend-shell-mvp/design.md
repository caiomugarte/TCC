# Prumo Frontend Shell and SaaS Vertical Slice Design

**Spec:** `.specs/features/frontend-shell-mvp/spec.md`  
**Context:** `.specs/features/frontend-shell-mvp/context.md`  
**Status:** Draft; spec and context approved

## Architecture Overview

One monorepo contains two deployable runtimes:

```text
Visitor/user
    |
    v
Next.js web app ----------------------+
    |                                  |
    | JSON over authenticated session  | public landing/assets
    v                                  |
FastAPI API                           |
    |                                  |
    +--> PostgreSQL                    |
    +--> recommendation adapter        |
              |
              +--> existing Python engine (`py/`)

Billing provider webhook --> FastAPI entitlement handler
```

Next.js owns rendering, navigation, route-level loading/error states, and
accessibility. FastAPI owns authentication verification, authorization,
validation, persistence, recommendation orchestration, drift calculations, and
entitlement checks. The browser never imports research modules or reads local
data/output artifacts.

## Repository Shape

```text
web/
  app/
    (public)/page.tsx
    (auth)/login/page.tsx
    (auth)/signup/page.tsx
    (protected)/app/onboarding/page.tsx
    (protected)/app/recommendation/page.tsx
    (protected)/app/portfolio/page.tsx
    (protected)/app/review/page.tsx
    (protected)/account/page.tsx
  components/
  lib/

api/
  app/
    main.py
    routers/
    schemas/
    services/
    adapters/
    db/

py/
  core/
  pipelines/
  allocation_profiles.py
```

Route groups are organizational only. `(public)`, `(auth)`, and
`(protected)` do not change the URL. The `/app` segment remains explicit in
protected page paths.

## Request and State Flow

1. Next.js renders the requested route and calls the API through a small typed
   client.
2. FastAPI verifies the session and derives `account_id` from it.
3. The API validates the request, loads the user's latest valid records, and
   invokes a domain service.
4. The domain service persists immutable profile/recommendation/snapshot
   records or calculates a review from existing records.
5. The API returns stable JSON with version, cutoff, and error metadata.
6. Next.js renders success, loading, unavailable, or validation state.

Recommendation generation starts synchronously. A worker is a later seam only
if measured execution time exceeds the API request budget.

## Code Reuse Analysis

| Existing component | Location | Use |
| --- | --- | --- |
| Public marketing prototype | `landing.html` | Port content and visual tokens into `/` |
| Suitability questionnaire | `suitability.html` | Port fields, score dimensions, consent wording, and result language into onboarding |
| Allocation anchors | `py/allocation_profiles.py` | Build Basic generic profile mapping |
| Allocation metrics/search | `py/core/allocation.py` | Preserve deterministic candidate evaluation |
| Allocation orchestration | `py/pipelines/asset_allocation.py` | Adapt result into API recommendation contract |
| Snapshot validation | `py/allocation_data.py` | Reuse documented snapshot and common-date rules |
| Recommendation contract | `.specs/features/portfolio-recommendation-mvp/spec.md` | Preserve provenance and compliance fields |
| Tracking contract | `.specs/features/portfolio-tracking-mvp/spec.md` | Preserve snapshot, drift, and contribution-first rules |
| Billing contract | `.specs/features/subscription-billing-mvp/spec.md` | Preserve provider-neutral entitlement behavior |

Fragile stock GA and legacy backtest modules remain outside the first API
adapter. The API must use the allocation path and canonical snapshot artifacts,
not infer a new stock portfolio from conflicting profile dictionaries.

## Components

### Web application shell

- **Purpose:** Render public, auth, protected, and account routes in one app.
- **Location:** `web/app/`, `web/components/`
- **Interfaces:** route components, shared layout, navigation, loading/error
  boundaries.
- **Dependencies:** Next.js, TypeScript, API client.
- **Reuses:** landing/suitability CSS language and semantic form patterns.

### Web API client

- **Purpose:** Keep HTTP calls and response/error normalization out of page
  components.
- **Location:** `web/lib/api-client.ts`, `web/lib/api-types.ts`
- **Interfaces:** `getMe()`, `getProfile()`, `saveProfile()`,
  `createRecommendation()`, `savePortfolio()`, `getReview()`.
- **Dependencies:** authenticated browser session and FastAPI JSON contract.
- **Reuses:** none; new boundary required between runtimes.

### FastAPI routers

- **Purpose:** Expose versioned account, profile, recommendation, portfolio,
  review, and health resources.
- **Location:** `api/app/routers/`
- **Interfaces:** `/v1/me`, `/v1/profile`, `/v1/recommendations`,
  `/v1/portfolio`, `/v1/review`, `/v1/account`, `/health`.
- **Dependencies:** auth dependency, Pydantic schemas, domain services.
- **Reuses:** existing Python modules through adapters only.

### Domain services

- **Purpose:** Own profile computation, recommendation orchestration,
  portfolio normalization, drift classification, and entitlement checks.
- **Location:** `api/app/services/`
- **Interfaces:** typed service functions accepting account-scoped inputs and
  returning typed records/results.
- **Dependencies:** repositories, engine adapter, configuration.
- **Reuses:** pure allocation and profile functions; no CLI subprocess.

### Persistence layer

- **Purpose:** Store account-owned, versioned product records.
- **Location:** `api/app/db/`
- **Interfaces:** SQLAlchemy models/repositories and migrations.
- **Dependencies:** PostgreSQL, SQLAlchemy, Alembic.
- **Reuses:** none; product persistence does not exist yet.

### Existing engine adapter

- **Purpose:** Translate API input into the existing allocation engine and
  translate results into the recommendation contract.
- **Location:** `api/app/adapters/allocation_engine.py`
- **Interfaces:** `generate_basic_recommendation(profile, capital, snapshot)`.
- **Dependencies:** importable `py/` modules and versioned snapshot artifacts.
- **Reuses:** `allocation_profiles.py`, `allocation_data.py`,
  `pipelines/asset_allocation.py`, `core/allocation.py`.
- **Constraint:** first package/import seam must preserve existing CLI behavior;
  no broad refactor of research code.

## Data Models

The API/database model is account-scoped. Recommendation and profile versions
are immutable; a new calculation creates a new run.

```typescript
interface Profile {
  id: string
  accountId: string
  version: number
  answers: Record<string, string>
  dimensions: Record<string, number>
  suitabilityScore: number
  genericProfile: "conservador" | "moderado" | "arrojado"
  investableCapitalBrl: number
  consentedAt: string
  createdAt: string
}

interface Recommendation {
  id: string
  profileVersion: number
  plan: "basic" | "premium"
  modelVersion: string
  snapshotId: string
  snapshotCutoff: string
  classes: AllocationClass[]
  assumptions: string[]
  risks: string[]
}

interface AllocationClass {
  key: string
  label: string
  targetWeight: number
  targetAmountBrl: number
}

interface PortfolioSnapshot {
  id: string
  source: "manual"
  capturedAt: string
  currency: "BRL"
  totalValueBrl: number
  classes: Record<string, number>
}

interface DriftItem {
  classKey: string
  currentWeight: number
  targetWeight: number
  drift: number
  valueGapBrl: number
  status: "within_range" | "underweight" | "overweight"
  suggestedAction: "hold" | "contribute" | "review_sale"
}
```

## Error Handling Strategy

| Scenario | API behavior | UI behavior |
| --- | --- | --- |
| Invalid form/input | `422` structured field errors | Mark fields and preserve entered values |
| Unauthenticated | `401` | Redirect to login with safe return path |
| Wrong account/resource | `404` or `403` without ownership disclosure | Show unavailable state |
| Missing/infeasible snapshot | `409` domain error with reason | Explain why result is unavailable |
| Missing prerequisite | `409` or explicit state response | Route user to next required step |
| Premium without entitlement | `403` stable error code | Show locked feature state |
| Unexpected server failure | `500` correlation-safe generic error | Show retry action; preserve prior valid state |

## Tech Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Frontend framework | Next.js + TypeScript | One app supports SEO landing, route layouts, and protected application |
| Backend framework | FastAPI | Thin API boundary; direct Python engine reuse; automatic OpenAPI |
| Persistence | PostgreSQL + SQLAlchemy/Alembic | Relational account/version/entitlement data and migrations |
| Contract source | FastAPI schemas/OpenAPI | One server-side validation boundary; frontend consumes stable JSON |
| Styling | Native CSS/CSS Modules first | Reuse existing visual language; avoid UI dependency before need exists |
| Auth | Clerk via `@clerk/nextjs` and `clerk-backend-api` | Managed identity UI/session issuance; API verifies tokens and maps `sub` to local accounts |
| Recommendation execution | Synchronous first | Avoid queue infrastructure until measured latency requires it |
| Portfolio scope | One manual class-level snapshot | Covers drift value with smallest valid data model |
| Deployment shape | One repo, separate web/API services | Matches JS/Python runtimes without premature microservices |

## Verification Strategy

- Web build/typecheck proves route tree and shared layouts compile.
- API health/OpenAPI smoke check proves service starts and contract is exposed.
- Domain unit tests cover profile mapping, weight/amount normalization, and
  drift classification with dependency-light fixtures.
- API tests cover account ownership, prerequisite states, and entitlement
  rejection.
- One Playwright smoke flow covers signup fixture, onboarding, recommendation,
  portfolio, and review once auth/persistence exist.
- Manual mobile and keyboard pass covers all shell routes before launch.

## Known Risks and Mitigations

- Existing `py/` imports assume execution from its directory. Isolate this in
  one adapter/package task; do not scatter `sys.path` changes across routers.
- Allocation artifacts can differ from current defaults. Persist exact
  snapshot/model metadata with every recommendation.
- Personalized outputs have a compliance gate. Keep Basic generic wording and
  do not enable public Premium recommendation until approved.
- Recommendation runtime may be too long for a request. Measure first; add a
  worker only at the measured boundary.
