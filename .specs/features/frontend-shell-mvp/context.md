# Prumo Frontend Shell and SaaS Vertical Slice Context

**Gathered:** 2026-08-11  
**Spec:** `.specs/features/frontend-shell-mvp/spec.md`  
**Status:** Approved

## Feature Boundary

One end-to-end web flow: account creation, suitability onboarding, Basic
five-class allocation, one manual class-level portfolio snapshot, and drift
review. Public landing and protected application share one frontend codebase.

## Implementation Decisions

### One product spec, two implementation workstreams

- Keep one feature spec because success is an end-to-end user flow.
- Split implementation tasks into `WEB-*`, `API-*`, `DATA-*`, and `VERIFY-*`
  workstreams.
- Create separate frontend/backend feature specs only when one side becomes an
  independently releasable product capability or has a different user goal.

### Stack and repository boundary

- `web/`: Next.js, TypeScript, App Router, and native CSS/CSS Modules first.
- `api/`: FastAPI, Python, Pydantic, SQLAlchemy/Alembic, and PostgreSQL.
- Existing `py/` calculation modules remain the source of recommendation
  logic; API adapters call them and return JSON.
- One monorepo and two deployable runtimes. No microservice split beyond web
  and API at this stage.

### Route and surface model

- Public and authenticated routes live in one frontend app.
- Authenticated routes share an application shell and server/API-backed route
  guards.
- `suitability.html` is a reference for onboarding behavior and visual language,
  not a separate application.

### Basic vertical slice

- Onboarding computes and stores a continuous score.
- Basic maps that result to a declared generic anchor and exposes only
  five-class allocation.
- Premium continuous personalization and Brazilian-stock detail remain
  entitlement-gated future work.

### Portfolio and review scope

- One primary portfolio per account.
- First slice accepts manual class-level BRL values only.
- Initial drift band is five percentage points, stored with review metadata.
- Review recommends contribution-first actions and never emits orders.

### Runtime simplicity

- Start recommendation generation synchronously.
- Add a worker/queue only if measured runtime exceeds the API request budget or
  real usage requires retries/progress state.
- Use versioned local or mounted snapshot artifacts for the first slice; store
  snapshot ID and cutoff in every recommendation.

### Auth and billing boundaries

- Use a managed or framework-supported authentication mechanism; do not write
  custom password hashing or client-only access control.
- Use Clerk for signup, login, session issuance, and user-facing account
  recovery. Keep Clerk-specific verification behind `api/app/auth/`.
- The API maps verified Clerk `sub` claims to local account records. Email is
  optional because it is not a default Clerk session claim.
- Checkout return pages never grant Premium; verified provider webhooks update
  the product entitlement.

## Existing Code to Reuse

- `landing.html` — public visual language and marketing content.
- `suitability.html` — questionnaire fields, score dimensions, consent copy,
  and accessibility baseline.
- `py/allocation_profiles.py` — generic allocation anchors and score
  interpolation.
- `py/core/allocation.py` — deterministic allocation metrics and candidate
  evaluation.
- `py/pipelines/asset_allocation.py` — allocation orchestration and output
  contract foundations.
- `.specs/features/portfolio-recommendation-mvp/spec.md` — recommendation
  provenance and compliance boundary.
- `.specs/features/portfolio-tracking-mvp/spec.md` — snapshot and drift rules.
- `.specs/features/subscription-billing-mvp/spec.md` — provider-neutral
  entitlement rules.

## Open Decisions

- Clerk email-verification policy and production authorized-party URLs.
- Billing provider and exact Basic/Premium entitlement matrix.
- Legal operating model for personalized recommendations.
- Final Basic presentation of target BRL amounts.
- Market snapshot deployment and refresh schedule.

## Deferred Ideas

- Separate frontend and backend feature specs for independently releasable
  capabilities.
- Stock-level portfolio input and Brazilian-stock drift review.
- CSV and B3 imports.
- Background recommendation jobs and progress UI.
- Multiple portfolios, broker integrations, and order execution.
