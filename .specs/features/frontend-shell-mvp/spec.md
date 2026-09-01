# Prumo Frontend Shell and SaaS Vertical Slice MVP

**Status:** In Progress  
**Depends on:** `landing-page-mvp`, `portfolio-recommendation-mvp`,
`portfolio-tracking-mvp`, `subscription-billing-mvp`, `caio-asset-allocation`

**Implementation target:** one repository with a Next.js/TypeScript web app,
a FastAPI/Python API, PostgreSQL persistence, and the existing Python
allocation engine.

## Problem Statement

Prumo currently has a public landing-page prototype and research HTML/scripts,
but no account, protected application, persistence, or end-to-end SaaS flow.
The MVP needs one frontend app with public and authenticated surfaces, then one
complete path from account creation to a basic allocation and portfolio drift
review.

The feature connects existing recommendation, tracking, and billing domain
specifications. It does not replace them or duplicate their calculation rules.

## Feature Boundary

Deliver these user-visible routes and vertical-slice states:

```text
signup/login
  -> suitability onboarding
  -> Basic class-level allocation
  -> manual primary-portfolio input
  -> current-versus-target drift review
```

Basic output uses a declared generic profile anchor and exposes five-class
allocation. The continuous questionnaire score is stored for later Premium
personalization, but stock-level recommendation is not part of this slice.

## Goals

- [ ] Build one frontend app with public and protected route layouts.
- [ ] Establish authenticated account ownership for profile, recommendation,
  and portfolio records.
- [ ] Persist suitability answers, computed dimensions, score, and version.
- [ ] Generate and persist one Basic five-class recommendation using the
  existing Python allocation logic and versioned snapshot metadata.
- [ ] Persist one manual primary portfolio snapshot at class level.
- [ ] Show reproducible current weight, target weight, signed drift, and a
  contribution-first review action for each class.
- [ ] Represent Basic/Premium entitlement state and protect Premium-only
  surfaces on the server.
- [ ] Keep API contracts independent from frontend implementation details.

## Routes

| Route | Surface | Access | MVP behavior |
| --- | --- | --- | --- |
| `/` | Public | None | Existing Prumo landing page |
| `/login` | Auth | None | Sign in and return to requested route |
| `/signup` | Auth | None | Create account and start onboarding |
| `/app/onboarding` | Application | Authenticated | Complete suitability and capital inputs |
| `/app/recommendation` | Application | Authenticated | Show stored Basic allocation |
| `/app/portfolio` | Application | Authenticated | Enter one class-level portfolio snapshot |
| `/app/review` | Application | Authenticated | Show drift and review actions |
| `/account` | Account | Authenticated | Show identity, plan, entitlement, and legal links |

Route prerequisites:

- Unauthenticated users requesting `/app/*` or `/account` SHALL be sent to
  `/login` with a safe return path.
- Authenticated users without a valid profile SHALL be sent to onboarding.
- Users without a recommendation SHALL be sent to recommendation setup after
  onboarding.
- Users without a portfolio snapshot SHALL be sent to portfolio input before
  review.
- Route guards SHALL not be the only protection; the API SHALL enforce the
  same ownership and entitlement rules.

## User Stories

### P1: Use public and protected surfaces ⭐ MVP

**Requirement ID:** `SHELL-01`

**User Story:** As a visitor, I want public pages and a protected application
in one product so that I can move from learning about Prumo to using it.

**Acceptance Criteria:**

1. WHEN a visitor opens `/` THEN the page SHALL render without an account or
   live market request.
2. WHEN an authenticated user navigates through the slice THEN the user SHALL
   remain in the shared application layout.
3. WHEN an unauthenticated user requests a protected route THEN the system
   SHALL redirect to login without exposing protected data.
4. WHEN a user reloads any completed step THEN the system SHALL resume from
   persisted server state.

**Independent Test:** Open the public page logged out, complete the protected
flow logged in, reload each route, and verify state and access boundaries.

### P1: Create and access an account ⭐ MVP

**Requirement ID:** `AUTH-01`

**User Story:** As an investor, I want an account so that my profile and
portfolio belong to me.

**Acceptance Criteria:**

1. WHEN valid signup data is submitted THEN the system SHALL create one user
   identity and establish an authenticated session.
2. WHEN invalid or duplicate signup data is submitted THEN the system SHALL
   show a useful error and SHALL not create a partial account.
3. WHEN valid login data is submitted THEN the system SHALL restore the user's
   own persisted state.
4. WHEN the user logs out THEN protected routes and API resources SHALL stop
   being accessible.
5. Credential storage and verification SHALL belong to the selected auth
   mechanism; the application SHALL not implement custom password hashing.

**Independent Test:** Create two accounts and verify each can only read and
modify its own profile and portfolio.

### P1: Complete suitability onboarding ⭐ MVP

**Requirement ID:** `PROF-01`

**User Story:** As an investor, I want to answer the profile questionnaire so
that Prumo can select an appropriate generic allocation anchor.

**Acceptance Criteria:**

1. WHEN required questions are incomplete or invalid THEN submission SHALL be
   rejected with field-level errors.
2. WHEN the form is accepted THEN the system SHALL store raw answers,
   computed dimensions, suitability score, selected generic anchor, consent
   state, profile version, and account ID.
3. WHEN the user reopens onboarding THEN the system SHALL reproduce the stored
   answers and score rather than silently recompute a different version.
4. WHEN profile computation changes in a later release THEN new submissions
   SHALL receive a new profile version and prior profiles SHALL remain intact.
5. The onboarding copy SHALL preserve the prototype's limitation that this is
   not, by itself, a complete regulated suitability process or a guarantee.

**Independent Test:** Submit a valid profile, reload it, and verify identical
answers, dimensions, score, anchor, and version.

### P1: Generate a Basic recommendation ⭐ MVP

**Requirement ID:** `REC-01`

**User Story:** As an investor, I want a clear Basic allocation so that I can
understand how my capital is distributed across asset classes.

**Acceptance Criteria:**

1. WHEN a valid profile, investable capital, and supported snapshot exist THEN
   the system SHALL generate five non-negative class weights summing to 100%.
2. The result SHALL cover Brazilian stocks, FIIs, international exposure,
   fixed income, and crypto using stable class keys.
3. WHEN capital is provided THEN each class SHALL include its BRL target
   amount; total target amounts SHALL equal the declared capital within the
   documented rounding rule.
4. The result SHALL include profile version, model version, snapshot ID,
   cutoff date, assumptions, risks, and plan level.
5. Basic SHALL expose class-level allocation only. Stock-level selection,
   continuous personalized output, and deeper Brazilian-stock analysis SHALL
   remain Premium-gated or deferred.
6. WHEN the snapshot is missing, stale beyond policy, or infeasible THEN the
   system SHALL report the recommendation as unavailable and SHALL not publish
   a partial or fabricated target.
7. Repeating the same profile, capital, snapshot, and model version SHALL
   reproduce the same stored result.

**Independent Test:** Run the existing allocation core with a synthetic
snapshot and valid Basic profile; verify weights, amounts, provenance, and
failure on missing data.

### P1: Maintain one manual portfolio snapshot ⭐ MVP

**Requirement ID:** `PORT-01`

**User Story:** As an investor, I want to enter my current allocation so that
Prumo can compare it with my target.

**Acceptance Criteria:**

1. The system SHALL support exactly one primary portfolio per account in this
   slice.
2. The user SHALL enter current BRL value for each supported asset class.
3. Values SHALL be finite and non-negative; a zero-total portfolio SHALL be
   rejected for drift review.
4. A saved snapshot SHALL include account ID, captured-at timestamp, currency,
   source (`manual`), class values, total value, and normalized weights.
5. Saving a new snapshot SHALL preserve the prior valid snapshot.
6. CSV import, ticker-level positions, B3 import, and price lookup SHALL stay
   outside this slice unless added by a later tracking task.

**Independent Test:** Save two manual snapshots and verify normalized values,
history, and account isolation.

### P1: Review portfolio drift ⭐ MVP

**Requirement ID:** `REVIEW-01`

**User Story:** As an investor, I want to see where my portfolio differs from
the target so that I can decide what to review next.

**Acceptance Criteria:**

1. WHEN a valid recommendation and portfolio snapshot exist THEN the system
   SHALL show current weight, target weight, signed drift, and BRL gap per
   class.
2. The default initial drift band SHALL be five percentage points and SHALL be
   stored as review metadata.
3. Drift inside the band SHALL be labeled `within_range`; outside drift SHALL
   be labeled `underweight` or `overweight`.
4. The review SHALL prioritize new contributions toward underweight classes
   before suggesting sales.
5. Any sale SHALL be labeled as a review action; the system SHALL never create
   or submit an order object.
6. WHEN a target or snapshot is unavailable THEN the system SHALL show tracking
   as unavailable rather than infer missing values.

**Independent Test:** Use a fixed target and known class values; verify drift,
band labels, contribution ordering, and absence of trade objects.

### P1: Expose a stable API contract ⭐ MVP

**Requirement ID:** `API-01`

**User Story:** As the frontend, I want versioned JSON resources so that UI
state does not contain business rules or direct file access.

**Acceptance Criteria:**

1. The API SHALL expose authenticated resources for the current account,
   profile, recommendation, portfolio, and review.
2. The API SHALL derive account ownership from the authenticated session; the
   client SHALL not choose a `user_id` for reads or writes.
3. The API SHALL validate all trust-boundary inputs and return structured
   errors without leaking stack traces or provider secrets.
4. Recommendation and review responses SHALL use stable keys and include
   version/cutoff metadata.
5. The frontend SHALL consume JSON through an API client and SHALL not import
   Python modules, execute CLI scripts, or read `data/`/`outputs/` directly.

### Minimum API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/v1/me` | Account identity and entitlement summary |
| `GET/PUT` | `/v1/profile` | Read or persist current suitability profile |
| `POST` | `/v1/recommendations` | Generate and persist Basic recommendation |
| `GET` | `/v1/recommendations/:id` | Read immutable recommendation run |
| `PUT/POST` | `/v1/portfolio` | Save a primary manual snapshot |
| `GET` | `/v1/review` | Calculate/read current drift review |
| `GET` | `/v1/account` | Account, plan, legal, and access state |

Auth provider endpoints and billing webhook endpoints remain integration
details outside this shell spec. Billing behavior follows
`subscription-billing-mvp`; checkout redirects SHALL never grant Premium by
themselves.

### Core resource shapes

```text
Profile
  id, account_id, version, answers, dimensions, suitability_score,
  generic_profile, investable_capital_brl, consented_at, created_at

Recommendation
  id, account_id, profile_version, plan, model_version, snapshot_id,
  snapshot_cutoff, classes[], assumptions, risks, created_at

AllocationClass
  key, label, target_weight, target_amount_brl

PortfolioSnapshot
  id, account_id, source, captured_at, currency, total_value_brl, classes[]

DriftItem
  class_key, current_weight, target_weight, drift, value_gap_brl,
  status, suggested_action
```

### P1: Enforce entitlement boundaries ⭐ MVP

**Requirement ID:** `ENT-01`

**User Story:** As Prumo, I want plan access enforced server-side so that
Premium features cannot be unlocked by changing frontend state.

**Acceptance Criteria:**

1. `/v1/me` SHALL expose current plan and entitlement status.
2. Protected Premium endpoints SHALL reject accounts without the required
   entitlement.
3. The frontend SHALL show a stable locked state for unavailable Premium
   features.
4. Subscription provider identity, event processing, and entitlement changes
   SHALL follow the separate billing specification.

**Independent Test:** Call a Premium endpoint as Basic and Premium users;
verify rejection and success respectively, independent of UI state.

### P1: Meet baseline UX and accessibility requirements ⭐ MVP

**Requirement ID:** `UX-01`

**Acceptance Criteria:**

1. All routes SHALL support keyboard navigation, visible focus, semantic
   headings, labels, and useful error messages.
2. The application SHALL work at supported mobile widths without horizontal
   scrolling.
3. Loading, empty, unavailable, and validation states SHALL be explicit; no
   blank screen SHALL represent a failed API request.
4. Sensitive values and plan state SHALL not be exposed in public HTML or
   client logs.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Full payment provider and webhook implementation | Separate `subscription-billing-mvp` |
| Premium stock-level recommendation | Later release after Basic slice and compliance review |
| CSV, B3, broker, or automatic order execution | Deferred tracking scope |
| Multiple portfolios or households | One primary portfolio keeps MVP small |
| Live/intraday market data | Versioned snapshots are sufficient for first slice |
| Background job infrastructure | Add only after measured recommendation latency requires it |
| Tax, costs, and tax-lot optimization | Separate financial model |
| Native mobile application | Responsive web first |
| Public personalized advice launch | Requires Brazil-first legal/compliance gate |

## Edge Cases

- Duplicate account: show conflict; do not create a second identity.
- Expired session: redirect to login and preserve only a safe return path.
- Cross-account resource ID: return not-found or forbidden without revealing
  whether another account owns it.
- Incomplete profile: block recommendation generation and identify missing
  fields.
- No feasible allocation: show unavailable/infeasible state; do not fabricate
  a target.
- Missing snapshot metadata: reject publication and report the missing input.
- Zero portfolio value: reject drift calculation with a field-level error.
- Missing recommendation: show review unavailable; do not infer target weights.
- API timeout or failure: preserve prior valid state and show retry guidance.

## Release Gate

The shell may run as a private research or controlled pilot. Public
personalized recommendations, especially Premium stock-level output, require
the Brazil-first legal/compliance operating model documented in the dependent
recommendation specification.

## Success Criteria

- [ ] A new user can complete signup, onboarding, Basic recommendation,
  portfolio input, and drift review in one session.
- [ ] Reloading or signing back in preserves each completed state.
- [ ] Two accounts cannot access each other's records.
- [ ] Same profile, capital, snapshot, and model version reproduce the same
  Basic result.
- [ ] Review shows exact current/target/drift values and no order object.
- [ ] Basic users cannot unlock Premium behavior through client-side changes.
- [ ] The vertical slice passes keyboard, mobile, and unavailable-state checks.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| SHELL-01 | Public and protected surfaces | Specify | Pending |
| AUTH-01 | Create and access account | Specify | Pending |
| PROF-01 | Suitability onboarding | Specify | Pending |
| REC-01 | Basic recommendation | Specify | Pending |
| PORT-01 | Primary portfolio snapshot | Specify | Pending |
| REVIEW-01 | Portfolio drift review | Specify | Pending |
| API-01 | Stable API contract | Specify | Pending |
| ENT-01 | Entitlement boundaries | Specify | Pending |
| UX-01 | Accessibility and mobile baseline | Specify | Pending |

**Coverage:** 9 requirements, 9 mapped to stories, 0 mapped to design/tasks.

## Open Decisions

- Final authentication provider and email-verification policy.
- Final billing provider, prices, trial, grace period, and webhook contract.
- Legal operating model and public wording for personalized outputs.
- Whether Basic displays class BRL amounts or only percentages when capital is
  supplied.
- Production storage and refresh process for versioned market snapshots.
