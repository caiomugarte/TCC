# Frontend Surface
> Existing visual prototypes; two SaaS surfaces, one MVP frontend app

Entry: `suitability.html`; generated dashboard:
`outputs/caio-allocation-dashboard.html`

Existing:
- Standalone root HTML with inline CSS and JavaScript.
- `landing.html` is a public static marketing prototype for Prumo; its CTAs
  still use in-page anchors until account routes exist.
- Questionnaire → continuous score/profile weights → result table → Python
  dictionary export.
- Prototype explicitly warns that it is not regulatory suitability or advice.
- Generated HTML/SVG dashboard exposes allocation charts and accessible labels.

Product surfaces:
- Public marketing surface — explain Prumo, Basic/Premium value, pricing,
  compliance wording, and entry CTA.
- Authenticated application — profile onboarding, recommendation, portfolio,
  drift/rebalance review, and account/plan state.
- MVP architecture: two surfaces in one frontend codebase/deployment, with
  public and protected routes/layouts. Separate apps can wait until SEO,
  release cadence, team ownership, or performance requires it.

Payment and account flow:
- Plan CTA → account creation/login → provider-hosted checkout → verified
  provider webhook → product entitlement → protected feature access.
- Checkout redirect alone does not grant Premium access.
- One account owns a versioned suitability profile, recommendation runs, one
  primary portfolio, and subscription state.

Suitability role:
- `suitability.html` belongs to the authenticated app's onboarding/profile
  surface, not the marketing page.
- Current prototype calculates in-browser and exports a Python dictionary; it
  does not authenticate, persist answers, or map output to an account.

SaaS references:
- `.specs/project/PROJECT.md:23-36` — Brazil-first portfolio decision
  assistant; one user/primary portfolio; five classes; manual input; monthly
  drift review; recurring subscription; compliance gate.
- `.specs/project/ROADMAP.md:18-57` — MVP goal and planned frontend states:
  onboarding/profile, recommendation, portfolio, rebalance, account/plan.
- `.specs/features/portfolio-recommendation-mvp/spec.md` — five-class target,
  Brazilian-stock sleeve, target amounts, provenance, no order execution.
- `.specs/features/portfolio-tracking-mvp/spec.md` — one portfolio, manual/CSV
  input, current-versus-target drift, contribution-first review, no trades.
- `.specs/features/subscription-billing-mvp/spec.md` — Basic/Premium,
  monthly/annual billing, provider-neutral entitlements.

Working product name:
- `Prumo` — balance and direction metaphor; fits allocation plus drift review;
  avoids promising returns.
- Positioning draft: “Prumo — decisões mais claras para sua carteira.”
- Trademark, domain, and app-store availability not checked.

MVP frontend boundary:
Public routes: landing, pricing, login/signup, checkout return/status.
Protected routes: onboarding/profile, recommendation, portfolio input, drift/
rebalance review, account/plan.

Profile re-entry:
- `web/components/onboarding/questionnaire.tsx:OnboardingQuestionnaire()` loads
  `getProfile()` before rendering the form when user revisits onboarding.
- Existing answers, restrictions, investable capital, and prior consent hydrate
  from account-scoped `GET /v1/profile`; missing profile keeps blank defaults.
- Profile-load failure leaves form usable and shows fallback guidance.

Deferred: B3 import, multiple portfolios, broker integrations, order execution,
intraday tracking, and complex plan/add-on packaging.

Not found:
- `package.json`, React/Next/Vite app, web API, authentication, persistence,
  subscription UI, or billing integration.

Boundary: reuse prototype visual language and questionnaire/chart ideas; SaaS
shell, account state, API contracts, entitlements, and production legal links
still need Design/implementation. `landing.html` is the first public-surface
prototype.

Updated: 2026-08-12
