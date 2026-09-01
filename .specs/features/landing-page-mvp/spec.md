# Prumo Public Landing Page MVP Specification

**Status:** Draft  
**Depends on:** SaaS MVP direction, `subscription-billing-mvp`, Prumo identity v0.1

**Copy and wireframe:** `docs/prumo-landing-wireframe.md`
**Implementation:** `web/app/(public)/page.tsx` and
`web/components/landing-motion.tsx`; `landing.html` remains the static content
reference.

## Problem Statement

Prumo has research prototypes but no public page that explains the product,
sets the product boundary, compares Basic and Premium, and gives a visitor a
clear path to create an account. The landing page must build trust without
presenting historical research or personalized outputs as guaranteed advice.

## Product Lane

The page is public marketing for a Brazil-first portfolio decision SaaS. It
does not calculate a profile, publish a user recommendation, execute orders,
or grant paid access. Those actions belong to the authenticated application
and billing flow.

Page language: Brazilian Portuguese (`pt-BR`).

Primary conversion: create an account.  
Secondary conversion: view plans or methodology.

## Scope

- One responsive public landing page.
- Prumo v0.1 logo, typography, colors, and UI tokens.
- Hero, product explanation, product preview, plans, trust/methodology, FAQ,
  final CTA, and legal footer.
- Basic/Premium value framing from the existing billing hypothesis.
- CTA handoff to account creation/login; hosted checkout remains a separate
  authenticated/billing surface.
- Motion limited to non-essential opacity/transform entrance and reveal effects;
  GSAP only orchestrates the small client-side timeline and scroll reveals.

## User Stories

### LAND-01 — Understand Prumo ⭐ MVP

**User Story:** As a first-time visitor, I want to understand what Prumo does
and who it is for so that I can decide whether to create an account.

**Acceptance Criteria:**

1. WHEN the hero is visible THEN it SHALL identify Prumo as a portfolio
   decision assistant for individual investors.
2. WHEN the visitor reads the first viewport THEN the primary CTA SHALL be
   visible and the product SHALL not be described as a broker, custodian, or
   guaranteed-return service.
3. The page SHALL use plain Brazilian Portuguese and avoid unexplained
   technical terms.

### LAND-02 — Explain the product journey ⭐ MVP

**User Story:** As a visitor, I want to see how Prumo works so that the value
   is clear before I subscribe.

**Acceptance Criteria:**

1. The page SHALL explain three steps: mapear perfil, definir alocação-alvo,
   and revisar desvios mensalmente.
2. The page SHALL state that the MVP supports one primary portfolio, manual
   input first, and no order execution.
3. The page SHALL identify the five allocation classes without implying that
   every class receives stock-level selection.

### LAND-03 — Show credible product evidence ⭐ MVP

**User Story:** As a visitor, I want to see the product output so that I can
   judge whether the promise is concrete.

**Acceptance Criteria:**

1. The page SHALL show a real research artifact or a faithful product preview
   covering target allocation, current-versus-target drift, or rebalance review.
2. Any example data SHALL be labeled as an example, historical result, proxy,
   or research output where applicable.
3. The preview SHALL not imply that the visitor already has a personalized
   recommendation.

### LAND-04 — Compare plans ⭐ MVP

**User Story:** As a visitor, I want to understand Basic and Premium so that I
   can choose the appropriate next step.

**Acceptance Criteria:**

1. Basic SHALL describe generic conservative, moderate, or aggressive profiles,
   class-level allocation, and a basic portfolio diagnostic.
2. Premium SHALL describe a customized profile, deeper Brazilian-stock
   analysis, exact class/stock target amounts, and rebalance tracking.
3. The page SHALL not invent prices, trial rules, or payment methods while
   those decisions remain open.
4. A paid-plan CTA SHALL lead to account creation/login before hosted checkout;
   a redirect alone SHALL not be treated as proof of payment.

### LAND-05 — Build trust without overclaiming ⭐ MVP

**User Story:** As a visitor, I want to understand the method and limits so
   that I can evaluate Prumo responsibly.

**Acceptance Criteria:**

1. The page SHALL explain the deterministic research/model basis at a level
   understandable to an investor.
2. The page SHALL expose assumptions, risks, data freshness/cutoff, and the
   absence of automatic order execution.
3. The page SHALL include real privacy, terms, contact, and responsible-party
   links before public launch; placeholders SHALL not be presented as complete
   legal coverage.
4. Personalized recommendation copy SHALL remain behind the documented
   Brazil-first legal/compliance gate.

### LAND-06 — Use restrained, accessible motion ⭐ MVP

**User Story:** As a visitor, I want motion to support hierarchy without
   slowing me down or excluding me.

**Acceptance Criteria:**

1. Motion SHALL be limited to opacity/transform transitions, small hover/focus
   states, and optional one-time hero/product reveals.
2. No information SHALL depend on animation, autoplay, parallax, looping
   charts, or a moving stock ticker.
3. `prefers-reduced-motion: reduce` SHALL disable or materially reduce
   non-essential motion.
4. Focus states, keyboard navigation, contrast, semantic headings, and image
   alternatives SHALL remain usable without motion.

### LAND-07 — Work across devices ⭐ MVP

**User Story:** As a visitor, I want the page to work on mobile and desktop so
   that I can understand and join from either device.

**Acceptance Criteria:**

1. The page SHALL use `lang="pt-BR"`, responsive layout, and no horizontal
   scrolling at supported viewport sizes.
2. Navigation, plan cards, CTAs, preview content, and footer links SHALL remain
   usable on narrow screens.
3. The page SHALL load without requiring authenticated state or live market
   data.

## Page Structure

1. Header: Prumo mark, `Como funciona`, `Planos`, `Metodologia`, `Entrar`.
2. Hero: one promise, one product sentence, `Criar conta` primary CTA, and
   `Ver planos` secondary CTA.
3. Three-step product journey.
4. Product preview using allocation and drift concepts.
5. Basic/Premium comparison.
6. Methodology, assumptions, risks, and compliance boundary.
7. FAQ.
8. Final CTA and legal footer.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Authenticated profile form | Belongs to the application onboarding flow |
| Recommendation generation | Requires account, stored profile, and backend contract |
| Payment provider integration/webhooks | Separate subscription-billing implementation |
| B3/broker import or order execution | Deferred SaaS scope |
| Live market data on the public page | Adds freshness and performance complexity |
| CMS, blog, multilingual content | Not needed for first conversion test |
| Testimonials, ratings, badges, or usage numbers | Require substantiated evidence |
| Complex animation system | Small GSAP boundary covers MVP motion; no parallax, autoplay, or looping charts |

## Success Criteria

- A first-time visitor can state what Prumo does, who it serves, and the next
  action after reading the hero.
- A visitor can distinguish Basic from Premium without seeing unsupported
  pricing or promises.
- Product preview, methodology, risks, and compliance boundary are visible
  before the final CTA.
- Page remains understandable and operable with reduced motion, keyboard
  navigation, and mobile layout.
- No authenticated state, live market request, payment secret, or order object
  is needed to render the public page.

## Open Decisions

- Exact monthly/annual prices and trial policy.
- Final legal operating model and responsible party for personalized outputs.
- Final vector logo asset and typography choice.
- Production domain and route names.
- Whether the first CTA should emphasize free Basic signup or plan selection.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| LAND-01 | Understand Prumo | Specify/Execute | Implemented |
| LAND-02 | Explain product journey | Specify/Execute | Implemented |
| LAND-03 | Show credible product evidence | Specify/Execute | Implemented |
| LAND-04 | Compare plans | Specify/Execute | Implemented |
| LAND-05 | Build trust without overclaiming | Specify/Execute | Partial: legal links pending |
| LAND-06 | Use restrained, accessible motion | Specify/Execute | Implemented |
| LAND-07 | Work across devices | Specify/Execute | Implemented: source-level |

**Coverage:** 7 requirements, 7 mapped to stories, 6 implemented, 1 partial.
