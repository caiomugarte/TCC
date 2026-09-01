# Public landing references
> Four first-party references for Prumo's public SaaS landing page

Reviewed: 2026-08-10

Scope: Brazil-first portfolio decision SaaS for individual investors. These
are pattern references, not copy or visual assets.

## Reference set

### Warren — profile-led investment story

[Official model page](https://warren.com.br/nosso-modelo)

- **Value proposition:** starts with simpler investing, goals, fee
  transparency, and a portfolio matched to the investor profile.
- **Trust and compliance:** places transparency, risk language, legal entity,
  regulatory status, and the investor's responsibility in the same public
  surface.
- **Reusable pattern:** explain the decision journey before listing products:
  objective and profile, portfolio, then ongoing review.
- **Prumo caution:** Warren's regulated, managed-investment claims are not
  transferable to Prumo. Prumo must not imply custody, order execution, or
  guaranteed results.

### Kinvo — local plan conversion

[Official plans page](https://consolidador.kinvo.com.br/planos/)

- **Pricing and CTA:** shows Free at R$0, a Premium annual price, a free trial
  without a card, separate account and subscription actions, and a feature
  comparison.
- **Proof:** places customer testimonials after the plan value and repeats the
  signup action.
- **Reusable pattern:** show Basic/Premium value in BRL, with one clear action
  per plan and the account handoff visible.
- **Prumo caution:** avoid a long feature matrix before explaining the decision
  problem; exact prices remain a pilot decision in the product spec.

### Gorila — product-first proof

[Official homepage](https://gorila.com.br/)

- **Hero and hierarchy:** leads with a short investment-journey promise,
  identifies investors as an audience, places account/demo CTAs near the hero,
  and follows with product visuals and capabilities.
- **Trust and proof:** includes product examples, privacy and terms links, and
  company identification in the footer.
- **Reusable pattern:** show a believable product view near the first CTA.
  For Prumo, that view should show target allocation, current-versus-target
  drift, and a review action list.
- **Prumo caution:** do not promise B3, broker, or real-time coverage before
  those integrations exist.

### Linear — focused SaaS hierarchy

[Official homepage](https://linear.app/)

- **Hero:** states audience, product category, and primary promise in a short
  headline, then shows the product interface before deeper feature detail.
- **Visual hierarchy:** follows the hero with a small set of differentiators,
  product demonstrations, and repeated paths to pricing, login, and signup.
- **Reusable pattern:** one clear promise, one product artifact, three proof
  blocks, then conversion paths.
- **Prumo caution:** do not let animation, charts, or a dense dashboard hide
  the financial scope and compliance boundary.

## Prumo implications

- **Positioning:** use “Prumo — decisões mais claras para sua carteira.”
  Explain the MVP as profile, target allocation, and monthly drift review—not
  as a broker or return predictor. See the [recommendation
  boundary](../.specs/features/portfolio-recommendation-mvp/spec.md) and
  [tracking boundary](../.specs/features/portfolio-tracking-mvp/spec.md).
- **Page order:** hero and CTA; three-step explanation; allocation/dashboard
  preview; Basic/Premium comparison; methodology and trust; FAQ and final CTA.
- **Trust:** state data cutoff, model/profile version, assumptions, risks, and
  “no order execution.” Add legal identity, privacy, terms, and the required
  Brazil-first review before public personalized recommendations.
- **Conversion:** let **Ver planos** and **Criar conta** lead to account
  creation. After login, route users to the suitability form; send paid users
  to hosted checkout and grant access only after verified payment state. See
  the [billing flow](../.specs/features/subscription-billing-mvp/spec.md).
- **Proof:** use real example outputs and methodology first. Add testimonials,
  usage numbers, badges, or ratings only when Prumo can substantiate them.

## What to avoid

- Return, performance, or “best investment” promises.
- Vague “AI” messaging without showing the decision produced.
- Fake social proof, unsupported regulatory badges, or copied competitor
  metrics.
- Brokerage language that suggests custody, automatic trades, or live market
  coverage.
- A dense feature/pricing table before the visitor understands the recurring
  problem Prumo solves.

## Research limitations

This is a qualitative review of four official pages, not a conversion study.
Prices, customer counts, claims, and page layouts can change. The references
serve different markets and business models, so their regulatory language and
pricing are examples only. No user testing, analytics, or independent claim
verification was performed.

Updated: 2026-08-10
