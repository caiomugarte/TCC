# Roadmap

## Current milestone: Caio asset-class allocation (initial run complete)

1. Specify the allocation vocabulary, data rules, objective, and output contract.
2. Build pure allocation metrics and deterministic weight-grid search.
3. Add benchmark snapshot loaders and a common BRL daily return frame.
4. Add annual rebalancing, walk-forward evaluation, frontier, knee selection, and baselines.
5. Run the analysis when complete data snapshots are available and report limitations. **Complete for the first documented snapshot; review remains.**

## Later

- Add a variable risk-free series and real-return diagnostics.
- Add alternative international, fixed-income, and crypto proxies.
- Add taxes and operating costs only as a separately specified scenario.
- Revisit the Caio stock sleeve as a time-varying ex-ante strategy if the research question changes.

## SaaS MVP (planned)

**Goal:** Let one investor create a profile, receive a five-class target allocation with deeper Brazilian-stock analysis, track one portfolio, and review monthly drift through a paid account.

### Features

**Portfolio recommendation** - PLANNED

- Reuse existing five-class allocation and Brazilian-stock analysis flows.
- Combine class targets with stock weights inside the Brazilian-equity sleeve.
- Preserve profile answers, data snapshot, model version, assumptions, and output history.

**Portfolio tracking and rebalancing** - PLANNED

- Manual position and class-level input.
- Current-versus-target weights and drift thresholds.
- Contribution-first rebalance guidance without order execution.

**Subscription billing** - PLANNED

- Basic free model profiles plus one Premium subscription with monthly and annual billing during validation.
- Provider-neutral subscription state and webhook-driven entitlements.
- Stripe, Mercado Pago, and Asaas remain provider candidates pending a Brazil-first payment spike.

**Frontend shell** - PLANNED

- Reuse the responsive card/form language from `suitability.html` and the
  accessible allocation chart patterns from the generated dashboard.
- Build only the onboarding/profile, recommendation, portfolio, rebalance, and
  account/plan states needed for the first vertical slice.
- Keep the research HTML artifacts as references; they are not the SaaS app.

### Deferred

- B3/Área do Investidor import.
- Multiple portfolios, broker integrations, taxes, transaction costs, and automatic trading.
- Stock-level selection for FIIs, international assets, fixed income, and crypto.
