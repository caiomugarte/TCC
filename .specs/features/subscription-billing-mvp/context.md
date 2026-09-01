# Context

**Status:** Draft; awaiting user confirmation before Design.

## User direction

- Recurring payment is required for the SaaS path.
- Stripe and Mercado Pago are candidates; Asaas should also be considered.
- Generic conservative, moderate, and aggressive profiles may be Basic, while a
  customized profile may be Premium.
- Trial versus paid-first is not decided.

## Proposed packaging

### Basic

- Generic model profile: conservative, moderate, or aggressive.
- Five-class model allocation.
- Basic portfolio diagnostic.

### Premium

- Continuous customized profile from the questionnaire.
- Deep Brazilian-stock analysis.
- Exact class and stock target amounts.
- Portfolio tracking and rebalance review.

## Proposed validation sequence

1. Run a small paid-first pilot with a founder price or manual collection.
2. Measure whether users complete the profile, add a portfolio, and accept the
   recommendation before building a larger acquisition funnel.
3. Publicly launch Basic plus a seven-day Premium trial after profile and
   portfolio completion.

## Provider notes

- Keep product entitlements independent of the provider.
- Implement only one provider after a short Brazil-first spike.
- Mercado Pago and Asaas fit a Brazil-first payment mix; Stripe is attractive
  if international expansion or its developer ecosystem is a priority.
- The domain should store provider, external customer/subscription IDs,
  entitlement status, period end, and processed event IDs; it must not store
  card data.

## Open decisions

- Monthly and annual prices.
- Paid-first cohort size and founder offer.
- Trial card requirement and expiry behavior.
- First provider: Mercado Pago, Asaas, or Stripe.
- Grace period and failed-payment messaging.
