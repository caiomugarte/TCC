# Subscription Billing MVP Specification

**Status:** Draft
**Depends on:** account access and product entitlements

## Problem Statement

The product needs recurring revenue without coupling portfolio logic to a payment vendor. MVP billing should sell access to analysis and tracking, map asynchronous payment events to entitlements, and support cancellation and failed-payment handling.

## Pricing Hypothesis

- Buyer: individual Brazilian investor.
- Value metric: one active primary portfolio with recurring analysis, not assets under management.
- Model proposal: a free Basic tier plus one paid Premium subscription during validation.
- Basic: generic conservative, moderate, or aggressive model profile; class-level allocation and basic portfolio diagnostic.
- Premium: continuous customized profile, deep Brazilian-stock analysis, exact stock and class target amounts, and rebalance tracking.
- Annual plan: test a 15–20% discount after willingness-to-pay evidence.
- No AUM fee, broker commission, or affiliate revenue in MVP.

Exact price remains a pilot decision. Provider choice remains a technical/business spike.

## Trial and validation proposal

- Pilot: start with a small paid-first cohort using a founder price or manual
  invoice. Validate willingness to pay before optimizing a free funnel.
- Public launch: keep Basic available without a trial timer, then offer a
  seven-day Premium trial only after the user completes the profile and adds a
  portfolio. This exposes the value before payment without making the generic
  model profiles feel like a bait-and-switch.
- The exact trial payment-method rule remains open. If no card is collected,
  the product must explicitly end or pause access at trial expiry.

## Goals

- [ ] Let a user start a paid subscription through hosted checkout.
- [ ] Grant or revoke product access from verified provider events.
- [ ] Handle cancellation, renewal failure, grace period, and reactivation.
- [ ] Keep billing provider identifiers separate from product subscription state.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Marketplace or revenue split | No third-party seller in MVP |
| AUM or performance-based fee | Conflicts and regulatory complexity |
| Multiple plans and complex add-ons | Validate one value proposition first |
| Custom card-data storage | Payment provider handles sensitive payment data |
| Invoicing/tax automation beyond provider capability | Separate accounting requirement |

## User Stories

### P1: Subscribe to plan ⭐ MVP

**User Story:** As a user, I want to subscribe monthly or annually so that I can access recommendation and tracking features.

**Acceptance Criteria:**

1. WHEN the user selects a plan THEN the system SHALL open hosted checkout for that plan.
2. WHEN payment is confirmed by the provider THEN the system SHALL grant the matching entitlement.
3. WHEN checkout is abandoned or rejected THEN the system SHALL not grant paid access.

**Independent Test:** Complete a sandbox checkout and verify entitlement changes only after the provider confirmation event.

### P1: Process subscription events ⭐ MVP

**User Story:** As the product, I want to process subscription events idempotently so that access reflects payment state.

**Acceptance Criteria:**

1. WHEN a valid webhook is received THEN the system SHALL verify its authenticity before changing access.
2. WHEN the same event is received twice THEN the system SHALL apply it once.
3. WHEN a renewal fails THEN the system SHALL apply the configured grace state and notify the user.
4. WHEN a subscription is canceled THEN the system SHALL keep access until the paid period ends, unless provider state requires immediate termination.

**Independent Test:** Replay the same sandbox event and verify one state transition and one audit record.

### P1: Manage subscription ⭐ MVP

**User Story:** As a subscriber, I want to cancel or reactivate my subscription so that I control recurring charges.

**Acceptance Criteria:**

1. WHEN the user chooses cancel THEN the system SHALL request cancellation through the provider's supported flow.
2. WHEN cancellation is confirmed THEN the system SHALL show end date and access state.
3. WHEN the user returns to an active plan THEN the system SHALL restore paid entitlements from provider state.

**Independent Test:** Cancel and reactivate a sandbox subscription and verify access state after each provider event.

## Provider Decision Spike

Compare Stripe, Mercado Pago, and Asaas before implementation using:

- BRL support;
- card, Pix, and boleto suitability for recurring plans;
- hosted checkout and customer self-service;
- webhook quality and retry behavior;
- sandbox availability;
- fees, settlement, refunds, and Brazilian tax/accounting needs;
- operational fit for a Brazil-first launch and future international sales.

Current documentation indicates Stripe supports subscription Checkout and
webhooks, but Stripe documents Pix Automático as unavailable for Brazilian
accounts except by invitation; do not assume Pix recurring at launch. Mercado Pago documents
recurring subscriptions, free trials, automatic retries, and local payment
options. Asaas supports subscriptions and Brazilian payment methods, but its
subscription lifecycle is tracked through billing webhooks rather than a
separate subscription-webhook stream. [Stripe subscription webhooks](https://docs.stripe.com/billing/subscriptions/webhooks?locale=pt-BR), [Stripe Pix](https://docs.stripe.com/payments/pix), [Mercado Pago subscriptions](https://www.mercadopago.com.br/developers/pt/docs/subscriptions/overview), [Asaas subscriptions](https://docs.asaas.com/docs/subscriptions).

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| BILL-01 | P1: Subscribe to plan | Specify | Pending |
| BILL-02 | P1: Process subscription events | Specify | Pending |
| BILL-03 | P1: Manage subscription | Specify | Pending |
| BILL-04 | Provider decision spike | Design | Pending |

**Coverage:** 4 requirements, 4 mapped to stories, 0 mapped to implementation tasks.

## Success Criteria

- [ ] Paid pilot can subscribe without manual access changes.
- [ ] Duplicate and out-of-order webhook events do not corrupt entitlements.
- [ ] Cancellation and failed renewal behavior is visible and testable.
- [ ] Switching provider does not require changing recommendation or tracking logic.

## Open Decisions

- Exact monthly and annual price.
- Whether to run the paid-first pilot before the public Basic/Premium funnel.
- Trial payment-method and expiry behavior.
- Stripe, Mercado Pago, or Asaas as the first provider.
- Grace-period length and user communication channels.
