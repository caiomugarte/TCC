# Portfolio Tracking and Rebalancing MVP Specification

**Status:** Draft
**Depends on:** `portfolio-recommendation-mvp`

## Problem Statement

A one-time recommendation has weak recurring value. The MVP needs one current portfolio snapshot so the user can compare actual weights with target weights and review the next rebalance action without requiring live brokerage execution.

Tracking consumes the recommendation's explicit class and stock targets. It
must distinguish a stock's weight inside the Brazilian-stock sleeve from its
weight in the total portfolio, and it must show the corresponding BRL gap
when valuation data is available.

## Goals

- [ ] Let one user maintain one primary portfolio.
- [ ] Support manual position input and CSV import.
- [ ] Compare current and target allocation at class and stock level.
- [ ] Produce contribution-first rebalance guidance and preserve history.

## Out of Scope

| Feature | Reason |
| --- | --- |
| B3/Área do Investidor import | Separate integration after manual MVP validation |
| Multiple portfolios or households | One primary portfolio keeps first release small |
| Automatic orders or broker credentials | Execution and security scope |
| Tax lots, realized gains, and IR calculation | Separate tax feature |
| Dividend and corporate-action reconciliation | Separate transaction-ledger feature |
| Real-time tracking | Daily/monthly decision cycle is sufficient initially |

## User Stories

### P1: Maintain portfolio snapshot ⭐ MVP

**User Story:** As an investor, I want to enter my holdings so that the system can compare my current portfolio with my target.

**Acceptance Criteria:**

1. WHEN the user adds a stock position THEN the system SHALL validate ticker, quantity, and supported asset class.
2. WHEN the user adds a non-stock sleeve THEN the system SHALL accept a class-level value for supported MVP classes.
3. WHEN a snapshot is saved THEN the system SHALL store source, timestamp, currency, and input values.
4. WHEN the user imports a CSV THEN the system SHALL show validation errors per row and SHALL not partially replace the previous valid snapshot.

**Independent Test:** Create a portfolio manually and through CSV, then verify identical normalized positions.

### P1: Show current versus target ⭐ MVP

**User Story:** As an investor, I want to see drift from my recommendation so that I know what requires attention.

**Acceptance Criteria:**

1. WHEN a portfolio snapshot and recommendation exist THEN the system SHALL show current weight, target weight, and signed drift for each class and stock sleeve.
2. WHEN no compatible target exists THEN the system SHALL show tracking as unavailable rather than infer a target.
3. WHEN prices or values are stale THEN the system SHALL display the cutoff and stale-data warning.

**Independent Test:** Use a fixed target and two snapshots; verify drift calculations and missing-target behavior.

### P1: Generate rebalance review ⭐ MVP

**User Story:** As an investor, I want a small action list so that I can decide how to bring my portfolio closer to target.

**Acceptance Criteria:**

1. WHEN absolute drift is inside the configured band THEN the system SHALL mark the item as within range.
2. WHEN drift exceeds the band THEN the system SHALL mark the item as underweight or overweight.
3. WHEN new contribution amount is provided THEN the system SHALL prioritize allocating new money to underweight items before suggesting sales.
4. WHEN a sale may be required THEN the system SHALL label it as a review action and SHALL not create an order.

**Independent Test:** Run a portfolio with known drifts and contribution amount; verify action labels and no order object is emitted.

### P2: Preserve tracking history

**User Story:** As an investor, I want historical snapshots and reports so that I can see how my portfolio changed.

**Acceptance Criteria:**

1. WHEN a new snapshot is saved THEN the previous valid snapshot SHALL remain available.
2. WHEN a new recommendation is generated THEN the dashboard SHALL retain the prior target for comparison.

## Edge Cases

- Duplicate ticker rows: combine only when the user confirms, or reject with a clear message.
- Unknown ticker: reject row; do not silently map to another asset.
- Zero or negative quantity/value: reject input.
- Missing price: preserve position but mark current valuation unavailable.
- Portfolio value is zero: show input error instead of dividing by zero.
- Target has stock sleeve but no stock positions: show full underweight state, not an empty chart.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| PORT-01 | P1: Maintain portfolio snapshot | Specify | Pending |
| PORT-02 | P1: Show current versus target | Specify | Pending |
| REBAL-01 | P1: Generate rebalance review | Specify | Pending |
| HIST-01 | P2: Preserve tracking history | Specify | Pending |

**Coverage:** 4 requirements, 4 mapped to stories, 0 mapped to implementation tasks.

## Success Criteria

- [ ] Pilot user can create or import a valid portfolio without support.
- [ ] Current-versus-target view is reproducible from stored snapshots.
- [ ] Rebalance review prefers contributions and never executes trades.
- [ ] A second snapshot does not destroy the first snapshot.

## Open Decisions

- Initial drift band; proposed default is 5 percentage points.
- Whether class-level non-stock values are enough for first release.
- Whether tracking refresh is user-triggered or scheduled monthly.
- Whether the basic plan receives class-level tracking or only the premium plan.
