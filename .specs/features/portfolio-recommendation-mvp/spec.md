# Portfolio Recommendation MVP Specification

**Status:** Draft
**Depends on:** `caio-asset-allocation`, existing Brazilian-stock selection pipeline

## Problem Statement

An investor can answer a profile questionnaire, but the current repository only produces research outputs through local scripts. The MVP must turn those outputs into one understandable recommendation: a general target across five asset classes, with deeper selection and weighting for the Brazilian-stock sleeve.

The product must preserve enough input, data, and model provenance for the user to understand what generated each result. It must not imply guaranteed returns or execute trades.

## Product and compliance lane

The intended paid output is individualized: it uses the user's profile and
investable capital to produce target weights and target amounts for asset
classes and supported stocks. This is treated as the personalized-advice path
until Brazil-first legal review says otherwise.

The result contract should expose both:

- each class target percentage and BRL amount;
- each stock's weight inside the Brazilian-stock sleeve, weight in the total
  portfolio, and BRL target amount.

The amount is a target allocation calculated from declared investable capital,
not an order, guarantee, or instruction to execute automatically. Tracking may
show a tolerance band around the target; the target itself remains explicit.

## Goals

- [ ] Generate a personalized five-class target allocation for one user.
- [ ] Select and weight Brazilian stocks inside the Brazilian-equity target sleeve.
- [ ] Convert class and stock targets into BRL amounts from investable capital.
- [ ] Explain assumptions, risks, data cutoff, and model version for every run.
- [ ] Produce a stable result that can be stored and compared over time.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Stock-level selection for FIIs, international assets, fixed income, or crypto | General allocation only for those sleeves in MVP |
| Automatic order execution | Compliance and brokerage scope |
| Tax and transaction-cost optimization | Requires a separate financial/tax model |
| Intraday or guaranteed predictions | Not needed for monthly portfolio decisions |
| Free-form LLM asset selection | Deterministic research rules remain the decision source |

## User Stories

### P1: Create investor profile ⭐ MVP

**User Story:** As an investor, I want to answer the profile form so that the system can constrain recommendations to my objectives, capacity, horizon, liquidity, knowledge, and restrictions.

**Acceptance Criteria:**

1. WHEN the user submits the form THEN the system SHALL validate required answers and reject incomplete or invalid values.
2. WHEN the form is accepted THEN the system SHALL store raw answers, computed dimensions, score, restrictions, and profile version.
3. WHEN a recommendation is generated THEN the system SHALL reference the exact profile version used.

**Independent Test:** Submit a valid form, reload the account, and reproduce the same stored profile.

### P1: Generate five-class allocation ⭐ MVP

**User Story:** As an investor, I want a target allocation across stocks, FIIs, international exposure, fixed income, and crypto so that I can see how my investable capital should be distributed.

**Acceptance Criteria:**

1. WHEN a valid profile and supported market snapshot exist THEN the system SHALL generate non-negative class weights that sum to 100%.
2. WHEN investable capital is provided THEN the system SHALL return a target amount for each class and SHALL use that capital rather than total declared patrimony as the calculation base.
3. WHEN a candidate violates the profile's hard risk constraints THEN the system SHALL exclude it from the selected target.
4. WHEN a required snapshot or class input is missing THEN the system SHALL report the run as unavailable instead of silently filling data.

**Independent Test:** Run the existing deterministic allocation core with a synthetic snapshot and verify valid weights, risk constraints, and failure on missing data.

### P1: Analyze Brazilian stocks inside equity sleeve ⭐ MVP

**User Story:** As an investor, I want deeper analysis of Brazilian stocks so that the stock portion of my target allocation has explainable constituents and weights.

**Acceptance Criteria:**

1. WHEN the stock universe and profile parameters are valid THEN the system SHALL return selected supported tickers and weights for the Brazilian-stock sleeve.
2. WHEN the class target is `w` THEN each stock SHALL expose its sleeve weight, total-portfolio weight after applying `w`, and BRL target amount.
3. WHEN an asset is unavailable, stale, or disallowed by a restriction THEN the system SHALL exclude it and record the reason.

**Independent Test:** Run the existing stock-selection pipeline against a fixed fixture and verify ticker eligibility, weight validity, and sleeve-to-total conversion.

### P1: Explain and store recommendation run ⭐ MVP

**User Story:** As an investor, I want to understand and revisit my result so that I can distinguish a model output from a promise of performance.

**Acceptance Criteria:**

1. WHEN a run completes THEN the system SHALL show class targets, stock constituents, assumptions, risks, data cutoff, and model/profile versions.
2. WHEN a run completes THEN the system SHALL store immutable input and output references.
3. WHEN a run cannot complete THEN the system SHALL show a useful error and SHALL not publish a partial recommendation.

**Independent Test:** Reopen a stored run after a newer run exists and verify the old result remains unchanged.

### P2: Refresh recommendation

**User Story:** As an active subscriber, I want to rerun my recommendation with a newer snapshot so that my report stays current.

**Acceptance Criteria:**

1. WHEN a newer supported snapshot is available THEN the user SHALL be able to request a new run.
2. WHEN a new run completes THEN the system SHALL preserve the previous run for comparison.

## Edge Cases

- Profile answers conflict with hard constraints: keep the stricter constraint and explain it.
- No feasible allocation exists: show infeasibility and closest diagnostic candidate; do not invent a target.
- Stock sleeve is zero: omit stock-level recommendations and explain that no equity sleeve was allocated.
- Stock sleeve is too small for the configured number of assets: reduce the usable universe or report infeasibility.
- Data snapshot dates differ: align only documented common dates.
- User asks for a result after profile or data expiry: require refresh before publishing.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| PROF-01 | P1: Create investor profile | Specify | Pending |
| ALLOC-01 | P1: Generate five-class allocation | Specify | Pending |
| STOCK-01 | P1: Analyze Brazilian stocks | Specify | Pending |
| RECO-01 | P1: Explain and store run | Specify | Pending |
| RECO-02 | P2: Refresh recommendation | Specify | Pending |

**Coverage:** 5 requirements, 5 mapped to stories, 0 mapped to implementation tasks.

## Release Gate

Personalized stock recommendations for public paid users require a Brazil-first legal/compliance review. Until approved, this feature is limited to private research or a controlled pilot with approved product wording and operating responsibility.

## Success Criteria

- [ ] A pilot user can complete profile and receive one stored recommendation.
- [ ] Recommendation combines five-class targets with stock-level sleeve output.
- [ ] Every published result shows data cutoff, assumptions, risks, and versions.
- [ ] Same profile, input snapshot, and model version reproduce same output.

## Open Decisions

- Final product wording: portfolio analytics, regulated advice, or partner-delivered advice.
- Legal operating model for personalized recommendations: own authorization, authorized partner, or private research pilot.
- Initial tolerance band for tracking; exact target amounts remain the recommendation output.
- Initial supported stock universe and minimum data freshness.
