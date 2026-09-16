# Profile v1 to Premium Optimization Specification

**Status:** Ready for Design
**Issue:** [#3](https://github.com/caiomugarte/TCC/issues/3)
**Depends on:** `portfolio-recommendation-mvp`, `caio-asset-allocation`, `status-invest-fii-selection`, and `subscription-billing-mvp`

## Problem Statement

Profile v1 already stores questionnaire answers, computed dimensions, a
continuous suitability score, a generic profile, and restrictions. The current
recommendation path discards the score and restrictions, always uses the
static `caio` allocation profile, and never runs stock or FII selection for an
account.

Premium needs one account-owned execution path that resolves the persisted
profile into explicit, versioned inputs for class allocation, Brazilian-stock
selection, and FII selection. The result must be reproducible and must not
change the existing Basic behavior.

## Confirmed Direction

- Premium runs the stock and FII selectors for each requested recommendation.
- Fixed consensus artifacts are not a substitute for Premium personalization.
- `POST /v1/premium/recommendations` starts an asynchronous run and returns a
  run ID; the existing account-scoped recommendation read path exposes status
  and the completed result.
- Every profile-dependent stock and FII parameter is generated deterministically
  from the persisted form profile. `generic_profile` is a baseline label, not
  the downstream authority.
- The v1 restriction policy is fixed as follows: `evitar_cripto` and
  `evitar_exterior` set their class weight to zero; `priorizar_renda` requires
  at least 40% fixed income; `limitar_concentracao` requires class HHI <= 0.25;
  and `evitar_illiquidez` reuses the current stock/FII liquidity and size
  eligibility thresholds.
- Each run uses the latest compatible immutable snapshot manifest. No live data
  fetch occurs during optimization.
- A completed run is stored in one `RecommendationRun` with structured JSON
  for the resolved policy, full outputs, and provenance.
- Public launch is blocked; v1 is restricted to a private controlled pilot
  until Brazil-first legal review approves the operating model and wording.
- GA execution knobs remain versioned system configuration by default:
  population, generations, mutation rate, crossover rate, and run count are
  not personalized by questionnaire answers.
- Class-allocation parameters remain separate from stock and FII selection
  parameters.

## Goals

- [ ] Resolve a valid Profile v1 record into explicit allocation, stock, and
  FII policies.
- [ ] Pass the normalized continuous suitability score to the allocation
  engine and preserve the raw score for provenance.
- [ ] Enforce supported restrictions as hard constraints, without silently
  relaxing them.
- [ ] Run all three optimization layers for a Premium account from the
  application API.
- [ ] Persist an immutable, account-scoped result with policy, model, seed,
  and source-snapshot provenance.
- [ ] Keep the existing Basic route and output behavior unchanged.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Rewriting the allocation, stock, or FII optimization algorithms | This feature connects and configures existing engines. |
| Personalizing GA execution knobs | These remain controlled system configuration unless separately approved. |
| Refreshing market data or adding a scheduler | The run consumes an available, versioned snapshot; data refresh remains a separate workflow. |
| Selection inside international equity, fixed income, or crypto sleeves | Only Brazilian stocks and FIIs receive security-level selection. |
| Automatic orders, brokerage integration, taxes, or transaction costs | Existing product and compliance boundary. |
| Changing billing provider or entitlement lifecycle | Premium access continues to use the existing entitlement boundary. |
| A new visual recommendation experience | Only the minimum client/API wiring needed to invoke and read the Premium result is in scope. |

## Resolved Policy Contract

The resolver SHALL return a serializable `ResolvedOptimizationPolicy` with
these logical sections. Exact field names may follow the project's Python
conventions, but no section may be implicit in a named profile.

```text
profile:
  schema_version
  profile_revision
  raw_score
  score
  dimensions
  generic_profile
  restrictions
  applied_rules

allocation:
  score
  volatility_cap
  drawdown_cap
  crypto_risk_contribution_cap
  hhi_penalty
  risk_adjusted_weights
  class_constraints

stocks:
  selection_preset
  n_assets
  factor_weights
  liquidity_and_size_filters
  lambda_hhi
  system_ga_config

fiis:
  selection_preset
  n_assets
  factor_weights
  liquidity_and_size_filters
  lambda_hhi
  system_ga_config

provenance:
  policy_version
  model_versions
  source_snapshot_ids
  source_snapshot_hashes
  cutoff_date
  random_seed
```

The stock factor keys SHALL remain compatible with the existing scoring groups
(`liquidez`, `rent`, `value`, `growth`, `div`). FII factor keys SHALL remain
compatible with (`liquidity`, `size_cash`, `value`, `growth`, `dividend`).

`generic_profile` may identify the base preset used to resolve a policy, but it
SHALL NOT be the only Premium input and SHALL NOT be passed downstream as an
unresolved static profile such as `caio`.

All profile-dependent fields in the stock and FII sections SHALL be produced
by a versioned rule table that consumes the persisted form profile's score,
dimensions, and restrictions. The table SHALL emit explicit values for every
field required by the downstream engines; omitted values are invalid.

## User Stories

### P1: Resolve Profile v1 into Optimization Policy - MVP

**User Story:** As the Premium recommendation flow, I want a deterministic
policy resolver so that the same profile produces explicit and testable inputs
for every optimization layer.

**Acceptance Criteria:**

1. WHEN a valid persisted Profile v1 is resolved THEN the system SHALL use its
   normalized `score` in the range `0..1` as the allocation score.
2. WHEN a profile is resolved THEN the system SHALL preserve `raw_score`,
   dimensions, generic profile, restrictions, applied rules, and profile
   revision in the resolved policy.
3. WHEN a profile is resolved THEN the system SHALL produce explicit stock and
   FII `selection_preset`, `n_assets`, factor weights, liquidity/size filters,
   HHI penalty, and system GA configuration.
4. WHEN the same profile revision, policy version, and system configuration
   are resolved twice THEN the resulting policy SHALL be identical.
5. WHEN the policy is serialized THEN allocation parameters SHALL be
   distinguishable from stock/FII selection parameters.

**Independent Test:** Resolve fixture profiles for each generic profile and
compare the complete serialized policy against deterministic expected fixtures.

### P1: Apply Profile Restrictions as Hard Constraints - MVP

**User Story:** As an investor, I want my declared restrictions to constrain
the recommendation so that disallowed exposure is never returned silently.

**Acceptance Criteria:**

1. WHEN `evitar_cripto` is present THEN the Premium allocation SHALL assign
   zero weight to `crypto` and SHALL not select crypto exposure.
2. WHEN `evitar_exterior` is present THEN the Premium allocation SHALL assign
   zero weight to `international_equity`.
3. WHEN `priorizar_renda` is present THEN the resolved policy SHALL require at
   least 40% fixed income and SHALL expose that floor in provenance.
4. WHEN `limitar_concentracao` is present THEN the class allocation SHALL have
   HHI no greater than 0.25 and applicable security-selection concentration
   controls SHALL be applied from the versioned policy.
5. WHEN `evitar_illiquidez` is present THEN stock and FII eligibility filters
   SHALL exclude instruments below the configured liquidity threshold and
   record exclusion reasons.
6. WHEN multiple restrictions are present THEN the effective policy SHALL be
   their intersection; no restriction SHALL be relaxed to obtain a feasible
   result.
7. WHEN an explicit restriction conflicts with the allocation engine's default
   minimum class weight THEN the restriction SHALL take precedence, including
   allowing an excluded class to have zero weight.
8. WHEN no feasible policy satisfies the restrictions THEN the run SHALL be
   reported as unavailable and SHALL not publish a recommendation.

**Independent Test:** Resolve fixtures for every supported restriction and
assert class constraints, eligibility filters, conflict behavior, and
infeasibility reporting.

### P1: Run a Premium Recommendation from the Application - MVP

**User Story:** As an entitled Premium investor, I want to request a complete
recommendation without running offline scripts so that my profile is applied
to the actual optimization engines.

**Acceptance Criteria:**

1. WHEN an authenticated Premium account with a valid Profile v1 requests the
   Premium recommendation path THEN the system SHALL authorize the request,
   create a run record, and return a run ID without waiting for optimization to
   finish.
2. WHEN an accepted run is processed THEN the system SHALL resolve the
   account's profile and run class allocation, Brazilian-stock selection, and
   FII selection using that policy.
3. WHEN a Premium run completes THEN the result SHALL include five-class
   targets and selected stock and FII constituents with sleeve weight, total
   portfolio weight, and BRL target amount.
4. WHEN a run uses market data THEN all optimization layers SHALL use the
   documented common snapshot set and cutoff date for that run.
5. WHEN stock or FII selection is executed THEN the call SHALL receive
   explicit resolved values rather than relying on an implicit static profile.
6. WHEN a required input is missing, stale, materially incomplete, or
   infeasible THEN the endpoint SHALL return an unavailable/error result and
   SHALL not publish partial output.

**Independent Test:** Call the Premium route with injected deterministic
fixtures for profile, snapshots, and engines; assert all three engine calls
receive the resolved policy and the response contains complete targets.

### P1: Protect Premium Access and Account Ownership - MVP

**User Story:** As the product, I want Premium execution and stored results to
be entitlement- and account-scoped so that Basic users cannot access paid
optimization and accounts cannot read one another's data.

**Acceptance Criteria:**

1. WHEN a user without an active or grace-period Premium entitlement calls the
   Premium path THEN the system SHALL deny access before starting any
   optimization.
2. WHEN a Premium user requests a run THEN the system SHALL use only that
   account's profile and SHALL never accept another account identifier as an
   authority for profile or result ownership.
3. WHEN an account requests a stored recommendation by ID THEN the system
   SHALL return only a run owned by that account.
4. WHEN a Basic user uses the existing Basic path THEN the system SHALL retain
   its current generic-profile behavior and SHALL not execute the Premium
   stock/FII flow.

**Independent Test:** Exercise Premium denial, authorized execution, cross-
account reads, and Basic regression cases with two isolated accounts.

### P1: Persist Reproducible Recommendation Provenance - MVP

**User Story:** As an investor and operator, I want each Premium result to
record what generated it so that a completed recommendation can be reproduced
and audited later.

**Acceptance Criteria:**

1. WHEN a Premium run starts THEN the system SHALL bind it to an account,
   profile revision, policy version, model versions, system GA configuration,
   random seed, source snapshot IDs/hashes, and cutoff date.
2. WHEN a run completes THEN the system SHALL persist the resolved policy,
   complete class/stock/FII outputs, and provenance as immutable structured JSON
   owned by its `RecommendationRun`.
3. WHEN a newer run is created THEN previous completed runs SHALL remain
   readable and unchanged.
4. WHEN the same inputs, snapshots, policy version, and seed are replayed THEN
   the output SHALL be reproducible within the numerical tolerance declared by
   the engines.
5. WHEN a run fails THEN no partial recommendation SHALL be visible through
   the recommendation read path.

**Independent Test:** Complete two runs, replay the first fixture, compare
outputs and provenance, and verify that a failed run cannot be read as a
published recommendation.

## Edge Cases

- WHEN `nenhuma` is combined with another restriction THEN the resolver SHALL
  reject the invalid combination or apply one documented canonicalization; it
  SHALL not silently change the user's effective restrictions.
- WHEN a profile revision is missing, invalid, or withdrawn THEN the Premium
  path SHALL fail before optimization.
- WHEN a stock or FII universe has fewer eligible instruments than `n_assets`
  THEN the run SHALL report the shortage or use only a policy-approved reduced
  count with the change recorded; it SHALL not substitute unknown instruments.
- WHEN the Brazilian-stock class target is zero THEN stock output SHALL be
  empty and the result SHALL explain why no stock sleeve was allocated.
- WHEN the FII class target is zero THEN FII output SHALL be empty and the
  result SHALL explain why no FII sleeve was allocated.
- WHEN source snapshots have different date ranges THEN only documented common
  dates SHALL be used; no artificial price filling is allowed.
- WHEN two restrictions or a restriction and a risk cap make the problem
  infeasible THEN the run SHALL expose the diagnostic and SHALL not relax a
  limit silently.
- WHEN the selector or allocation engine raises an error or times out THEN the
  recommendation SHALL remain unpublished and the failure SHALL be traceable
  to its run record.

## API Boundary

The feature adds the Premium-protected command
`POST /v1/premium/recommendations`. It returns `202 Accepted` with a run ID;
optimization continues asynchronously. `GET /v1/recommendations/{id}` is the
status/result read path and SHALL return only the requesting account's run.
The completed result contract is the same regardless of run status.

The result SHALL expose:

- recommendation/run ID and status;
- profile revision, normalized score, generic profile, and applied restrictions;
- each class target weight and BRL amount;
- each selected stock and FII ticker, sleeve weight, total-portfolio weight,
  and BRL amount;
- assumptions, risk limits, data cutoff, source references, policy/model
  versions, and reproducibility seed.

The existing `GET /v1/premium` entitlement check and Basic recommendation routes
remain compatible unless a later approved design explicitly changes them.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| POL-01 | P1: Resolve Profile v1 | Specify | Pending |
| POL-02 | P1: Resolve Profile v1 | Specify | Pending |
| REST-01 | P1: Apply restrictions | Specify | Pending |
| REST-02 | P1: Apply restrictions | Specify | Pending |
| EXEC-01 | P1: Run Premium recommendation | Specify | Pending |
| EXEC-02 | P1: Run Premium recommendation | Specify | Pending |
| AUTH-01 | P1: Protect access and ownership | Specify | Pending |
| AUTH-02 | P1: Protect access and ownership | Specify | Pending |
| PROV-01 | P1: Persist provenance | Specify | Pending |
| PROV-02 | P1: Persist provenance | Specify | Pending |
| FAIL-01 | P1: Apply restrictions / Run recommendation | Specify | Pending |

**Coverage:** 11 requirements, 11 mapped to stories, 0 mapped to design/tasks.

## Design Constraints

1. The design SHALL define the versioned rule-table constants and fixtures for
   mapping score/dimensions to stock/FII presets, `n_assets`, factor weights,
   liquidity/size filters, and HHI penalties. The form is the source of every
   profile-dependent value.
2. The design SHALL define the immutable snapshot-manifest schema and the
   compatibility check used to choose the latest manifest. Missing compatible
   data fails the run; it does not trigger a live fetch or fixed-artifact
   fallback.
3. The design SHALL add the minimum persistence fields or JSON columns needed
   for one immutable `RecommendationRun`, including queued/running/failed/
   completed status and account ownership.
4. The design SHALL define the asynchronous execution lifecycle, including
   failure status, retry policy, duplicate requests, and client polling, while
   keeping the result contract above.
5. The design SHALL implement the private-pilot gate without changing the
   entitlement rule: only approved Premium accounts may start a run.

## Success Criteria

- [ ] A Premium account with a valid Profile v1 can request a complete
  recommendation without invoking an offline script.
- [ ] The normalized profile score, restrictions, and resolved policy are
  visible in the stored run and are used by the engines.
- [ ] Stock and FII selectors receive explicit, testable profile-derived
  configuration; no hidden static Premium profile is used.
- [ ] Basic users retain their current behavior and cannot invoke Premium
  execution.
- [ ] Cross-account reads are denied and failed runs never publish partial
  recommendations.
- [ ] A completed run records enough policy, model, seed, and snapshot metadata
  to reproduce it.
