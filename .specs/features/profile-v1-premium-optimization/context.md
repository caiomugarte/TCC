# Profile v1 to Premium Optimization Context

**Gathered:** 2026-09-15
**Spec:** `.specs/features/profile-v1-premium-optimization/spec.md`
**Status:** Ready for design

## Feature Boundary

Connect a persisted Profile v1 record to an account-owned Premium execution
that resolves every profile-dependent stock/FII and class-allocation input,
runs the stock and FII selectors for each accepted request, runs the class
allocation, and stores a reproducible result. Basic behavior remains unchanged.

## Implementation Decisions

### Execution lifecycle

- `POST /v1/premium/recommendations` is asynchronous and returns `202 Accepted`
  with a run ID.
- `GET /v1/recommendations/{id}` is the account-scoped status/result path.
- Each accepted request creates a new run; fixed consensus artifacts do not
  replace selector execution.

### Profile-to-policy resolution

- The persisted form profile is the source for every profile-dependent input
  required by the stock and FII engines.
- The resolver consumes score, dimensions, and restrictions through a
  versioned deterministic rule table.
- `generic_profile` identifies the baseline preset but is not sufficient to
  configure Premium and is never passed downstream as an unresolved static
  profile.
- GA runtime knobs remain system-controlled and versioned.

### Restriction policy

- `evitar_cripto` sets the crypto class weight to zero.
- `evitar_exterior` sets the international-equity class weight to zero.
- `priorizar_renda` requires at least 40% fixed income.
- `limitar_concentracao` requires class HHI <= 0.25 and applies relevant sleeve
  concentration controls.
- `evitar_illiquidez` reuses the current stock/FII liquidity and size thresholds.
- Multiple restrictions intersect. No constraint is silently relaxed.

### Data boundary

- Every run uses the latest compatible immutable snapshot manifest.
- Stock selection, FII selection, and class allocation share the manifest's
  common cutoff.
- No live data fetch, forward-fill, or fixed-artifact fallback occurs inside a
  Premium run.

### Persistence

- One `RecommendationRun` owns account, status, profile revision, and immutable
  structured JSON for policy, complete outputs, and provenance.
- Provenance includes policy/model versions, system GA configuration, seed,
  snapshot IDs and hashes, and cutoff date.
- Failed or incomplete runs are never published as recommendations.

### Compliance

- V1 is restricted to a private controlled pilot with approved accounts.
- Public personalized recommendations require Brazil-first legal review of the
  operating model, responsible party, and wording.

## Agent's Discretion

- Exact Python type names, database column names, and internal module layout.
- Whether the asynchronous runner uses the existing application process or an
  existing repository-supported worker mechanism, provided the lifecycle and
  persistence guarantees are met.
- Exact mapping-table constants may be calibrated from existing system presets,
  but must be versioned, explainable, and covered by deterministic fixtures.

## Specific References

- Issue: https://github.com/caiomugarte/TCC/issues/3
- Profile computation: `api/app/services/profile.py`
- Profile schema: `api/app/schemas/profile.py`
- Premium entitlement: `api/app/entitlements/dependencies.py`
- Recommendation routes: `api/app/routers/recommendations.py`
- Allocation adapter: `api/app/adapters/allocation_engine.py`
- Stock configuration and pipeline: `py/config.py`, `py/pipelines/single_run.py`,
  `py/core/optimizer.py`
- FII selector: `py/fii_selection.py`
- Existing recommendation model: `api/app/db/models.py`

## Deferred Ideas

- Public launch before legal/compliance approval.
- Live market-data acquisition during a recommendation request.
- Personalization of GA population, generations, mutation, crossover, or run
  count.
- Security-level selection for international, fixed-income, or crypto sleeves.
- Normalized child tables or an external artifact store for recommendation
  payloads.
