# Profile v1 to Premium Optimization Design

**Spec:** `.specs/features/profile-v1-premium-optimization/spec.md`
**Context:** `.specs/features/profile-v1-premium-optimization/context.md`
**Status:** Draft

## Architecture Overview

The feature adds an account-scoped orchestration layer above the existing
allocation, stock-selection, and FII-selection engines. The API validates
entitlement, pilot access, profile ownership, and snapshot availability before
creating a queued run. A bounded application runner processes the run with
explicit policy values and persists one terminal result.

Class allocation remains a class-level operation. Stock and FII selection
produce constituents inside their respective sleeves; their fundamental
weights are never reused as cross-class weights.

## Execution Flow

1. The client calls `POST /v1/premium/recommendations`.
2. The route authenticates the Clerk account and checks Premium entitlement and
   private-pilot approval before doing optimization work.
3. The service loads the latest valid profile revision owned by that account,
   or the requested owned revision when the request supplies `profileId`.
4. The snapshot registry selects and validates the latest compatible immutable
   manifest. Missing or changed source files fail the request before enqueue.
5. The policy resolver creates and validates a complete
   `ResolvedOptimizationPolicy` from the stored profile and versioned rules.
6. The repository stores a `queued` `RecommendationRun` containing the account,
   profile revision, policy, manifest ID, and deterministic seed.
7. The runner submits the run after the transaction commits and returns
   `202 Accepted` with the run ID.
8. The worker marks the run `running`, executes the three engines with isolated
   inputs and output paths, combines class and sleeve results, and stores the
   immutable JSON result.
9. The worker marks the run `completed` only after the result transaction
   succeeds. Any failure marks it `failed` without a published result.
10. The client polls `GET /v1/recommendations/{id}` until the run is terminal.

The API process uses a standard-library, bounded single-worker executor for
the private pilot. A per-run workspace prevents shared checkpoints, caches,
and output files from crossing account boundaries. This is an intentional
single-process ceiling; a durable queue and external worker are deferred until
the product needs multiple API instances or higher throughput.

## Code Reuse Analysis

### Existing Components to Leverage

| Component | Location | Use |
| --- | --- | --- |
| Profile computation | `api/app/services/profile.py` | Persist the existing dimensions, scores, rules, and warnings without changing score semantics. |
| Profile persistence | `api/app/db/models.py` | Extend `ProfileRecord` with computed provenance fields and reuse its account/revision relationship. |
| Premium entitlement | `api/app/entitlements/dependencies.py` | Compose the existing active/grace-period check with the pilot allowlist. |
| Account ownership | `api/app/auth/dependencies.py` and route queries | Derive ownership from the authenticated account; never trust a request account ID. |
| Allocation profile | `py/allocation_profiles.py` | Build the explicit score-based `AllocationProfile` and retain validation/interpolation. |
| Allocation runner | `py/pipelines/asset_allocation.py` | Add explicit class constraints while preserving the current Basic defaults. |
| Allocation snapshot loader | `py/allocation_data.py` | Reuse common-date alignment and missing-data validation in strict Premium mode. |
| Stock scoring/GA | `py/core/scoring.py` and `py/core/optimizer.py` | Add explicit configuration inputs while retaining named-profile wrappers for existing offline callers. |
| FII scoring/GA | `py/fii_selection.py` | Reuse lower-level custom-weight and custom-GA seams; make top-level execution explicit and non-interactive. |
| API adapter injection | `api/app/adapters/allocation_engine.py` | Follow the existing injected-engine pattern for deterministic route tests. |
| Recommendation reads | `api/app/routers/recommendations.py` | Extend account-scoped ID reads to return queued, running, failed, and completed states. |

### Integration Points

| System | Integration Method |
| --- | --- |
| Profile API | Store a computed profile snapshot at submission time; resolve from that immutable revision. |
| Entitlements | Add a pilot-aware dependency on top of `require_premium`. |
| Database | Add run status, policy, result, provenance, and failure fields through an Alembic migration. |
| Allocation data | Add a manifest-backed strict loader with no IFIX or fixed-artifact fallback for Premium. |
| Stock/FII pipelines | Add explicit configuration objects and isolated output directories; do not pass static `caio` as the policy source. |
| Web client | Branch only the existing recommendation loader by plan, start the Premium run, poll its ID, and render the existing result shell. |

## Components

### Profile policy resolver

- **Purpose:** Convert one persisted Profile v1 revision into explicit
  allocation, stock, and FII configuration.
- **Location:** `api/app/services/premium_policy.py` and its versioned rules
  module.
- **Interface:**
  `resolve_premium_policy(profile_record, rules_version) -> ResolvedOptimizationPolicy`.
- **Dependencies:** `ProfileRecord`, existing allocation anchors, stock/FII
  configuration schemas, and the versioned profile-to-parameter rule table.
- **Rules:** Use normalized `score` for allocation interpolation; use score,
  dimensions, and restrictions for every profile-dependent stock/FII field;
  retain `generic_profile` only as the baseline label; reject incomplete or
  contradictory restrictions.
- **Reuses:** `compute_profile`, `AllocationProfile`, and
  `interpolate_profile`.

The rule table is plain versioned configuration, not a new rule engine. It must
emit explicit values for stock `n_assets`, factor weights, filters, and HHI
penalty, and the equivalent FII values. System GA fields are copied from the
versioned system configuration and are not profile-dependent.

### Snapshot registry

- **Purpose:** Select and validate the latest compatible immutable source set.
- **Location:** `py/snapshot_manifest.py` with a small API-facing registry
  adapter.
- **Interface:**
  `latest_compatible_manifest(registry_dir) -> SnapshotManifest` and
  `validate_manifest(manifest) -> None`.
- **Dependencies:** JSON manifests, source-file hashes, allocation snapshot
  loader, and configured registry path.
- **Manifest contents:** Manifest ID/version, creation and cutoff dates,
  allocation snapshot path and hash, stock fundamentals path and hash, FII
  fundamentals path and hash, source metadata, common-date metadata, and
  supported classes.
- **Rules:** Choose by manifest cutoff/creation metadata, verify every declared
  hash, reject changed or incomplete files, and never fetch or substitute a
  fallback source during a run.
- **Reuses:** Common-date and missing-data validation in `py/allocation_data.py`.

### Premium run service

- **Purpose:** Coordinate validation, run creation, submission, and status
  transitions.
- **Location:** `api/app/services/premium_recommendation.py`.
- **Interfaces:**
  `create_run(account, request) -> RecommendationRun` and
  `execute_run(run_id) -> None`.
- **Dependencies:** Entitlement/pilot dependency, policy resolver, snapshot
  registry, run repository, and engine orchestrator.
- **Rules:** Resolve and persist all inputs before submission; use a fresh
  SQLAlchemy session inside the worker; create a new run for every accepted
  request; do not automatically retry failed runs.
- **Reuses:** Existing recommendation adapter dependency-injection style and
  account-scoped query pattern.

### Run repository and state machine

- **Purpose:** Persist durable run state and enforce terminal-result
  visibility.
- **Location:** `api/app/repositories/recommendations.py` or the existing
  recommendation service boundary.
- **Interface:**
  `create_queued`, `mark_running`, `mark_completed`, `mark_failed`, and
  `get_owned_run`.
- **State transitions:** `queued -> running -> completed` or
  `queued/running -> failed`. A terminal run cannot be changed to another
  terminal state.
- **Dependencies:** `RecommendationRun`, SQLAlchemy session factory, and
  account ID.
- **Rules:** Store the policy before execution, write the complete result and
  terminal status in one transaction, and return no result JSON for failed or
  non-terminal runs.

### Pilot access dependency

- **Purpose:** Prevent public personalized recommendations before legal
  approval.
- **Location:** `api/app/entitlements/dependencies.py` and application
  settings.
- **Interface:** `require_premium_pilot(account) -> Account`.
- **Dependencies:** Existing `require_premium` and a configured allowlist of
  approved account IDs.
- **Rules:** Fail closed when the account is not allowlisted; entitlement is
  checked first; no database pilot model is added in v1.

### Engine orchestrator

- **Purpose:** Run allocation, stock selection, and FII selection with the
  explicit policy and combine their outputs.
- **Location:** `api/app/adapters/premium_optimization.py` plus explicit
  runners under `py/`.
- **Interface:**
  `run_premium_optimization(policy, manifest, workspace) -> PremiumResult`.
- **Dependencies:** Three engine adapters, isolated workspace, manifest data,
  and deterministic seed.
- **Rules:** Allocation receives only allocation parameters and class data;
  stock/FII runners receive only their selection parameters and universes;
  selector output is converted to sleeve and total-portfolio amounts after
  class targets exist.
- **Reuses:** Existing allocation adapter injection and lower-level stock/FII
  scoring and optimizer functions.

### Stock selection runner

- **Purpose:** Execute the stock pipeline with explicit profile-derived values.
- **Location:** `py/pipelines/premium_stock_selection.py` or the existing
  stock pipeline after extracting a reusable explicit-config path.
- **Interface:**
  `run_stock_selection(input_path, config, workspace, seed) -> StockSelectionResult`.
- **Dependencies:** Stock fundamentals, explicit eligibility filters, factor
  weights, HHI penalty, system GA config, and run workspace.
- **Output:** Selected tickers, sleeve weights, scores/metrics, and exclusion
  reasons for instruments removed by data or restrictions.
- **Rules:** No interactive checkpoint prompt, shared profile output, or hidden
  module-level profile lookup.

### FII selection runner

- **Purpose:** Execute the FII selector with explicit profile-derived values.
- **Location:** `py/pipelines/premium_fii_selection.py` or
  `py/fii_selection.py` after extracting a reusable explicit-config path.
- **Interface:**
  `run_fii_selection(input_path, config, workspace, seed) -> FiiSelectionResult`.
- **Dependencies:** FII fundamentals, explicit factor weights, eligibility
  thresholds, HHI penalty, system GA config, and run workspace.
- **Output:** Selected tickers, sleeve weights, scores/metrics, and exclusion
  reasons.
- **Rules:** No static named profile as the source of values, no global output
  paths, and no IFIX substitution.

### Web recommendation client

- **Purpose:** Start and poll a Premium run without introducing a new result
  screen.
- **Location:** `web/lib/api-client.ts`, `web/lib/api-types.ts`, and
  `web/components/recommendation/recommendation-view.tsx`.
- **Interface:** `startPremiumRecommendation`,
  `getRecommendation(id)`, and a bounded polling loop for `queued`/`running`.
- **Dependencies:** Existing authenticated fetch helper and account plan.
- **Rules:** The client only changes presentation/orchestration; the API
  remains the source of authorization. Basic continues its current one-shot
  path.

## Data Models

### ProfileRecord additions

Reuse the existing profile revision as the immutable input version. Persist the
computed values that are currently discarded:

| Field | Purpose |
| --- | --- |
| `raw_score` | Preserve the pre-cap suitability signal. |
| `rules_json` | Preserve applied score-cap rules. |
| `warnings_json` | Preserve profile warnings used in explanation. |
| `restrictions_json` | Store normalized restriction values without reparsing answers. |
| `schema_version` | Identify the Profile v1 questionnaire/computation contract. |

The existing answers, dimensions, capped score, and generic profile remain
available for compatibility.

### RecommendationRun additions

Keep existing summary fields used by Basic responses. Add fields equivalent to:

| Field | Purpose |
| --- | --- |
| `status` | `queued`, `running`, `completed`, or `failed`. |
| `started_at` / `completed_at` | Lifecycle timestamps. |
| `failure_code` / `failure_message` | Safe terminal diagnostics without partial output. |
| `policy_version` | Version of the resolver rules. |
| `policy_json` | Immutable serialized `ResolvedOptimizationPolicy`. |
| `result_json` | Immutable class, stock, and FII result payload; null until completed. |
| `provenance_json` | Model versions, system GA config, seed, manifest IDs/hashes, and cutoff. |
| `output_hash` | Canonical result hash for integrity and replay comparison. |

The existing `snapshot_id` and `model_version` fields remain populated with
compatibility summaries. The new JSON fields are the authoritative Premium
payload.

### SnapshotManifest

```text
SnapshotManifest {
  manifest_id
  manifest_version
  created_at
  cutoff_date
  allocation_source { path, sha256, metadata }
  stock_source { path, sha256, metadata }
  fii_source { path, sha256, metadata }
  common_dates
  supported_classes
}
```

The registry treats referenced source files as immutable. The worker validates
hashes again before reading data, so a file changed after enqueue fails the run
instead of producing an untraceable result.

### Premium result

```text
PremiumResult {
  classes: [{ key, target_weight, target_amount_brl, metrics }]
  stocks: [{ ticker, sleeve_weight, portfolio_weight, target_amount_brl, reasons }]
  fiis: [{ ticker, sleeve_weight, portfolio_weight, target_amount_brl, reasons }]
  assumptions
  risks
  provenance
}
```

`portfolio_weight` equals `class.target_weight * sleeve_weight`. The sum of
class weights is 1, and each non-empty sleeve sums to 1 before conversion.

## Policy and Constraint Handling

The resolver composes constraints before any engine starts:

| Restriction | Resolved constraint |
| --- | --- |
| `evitar_cripto` | `crypto.max_weight = 0`. |
| `evitar_exterior` | `international_equity.max_weight = 0`. |
| `priorizar_renda` | `fixed_income.min_weight = 0.40`. |
| `limitar_concentracao` | Class HHI `<= 0.25` plus applicable stock/FII HHI controls. |
| `evitar_illiquidez` | Current versioned stock/FII liquidity and size thresholds. |
| `nenhuma` | No additional restriction. It cannot coexist with another restriction. |

The Premium allocation path sets its default minimum class weight to zero so
an explicitly excluded class can be zero. All other class weights remain
non-negative and must sum to 1. Infeasible intersections fail the run.

## API Contracts

### Start run

`POST /v1/premium/recommendations`

- **Authorization:** `require_premium_pilot`.
- **Request:** Optional owned `profileId`; no account ID, arbitrary policy, or
  snapshot path from the client.
- **Success:** `202 Accepted` with `{ id, status: "queued", profile_version,
  created_at }`.
- **Validation failures:** Existing account/profile errors plus explicit
  `premium_pilot_required`, `snapshot_unavailable`, or `profile_invalid`.

### Read run

`GET /v1/recommendations/{id}`

- Query by both run ID and authenticated account ID.
- Return status and lifecycle metadata while queued or running.
- Return failure code/message without `result_json` when failed.
- Return the existing completed recommendation fields plus Premium policy,
  stock/FII outputs, and provenance when completed.
- Preserve the existing Basic completed response contract.

## Error Handling Strategy

| Scenario | Handling | User impact |
| --- | --- | --- |
| No Premium entitlement | Reject before run creation. | Paid feature unavailable. |
| Premium account not in pilot allowlist | Reject before run creation. | Feature unavailable during controlled pilot. |
| No valid profile | Return the existing profile-required/invalid error. | User must submit or refresh profile. |
| No compatible manifest | Return `snapshot_unavailable`; do not enqueue. | User sees that recommendation data is unavailable. |
| Source hash changed after enqueue | Mark run failed with `snapshot_changed`. | No result is published. |
| No feasible constrained allocation | Mark run failed with `infeasible_constraints`. | Show diagnostic, not a fabricated target. |
| Selector or allocation failure | Mark run failed with a safe engine error code. | No partial recommendation is visible. |
| Worker interrupted/restarted | Mark stale queued/running runs failed on recovery. | User can submit a new run. |
| Cross-account run ID | Return the existing account-scoped not-found response. | No ownership information leaks. |

## Tech Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Async mechanism | Standard-library bounded single-worker executor. | No new queue dependency for a private pilot; isolates HTTP latency. |
| Job durability | Persist status in `RecommendationRun`; fail stale work on recovery. | Simple behavior with explicit restart limitation. |
| Run isolation | Unique temporary/configured workspace per run. | Existing runners write files and checkpoints globally. |
| Policy representation | Frozen internal objects plus canonical JSON. | Keeps engine calls typed while making provenance serializable. |
| Snapshot selection | Latest compatible immutable manifest, validated twice. | Reproducibility without live network calls or silent fallback. |
| Retry behavior | No automatic retry; a new request creates a new run. | Avoid duplicate expensive GA work and ambiguous provenance. |
| Persistence shape | One `RecommendationRun` with structured JSON payloads. | Smallest schema surface that preserves complete immutable output. |
| Pilot approval source | Configured account-ID allowlist. | Avoids a new billing/pilot table before legal launch decisions. |

## Requirement Mapping

| Requirement | Design coverage |
| --- | --- |
| POL-01, POL-02 | Profile policy resolver and versioned rule table. |
| REST-01, REST-02 | Policy and constraint handling. |
| EXEC-01, EXEC-02 | Execution flow, run service, and engine orchestrator. |
| AUTH-01, AUTH-02 | Pilot dependency and account-scoped read path. |
| PROV-01, PROV-02 | Run model, snapshot manifest, and canonical result storage. |
| FAIL-01 | State machine and error handling strategy. |

## Design Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Existing stock/FII functions resolve config from module globals. | Add explicit-config seams and retain named-profile wrappers only for offline callers. |
| Existing GA does not seed Python `random`. | Seed every random source from the persisted base seed and derive per-run seeds. |
| Current FII runner has no configurable liquidity thresholds. | Extract its current eligibility rules into versioned explicit configuration before applying restriction overrides. |
| Current allocation loader falls back to IFIX. | Add strict Premium manifest loading and make fallback unavailable in that path. |
| Application restart loses in-flight work. | Persist status, mark stale runs failed on recovery, and document durable workers as the scale-up boundary. |
| Multiple API workers could submit concurrent process-local jobs. | Restrict the private-pilot deployment to one API worker until a durable executor exists. |
