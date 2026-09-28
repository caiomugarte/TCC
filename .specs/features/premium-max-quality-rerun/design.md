# Design: Premium Max-Quality Adaptive Runs and Explicit Rerun

## Existing contracts

- `api/app/services/premium_policy.py` resolves the profile and currently emits
  `system_ga_config` for both selectors.
- `api/app/adapters/premium_optimization.py` invokes
  `run_multi_execution_profile` for stocks and the FII runner sequentially.
- `py/pipelines/multi_run.py` supports `adaptive_mode`, `min_runs`,
  `target_cv`, and `target_jaccard`; adaptive convergence is checked after the
  minimum run count.
- `py/core/optimizer.py` independently performs generation-level early stopping
  after a patience window. This remains distinct from cross-run adaptive
  convergence.
- `api/app/services/premium_recommendation.py` creates account-owned queued
  runs; `RecommendationRun` stores policy/result/provenance.
- `web/components/recommendation/recommendation-view.tsx` starts Premium runs
  and polls their account-scoped IDs.

## Policy model

Extend the versioned system configuration with explicit cross-run fields:

```text
population: 300
generations: 400
mutation_rate: 0.02
crossover_rate: 0.8
run_count: 150
adaptive_mode: true
min_runs: 40
target_cv: 0.02
target_jaccard: 0.75
```

The adapter maps these values to the existing pipeline signature rather than
creating a second optimizer. The policy JSON remains the source of truth for
replay and audit.

## Run lifecycle and force semantics

Add `force: bool = false` to `PremiumRecommendationRequest`.

Before creating a run, query the latest owned run for the current profile:

- `queued`/`running` + `force=false`: return the existing run.
- `queued`/`running` + `force=true`: return a conflict or existing run; never
  start a second worker for the same profile.
- terminal + `force=false`: return/reuse the completed or failed run according
  to the current read path.
- terminal + `force=true`: create one new queued run.

The repository/service must preserve account and profile ownership and must not
use a request-supplied account ID as authority.

## Web behavior

- Initial load uses the latest Premium run and does not force a new run.
- Retry sends `{ force: true }` only when the current run is terminal/failed.
- When the bounded browser polling window expires, retain the latest queued or
  running object and show that the server continues processing; do not throw a
  generic unavailable error.

## Runtime risks

Max-quality may take substantially longer than the current 30-second browser
polling window and may consume significant CPU. The single API worker remains
bounded to one active optimization. A future durable queue is out of scope.

## Verification

- Unit tests for resolved policy fields and stable serialization.
- Adapter tests assert adaptive arguments for stock and FII calls.
- Service/route tests cover reuse, forced terminal rerun, and active-run guard.
- Web source test covers force retry and timeout state.
- Existing API, research, typecheck, build, and smoke suites remain green.
