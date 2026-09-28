# Premium Max-Quality Adaptive Runs and Explicit Rerun

**Status:** Implemented and locally verified
**Parent feature:** `.specs/features/profile-v1-premium-optimization/`

## Problem

The Premium API currently resolves a fixed selector configuration of 30 runs and
400 maximum generations. The existing terminal workflow also supports an
adaptive `max-quality` mode: up to 150 runs, at least 40 runs before checking
stability, and convergence targets of CV <= 2% and mean Jaccard >= 75%.

The application must use the max-quality adaptive workflow for both Brazilian
stocks and FIIs when a Premium run is requested. Users also need an explicit
rerun action for the current profile. A browser refresh must not create a new
run when a completed or active run already exists.

## Goals

- Apply the max-quality adaptive configuration to both Premium selectors.
- Persist the full adaptive configuration in the resolved policy/provenance.
- Expose an explicit `force` rerun request path for the current owned profile.
- Prevent accidental duplicate queued/running runs when `force` is false.
- Keep Basic behavior unchanged.
- Keep non-terminal UI states truthful when a long run exceeds browser polling.

## Requirements

1. The Premium system GA configuration SHALL use:
   - `population: 300`
   - `generations: 400`
   - `mutation_rate: 0.02`
   - `crossover_rate: 0.8`
   - `run_count: 150` maximum selector runs
   - `adaptive_mode: true`
   - `min_runs: 40`
   - `target_cv: 0.02`
   - `target_jaccard: 0.75`
2. Stocks and FIIs SHALL receive the same max-quality adaptive controls while
   retaining their profile-derived factor weights, filters, asset count, and HHI
   penalty.
3. The persisted policy SHALL identify the adaptive mode and all convergence
   thresholds so a run is reproducible/auditable.
4. `POST /v1/premium/recommendations` SHALL accept an explicit `force` flag.
5. With `force=false`, an existing queued/running run for the current account
   and profile SHALL be reused rather than duplicated.
6. With `force=true`, a new run MAY be created only after ownership/profile and
   entitlement/pilot checks pass; an active run SHALL not be duplicated.
7. The web retry action SHALL send `force=true`; ordinary load/refresh SHALL not.
8. A polling timeout SHALL leave queued/running status visible and SHALL not
   render a false unavailable/failed state.
9. Failed or unavailable runs SHALL not publish partial results.
10. The existing `qa-local-premium` manifest remains explicitly QA-only and is
    outside this feature's production data provisioning.

## Acceptance criteria

- A new Premium run policy contains the max-quality fields above for both
  `stocks` and `fiis`.
- The adapter passes adaptive mode and thresholds into both selector pipelines.
- A non-forced duplicate request does not create a second active run.
- An explicit forced rerun creates one new run after the previous run is
  terminal; an active run is rejected/reused safely.
- API and web tests cover policy, adapter arguments, force semantics, and the
  long-running UI state.
- Basic tests remain green.
- A controlled local rerun can be started and read back by run ID without
  creating concurrent workers.
