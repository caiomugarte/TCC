# Tasks: Premium Max-Quality Adaptive Runs and Explicit Rerun

**Status:** Completed

## Phase 1 — policy and engine wiring

- [x] T1. Replace the Premium selector system GA configuration with the
      max-quality adaptive contract: 150 max runs, minimum 40, CV 0.02,
      Jaccard 0.75, while preserving 400 generations and population 300.
- [x] T2. Pass adaptive mode and convergence thresholds through the stock and
      FII adapters into the existing multi-run pipelines.
- [x] T3. Persist/serialize the expanded configuration and add policy tests.

## Phase 2 — explicit rerun lifecycle

- [x] T4. Add `force` to the Premium request schema and implement account/profile
      scoped active-run reuse/guarding in the service/repository.
- [x] T5. Add API tests for non-forced idempotency, active-run protection, and
      forced rerun after a terminal run.

## Phase 3 — web behavior

- [x] T6. Send `force=true` only from the explicit retry action; keep refresh
      read-only when a completed/active run exists.
- [x] T7. Keep long-running queued/running state visible after polling timeout
      and update the web test coverage.

## Phase 4 — controlled verification

- [x] T8. Run focused API/adaptor/service tests.
- [x] T9. Run full API/research/web typecheck/build/smoke suites.
- [ ] T10. Restart the local API with the QA manifest, force one rerun for the
       existing profile, monitor the single worker, and verify terminal result,
       persisted provenance, and UI read-back.

       Backend terminal result and persisted provenance are verified. The final
       authenticated visual read-back requires a browser refresh by the user.

## Constraints

- Do not modify Basic behavior.
- Do not create a second worker or concurrent optimizer.
- Do not treat `qa-local-premium` as an official production snapshot.
- Do not commit or push unless separately requested.
