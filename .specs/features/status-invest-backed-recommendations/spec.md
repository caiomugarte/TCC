# Status Invest-backed Recommendations Specification

## Problem Statement

Basic API recommendations currently return class allocations without selecting stock/FII constituents. Premium recommendations can select constituents from manifest-backed local files, but the API does not refresh Status Invest before a new run. Users need every new API recommendation to use fresh Status Invest fundamentals for stock/FII selection while retaining market-history snapshots for return and risk calculations.

## Goals

- [ ] Make every new Basic and Premium API recommendation include profile-based Status Invest stock/FII selection and class allocations.
- [ ] Refresh and freeze Status Invest inputs for each run; never silently use stale or alternate stock/FII data.
- [ ] Preserve account/profile linkage and reproducible provenance for successful and failed runs.
- [ ] Derive Premium stock/FII and supported GA settings from the user's saved questionnaire profile and persist the resolved policy per run.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Replacing dated market-history sources with Status Invest fundamentals | Status Invest selection data is not a daily total-return history. |
| Applying today's selected tickers to historical windows as if selected then | Project has no point-in-time Status Invest fundamentals; this would introduce look-ahead bias. |
| Changing allocation objectives, benchmark definitions, risk constraints, or rebalance rules | Existing class-allocation contract remains in force. |
| Changing standalone CLI refresh behavior | This feature covers authenticated app/API recommendation flows. |
| Changing plan entitlements or exposing additional account data | Existing access boundaries remain in force. |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Recommendation contents | Basic and Premium return class allocations plus Status Invest-selected stocks and FIIs. | User chose combined output for all API routes. | Yes |
| Source freshness | Refresh stock and FII datasets for each new run; freeze inputs per run. | User chose per-run refresh and immutable snapshot. | Yes |
| Refresh failure | Fail new run; no stale or alternate-provider fallback. Keep previous completed results available. | User explicitly rejected fallback. | Yes |
| Selector engine | Reuse Premium `multi_run` with app-profile policy, not literal old `profiles.py`/`ga.py` profile configs. | User chose API Premium engine to preserve profile-specific policy. | Yes |
| Premium GA personalization | Derive supported selector and GA settings from saved suitability score, dimensions, and restrictions; store the resolved policy with the run. | User confirmed Premium questionnaire should produce policies and GA parameters for that profile. | Yes |
| Identical profile values | Users with identical policy-driving profile values may resolve to the same deterministic policy; account ID is not a random tuning input. | Makes runs reproducible and personalization explainable. | Assumption |
| Shared GA quality bounds | Keep existing max-quality/adaptive execution caps shared; profile policy operates within those bounds. | Preserves existing selector quality controls while allowing profile-specific settings. | Assumption |
| Historical class calculation | Keep existing reference sleeve and dated market histories; do not apply current Status Invest selections backward. | User chose separate reference history; current fundamentals lack historical vintages. | Yes |
| Refresh trigger | Refresh only for a new run request/action; GET and page loads read stored runs. | Avoid unexpected repeated external calls and expensive optimizations. | Assumption |
| Profile and plan behavior | Preserve existing account-owned profile checks and plan gates; Basic and Premium keep their current profile-policy inputs. | No access or entitlement change requested. | Assumption |
| Failed-run history | Persist failed status and diagnostic; do not replace last completed run. | Preserves audit trail and usable prior result. | Assumption |
| Last-success display | While a new run is queued, running, or failed, expose the last completed result separately from current run status. | User approved keeping the prior result visible after failure. | Yes |
| Snapshot lifecycle | Retain run-specific input snapshots for at least as long as the associated recommendation record. | Reproduction requires the exact input used. | Assumption |
| Both selectors | Refresh and run stock and FII selection for every combined recommendation. | Both are part of the agreed output. | Assumption |
| Standalone CLI | Do not change CLI behavior in this feature. | Agreed scope is the app/API routes. | Assumption |

**Open questions:** none. Undiscussed behavior is recorded above as assumptions.

---

## User Stories

### P1: Generate combined profile-based recommendation ⭐ MVP

**User Story**: As an authenticated app user, I want one recommendation combining class weights with Status Invest-selected Brazilian stocks and FIIs so that my result includes both allocation and constituents.

**Why P1**: This is the requested end-to-end behavior for all app recommendation routes.

**Acceptance Criteria**:

1. SIR-01: WHEN an authenticated user starts a new Basic or Premium recommendation THEN the system SHALL create a run linked to that account and its selected account-owned profile.
2. SIR-02: WHEN a valid run reaches selection THEN the system SHALL use the API `multi_run` selector with the selection policy derived from the app profile.
3. SIR-03: WHEN a run completes THEN the Basic and Premium API responses SHALL include class allocations, selected stock/FII constituents, and BRL target amounts.
4. SIR-04: WHEN class return and risk are calculated THEN the system SHALL use the existing dated market-history reference inputs and SHALL NOT apply current Status Invest selections retrospectively.

**Independent Test**: Submit Basic and Premium requests with an authenticated profile and mocked fresh source datasets; both produce class allocations and stock/FII constituents tied to that profile.

---

### P1: Personalize Premium selector and GA policy ⭐ MVP

**User Story**: As a Premium user, I want my saved questionnaire profile to determine stock/FII selection policy and supported GA settings so that the optimization reflects my preferences.

**Why P1**: Profile-specific policy is what makes the Premium result personalized rather than a shared default portfolio.

**Acceptance Criteria**:

1. SIR-15: WHEN a Premium run is created THEN the system SHALL derive stock/FII selector policy and supported GA settings from that account's saved suitability score, profile dimensions, and restrictions, and SHALL persist the resolved policy with the run.
2. SIR-16: WHEN two Premium profile versions differ on a policy-driving field THEN the policy resolver SHALL apply that field's configured mapping to the corresponding selector or GA parameter.
3. SIR-17: WHEN the same supported profile values and policy version are resolved twice THEN the system SHALL produce the same policy configuration.

**Independent Test**: Resolve policy for two fixtures that differ on a mapped questionnaire field and assert the corresponding parameter changes; repeat one fixture and assert identical policy JSON is persisted.

---

### P1: Refresh and freeze Status Invest inputs ⭐ MVP

**User Story**: As a user, I want every new optimization to use fresh Status Invest stock and FII data so that results do not silently depend on old local files.

**Why P1**: Data freshness is the core requirement.

**Acceptance Criteria**:

1. SIR-05: WHEN a new recommendation run starts THEN the system SHALL refresh the Status Invest stock dataset for that run.
2. SIR-06: WHEN a new recommendation run starts THEN the system SHALL refresh the Status Invest FII dataset for that run.
3. SIR-07: WHEN both refreshed datasets pass their existing schema and selector validation THEN the system SHALL freeze run-specific inputs and persist provider, retrieval timestamp, and SHA-256 for each input.
4. SIR-08: IF either refresh or validation fails THEN the system SHALL mark the run failed and SHALL NOT produce a completed recommendation.
5. SIR-09: IF either refresh or validation fails THEN the system SHALL NOT substitute stale local inputs or inputs from another provider for stock/FII selection.

**Independent Test**: Mock successful and failed stock/FII refreshes; verify hashes and provenance on success, failed status and zero completed result on failure, and no fallback calls.

---

### P1: Preserve history and expose run state ⭐ MVP

**User Story**: As a user, I want to see recommendation status, data provenance, and prior completed results so that a failed refresh does not erase usable history.

**Why P1**: Per-run external refresh can fail; run history must remain trustworthy.

**Acceptance Criteria**:

1. SIR-10: WHEN a client reads a stored recommendation or run status THEN the system SHALL return persisted data without refreshing Status Invest or rerunning optimization.
2. SIR-11: IF a new run fails THEN the system SHALL preserve the last completed recommendation and persist the failed run against the same account/profile.
3. SIR-12: WHEN the current run is queued, running, or failed and a completed run exists for the same account/profile/plan THEN the system SHALL return the completed result separately from the current run status.
4. SIR-13: WHILE a recommendation is queued or running THEN the app SHALL display its current state.
5. SIR-14: WHEN a recommendation completes THEN the app SHALL display class weights, BRL targets, stock/FII constituents, and Status Invest retrieval time.

**Independent Test**: Load completed and failed runs in the app; confirm existing results remain visible after a new run fails and GET/poll requests cause no source refresh.

---

## Edge Cases

- IF one Status Invest dataset succeeds and the other fails THEN the run fails as a whole; no partial combined recommendation is completed.
- IF refreshed CSVs are empty, malformed, or fail existing schema validation THEN the run fails before selector execution.
- IF two runs overlap THEN each run SHALL read its own immutable inputs and SHALL NOT overwrite or consume the other run's files.
- IF the submitted profile is not owned by the authenticated account THEN the system SHALL reject the request before starting an external refresh.
- IF a user requests a historical recommendation THEN the system SHALL return the stored result and its original provenance without refreshing.

## Implicit Requirement Decisions

| Dimension | Requirement or explicit N/A |
| --- | --- |
| Input validation & bounds | Validate refreshed stock/FII files using existing schema and selector validators; reject empty or malformed input before optimization. |
| Failure / partial-failure states | Any required refresh or validation failure fails the whole run; preserve previous completed result. |
| Idempotency / retry / duplicate handling | Each accepted new-run request creates one independently tracked run; retries must not mark the same failed run completed from stale inputs. |
| Auth boundaries & rate limits | Require existing authentication, account-owned profile, and existing plan entitlement; no entitlement expansion. |
| Concurrency / ordering | Use immutable run-specific inputs and isolated run workspaces; results remain linked to their initiating account/profile. |
| Data lifecycle / expiry | Retain snapshots for at least the lifetime of their recommendation run; no new expiry policy in scope. |
| Observability | Persist provider, retrieval time, hashes, profile/account linkage, run state, and a sanitized failure reason. |
| External-dependency failure | Fail closed; no stale-cache or alternate-provider fallback for stock/FII selection. |
| State-transition integrity | Runs transition queued → running → completed or failed; completed/failed runs are terminal. |

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| SIR-01 | P1: Generate combined profile-based recommendation | Tasks | In Tasks |
| SIR-02 | P1: Generate combined profile-based recommendation | Tasks | In Tasks |
| SIR-03 | P1: Generate combined profile-based recommendation | Tasks | In Tasks |
| SIR-04 | P1: Generate combined profile-based recommendation | Tasks | In Tasks |
| SIR-05 | P1: Refresh and freeze Status Invest inputs | Tasks | In Tasks |
| SIR-06 | P1: Refresh and freeze Status Invest inputs | Tasks | In Tasks |
| SIR-07 | P1: Refresh and freeze Status Invest inputs | Tasks | In Tasks |
| SIR-08 | P1: Refresh and freeze Status Invest inputs | Tasks | In Tasks |
| SIR-09 | P1: Refresh and freeze Status Invest inputs | Tasks | In Tasks |
| SIR-10 | P1: Preserve history and expose run state | Tasks | In Tasks |
| SIR-11 | P1: Preserve history and expose run state | Tasks | In Tasks |
| SIR-12 | P1: Preserve history and expose run state | Tasks | In Tasks |
| SIR-13 | P1: Preserve history and expose run state | Tasks | In Tasks |
| SIR-14 | P1: Preserve history and expose run state | Tasks | In Tasks |
| SIR-15 | P1: Personalize Premium selector and GA policy | Tasks | In Tasks |
| SIR-16 | P1: Personalize Premium selector and GA policy | Tasks | In Tasks |
| SIR-17 | P1: Personalize Premium selector and GA policy | Tasks | In Tasks |

**Coverage:** 17 total, 17 mapped to tasks, 0 unmapped.

## Success Criteria

- [ ] Every new Basic/Premium API run either completes with fresh Status Invest stock/FII provenance or ends failed without fallback.
- [ ] Every completed run returns combined class allocation and stock/FII constituents for its account-owned profile.
- [ ] Premium run policy is deterministically derived from saved questionnaire fields and persisted with that run.
- [ ] A failed new run does not remove or overwrite the user's last completed result.
- [ ] Reading run history never triggers source refresh or optimization.
