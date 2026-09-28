# Status Invest-backed Recommendations Context

**Gathered:** 2026-09-28

**Spec:** `.specs/features/status-invest-backed-recommendations/spec.md`

**Status:** Approved for design

---

## Feature Boundary

Every new Basic or Premium API recommendation refreshes Status Invest stock and FII fundamentals, freezes run-specific inputs, selects Brazilian stock and FII constituents using the app profile, and returns those constituents together with class allocations. Existing market-history snapshots continue to calculate historical return and risk. Old recommendation runs remain readable.

---

## Implementation Decisions

### Recommendation scope

- Basic and Premium API routes both produce combined class allocations and Status Invest-selected Brazilian stock/FII constituents.
- All new recommendation runs use refreshed Status Invest data. No new run falls back to an old snapshot or another provider for stock/FII selection.
- Existing class-allocation market histories remain inputs for return and risk. Current Status Invest selections are not projected backward into historical class-allocation windows.

### Engine and profile

- Reuse the API Premium `multi_run` selector with app-profile-derived policy.
- Do not invoke literal legacy `profiles.py`/`ga.py` named-profile settings. Those settings are not account-profile mappings.
- For Premium, derive supported selector/GA settings from the saved questionnaire profile; persist the exact resolved policy with the run. Same policy-driving profile values resolve deterministically; global max-quality bounds remain shared.
- Preserve existing authentication, profile ownership, and plan entitlements.
- Use one asynchronous recommendation coordinator for Basic and Premium routes; route-specific policy and access remain separate.
- While a new run is pending or failed, show the current run state beside the last completed result for the same account/profile/plan.

### Refresh and run behavior

- Refresh both Status Invest stock and FII inputs for each new optimization run.
- Freeze source inputs per run and retain source, retrieval time, and content hashes.
- If either refresh or validation fails, fail that run without fallback. Keep prior completed recommendations available.
- Reading a stored recommendation does not refresh data or start another optimization.

### Agent's Discretion

- Choose bounded timeout/retry and concurrency behavior that protects Status Invest and isolates run inputs.
- Store failed-run diagnostics without exposing credentials or leaking data across accounts.
- Preserve snapshots for at least the lifetime of their recommendation run.

### Declined / Undiscussed Gray Areas → Assumptions

- A new optimization is an explicit API run request or equivalent “new recommendation” action. Page loads and GET/history requests do not refresh data.
- The existing standalone CLI remains outside this API feature; its data-refresh behavior does not change.
- A failed run is persisted as failed, and the last completed run remains available in history.
- Both stock and FII selectors run for the combined recommendation; each selector follows the account profile policy.

---

## Specific References

- Status Invest is the fundamental-data source for stock/FII selection.
- Class allocation remains a separate return/risk calculation using dated market-history snapshots.
- No historical point-in-time Status Invest fundamentals are available in current project data.

---

## Deferred Ideas

- Archive point-in-time Status Invest fundamentals for unbiased historical stock/FII reselection.
- Change or automate standalone CLI refresh behavior.
