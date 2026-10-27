# FII Universe Feasibility — Context (gray-area decisions)

**Issue:** caiomugarte/TCC#12 — P0-05
**Route:** DRIVEN. The gray areas below were resolved by evidence recomputed in
this worktree with the production functions; the two policy rows await the human
checkpoint (approved 2026-10-27; implementation parked until issue #7 lands).

## Gray areas and resolutions

### 1. Which count is "elegível"? (eligible vs ranked)

`build_fii_scores` drops FIIs with unusable fundamentals after
`apply_fii_eligibility`, so the pickable universe is smaller than the eligible
one (measured: 13 eligible → 12 ranked at `size_min=3e9`; 10 → 9 at the
3.4e9/1.4e6 band point). **Decision:** the precondition judges the ranked
universe (the GA's actual input, matching `py/fii_selection.py:393`); diagnostics
report both counts. The matrix pins both series.

### 2. Normative matrix: sweep or anchors?

The issue's `(7, 13, 28, 50)` is a size sweep at constant `liq_min=2e6`, not the
profile anchors (61 moderado / 162 arrojado). **Decision:** both series are
normative in one table (sweep + anchors + the 3.4e9/1.4e6 premium band point),
with provenance = committed snapshot + production functions. See spec
Assumption 2 and T2.

### 3. Policy option (a) vs (b) — APPROVED 2026-10-27 (option (a))

- (b) `n_assets=7` at 5e9: hard-infeasible under `limitar_concentracao` — sector
  HHI fixed at 15/49 = 0.3061 > 0.25 (the 7-asset set *is* the portfolio).
- (a) `size_min=3e9`: 13 eligible / 12 ranked ≥ 10, min sector HHI 0.24 ≤ 0.25,
  and 0/21 failing premium-band points vs 11/21 today.

**Chosen:** (a), recorded in `docs/adr/0002-fii-universe-feasibility-policy.md`
with full sensitivity. Approval of that ADR is the feature's human checkpoint.

### 4. Policy-version bump with the calibration — APPROVED 2026-10-27

Changing rule values under an unchanged version string would make provenance
silent. **Chosen:** `premium-policy-v2` / `basic-policy-v2` alongside the
calibration (spec Assumption 4). If the maintainer declines the bump, POL-01's
version clause drops and the pin test alone records the change.

### 5. What is the "published batch" (AC4)?

**Decision:** `GET /v1/recommendations/latest-completed` = the published batch;
`GET /v1/recommendations` may show the latest run as a closed failure. Matches
current API (`api/app/routers/recommendations.py:62-100`) and web
(`recommendation-view.tsx:149`); no new table.

### 6. suitability.html drift

Its `BASE_PROFILES` mirrors `STOCK_RULES` (stock policy), not `FII_RULES`.
**Decision:** no value change; drift-guard test only (DFT-01).

### 7. `enforce_liquidity_and_size` no-op

Computed (`premium_policy.py:228`) but unread by the engines. **Decision:** leave
exactly as-is; contract test asserts it does not affect eligibility.

### 8. Offline named-profile path

`filters=None` (`py/fii_selection.py:558-567`). **Decision:** precondition stays
in the explicit/premium path; committed artifacts
(`data/processed/fii_clean_*.csv`, `outputs/carteira_fii_*_consensus.json`,
`data/allocation/caio_fiis.csv`) must remain byte-identical. On shortage the
legacy check in `optimize_fii_portfolio` (`py/fii_selection.py:392-396`) stays
byte-identical (same `FiiSelectionError`, same message) and is the only shortage
guard on that path.

### 9. Failure message contract

Portuguese, structured-fields-only template frozen in `design.md`; six mandated
fields; no paths/raw exception text; fits `String(64)`/`String(512)`. Existing
sanitisation tests are the guardrail.

## Decided since round 1 (FIX round)

- **Field transport frozen:** the six shortage fields travel in a `details` dict
  on `PremiumOptimizationError`; the executor seam is
  `_fail(run_id, code, exc: Exception)` reading `getattr(exc, "details", None)`
  (design component 3, T5).
- **Offline guard frozen (binding decision):** the `optimize_fii_portfolio`
  check keeps its legacy `FiiSelectionError` and message byte-identically; the
  structured raiser is not shared with it (`eligible_count` is genuinely
  unavailable there), so "named-profile calls unaffected" is a testable claim
  (T3).

## Explicitly NOT decided here (left to IMPLEMENT)

- The exact shape of the count-based DP for minimum achievable sector HHI (T8
  pins the method and its sanity values, not the implementation).
