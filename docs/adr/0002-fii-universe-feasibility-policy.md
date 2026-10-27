# ADR 0002: Calibrate the conservative FII size floor to 3e9 and fail fast on universe shortage

## Status

Accepted — 2026-10-27, by explicit human approval of the caiomugarte/TCC#12
checkpoint: option (a) (calibrate `size_min` 5e9 → 3e9) and the policy-version bump
(`premium-policy-v2` / `basic-policy-v2`) were both approved. Implementation is
parked until issue #7 (P0-02) lands — see "Implementation status" below.
Superseding this ADR requires a new ADR number.

## Implementation status

Approved but not implemented: the owner chose to land #7 (P0-02, fitness↔chromosome
and regenerated artifacts) before #12. The SPEC/design/tasks in
`.specs/features/fii-universe-feasibility/` are complete and approved; T1 is the
first task to execute on resume. The FII matrix counts pinned here do not depend
on the GA, but the T9 anchor-executability evidence should be re-measured after
#7 lands.

## Context

The conservative FII policy (`FII_RULES["conservador"]`, `api/app/services/premium_policy.py:69-75`)
requests `n_assets=10` with `size_min=5e9` and `liq_min=2e6`. Over the committed
snapshot `data/raw/status_invest_fii.csv` (588 FIIs) that threshold pair leaves
only 7 eligible FIIs, so every run dies in
`py/fii_selection.py:392-396` (`FiiSelectionError("FII universe has N assets; M required)`),
and the failure path loses the numbers before they reach
`RecommendationRun.failure_message` (`api/app/adapters/premium_optimization.py:539-546`,
`api/app/services/premium_executor.py:203-212`). The issue forbids silently
reducing `n_assets`, so the shortage must become an explicit policy decision:
(a) calibrate `size_min` to 3e9, or (b) keep 5e9 and set `n_assets=7` if sector
diversification sustains it.

All counts below were recomputed in this worktree with the production functions
(`load_fii_data` + `apply_fii_eligibility` + `build_fii_scores` from
`py/fii_selection.py`) over the committed snapshot. Two universe series matter:
**eligible** (post-eligibility) and **ranked** (post-scoring — the rows the GA can
actually pick; scoring drops FIIs with unusable fundamentals). Feasibility is
judged on the ranked series, matching the runtime check.

### Sensitivity analysis

**1. Eligibility counts vs `size_min` (rows 1, 3, 4, 5 at constant
`liq_min=2e6` — the issue's sweep; row 2 is the premium band point):**

| size_min | liq_min | eligible | ranked | `n_assets=10` feasible (ranked)? | sector HHI of eligible set |
| --- | --- | --- | --- | --- | --- |
| 5.0e9 | 2e6 | 7 | 7 | no (7 < 10) | fixed 0.3061 (7 of 7 taken) |
| 3.4e9 | 1.4e6 | 10 | 9 | no (9 < 10) | — |
| 3.0e9 | 2e6 | 13 | 12 | yes (12 ≥ 10) | min achievable 0.24 (n=10), max 0.38 |
| 2.0e9 | 2e6 | 28 | 26 | yes | min 0.14 (n=10) |
| 1.0e9 | 2e6 | 50 | 47 | yes | min 0.10 (n=10) |

Profile anchors for reference: moderado `1e9/5e5` → 61 eligible / 57 ranked
(`n_assets=12`); arrojado `2e8/5e4` → 162 / 137 (`n_assets=15`). The issue's
matrix `(7, 13, 28, 50)` is the eligible column of the size sweep; the ranked
column is `(7, 12, 26, 47)`.

**2. Sector HHI vs the `limitar_concentracao` cap (`hhi_max=0.25`,
`api/app/services/premium_policy.py:229-230`, hard check
`py/fii_selection.py:639-640`):**

- Option (b) with `n_assets=7`: the eligible set *is* the portfolio (Ind/Log 3,
  Papéis 2, Imóveis Com. Outros 1, Shoppings 1 → sector weights 3/7, 2/7, 1/7,
  1/7). Sector HHI is fixed at 15/49 = **0.3061 > 0.25** — no subset exists
  because the subset is unique. Option (b) is therefore **unreachable for every
  profile carrying `limitar_concentracao`**, regardless of the GA.
- Option (a) with `n_assets=10`: on the 13-asset **eligible** set (Papéis 5,
  Ind/Log 3, Shoppings 2, Misto 2, Outros 1) the minimum achievable sector HHI
  over all 10-subsets is **0.22** (e.g. Papéis 2/5, Ind/Log 3/3, Shoppings 2/2,
  Misto 2/2, Outros 1/1). On the 12-asset **ranked** set the GA actually draws
  from (Papéis 5, Ind/Log 3, Shoppings 2, **Misto 1**, Outros 1 — scoring drops
  one Misto) the minimum is **0.24 ≤ 0.25** (Papéis 3/5, Ind/Log 3/3,
  Shoppings 2/2, Misto 1/1, Outros 1/1 → (9+9+4+1+1)/100 = 0.24). Feasible, with
  one swap of slack — thin but real. Both minima are computed exactly with a
  count-based DP over sector multiplicities (never subset enumeration).

**3. Effect on the premium interpolation band** (`n_assets` and thresholds are
interpolated between anchors, `api/app/services/premium_policy.py:222-243`;
band sampled at segment-fraction steps of 0.05 = score steps of 0.025, 21 points
per segment, feasibility =
ranked count ≥ interpolated `n_assets`, plus HHI reachability under 0.25):

| Band segment | Today (`size_min` base 5e9) | After option (a) (base 3e9) |
| --- | --- | --- |
| conservador → moderado | **11/21 points fail** (10 with ranked < `n_assets` at f=0.00–0.45, incl. the issue's 3.4e9/1.4e6 → 9 ranked vs 11 required; 1 more, f=0.50, count-feasible but min sector HHI > 0.25) | **0/21 fail**; min sector HHI ≤ 0.24 at every point |
| moderado → arrojado | 0/21 fail (61→162 eligible) | unchanged, 0/21 fail |

**2b. The cap only binds if `limitar_concentracao` reaches the nested filters
(measured).** `hhi_max=0.25` is injected into the selector's
`liquidity_and_size_filters` only when the profile carries
`limitar_concentracao` (`api/app/services/premium_policy.py:229-230`); it then
flows into the GA settings (`py/fii_selection.py:376-377` →
`GeneticAlgorithm(hhi_max=...)` at `:405`) and is re-checked after selection
(`:637-640`). With the wiring in place the GA reproduces sector HHI **0.2400
for 4/4 seeds** (consensus 0.2400) at the anchor; with `hhi_max` withheld from
the nested filters the same seeds produce **0.2800**, which the post-check would
reject. The "anchor becomes executable" claim below therefore rests on this
wiring: the band grid is pinned end-to-end by the feature's band test (task T8)
and the wiring by its anchor test (task T9, restricted vs unrestricted).

## Decision

Adopt **option (a)**: set `FII_RULES["conservador"]["filters"]["size_min"]` from
`5_000_000_000` to `3_000_000_000`, keeping `liq_min=2_000_000` and
`n_assets=10`, and record the change with bumped policy versions
(`premium-policy-v1` → `premium-policy-v2`, `basic-policy-v1` → `basic-policy-v2`).
Option (b) is rejected: it is structurally infeasible under
`limitar_concentracao` (sector HHI fixed at 0.3061 > 0.25) and it would keep the
interpolation band broken.

Additionally — and independently of the calibration — the system gains an
explicit universe-feasibility precondition in the explicit/premium FII path:
`ranked_count >= n_assets` or the run fails as `infeasible_universe` with the
counts, thresholds, and profile in the message. A shortage is never absorbed by
shrinking `n_assets` at runtime; changing `n_assets` or a threshold is only
possible as a recorded policy change (new ADR + policy-version bump + pinned test
update in the same change).

## Consequences

Positive:

- The conservative anchor becomes executable: 12 ranked ≥ 10 with sector
  diversity that can satisfy `hhi_max=0.25` (min 0.24).
- The whole conservador→moderado interpolation band becomes feasible (0/21 vs
  11/21 failing sampled points), so personalized scores near the conservative
  end stop failing at run time.
- The 3e9 floor still excludes 575 of 588 FIIs (13 eligible): the "large,
  liquid" intent of the conservative filter is preserved at a lower size floor.
- Failures that do occur name their numbers, and `n_assets` can never drift
  downward without an explicit, versioned record.

Trade-offs:

- The feasibility margin at the anchor is thin: 12 ranked vs 10 required and a
  minimum sector HHI of 0.24 against a 0.25 cap. A data refresh that drops two
  scorables or shifts sector mix can re-break conservador — by design this now
  fails loudly (`infeasible_universe`) instead of silently; the pinned matrix
  test on the committed snapshot is the tripwire.
- Eligible FIIs between 3e9 and 5e9 of equity are smaller than the original
  policy intended; the sensitivity table above is the recorded justification.
- The calibration diverges from `STOCK_RULES["conservador"]` (`cap_min=5e9`),
  which stays as-is; `suitability.html` mirrors the stock policy and must not
  absorb the FII change (guarded by test).
- Basic and Premium share `FII_RULES`, so Basic runs inherit the calibration;
  the version bump keeps that visible in provenance.

## Revisit when

The FII snapshot refreshes and the pinned matrix counts change (7/13/28/50 or
the ranked series), when `limitar_concentracao`'s `hhi_max=0.25` is redefined, or
when the conservative profile's diversification intent changes. Any of these
warrants a new sensitivity pass and a new ADR rather than editing `FII_RULES`
in place.

## Source references

- Counts computed with `py/fii_selection.py` (`load_fii_data`, `apply_fii_eligibility`,
  `build_fii_scores`) over `data/raw/status_invest_fii.csv` (588 rows) in the
  `t3/issue-12` worktree.
- Policy source: `api/app/services/premium_policy.py:69-88` (`FII_RULES`),
  `:199-243` (interpolation), `:229-230` (`hhi_max=0.25`).
- Failure path: `py/fii_selection.py:392-396`, `api/app/adapters/premium_optimization.py:537-546`,
  `api/app/services/premium_executor.py:174-212`,
  `api/app/repositories/recommendations.py:196-207`.
- Prior spec obligation: `.specs/features/profile-v1-premium-optimization/spec.md:281-283`,
  `:293-295`.
