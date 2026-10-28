# FII Universe Feasibility Policy Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and
follow its Execute flow and Critical Rules.** Do not search for skill files by
filesystem path. The skill is the source of truth for the full flow (per-task
cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Design**: `.specs/features/fii-universe-feasibility/design.md`
**Spec**: `.specs/features/fii-universe-feasibility/spec.md`
**Decision record**: `docs/adr/0002-fii-universe-feasibility-policy.md`
**Status**: Draft (execute only after the ADR-0002 human checkpoint is approved)

---

## Test Coverage Matrix

> Generated from codebase sampling (`tests/test_fii_selection.py`,
> `tests/test_premium_research_contracts.py`, `api/tests/test_premium_*.py`,
> `api/tests/test_product_routes.py`) and project guidelines
> (`.specs/codebase/CONVENTIONS.md`) - confirm before Execute. Provenance note:
> `.specs/codebase/TESTING.md`'s "no maintained pytest/unittest suite" claim is
> stale - a maintained stdlib `unittest` suite exists and is green (7 tests in
> `tests/test_fii_selection.py`, 89 across `api/tests/`, 7 in
> `tests/test_premium_research_contracts.py`). Framework: stdlib `unittest` (no
> pytest config, no CI).

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| --- | --- | --- | --- | --- |
| py/ selector domain logic (precondition, structured error) | unit | 1:1 to UNIV-01/UNIV-02/POL-02; every matrix row and edge case | `tests/test_fii_selection.py` | quick |
| api policy service (calibration, band, guards) | unit | 1:1 to POL-01/POL-03/DFT-01; pin assertions + band grid (21 points/segment, both checks per point) | `api/tests/test_premium_policy.py` | full |
| cross-layer selector wiring (policy → GA `hhi_max`) | integration | POL-03 executability: conservador anchor completes under `hhi_max ≤ 0.25` with `limitar_concentracao`; negative case without the restriction | `api/tests/test_premium_policy.py` | full |
| api adapter (error classification) | unit | ERR-01/ERR-03 branches: shortage code vs generic mapping | `api/tests/test_premium_optimization.py` | full |
| api executor service (message rendering) | unit | ERR-01/ERR-02: mandated fields present, sanitisation constraints | `api/tests/test_premium_executor.py` | full |
| api routes / read path | integration | PUB-01/PUB-02: happy + failed + latest-completed stability | `api/tests/test_product_routes.py` | full |

**Test environment (verified in this worktree):** `/tmp/opencode/tcc-venv` already
contains `api/requirements.txt` (sqlalchemy, fastapi, …) plus `tqdm`, numpy and
pandas; all three gates below pass green today (89 API + 7 selector + 7 research
tests). System `python3` has no pandas, so always invoke the venv interpreter.
No install step is required; if the venv is ever rebuilt, install
`api/requirements.txt` + `tqdm` into it (house convention alternative:
`api/.venv` as in `.specs/features/profile-v1-premium-optimization/tasks.md`).

## Gate Check Commands

> Generated from codebase (verified runnable in this worktree) - confirm before Execute.

| Gate Level | When to Use | Command (from repo root) |
| --- | --- | --- |
| Quick | After py/ unit-test tasks | `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_fii_selection -v` |
| Full | After api tasks | `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q` |
| Build | After phase completion | quick + full + `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_premium_research_contracts -q` |

No linter/formatter is configured in this repository (`.specs/codebase/CONVENTIONS.md`
notes code is not uniformly formatted); the build gate is test-only.

---

## Execution Plan

Phases are ordered and run sequentially - each phase completes before the next
begins, and tasks within a phase execute in order.

### Phase 1: Policy calibration and pins

T1 and T2 are independent; both must land before the guard work.

```text
T1
T2
```

### Phase 2: Feasibility guard and failure contract

T2's pinned matrix feeds the structured error first, then classification, then
rendering.

```text
T2 -> T3 -> T4 -> T5
```

### Phase 3: Publication semantics, drift guards, and ADR evidence

Freeze the read-path contract, the duplicated-constant guards, and the ADR's
executable evidence (band grid + anchor wiring).

```text
T4 -> T6
T5 -> T6
T1 -> T7
T1 -> T8
T1 -> T9
```

---

## Task Breakdown

### T1: Calibrate conservador FII rules and bump policy versions

**What**: Apply ADR-0002 option (a) in `FII_RULES`
(`size_min` 5e9 → 3e9 for conservador; `liq_min`/`n_assets` unchanged) and bump
`POLICY_VERSION`/`BASIC_POLICY_VERSION` to v2; add pin tests asserting the
calibrated values, the version strings, and that `n_assets` values equal the
ADR-declared ones.

**Where**: `api/app/services/premium_policy.py`

**Depends on**: None

**Reuses**: `FII_RULES` structure, `api/tests/test_premium_policy.py` test style.

**Requirement**: `POL-01`, `POL-03` (change-control half)

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] `FII_RULES["conservador"]["filters"] == {"size_min": 3_000_000_000, "liq_min": 2_000_000}` and `n_assets == 10`; moderado/arrojado and `STOCK_RULES` untouched.
- [x] `POLICY_VERSION == "premium-policy-v2"` and `BASIC_POLICY_VERSION == "basic-policy-v2"`; `resolve_premium_policy`/`resolve_basic_policy` reject the old version strings.
- [x] Pin tests in `api/tests/test_premium_policy.py` assert the exact values above and the persisted `policy_version`, so any value change breaks a test until the pin is updated with it.
- [x] Post-approval bookkeeping (ADR-0002 decision = option (a) + version bump): record it in `.specs/project/STATE.md` Confirmed decisions as AD-0002. Execute runs only after the human checkpoint approves the ADR, so this checkbox can never self-approve it; until then `.specs/project/STATE.md` stays deliberately untouched.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: existing `api/tests/test_premium_policy.py` tests plus ≥ 3 new pin tests pass (no silent deletions).

**Tests**: unit
**Gate**: full
**Commit**: `feat(policy): calibrate conservador FII size floor to 3e9`

---

### T2: Pin the normative eligibility matrix

**What**: Add the normative matrix regression test over the committed snapshot
`data/raw/status_invest_fii.csv` using the **production** pipeline
(`load_fii_data` → `preprocess_fii` → `build_fii_scores`,
`py/fii_selection.py:566-572` — the same three calls `run_fii_selection`
makes; `preprocess_fii` wraps `apply_fii_eligibility` plus winsorization and
z-scores, so the eligibility-only path must not be trusted alone): eligible/ranked
counts (7/7, 13/12, 28/26, 50/47 at `liq_min=2e6` for size_min
5e9/3e9/2e9/1e9; 10/9 at 3.4e9/1.4e6; 61/57 at 1e9/5e5; 162/137 at 2e8/5e4)
with provenance comment. Counts **only** — per-row precondition behavior is
asserted in T3 (against these pinned rows), not here. Additionally pin the
production entry point itself: `run_fii_selection(...)` for the 3e9/2e6 row
(explicit config, `n_runs=1`, reduced `generations`/`pop_size` via `ga_config`,
no output artifacts) returns `n_candidates == 12`.

**Where**: `tests/test_fii_selection.py`

**Depends on**: None

**Reuses**: `_fixture_frame` pattern is not used here — this test pins the real
committed CSV; helpers from `tests/test_fii_selection.py` imports.

**Requirement**: `POL-02`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Matrix test asserts every count in the table above (both series) and fails loudly if the snapshot drifts.
- [x] The counts come from the production pipeline (`load_fii_data` → `preprocess_fii` → `build_fii_scores`); where the test also computes the `apply_fii_eligibility`-only counts it asserts equality with the `preprocess_fii` counts rather than trusting either path alone.
- [x] Production entry-point pin: `run_fii_selection(...)` for the 3e9/2e6 row reports `n_candidates == 12` (the ranked count), with `n_runs=1`, reduced GA budget, and no artifacts written.
- [x] Gate check passes: `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_fii_selection -v`
- [x] Test count: ≥ 8 new tests pass (one per matrix row plus the pipeline-equality and `n_candidates` pins).

**Tests**: unit
**Gate**: quick
**Commit**: `test(fii): pin normative eligibility matrix`

---

### T3: Add the universe-feasibility precondition with a structured error

**What**: Add `FiiUniverseInfeasibleError(FiiSelectionError)` with frozen fields
(profile label, eligible_count, universe_count/ranked, required_count, size_min,
liq_min) and a shared `_require_feasible_universe` raiser; call it **only** in
the explicit branch of `run_fii_selection`, right after the production pipeline
(`py/fii_selection.py:566-573`) where both counts exist — before the run loop
and before any workspace writes. The legacy check in `optimize_fii_portfolio`
(`py/fii_selection.py:392-396`) stays **AS IS** (same `FiiSelectionError`, same
message) and remains the only shortage guard for named-profile/offline calls —
its `eligible_count` is genuinely unavailable there (only `df_ranked` exists),
which is why the structured raiser is not shared with it. The named-profile
offline path keeps `filters=None` behavior and writes nothing new. Freeze
`__reduce__` (fields via `self.args`) so the new type round-trips through
`pickle` — `FiiSelectionError` subclasses cross the `mp.Pool` worker boundary
today (`py/fii_selection.py:594`; `py/run_fii.py --quick/--production` use
`parallel=True`), and any exception crossing it must survive pickling.

**Where**: `py/fii_selection.py`

**Depends on**: T2 (reuses the pinned matrix rows for the per-row behavior assertions)

**Reuses**: `FiiSelectionError`, `_normalise_fii_filters`, the shortage
condition shape at `py/fii_selection.py:393` (the legacy check itself is
deliberately not refactored), fixture style of `tests/test_fii_selection.py`.

**Requirement**: `UNIV-01`, `UNIV-02`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Shortage in explicit mode raises before any GA execution and before any file write; `n_assets` is never reduced anywhere.
- [x] The raised error carries all six structured fields and its exception text contains no file paths; `pickle.loads(pickle.dumps(err))` reconstructs the same type with all six fields intact.
- [x] Synthetic-fixture unit tests cover: shortage raises; exact boundary (ranked == n_assets) proceeds; and the failing variant where the eligible set satisfies `n_assets` but scoring shrinks the universe — eligible 13 / ranked 12 / `n_assets` 13 raises and reports `12 ativos utilizáveis (13 elegíveis) para 13 exigidos` (both counts in the diagnostic).
- [x] Per-row behavior over the normative matrix (rows pinned by T2): ranked ≥ `n_assets` passes the precondition; ranked < `n_assets` raises `FiiUniverseInfeasibleError` carrying both counts.
- [x] Named-profile calls are testably unaffected: a named-profile shortage raises the exact legacy `FiiSelectionError("FII universe has {N} assets; {M} required")` (type and message byte-identical), never `FiiUniverseInfeasibleError`, and writes no artifacts.
- [x] Gate check passes: `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_fii_selection -v`
- [x] Test count: existing 7 tests plus ≥ 6 new tests pass (no silent deletions).

**Tests**: unit
**Gate**: quick
**Commit**: `feat(fii): fail fast on infeasible universe with structured error`

---

### T4: Classify the shortage as `infeasible_universe` in the adapter

**What**: In the FII branch of `run_premium_optimization`, catch
`FiiUniverseInfeasibleError` before the generic `except Exception` and raise
`PremiumOptimizationError(<structured message>, "infeasible_universe")`, copying
the six shortage fields into the frozen `details` dict on the
`PremiumOptimizationError` (design component 2) for the executor; the stock
branch and the existing `infeasible_constraints`/`engine_error` heuristic stay
untouched.

**Where**: `api/app/adapters/premium_optimization.py`

**Depends on**: T3

**Reuses**: `PremiumOptimizationError(message, code)` and the `exc.code`
short-circuit in `premium_executor._failure_code`
(`api/app/services/premium_executor.py:177-179`).

**Requirement**: `ERR-01`, `ERR-03`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] A shortage raised by the FII engine surfaces as `PremiumOptimizationError` with `code == "infeasible_universe"` and the structured fields intact.
- [x] Non-shortage `FiiSelectionError`s keep today's classification; the HHI breach text path is unchanged.
- [x] Unit tests in `api/tests/test_premium_optimization.py` cover both branches with a stub `fii_engine`.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: existing tests plus ≥ 3 new tests pass.

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): map universe shortage to infeasible_universe`

---

### T5: Render the sanitized diagnostic message

**What**: Freeze the executor seam so the structured diagnostic reaches storage.
Today `_fail(run_id, code, message: str)` ignores its `message` argument and
persists `_safe_failure_message(code)` — raw text never reaches storage, and all
three call sites pass `str(exc)` at `:96`/`:117`/`:130`, so the exception's
fields are gone by then. Change the seam to `_fail(run_id, code, exc: Exception)`:
the call sites become `self._fail(run_id, self._failure_code(exc), exc)`; `_fail`
extracts `fields = getattr(exc, "details", None)` and persists
`_safe_failure_message(code, fields)`; `_safe_failure_message(code, fields=None)`
renders the frozen Portuguese template from `design.md` (ranked count, eligible
count, required count, `size_min`, `liq_min`, profile label) for
`infeasible_universe` and keeps the current fixed strings for every other code.

**Where**: `api/app/services/premium_executor.py` — `_fail` (`:189-196`),
`_safe_failure_message` (`:203-212`), call sites `:96`, `:117`, `:130`.

**Depends on**: T4

**Reuses**: `_safe_failure_message` map, sanitisation assertions in
`api/tests/test_premium_executor.py:129-131`, `:194-195`.

**Requirement**: `ERR-01`, `ERR-02`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Persisted message equals the frozen template rendering — asserted as the exact full string, e.g. `Universo FII insuficiente: 7 ativos utilizáveis (7 elegíveis) para 10 exigidos; limiares size_min=5000000000, liq_min=2000000; perfil conservador.` for the 5e9/2e6 shortage — containing all six mandated fields (this exact-string assertion supersedes substring checks).
- [x] Message contains no `/`-path fragments and no raw exception text (extend the existing sanitisation tests); length ≤ 512 and code ≤ 64 as stored.
- [x] Executor end-to-end test: a run whose optimization raises the typed shortage stores `failure_code="infeasible_universe"` and the rendered message; all other codes keep their existing fixed strings (existing tests unchanged).
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: existing tests plus ≥ 3 new tests pass.

**Tests**: unit
**Gate**: full
**Commit**: `feat(api): render universe shortage diagnostics safely`

---

### T6: Freeze Basic closed-failure publication semantics

**What**: Add integration tests on the recommendation read path asserting the
AC4 restatement: a failed Basic run is terminal with cleared result payload and
populated failure fields (no class/stock/FII payload anywhere in the read path),
`GET /v1/recommendations/latest-completed` keeps returning the previous completed
run byte-identical, and a new Basic run leaves prior completed runs unchanged.
The failure must be **induced through the executor/adapter path** — a stub
`fii_engine` raising the typed `FiiUniverseInfeasibleError` shortage, flowing
through adapter classification and executor rendering — never by calling
`repository.mark_failed` directly; and the stored code must be asserted as
`failure_code == "infeasible_universe"` specifically (exact match, not merely
truthy) before the endpoint assertions.

**Where**: `api/tests/test_product_routes.py`

**Depends on**: T4, T5

**Reuses**: existing route-test fixtures and the
`test_queued_and_failed_responses_do_not_expose_snapshot_paths` pattern
(`api/tests/test_product_routes.py:343-352`).

**Requirement**: `PUB-01`, `PUB-02`, `ERR-03`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] The failed run is produced through the executor/adapter path: a stub `fii_engine` raises `FiiUniverseInfeasibleError`, and the stored `failure_code == "infeasible_universe"` is asserted as an exact match before any endpoint assertion (the test never calls `repository.mark_failed` directly).
- [x] Tests cover completed → failed sequence for Basic: through `GET /v1/recommendations` the failed run exposes `failure_code="infeasible_universe"`, `failure_message` and empty `classes`/`stocks`/`fiis`; `GET /v1/recommendations/latest-completed` continues to return the previous completed run with its non-empty payload and `failure_code=None` (a failed run can never be returned there: `api/app/repositories/recommendations.py:147-159`).
- [x] Sanitisation assertions from `ERR-02` hold at route level.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: existing tests plus ≥ 3 new tests pass.

**Tests**: integration
**Gate**: full
**Commit**: `test(api): freeze basic closed-failure publication semantics`

---

### T7: Add drift guards for duplicated policy constants

**What**: Add a guard test asserting `suitability.html` `BASE_PROFILES` filter
anchors equal `STOCK_RULES` filters (unchanged by this feature), and a contract
test asserting `filters["enforce_liquidity_and_size"]` does not alter the
eligible set when present.

**Where**: `api/tests/test_premium_policy.py`

**Depends on**: T1

**Reuses**: `STOCK_RULES` import, `apply_fii_eligibility` via `py` on `PYTHONPATH`.

**Requirement**: `DFT-01`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Guard test reads `suitability.html` and fails if its three anchor filter pairs diverge from `STOCK_RULES`.
- [x] Contract test proves eligibility is identical with and without `enforce_liquidity_and_size` in the filters mapping.
- [x] Review checklist (spec Out of Scope, not observable behavior): the diff does not alter how `filters["enforce_liquidity_and_size"]` is computed (`api/app/services/premium_policy.py:228`) or consumed.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: ≥ 2 new tests pass.

**Tests**: unit
**Gate**: full
**Commit**: `test(api): guard duplicated policy constants from drift`

---

### T8: Pin the premium interpolation band feasibility (POL-03 band half)

**What**: Make the ADR-0002 band claim executable. On the ADR's
21-point-per-segment grid — segment fraction `f = 0.00…1.00` step 0.05,
resolved through the real policy resolution path (`_selector_policy`/
`_interpolate`, `api/app/services/premium_policy.py:199-243`) with
`limitar_concentracao` in the restrictions, as `selector_score = 0.5·f` for
conservador→moderado (the score axis spans 0.00…0.50) and
`selector_score = 0.5 + 0.5·f` for moderado→arrojado — assert per point
(i) the ranked universe count ≥ the interpolated `n_assets` and (ii) the minimum
achievable sector HHI at that `n_assets` ≤ 0.25. The minimum HHI must be computed
by an **exact DP/count-based method over sector multiplicities** (minimize
`Σ c_s²` subject to `Σ c_s = n_assets`, `0 ≤ c_s ≤ m_s`; HHI = `Σ c_s² / n_assets²`),
never by enumerating `C(m, n)` subsets — `C(57, 12)` is infeasible.

**Where**: `api/tests/test_premium_policy.py` (counts via the `py/` production
pipeline on `PYTHONPATH`)

**Depends on**: T1 (the grid is only feasible post-calibration: 0/21 failing
points per segment)

**Reuses**: `_selector_policy`/`_interpolate` (`api/app/services/premium_policy.py:199-243`),
the production pipeline for counts (`py/fii_selection.py:566-572`).

**Requirement**: `POL-03`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Band-grid test asserts both checks at every one of the 21 conservador→moderado points and all 21 moderado→arrojado points (0/21 failing per segment, matching ADR-0002); the coarse score grid `0.00…0.50` step 0.05 is a subset of the swept points.
- [x] Min-HHI is computed by the exact count-based DP and the method is sanity-pinned against hand-computed values (10-of-12 at the 3e9 anchor = 0.2400; the 7-asset set = 15/49 = 0.3061); subset enumeration appears nowhere.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: ≥ 3 new tests pass.

**Tests**: unit
**Gate**: full
**Commit**: `test(policy): pin premium interpolation band feasibility`

---

### T9: Prove the conservador anchor executes under `limitar_concentracao`

**What**: Make the ADR-0002 central claim ("the conservador anchor becomes
executable") executable end-to-end. Resolve the conservador FII policy through
the real resolution path (`_selector_policy`/`resolve_premium_policy` at the
conservador anchor, `api/app/services/premium_policy.py:199-243`) with
`limitar_concentracao` in the restrictions, run the real selector
(`run_fii_selection` explicit mode, `py/fii_selection.py:488+`) with a reduced
GA budget (override `generations`/`pop_size` via `ga_config`, `:543-544`), and
assert completion with consensus HHI ≤ 0.25. Add the negative case resolving the
same policy **without** `limitar_concentracao`, same seed and budget, showing the
wiring is exactly what differs: the nested filters carry no `hhi_max`
(`premium_policy.py:228-230`) so the GA runs unconstrained and produces a
consensus HHI > 0.25 (measured 0.2800 with the production budget), while the
restricted run carries `hhi_max=0.25` through the nested filters into the GA
settings (`py/fii_selection.py:376-377`, `:405`) and stays green at the
post-check (`:637-640`).

**Where**: `api/tests/test_premium_policy.py`

**Depends on**: T1 (the calibrated `size_min=3e9` anchor)

**Reuses**: `resolve_premium_policy`/`_selector_policy`, `run_fii_selection`
explicit mode and its `ga_config` override (`py/fii_selection.py:543-544`).

**Requirement**: `POL-03`

**Tools**:

- MCP: NONE
- Skill: `tlc-spec-driven`

**Done when**:

- [x] Restricted run (conservador + `limitar_concentracao`): completes; `hhi_max == 0.25` present in `liquidity_and_size_filters` and reaching the GA settings; consensus HHI ≤ 0.25 (measured 0.2400 for 4/4 seeds with the production budget).
- [x] Unrestricted run (restriction absent, same pinned seed and budget): nested filters carry no `hhi_max`; the run completes; consensus HHI > 0.25 — the measured direction (0.2800) is asserted with the pinned seed, not the exact production value.
- [x] Gate check passes: `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q`
- [x] Test count: ≥ 2 new tests pass.

**Tests**: integration
**Gate**: full
**Commit**: `test(policy): prove conservador anchor executability under hhi cap`

---

## Phase Execution Map

```
Phase 1 -> Phase 2 -> Phase 3

Phase 1:  T1   T2
Phase 2:  T2 ------→ T3 ------→ T4 ------→ T5
Phase 3:  T4 ------→ T6
          T5 ------→ T6
          T1 ------→ T7
          T1 ------→ T8
          T1 ------→ T9
```

Execution is strictly sequential - there is no intra-phase parallelism. With 9
tasks the feature packs into two task-budgeted batches (Phases 1+2 = 5 tasks;
Phase 3 = 4 tasks), so Execute presents the skill's sub-agent batch offer before
starting and runs inline only if the offer is declined; the always-on Verifier
still runs after the final task.

---

## Task Granularity Check

| Task | Scope | Status |
| --- | --- | --- |
| T1: Calibrate FII rules + version bump | 1 file + co-located tests | ✅ Granular |
| T2: Pin eligibility matrix | 1 test file | ✅ Granular |
| T3: Precondition + structured error | 1 file + co-located tests | ✅ Granular |
| T4: Adapter classification | 1 file + co-located tests | ✅ Granular |
| T5: Failure rendering | 1 file + co-located tests | ✅ Granular |
| T6: Publication semantics tests | 1 test file | ✅ Granular |
| T7: Drift guard tests | 1 test file | ✅ Granular |
| T8: Pin premium interpolation band | 1 test file | ✅ Granular |
| T9: Anchor wiring e2e (conservador + `limitar_concentracao`) | 1 test file | ✅ Granular |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| --- | --- | --- | --- |
| T1 | None | alone in Phase 1 | ✅ Match |
| T2 | None | alone in Phase 1; source of the `T2 -> T3` cross-phase edge in Phase 2 | ✅ Match |
| T3 | T2 | `T2 -> T3` (cross-phase edge in Phase 2 block and map) | ✅ Match |
| T4 | T3 | `T3 -> T4` | ✅ Match |
| T5 | T4 | `T4 -> T5` | ✅ Match |
| T6 | T4, T5 | `T4 -> T6`, `T5 -> T6` | ✅ Match |
| T7 | T1 | `T1 -> T7` | ✅ Match |
| T8 | T1 | `T1 -> T8` | ✅ Match |
| T9 | T1 | `T1 -> T9` | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| --- | --- | --- | --- | --- |
| T1 | api policy service | unit | unit | ✅ OK |
| T2 | py/ selector tests | unit | unit | ✅ OK |
| T3 | py/ selector domain logic | unit | unit | ✅ OK |
| T4 | api adapter | unit | unit | ✅ OK |
| T5 | api executor service | unit | unit | ✅ OK |
| T6 | api routes / read path | integration | integration | ✅ OK |
| T7 | api policy service | unit | unit | ✅ OK |
| T8 | api policy service (band feasibility) | unit | unit | ✅ OK |
| T9 | cross-layer selector wiring (policy → GA `hhi_max`) | integration | integration | ✅ OK |
