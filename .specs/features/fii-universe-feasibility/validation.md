# Validation — FII universe feasibility policy (issue #12, P0-05)

## Validation: PASS — 0 blocking, 3 nonblocking

| Field | Value |
| --- | --- |
| Feature | `fii-universe-feasibility` |
| Branch / HEAD | `t3/issue-12` @ `d3226cedb4d7aaaea7916f0e9d16799a6599ca2b` |
| Diff range | `aa5e1bc..HEAD` (11 commits: 5 feat/test + 1 bookkeeping, then 4 test + 1 bookkeeping) |
| Verifier | independent (did not author the feature); all numbers recomputed in this worktree |
| Worktree state | `git status --porcelain` empty before and after verification; sensor ran only in `/tmp/opencode/verify-mut/repo` |
| Interpreter | `/tmp/opencode/tcc-venv/bin/python3` (pandas 3.1.0, numpy, fastapi, sqlalchemy, tqdm present) |

Scope touched by the feature — exactly four production files, no others:
`api/app/services/premium_policy.py` (+3/−3), `api/app/services/premium_executor.py` (+21/−6),
`api/app/adapters/premium_optimization.py` (+24/−1), `py/fii_selection.py` (+96/−0).
`git diff --stat aa5e1bc..HEAD -- suitability.html` is empty (stock calculator untouched).

---

## Gates (observed, not inherited)

| Gate | Command | Observed |
| --- | --- | --- |
| Full | `PYTHONPATH=api:py /tmp/opencode/tcc-venv/bin/python3 -m unittest discover -s api/tests -q` | `Ran 110 tests in 55.286s` / `OK` |
| Quick | `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_fii_selection -q` | `Ran 23 tests in 22.162s` / `OK` |
| Build | `/tmp/opencode/tcc-venv/bin/python3 -m unittest tests.test_premium_research_contracts -q` | `Ran 7 tests in 0.426s` / `OK` |

Baseline before the feature was 89 / 7 / 7; the deltas are +21 API tests, +16 selector tests,
0 research tests. No test file in the range shows a deletion (`git diff --stat` shows additive-only
hunks for `api/tests/*` and `tests/test_fii_selection.py`), and `tasks.md` changed only by flipping
`- [ ]` to `- [x]` (46/46 one-for-one, no criterion text changed).

All three gates also pass unmodified in the scratch copy, which is what makes the mutation results
below attributable to the mutations rather than to the environment.

---

## Requirement-by-requirement verdict

| Req | Verdict | Evidence |
| --- | --- | --- |
| UNIV-01 | MET | Gate `ranked >= n_assets` at `py/fii_selection.py:227`; called only from the explicit branch at `py/fii_selection.py:662-669`, after `load_fii_data → preprocess_fii → build_fii_scores` (`:654-660`) and before the workspace mkdir (`:674-675`) and the run loop (`:686-731`). Boundary case proven by `tests/test_fii_selection.py:292-306` (`ranked == n_assets` proceeds, `n_selected == 6`). Row-by-row proof over the pinned matrix at `tests/test_fii_selection.py:328-352`. |
| UNIV-02 | MET | No shortage is ever absorbed: `py/fii_selection.py:230-237` raises before any write; the failure test asserts `workspace`/`portfolio.json`/`clean.csv` do not exist (`tests/test_fii_selection.py:266-290`). `n_assets` is read once per call (`py/fii_selection.py:480`, `:661`) and never re-assigned downward anywhere (`grep -rn "n_assets\s*=\s*min\|n_assets\s*=\s*max\|min(n_assets\|n_assets -= \|n_assets = len" api py` outside tests → only `py/backtest_analysis.py:612`, unrelated); the GA itself raises instead of shrinking when `m < n_assets` (`py/core/optimizer.py:208-210`). Silent-shrink mutant M13 killed. |
| ERR-01 | MET | Typed error carries all six fields (`py/fii_selection.py:120-178`); adapter maps it to `infeasible_universe` **before** the generic handler (`api/app/adapters/premium_optimization.py:549-561`, generic at `:562`); executor persists the code and renders the six fields (`api/app/services/premium_executor.py:189-218`). Exact-string persistence proven by `api/tests/test_premium_executor.py:301-323` and end-to-end through adapter+executor at `api/tests/test_product_routes.py:509-527`. |
| ERR-02 | MET | Message is built only from `details`, never `str(exc)` (`api/app/services/premium_executor.py:205-218`); rendered example is 146 chars, contains no `/`, and equals the design template byte-for-byte (recomputed, see Invariants §4). Route-level sanitisation `api/tests/test_product_routes.py:572-574`; executor-level `api/tests/test_premium_executor.py:326-352`; column bounds `String(64)`/`String(512)` at `api/app/db/models.py:91-92` with the unchanged truncation at `api/app/repositories/recommendations.py:205-206`. Raw-text mutant M12 killed, including by the two pre-existing sanitisation tests. |
| ERR-03 | MET | The shortage is never mapped to `engine_error`/`infeasible_constraints` (`api/tests/test_premium_optimization.py:482-505`) while the HHI text path keeps its old classification (`:506-527`). No partial recommendation is visible: `mark_failed` clears `result_json`/`classes`/`assumptions`/`risks` (`api/app/repositories/recommendations.py:207-212`), asserted at `api/tests/test_product_routes.py:509-527` (`classes/stocks/fiis == []`, `provenance is None`). Wrong-code mutant M5 killed by 5 tests. |
| POL-01 | MET | `api/app/services/premium_policy.py:73` = `{"size_min": 3_000_000_000, "liq_min": 2_000_000}`, `n_assets` 10 at `:70`; versions `:12-13` = `premium-policy-v2` / `basic-policy-v2`. Pinned by `api/tests/test_premium_policy.py:222-256` and `:258-271` (old v1 strings rejected). Reverting the floor is killed by 14 failures/2 errors (M1). |
| POL-02 | MET | Production pipeline independently recomputed over `data/raw/status_invest_fii.csv`: eligible/ranked = 7/7, 13/12, 28/26, 50/47, 10/9, 61/57, 162/137 — identical to the ADR table and to `NORMATIVE_MATRIX` (`tests/test_fii_selection.py:174-183`), asserted row-by-row at `:195-231` plus the entry-point pin `n_candidates == 12` with no artifacts (`:233-252`). |
| POL-03 | MET | Band: 21 points per segment through the real `_selector_policy`/`_interpolate` (`api/tests/test_premium_policy.py:379-394`, `:430-443`); independent recomputation gave 0/21 failing points on both segments (`cons->mod` starts 10/12 @0.24, `mod->arro` ends 15/137 @0.0667). Min-HHI oracle is an exact DP over sector multiplicities (`:333-353`), cross-checked by me against brute-force subset enumeration on 300 random multiplicity vectors — all equal; pins 24/100 = 0.2400 and 15/49 = 0.3061 are computed from the real ranked sets (`:396-428`). Anchor executability: restricted run gives GA `hhi_max=0.25` and consensus HHI 0.2400, unrestricted gives `None` and 0.2800 with the same seed/budget (`:457-523`, values reproduced independently). Change control pinned by the policy test (M1/M9 killed). |
| PUB-01 | MET | Failure induced through the real adapter + real executor with a stub `fii_engine` (`api/tests/test_product_routes.py:442-507`); `repository.mark_failed` is never called by the new tests (only pre-existing tests at `:404`, `:436`). `failure_code == "infeasible_universe"` is asserted as an exact string on the stored row *before* any read-path call (`:516-518`, `:537-539`, `:562-564`). Closed-failure payload asserted at `:520-527`. |
| PUB-02 | MET | `latest-completed` returns a byte-identical `model_dump_json()` before vs after the failure (`api/tests/test_product_routes.py:529-547`) and the prior completed run stays readable and unchanged (`:555-577`). |
| DFT-01 | MET | `suitability.html` guard parses the three anchors out of `BASE_PROFILES` and compares to `STOCK_RULES` (`api/tests/test_premium_policy.py:277-301`); the page is unchanged in the range, and `FII_RULES` is defined exactly once (`grep -rn "FII_RULES\s*="` → `api/app/services/premium_policy.py:69`). `enforce_liquidity_and_size` inertness asserted for True/False at two thresholds (`:303-319`). |

---

## Load-bearing invariants (independently recomputed)

1. **Calibration / versions.** `FII_RULES["conservador"]["filters"] == {'size_min': 3000000000, 'liq_min': 2000000}`,
   `n_assets == 10`, `POLICY_VERSION == "premium-policy-v2"`, `BASIC_POLICY_VERSION == "basic-policy-v2"`
   (imported and printed, `api/app/services/premium_policy.py:12-13,69-75`). moderado (`1e9/5e5`),
   arrojado (`2e8/5e4`) and `STOCK_RULES` (`cap_min` 5e9/1e9/2e8) are unchanged — the range diff for
   that file is 3 replaced lines and nothing else.
2. **Legacy shortage check byte-identical.** `optimize_fii_portfolio` extracted from `aa5e1bc` and from
   `HEAD` by function boundaries: `diff -u` empty, `md5sum` identical (`44e7c333814dbae013d66a5247a7e089`,
   61 lines each). `git diff aa5e1bc..HEAD -- py/fii_selection.py` is `+96 −0`: nothing was removed or
   rewritten, so the legacy guard at `py/fii_selection.py:481-484` is untouched.
3. **Precondition placement.** Sole call site `py/fii_selection.py:663`; reachable only under
   `if explicit_mode:` (`:662`), after `ranked` exists (`:656-660`), before `workspace_path.mkdir`
   (`:674-675`) and before the first `run(...)` (`:686`). `grep -rn "_require_feasible_universe"` shows
   exactly one production caller. Moving it after the loop (M3) breaks 2 selector tests while the whole
   API suite stays green — i.e. the placement is pinned by the layer that owns it.
4. **Frozen executor template.** Rendered output equals `design.md:88` after substituting the fields
   (byte-for-byte comparison against the line read from the spec artifact): equal. `:.0f` behaviour
   confirmed for both integers and fractional thresholds (`3.4e9 → 3400000000`, `1.4e6 → 1400000`).
   All other codes keep their `aa5e1bc` strings — the message map is context, not a changed hunk, and
   every code is pinned in `api/tests/test_premium_executor.py:354-372`.
5. **No path / no raw text.** `_safe_failure_message` reads `fields[...]` only; `details` is produced
   from the typed exception's frozen fields (`api/app/adapters/premium_optimization.py:553-560`), which
   are ints/floats plus `selection_preset` (bounded vocabulary). The production label was traced end to
   end: `resolve_recommendation_policy(...).fiis` → `_selector_config` (`:304-311`) → `run_fii_selection`
   → `perfil conservador`, verified by running that chain against the real snapshot with `n_assets`
   forced to 13 (raised `FiiUniverseInfeasibleError(profile='conservador', eligible=13, ranked=12,
   required=13, size_min=3e9)`).
6. **`details` immutability.** `PremiumOptimizationError.__init__` wraps in `MappingProxyType`
   (`api/app/adapters/premium_optimization.py:96-98`); item assignment raises `TypeError` (verified).

---

## Test quality review

- **T6 inducer is real.** `induce_infeasible_universe_failure` (`api/tests/test_product_routes.py:442-507`)
  builds the policy with the real `resolve_recommendation_policy`, queues a real `RecommendationRun`, and
  calls the real `run_premium_optimization` **and** the real `PremiumExecutor` with a stub `fii_engine`
  that raises `FiiUniverseInfeasibleError`. `repository.mark_failed` appears in that file only in
  pre-existing tests (`:404`, `:436`). Code asserted as an exact string first, endpoints after.
- **T8 is an exact DP, not enumeration.** `_min_sector_hhi_sq` (`api/tests/test_premium_policy.py:333-353`)
  is a 0/1 knapsack over multiplicities; `grep` for `itertools|combinations|product(` in the two new test
  modules returns nothing. I validated the oracle against `itertools.combinations` brute force on 300
  random cases — all agree. The 0.2400 and 0.3061 pins come from `build_fii_scores` output on the
  committed CSV, not from constants.
- **T8 sweep size.** `range(21)` per segment = 21 + 21 points, both checks asserted per point, and the
  coarse 0.00–0.50 grid is proven a subset (`:436-438`). Independent rerun reproduced 0/21 failing points
  on each segment.
- **T9 is not vacuous.** It resolves the policy through `resolve_premium_policy`, runs the real selector,
  records the `hhi_max` actually passed to `GeneticAlgorithm` (`[0.25]` vs `[None]`), and asserts
  measured consensus HHI `0.2400` vs `0.2800`. I also diffed every field of the two resolved `fiis`
  policies: only `liquidity_and_size_filters["hhi_max"]` differs (see nonblocking finding 3).
- **No tautologies identified.** Every new assertion compares an observed value to a spec-defined
  constant (counts, codes, exact message, HHI bounds) or to an independently derived baseline
  (`model_dump_json()` before/after). No test restates an implementation expression, no `try/except/pass`,
  and no assertion is reachable only when the feature is reverted (all 13 mutants below die).

---

## Discrimination sensor

Scratch copy at `/tmp/opencode/verify-mut/repo` (full tree, `.git` excluded, `__pycache__` purged);
baseline green (110/23). Each mutant applied alone with an exact-match substitution guard, gates re-run,
then restored from a pristine snapshot. Real worktree never touched.

| # | Mutant | Result | Killed by |
| --- | --- | --- | --- |
| M1 | conservador `size_min` reverted to `5_000_000_000` | killed | `test_conservador_fii_rules_pin_the_adr_0002_calibration`, `test_min_sector_hhi_dp_matches_the_hand_computed_pins`, band test (11/21 points, matching ADR) — 14 failures + 2 errors |
| M2 | `filters["hhi_max"] = 0.25` dropped from nested filters | killed | both `ConservadorAnchorExecutabilityTests`, `test_concentration_restriction_is_explicit_and_infeasible_intersection_fails` |
| M3 | precondition moved after the run loop | killed | `test_explicit_shortage_fails_before_execution_and_writes`, `test_scoring_shrunk_universe_reports_both_counts` (API suite stays green — expected, it stubs the engine) |
| M4 | gate on `eligible_count` instead of ranked count | killed | `test_precondition_behaviour_matches_normative_matrix` (5 sub-cases), `test_scoring_shrunk_universe_reports_both_counts` |
| M5 | adapter returns `engine_error` | killed | 2 adapter tests + all 3 route tests |
| M6 | one key (`liq_min`) dropped from `details` | killed | 3 route tests (KeyError escapes `execute_run`) + adapter details test |
| M7 | `:.0f` replaced by raw value | killed | both new executor tests + 2 route tests (`5000000000.0` leaks) |
| M8 | `_fail` stops forwarding `details` | killed | both new executor tests + 2 route tests (generic string persisted) |
| M9 | policy versions reverted to v1 | killed | `test_policy_versions_are_bumped_and_reject_the_v1_strings`, `test_persisted_provenance_reports_the_bumped_policy_versions` |
| M10 | boundary `>=` → `>` | killed | `test_precondition_behaviour_matches_normative_matrix`, boundary test (8 errors) |
| M11 | typed `except` moved after the generic `except Exception` | killed | 2 adapter tests + 3 route tests |
| M12 | `str(exc)[:512]` persisted instead of the template | killed | 2 new executor tests + 2 route tests + 2 **pre-existing** sanitisation tests |
| M13 | silent `n_assets = min(n_assets, len(ranked))` | killed | `test_explicit_shortage_fails_before_execution_and_writes`, `test_scoring_shrunk_universe_reports_both_counts` |

**13 injected, 13 killed, 0 survived.** No survivor on the failure path, the feasibility gate, the policy
calibration, or the message-sanitisation path.

---

## Findings

### 1. `_safe_failure_message` assumes a complete `details` mapping — nonblocking

`api/app/services/premium_executor.py:209-218` indexes `fields['universe_count']`, `'eligible_count'`,
`'required_count'`, `'size_min'`, `'liq_min'`, `'profile'` whenever `code == "infeasible_universe"` and
`fields` is any `Mapping`. A partial mapping raises `KeyError` inside `_fail`, and `_fail` has no
surrounding `try` that would convert it — it escapes `execute_run` (proof: mutant M6, where dropping the
`liq_min` key produced `ERROR` in all three route tests instead of a persisted failure).

Reachability traced rather than assumed:

- The only producer of `code="infeasible_universe"` is `api/app/adapters/premium_optimization.py:550-561`,
  which always passes all six keys in a single dict literal (`grep -rn "infeasible_universe" --include=*.py api py` → 2 hits: this producer and the renderer).
- `_failure_code` (`api/app/services/premium_executor.py:174-187`) returns that code only via
  `getattr(exc, "code", None)`; the only exception types with a `.code` attribute are
  `PremiumOptimizationError` and `PremiumRecommendationError` (`api/app/services/premium_recommendation.py:32-37`),
  whose codes are `snapshot_unavailable` / `plan_invalid` / `profile_invalid` / `profile_not_found`
  (`:82,145,199,270`). Neither the text heuristics nor `StatusInvestInputError` can produce the string.
- `details=None` is safe (`isinstance(None, Mapping)` is false → generic string).

So no reachable production path yields a partial mapping today. Classified **nonblocking** (hardening:
use `fields.get(...)` with defaults, or build the message only when all six keys are present).

### 2. `details` immutability is implemented but unpinned — nonblocking

`MappingProxyType` at `api/app/adapters/premium_optimization.py:96-98` is correct (verified: assignment
raises `TypeError`), but no test asserts it. A regression to `dict(details)` would keep all 110 tests
green. Cheap fix: one `assertRaises(TypeError)` on `failure.details["profile"] = "x"` in
`api/tests/test_premium_optimization.py`.

### 3. T9 pins the filter diff but not `lambda_hhi` — nonblocking

`api/tests/test_premium_policy.py:514-518` pops `hhi_max` from the restricted filter dict and compares
the rest; it does not compare `lambda_hhi`. Today `limitar_concentracao` also applies
`lambda_hhi = max(lambda_hhi, 0.25)` (`api/app/services/premium_policy.py:233-234`), which is a no-op at
the conservador anchor (0.50 > 0.25) — I confirmed all seven `SelectorPolicy` fields: only
`liquidity_and_size_filters["hhi_max"]` differs. The spec claim therefore holds, but the test would not
notice if the anchor's `lambda_hhi` ever dropped below 0.25, at which point the restricted/unrestricted
comparison would no longer isolate `hhi_max`. Strengthen by asserting `restricted.lambda_hhi ==
unrestricted.lambda_hhi`.

---

## Verdict

Every one of the 11 requirements is MET with executable evidence; all load-bearing invariants were
recomputed rather than read from the commit messages; 13/13 mutants were killed; all three gates pass on
the real tree and on an untouched copy. Three nonblocking hardening items are recorded above; none of
them weakens a spec obligation on a reachable path.

iw-verdict: APPROVED blocking=0 nonblocking=3
