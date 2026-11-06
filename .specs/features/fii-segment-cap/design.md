# FII Segment Cap and Policy v4 Design

**Spec**: `.specs/features/fii-segment-cap/spec.md`
**Status**: Draft
**Conforms to**: AD-026 (protection-zone contract; extended, not superseded, by AD-027), AD-022 (zone semantics, unchanged when no cap is given), AD-025 (NaN-score order in `rank_candidates`, untouched).

No approach exploration: the issue's `## Decisões` fixes the mechanism at every stage (GA infeasibility like `hhi_max`, consensus skip, zone exit reason). This design only fixes names, signatures and order.

---

## Architecture Overview

One integer, `max_per_segment`, travels from the policy to the three places that decide FII membership. Every new parameter defaults to `None`, and `None` takes the exact pre-change code path.

```mermaid
graph TD
    P["premium_policy.resolve_premium_policy<br/>fiis.max_per_segment = 2<br/>FII n 4/5/6, v4, fii-selection-v2"] --> PJ["policy_json (persisted)"]
    PJ --> A["adapter _selector_config<br/>copies + validates max_per_segment"]
    A --> R["fii_selection.run_fii_selection<br/>capacity check, post-check"]
    R --> O["optimize_fii_portfolio -> GeneticAlgorithm(max_per_segment)<br/>fitness: over cap = -inf"]
    R --> C["_consensus(max_per_segment)"]
    C -->|no incumbents| T["protection_zone.take_within_segment_cap"]
    C -->|incumbents + zone| S["protection_zone.select_with_fallback(segment_of, max_per_segment)"]
    S --> Z["apply_protection_zone: segment cut, capped fill, capped challengers"]
    S --> Z0["apply_protection_zone K=0: capped consensus / HHI fallback"]
    B["resolve_basic_policy<br/>FII 10/12/15, no cap, fii-selection-v1"] -.-> PJ
```

Stocks never receive the key: `stocks.max_per_segment` is `None`, so `_selector_config` emits no key and `py/pipelines/multi_run.py` is not touched.

---

## Code Reuse Analysis

### Existing Components to Leverage

| Component | Location | How to Use |
| --------- | -------- | ---------- |
| `hhi_max` infeasibility | `py/core/optimizer.py:92-95` (validation), `:132-135` (fitness) | Same shape for `max_per_segment`: validate in `__init__`, return `-np.inf` in `fitness`. |
| `_is_integer` | `py/core/protection_zone.py:71-72` | Integer check that rejects `bool`; reuse in the zone validator. The GA gets the same three-line check (it cannot import from `protection_zone` without a cycle risk: `protection_zone` must stay pandas-free and import-light). |
| `rank_candidates` / sort key | `py/core/protection_zone.py:30-54`, `py/fii_selection.py:588-589` | Rank order is unchanged; the cap only filters while walking it. |
| `select_with_fallback` HHI fallback | `py/core/protection_zone.py:198-253` | Pass the cap to both the zone call and the `K = 0` call; fallback logic unchanged. |
| FII HHI post-check | `py/fii_selection.py:768-770` | Same pattern for the segment post-check (SEG-29). |
| `_require_feasible_universe` | `py/fii_selection.py:214-238` | Sibling `_require_segment_capacity` raising plain `FiiSelectionError` (SEG-28). |
| `PROTECTION_ZONE` / `_protection_zone` / `_selector_dict` | `api/app/services/premium_policy.py:84-87`, `:173-190` | `max_per_segment` follows `protection_zone`: optional dataclass field, dropped from the dict when `None`. |
| `_selector_config` zone copy | `api/app/adapters/premium_optimization.py:311-316` | Same copy block for `max_per_segment`, with an integer check. |
| Pre-change helper copies in tests | `tests/test_protection_zone_selectors.py:421-441` (`_pre_change_fii_consensus`) | Regression oracle for SEG-18. |
| Real-data selector harness | `api/tests/test_premium_policy.py` `ConservadorAnchorExecutabilityTests` (`_RecordingGeneticAlgorithm`, `GA_BUDGET`) | Pattern for the real-data v4 executability test (T11). |

### Integration Points

| System | Integration Method |
| ------ | ------------------ |
| Persisted runs (`recommendation_runs.policy_json`) | New optional key `fiis.max_per_segment`; old rows lack it and execute uncapped (SEG-32). No migration. |
| Run seed | `derive_premium_seed` hashes `POLICY_VERSION`; v4 seeds differ from v3 (A-11). |
| #9 tooling (`scripts/multirun_convergence.py`, `scripts/protection_zone_turnover.py`, `scripts/benchmark_premium.py`) | Resolve the live policy; no code change. Tests follow (A-10). |

---

## Components

### `GeneticAlgorithm` segment cap

- **Purpose**: Make a candidate with more than `max_per_segment` tickers of one `SETOR` infeasible.
- **Location**: `py/core/optimizer.py`
- **Interfaces**:
  - `GeneticAlgorithm.__init__(..., early_stopping_min_delta=1e-6, max_per_segment: Optional[int] = None)` - new keyword appended last so no positional caller shifts. Raises `ValueError("max_per_segment must be an integer >= 1")` for `0`, negatives, floats, `bool`, strings.
  - `fitness(df, mask) -> float` - order: (1) `mask.sum() != n_assets` → `-inf` (unchanged); (2) `selected = df.iloc[mask.astype(bool)]` (unchanged); (3) **new, only when `self.max_per_segment is not None`**: `if "SETOR" not in df.columns: raise ValueError("max_per_segment requires a SETOR column")`; `if selected["SETOR"].value_counts().max() > self.max_per_segment: return -np.inf`; (4) HHI and `hhi_max` (unchanged); (5) return (unchanged).
- **Dependencies**: pandas (already imported).
- **Reuses**: `hhi_max` pattern.
- **Not changed**: `crossover`, `mutate`, `initialize_population`, `optimize`, `optimize_portfolio`, `_explicit_ga_config` (stocks never pass the cap; RNG call order is untouched, so `None` is byte-identical).

### Zone segment cap

- **Purpose**: Apply the cap inside the protection-zone rule and expose one capped greedy pick for the pure consensus.
- **Location**: `py/core/protection_zone.py`
- **Interfaces**:
  - `SEGMENT_CAP_DEPARTURE = "saiu: teto do segmento"` (module constant).
  - `validate_segment_cap(max_per_segment, segment_of, tickers) -> None` - `None` cap returns at once; else `max_per_segment` must pass `_is_integer` and be `>= 1`, `segment_of` must be a mapping containing every ticker in `tickers`; raises `ValueError` naming `max_per_segment` or `segment_of` (SEG-25).
  - `take_within_segment_cap(ordered: Sequence[str], n_assets: int, *, segment_of: Mapping[str, str] | None, max_per_segment: int | None) -> list[str]` - `None` cap returns `list(ordered[:n_assets])`; otherwise walks `ordered`, skips a ticker whose segment already has `max_per_segment` picks, stops at `n_assets` (SEG-15).
  - `apply_protection_zone(..., eligible, dedup_base_company, segment_of: Mapping[str, str] | None = None, max_per_segment: int | None = None) -> ZoneSelection` - new keyword-only parameters. With a cap:
    1. `validate_segment_cap(max_per_segment, segment_of, candidates)` after the eligible filter.
    2. Walk `qualifying` (rank order); a ticker whose segment already has `max_per_segment` protected entries goes to `segment_capped`; the rest form `qualifying_capped`. Then `protected = qualifying_capped[:n_assets]`, `capped = set(qualifying_capped[n_assets:])` (SEG-19, SEG-20, A-5).
    3. Fill: `for ticker in ranking: if len(held) >= n_assets: break; if ticker in held or segment_full(ticker): continue; held.add(ticker)` (SEG-21). Without a cap the loop is the current one verbatim.
    4. Swap: `challenger = next((t for t in ranking if t not in held and t not in replaced and fits(t, weakest)), None)` where `fits` is `True` without a cap, else "count of `segment_of[t]` in `held - {weakest}` < `max_per_segment`" (SEG-22). Swap test unchanged.
    5. Departures: `inelegível`, then `substituído por`, then `t in segment_capped` → `SEGMENT_CAP_DEPARTURE`, then `vagas reduzidas`, then `fora do top K` (A-6).
  - `select_with_fallback(..., hhi_of, hhi_max, segment_of=None, max_per_segment=None)` - forwards both keywords to the zone call and the `K = 0` call (SEG-23). Record fields unchanged (A-13).
- **Dependencies**: none new (stays pandas-free).
- **Reuses**: `_is_integer`, `rank_candidates`, existing loop structure.

### FII selector wiring

- **Purpose**: Read the cap from the explicit config and hand it to the GA and the consensus; fail fast or defensively when it cannot hold.
- **Location**: `py/fii_selection.py`
- **Interfaces**:
  - `optimize_fii_portfolio(...)` - in the `config is not None` branch copy `config["max_per_segment"]` into `settings` when present; construct `GeneticAlgorithm(..., hhi_max=settings.get("hhi_max"), max_per_segment=settings.get("max_per_segment"))`. The `ga_config`-only branch also honours the key if a caller puts it there.
  - `_consensus(portfolios, df_ranked, n_assets, *, incumbents=None, protection_zone=None, hhi_max=None, max_per_segment=None)` - builds `segment_of = dict(zip(df_ranked["TICKER"], df_ranked["SETOR"].astype(str)))` only when the cap is set; pure path = `take_within_segment_cap(<current sorted list>, n_assets, ...)`; zone path passes `segment_of` and the cap to `select_with_fallback` (SEG-15, SEG-26). Without a cap both paths are the current code.
  - `_require_segment_capacity(ranked: pd.DataFrame, n_assets: int, max_per_segment: int) -> None` - raises `FiiSelectionError(f"FII universe cannot fill {n_assets} under max_per_segment {cap}: capacity {c}")` (SEG-28).
  - `run_fii_selection(...)` - explicit mode only: `max_per_segment = resolved_config.get("max_per_segment")`; when set, call `_require_segment_capacity` right after `_require_feasible_universe`; pass it to `_consensus`; after the HHI post-check raise `FiiSelectionError("FII selection violates the configured segment cap")` if any segment count exceeds it (SEG-29). Named-profile mode never reads the key.
- **Dependencies**: `take_within_segment_cap` from `core.protection_zone`.
- **Reuses**: HHI post-check, `_require_feasible_universe` shape.

### Premium policy v4

- **Purpose**: Encode 4/5/6, the cap, the version and the method version; keep Basic byte-identical.
- **Location**: `api/app/services/premium_policy.py`
- **Interfaces / constants**:
  - `POLICY_VERSION = "premium-policy-v4"`.
  - `FII_RULES[band]["n_assets"]`: conservador 4, moderado 5, arrojado 6 (weights, filters, λ unchanged).
  - `BASIC_FII_ASSET_COUNTS = {"conservador": 10, "moderado": 12, "arrojado": 15}`; `resolve_basic_policy` builds its FII section from `{band: {**FII_RULES[band], "n_assets": BASIC_FII_ASSET_COUNTS[band]}}`.
  - `FII_MAX_PER_SEGMENT = 2`.
  - `PREMIUM_MODEL_VERSIONS = {"allocation": "allocation-v1", "stocks": "stock-selection-v1", "fiis": "fii-selection-v2"}`, `BASIC_MODEL_VERSIONS = {... "fiis": "fii-selection-v1"}`. `resolve_premium_policy` defaults to the former; `resolve_basic_policy` replaces `provenance.model_versions` with the latter when the caller passed none.
  - `SelectorPolicy.max_per_segment: int | None = None` (after `protection_zone`); `_selector_dict` pops it when `None`.
  - `resolve_premium_policy`: `fiis = replace(_protection_zone(_selector_policy(FII_RULES, ...), "fiis"), max_per_segment=FII_MAX_PER_SEGMENT)`.
- **Reuses**: `_protection_zone`, `_selector_dict` omission rule, `replace`.

### Adapter forwarding

- **Purpose**: Copy the cap from the policy section into the FII engine config.
- **Location**: `api/app/adapters/premium_optimization.py`
- **Interfaces**: `_selector_config(section)` - after the zone block: `cap = section.get("max_per_segment")`; if not `None`, require `isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1` else `PremiumOptimizationError("selector segment cap is invalid", "policy_invalid")`; set `config["max_per_segment"] = int(cap)`.
- **Not changed**: `_zone_record`, `_zone_reasons`, `_validate_selector_constraints`, error mapping (A-9).

---

## Data Models

### `SelectorPolicy` (persisted inside `policy_json`)

```python
@dataclass(frozen=True)
class SelectorPolicy:
    selection_preset: str
    n_assets: int
    factor_weights: Mapping[str, float]
    liquidity_and_size_filters: Mapping[str, object]
    lambda_hhi: float
    system_ga_config: Mapping[str, int | float | bool]
    protection_zone: Mapping[str, object] | None = None
    max_per_segment: int | None = None   # new; omitted from to_dict() when None
```

Premium v4 owner profile (selector score 0.70), `fiis` section, relevant keys:

```json
{"n_assets": 5, "max_per_segment": 2, "protection_zone": {"k_offset": 3, "K": 8, "delta": 0.11}}
```

### FII `selection_config` (adapter → selector)

Existing keys plus top-level `"max_per_segment": 2` (FIIs only).

---

## Error Handling Strategy

| Error Scenario | Handling | User Impact |
| -------------- | -------- | ----------- |
| Invalid cap in a policy section | `PremiumOptimizationError(..., "policy_invalid")` in the adapter | Run fails with `policy_invalid` (only reachable through a hand-edited stored policy). |
| Invalid cap reaching the GA or zone directly | `ValueError` naming `max_per_segment` / `segment_of` / `SETOR` | Programming error surfaced in tests. |
| Universe too small for the cap | `FiiSelectionError` before any GA run; adapter maps it to `engine_error` through its generic branch | Run fails with a clear message instead of "GA não convergiu". Unreachable on committed data (A-9). |
| Final FII selection over the cap | `FiiSelectionError("... segment cap")` | Run fails rather than deliver an over-cap sleeve; unreachable unless a bug lands. |
| Stored v3 policy executed after deploy | No key → uncapped path | Same output as before the deploy. |

---

## Risks & Concerns

| Concern | Location (file:line) | Impact | Mitigation |
| ------- | -------------------- | ------ | ---------- |
| Extra `value_counts` per fitness call when the cap is set | `py/core/optimizer.py:130-138` | FII GA wall-clock grows (FII stage only; up to 150 runs × 300 × 400 evaluations). | Accepted (issue picks fitness infeasibility over a repair operator). Stocks never pay it. T11 records the real-data runtime of the capped run next to the uncapped one in its commit body; if the FII stage is more than 1.5x slower, open a follow-up to cache factorized segment codes. |
| The zone with a cap no longer matches the AD-022 oracle (`scripts/convergence_analysis.zone_portfolio`) | `tests/test_protection_zone_equivalence.py` | An oracle check with the cap would fail by design. | The equivalence tests run without a cap and must pass unchanged (SEG-24); AD-027 records the divergence. |
| Stored FII `SETOR` labels vary in spelling | `data/raw/status_invest_fii.csv` | Two spellings of one segment would each get 2 slots. | Out of scope: the cap uses the same exact labels as `hhi_sector` (A-4); the committed file has 18 distinct labels over 588 FIIs, no spelling variants; "Logística" (1 eligible fund for arrojado) and "Imóveis Industriais e Logísticos" are separate source labels and are capped separately. |
| `ConservadorAnchorExecutabilityTests` subject drifts with n = 4 | `api/tests/test_premium_policy.py:604-622` | At n = 4 the uncapped GA reaches HHI 0.25 on its own, so `assertGreater(hhi, 0.25)` fails. | T9 fixes `n_assets` 10 in that test's config (its subject is "uncapped HHI exceeds 0.25"), keeping the assertion. |
| #9 tooling no longer replays its protocol | `scripts/multirun_convergence.py:208-216` | `check_n_assets` exits 2 for the owner profile under v4. | Intended guard (A-10). Frozen files untouched (SEG-36); Q2 logged. |
| `py/fii_selection.py` keeps growing (832 lines) | `py/fii_selection.py` | Readability. | Nit only (REVIEW.md); the change adds two small helpers and keyword plumbing. |
| Test gap: no existing test pins the GA output byte-for-byte | `py/core/optimizer.py` | SEG-14 would be unprovable. | T1 captures golden tickers/fitness/hhi/generations for two seeds on a synthetic frame at `a61164a` before editing the GA. |

---

## Tech Decisions (only non-obvious ones)

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| Where the cap filter sits in the zone | Before the `n_assets` cut | Keeps more incumbents (A-5). |
| Challenger that would break the cap | Skipped, next one tried | Same rule as fill and pure consensus; `K = 0` stays equal to the capped pure consensus (A-7). |
| GA keyword position | Appended after `early_stopping_min_delta` | No positional caller shifts. |
| Basic | Own FII counts, no cap, `fii-selection-v1` | A-2. |
| Too-small universe | Plain `FiiSelectionError`, no new error code | Unreachable on committed data; avoids adapter and web contract changes (A-9). |

Project-level decision recorded as **AD-027** in `.specs/STATE.md`.
