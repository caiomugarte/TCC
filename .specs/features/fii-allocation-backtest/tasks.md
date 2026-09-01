# FII Allocation Backtest Tasks

**Design:** `.specs/features/fii-allocation-backtest/design.md`
**Status:** Complete

## Execution Plan

### Phase 1: Snapshot contract

```text
T1 → T2 → T3
```

### Phase 2: Verification and integration

```text
T3 → T4 ─┐
T3 → T5 ─┼→ T6
```

## Task Breakdown

### T1: Add generic portfolio ticker loading

**What:** Extend `allocation_data.py` to validate the FII consensus artifact
while preserving the stock loader contract.
**Where:** `py/allocation_data.py`
**Depends on:** None
**Requirement:** FII-ALLOC-05, FII-ALLOC-08

**Done when:**

- [x] Valid stock/FII JSON lists yield unique non-empty tickers.
- [x] Missing, malformed, empty, duplicate, or blank ticker artifacts fail with
      `SnapshotError`.
- [x] Existing stock loader tests remain green.

**Verify:** `./py/.venv/bin/python -m unittest tests.test_allocation -v`

### T2: Fetch selected FII historical levels

**What:** Extend the allocation snapshot fetcher to fetch FII `.SA` symbols,
write `caio_fiis.csv`, record artifact/symbol/skip metadata, and expose an
optional IFIX toggle.
**Where:** `py/fetch_allocation_snapshot.py`
**Depends on:** T1
**Requirement:** FII-ALLOC-01, FII-ALLOC-05, FII-ALLOC-06, FII-ALLOC-07,
FII-ALLOC-10

**Done when:**

- [x] Selected FII tickers use the same adjusted Yahoo path as stocks.
- [x] Permanent and transient failures follow the documented behavior.
- [x] `caio_fiis.csv` and metadata are written only for valid snapshot data.
- [x] IFIX can be omitted without blocking the snapshot.

**Verify:** Mocked fetch tests cover success, permanent skip, transient abort,
malformed artifact, and optional IFIX.

### T3: Build the optimized FII sleeve in the snapshot bundle

**What:** Extend `load_snapshot_bundle()` to use FII ticker levels and
`build_equal_weight_sleeve()` for `ASSET_CLASSES[1]`, never IFIX.
**Where:** `py/allocation_data.py`
**Depends on:** T1, T2
**Requirement:** FII-ALLOC-02, FII-ALLOC-03, FII-ALLOC-04, FII-ALLOC-09,
FII-ALLOC-10

**Done when:**

- [x] FII class returns equal the fixture annual-rebalanced sleeve.
- [x] IFIX changes or absence do not change FII class returns.
- [x] Common-date and short-history behavior matches stocks.
- [x] Existing five-class row shape remains unchanged.

**Verify:** Fixture bundle test compares FII rows against a hand-calculated
equal-weight series and runs the existing allocation pipeline.

### T4: Wire FII portfolio paths into allocation CLIs [P]

**What:** Add explicit `--fii-portfolio` defaults to snapshot fetch and
allocation execution, and propagate the path into metadata.
**Where:** `py/fetch_allocation_snapshot.py`, `py/run_allocation.py`
**Depends on:** T3
**Requirement:** FII-ALLOC-01, FII-ALLOC-05

**Done when:**

- [x] Both CLIs accept custom FII artifacts.
- [x] Default points to `outputs/carteira_fii_caio_consensus.json`.
- [x] A run identifies the selected FII artifact in output metadata.

**Verify:** CLI parser tests and a synthetic end-to-end invocation pass.

### T5: Add FII allocation/backtest regression tests [P]

**What:** Add dependency-light tests for FII snapshot fetching, bundle
construction, IFIX independence, and allocation output shape.
**Where:** `tests/test_allocation.py`
**Depends on:** T3
**Requirement:** FII-ALLOC-01 through FII-ALLOC-10

**Done when:**

- [x] Tests cover selected portfolio identity, equal-weight sleeve, skips, and
      optional IFIX.
- [x] Stock fixtures still pass unchanged.
- [x] Full suite passes.

**Verify:** `./py/.venv/bin/python -m unittest discover -s tests -v`

### T6: Update project flow notes and specification status

**What:** Mark requirements verified, update the allocation notebook boundary,
and record the fixed-artifact/look-ahead decision.
**Where:** `.specs/features/fii-allocation-backtest/`, `.notebook/`
**Depends on:** T4, T5
**Requirement:** FII-ALLOC-05, FII-ALLOC-08, FII-ALLOC-10

**Done when:**

- [x] Spec traceability maps all implemented requirements.
- [x] Notebook points to the new snapshot and allocation flow.
- [x] No stock-flow contract is documented as changed.

**Verify:** `git diff --check` and manual pointer review.

## Execution Gate

Before implementation, confirm task order and tool choice. Planned tools:

- Shell/apply_patch for code and tests.
- `codenavi` for repository navigation.
- `coding-guidelines` for Python changes.
- `caveman` and `ponytail` for concise, minimal execution.
- No external MCP or web research required; the existing Yahoo integration is
  already the project source contract.
