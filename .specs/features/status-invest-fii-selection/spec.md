# Status Invest FII selection

## Objective

Add a separate FII selection flow using Status Invest segment exports and the
existing equal-weight genetic algorithm, without changing the stock flow.

## Scope

- Fetch the currently populated FII segments from Status Invest.
- Attach the official segment label as `SETOR`.
- Store FII raw data in `data/raw/status_invest_fii.csv`.
- Normalize and score FII-specific indicators with independent configuration.
- Select the best FII assets; leave portfolio allocation to the allocation
  optimizer.
- Support single, quick, production, and max-quality FII execution modes.
- Keep stock data, stock preprocessing, stock scoring, and stock runners intact.

## Decisions

- Status Invest FII exports use `Segment` and `CategoryType=2`.
- Current populated segment IDs are recorded in the collector as an explicit
  source map; an empty segment fails the refresh rather than silently changing
  the universe.
- FII profile weights mirror the existing profile proportions, mapping stock
  profitability to an explicit FII size/cash group. These are defaults, not a
  claim that FII fundamentals have the same meaning as stock fundamentals.
- The binary GA uses equal-weight assumptions while evaluating candidates, but
  the selector does not assign final portfolio weights.
- FII execution presets mirror stock run budgets: once (1), quick (20),
  production (up to 100 adaptive parallel runs), and max-quality (up to 150
  adaptive sequential runs).
- Existing allocation analysis remains unchanged; replacing its IFIX benchmark
  with a selected FII sleeve is a later, separately specified change.

## Requirements

### FII-01 — Segment refresh

The collector SHALL request each configured Status Invest FII segment with the
FII export filter and attach its segment name as `SETOR`.

### FII-02 — FII schema

The normalized dataset SHALL contain the FII source metrics, `TICKER`, and
`SETOR`, including a canonical `CAGR VALOR COTA 3 ANOS` name for the source's
current `CORA` header typo.

### FII-03 — Safe refresh

Any request, parse, schema, empty-segment, or duplicate-ticker failure SHALL
leave the existing FII raw dataset unchanged.

### FII-04 — Separate selection

The selector SHALL load only the FII dataset, apply FII eligibility and
normalization, calculate FII scores, and reuse the binary equal-weight GA.

### FII-05 — Selection/allocation boundary

The selector SHALL write selected FII assets without a final allocation weight.
The later allocation optimizer owns the amount assigned to each asset.

### FII-06 — Execution modes

The FII CLI SHALL expose once, quick, production, and max-quality modes with
the same run budgets and convergence thresholds as the stock runner.

## Verification

- Offline tests cover FII parsing, header aliasing, segment labeling, duplicate
  detection, atomic failure, FII scoring, selection-only output, and a small GA
  run.
- A live collector run may be used to populate the new FII raw path; it must
  not touch the stock raw path.
- The complete existing test suite remains green.
