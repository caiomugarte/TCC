# Status Invest sector data sync

## Problem

The project currently depends on manually downloaded, sector-separated Status
Invest CSVs. This feature adds a manual command that retrieves the current
fundamental data for all supported equity sectors and rebuilds the canonical
raw dataset consumed by preprocessing.

## Scope

- Fetch the ten Status Invest equity sectors through the public advanced-search
  export endpoint.
- Keep one row per ticker/share class, matching the current project contract.
- Normalize headers, Brazilian numeric values, and the `SETOR` column.
- Validate the complete result before replacing
  `data/raw/status_invest_fundamentals.csv`.
- Keep refresh manual; do not run it from the GA or preprocessing pipeline.

## Out of scope

- Scheduler or background job.
- Login, CAPTCHA solving, or browser automation.
- Historical snapshot retention.
- Changes to scoring, filtering, or portfolio optimization.
- Aggregating ticker classes into one company row.

## Requirements

### SI-01 — Manual refresh

WHEN the user runs `python py/fetch_status_invest.py` THEN the command SHALL
request all ten configured sectors and report the result for each sector.

### SI-02 — Canonical output

WHEN all sector downloads succeed THEN the command SHALL write one consolidated
CSV to `data/raw/status_invest_fundamentals.csv`.

### SI-03 — Existing schema

WHEN the downloaded data is normalized THEN the output SHALL contain
`TICKER`, `SETOR`, and every fundamental/filter column required by
`py/core/preprocessing.py`.

### SI-04 — Safe failure

WHEN any request, parsing step, or schema check fails THEN the command SHALL
fail without replacing the existing canonical CSV.

### SI-05 — Source compatibility

The client SHALL send the Status Invest advanced-search filter and category
type expected by the current export endpoint, with a browser-like user agent
and referer.

## Verification

- Offline tests cover CSV parsing, header normalization, sector labeling,
  required-column validation, and safe failure.
- A live manual refresh reports ten sectors and produces a readable canonical
  CSV without modifying downstream modules.

## Traceability

| ID | Status |
| --- | --- |
| SI-01 | Verified |
| SI-02 | Verified |
| SI-03 | Verified |
| SI-04 | Verified |
| SI-05 | Verified |
