# v0-legacy baseline bundle

Frozen scientific baseline for the v0 stock-selection protocol, archived from commit
`97ea3275293da4e1902fcfeb684c814f04f5368f` (2025-12-23, "last execution with profile").
Every byte under `code/`, `data/` and `reference/` was written with
`git show 97ea3275293da4e1902fcfeb684c814f04f5368f:<source_path>` and is never edited in place.
A new baseline is a new `baselines/<id>/` directory.

Directory `outputs/` in the current working tree no longer holds these numbers. This bundle is
the authoritative v0 evidence; cite it, not `outputs/`.

## What is here

| Path | Files | Role | Why it is archived |
| --- | --- | --- | --- |
| `code/` | 3 | executed | `ga.py`, `ga_multiple_runs.py`, `profiles.py` - the modules the replay calls |
| `data/processed/` | 5 | executed (4 CSVs), provenance_only (the parquet) | the cleaned fundamentals the protocol ranks, plus the universe filter behind them |
| `reference/` | 9 | comparison | the recorded outcome: per-profile consensus portfolios, per-profile run metrics, and the multi-run summary |
| `baseline.json` | - | authored | the manifest: `path`, `source_path`, `sha256`, `git_blob_sha1`, `bytes`, `mode`, `role` per covered file |
| `protocol.json` | - | authored | the parameters, the run and seed rule, the consensus rule and the comparison tolerance |
| `tools/` | 4 | authored | `build_manifest.py`, `verify_manifest.py`, `replay.py`, `_legacy_driver.py` |

`data/processed/eligibility_filters.parquet` is `provenance_only`: no covered module reads it.
It is the archived `ticker, cap, liq` audit of the filter that decided which tickers reached
`fundamentals_clean_caio.csv`, so it is the only record of the universe the replay ran on.

## Verify the bytes

```bash
python3 baselines/v0-legacy/tools/verify_manifest.py
```

Exit 0 with `status: VERIFIED` and `files: 17` means every covered file still matches its blob
at `97ea327`. Exit 1 means drift: each finding names the `path` and the `kind` - `missing`,
`modified` or `unexpected`. Exit 3 means `baseline.json` itself cannot be trusted.

## Replay the numbers

```bash
python3 baselines/v0-legacy/tools/replay.py
```

The report goes to `outputs/baseline_v0_legacy/replay_report.json` (untracked; it is a view of a
run, not evidence) and the same JSON object goes to stdout. Flags: `--bundle`,
`--report`, `--budget-seconds` (default 1800) and `--driver` (default
`tools/_legacy_driver.py`, resolved against the bundle).

`status` is one of three, and never a fourth:

- `REPRODUCED` (exit 0) - every compared field matched within `rtol`/`atol` `1e-9`
- `MISMATCH` (exit 1) - the replay executed and disagreed; `fields` lists each differing field
  with its `recorded` and `recomputed` value
- `UNVERIFIABLE` (exit 2) - the replay could not run; `cause` is one of `hash_mismatch`,
  `dependency_unavailable`, `dataset_unavailable`, `protocol_mismatch`, `runtime_error`,
  `runtime_budget_exceeded`

`dataset_unavailable` also covers a recorded CSV that cannot be read as `run_id` -> row - no
`run_id` column, a `run_id` that is not an integer, an unreadable or malformed file - because then
the recorded outcome is not available to compare against. Every such input is reported, never
crashing the CLI.

Exit 3 means no trustworthy report could be produced: either a covered byte moved while the driver
ran, or `--report` resolved inside a covered subtree (`code/`, `data/` or `reference/`), in which
case the replay refuses before running anything and names the path it refused. Nothing is ever
written under `reference/`; `golden_output_written` is always false.

## Two numbers that do not agree, on purpose

> **INVALID (P0-02).** The archived `fitness` values are invalid under the P0-02 capture defect.
> The record block (`ga.py:115,129,132-135`) pairs a generation's best fitness with the
> population it did not come from, so an archived `fitness` cell is never the fitness of the
> portfolio this bundle returns. `baselines/v0-corrected/status.json` is the verdict map:
> one entry per recorded value, with its evidence and its verdict. The corrected reference and
> the methodological note live under `baselines/v0-corrected/`. The map also names the values
> this defect does NOT explain: `score_provenance_inconsistent` and `not_verifiable_at_head`
> rows carry their own reason and do not borrow the defect's name.

The replay reports `reported_fitness` (the archived `fitness` cell for that run) beside
`recomputed_fitness` (the archived `ga.fitness` applied to the chromosome the run returned).
They disagree because `ga.py:132-134` pairs the new population with the previous generation's
scores. The replay records both and lets `reported_fitness` alone decide `status`: repairing the
baseline here would destroy the measurement that P0-02 needs.

## Known gaps

`97ea327` declares no dependency set, so `protocol.json` records `dependencies.status` as
`UNVERIFIABLE` with `7d8d6a0` and `py/requirements.txt` as post-hoc evidence. A machine without
`numpy` and `pandas` therefore reports `UNVERIFIABLE` / `dependency_unavailable` rather than
guessing versions. The probe applies to the archived driver only: a substitute passed with
`--driver` declares its own imports, which is what lets the `REPRODUCED`, `MISMATCH` and failure
branches be exercised without `numpy` and `pandas` installed.

## Next step for a reader

Read `protocol.json` for how the numbers were produced, `baseline.json` for what the bytes are,
and the report for whether they still reproduce. If you need the caio consensus portfolio
itself, it is `reference/carteira_caio_consensus.json`. If you need a figure in the thesis, cite
this directory plus the commit SHA above.
