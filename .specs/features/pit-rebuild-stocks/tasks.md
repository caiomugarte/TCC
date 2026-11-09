# PIT Rebuild Feasibility for Stocks Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/pit-rebuild-stocks/spec.md`
**Design**: `.specs/features/pit-rebuild-stocks/design.md`
**Decision record**: AD-031 in `.specs/STATE.md` (proposed)
**Status**: Draft (execute only after the human approves spec A-1..A-14, the design approach and AD-031)

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `REVIEW.md` (every CLI exit code in the documented table with a test through the real path; docs numbers equal the committed artifact; research tools never write into `data/`, `snapshots/`, `baselines/` or the registry; a new test must fail when the behaviour it names is removed), `.specs/codebase/TESTING.md` (deterministic functions, no network). Samples: `tests/test_pit_validation.py` (drives `main()`, snapshots the out-dir to prove nothing is written on failure, one test per exit code), `tests/test_fetch_status_invest.py`. Runner: stdlib `unittest`, no pytest, no CI, no coverage threshold, no linter or formatter. Strong defaults applied: every spec AC and edge case maps to at least one test; every guard gets its own input.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Pure rules (`fields`, `status_invest`, `compare`, `coverage`, `indicators`, `safety`) | unit | 1:1 to the spec ACs they own; each guard with its own input; boundary values (90/70 %, 5 %, 0.5 p.p., zero reference, zero divisor) | `tests/test_pit_rebuild_<module>.py` | `python3 -m unittest discover -s tests -p "test_pit_rebuild_<module>.py" -v` |
| Parsers (`cvm`, `cotahist`, `dividends`, `mapping`) | unit on committed text fixtures zipped in a temp dir | Every field read from the real layouts in design.md "Sources"; PIT selection boundaries (`DT_RECEB == d`, `d + 1`; session on `d`, after `d`); every unmapped reason | `tests/test_pit_rebuild_<module>.py`, fixtures `tests/fixtures/pit_rebuild/` | same |
| Network boundary (`cache`, `sources`) | unit with a fake `opener` (no network) | Cache hit, miss, hash mismatch, failed download, bad zip, offline miss; year ranges of SRC-01 | `tests/test_pit_rebuild_cache.py`, `tests/test_pit_rebuild_sources.py` | same |
| CLI (`cli`, `__main__`) | integration through `main([...])` on a fixture cache | Each exit code 0, 1, 2, 3, 4, 5 via the real subcommand path; nothing written on failure; SI CSVs and protected trees byte-identical; `--offline` makes no request | `tests/test_pit_rebuild_cli.py` | `python3 -m unittest discover -s tests -p "test_pit_rebuild_cli.py" -v` |
| Committed report | artifact test | Markdown re-rendered from the committed JSON equals the committed Markdown; JSON has both dates, the class verdict and provenance | `tests/test_pit_rebuild_artifact.py` | same pattern |
| Docs (`docs/validation/README.md` section) | none | build gate only (no numbers in the section) | - | build gate |

**Test environment:** the tool and its tests are stdlib-only (A-14), so the system `python3` (3.14.4, no pandas) runs every new test. `$PY` = `/home/caio/Documents/projetos/TCC/.venv/bin/python` (3.14.4 with numpy and pandas, shared by worktrees) runs the root suite. Commit with `git commit -F <file>`: the shell hook rewrites a `-m` message containing the word "make".

## Gate Check Commands

> Generated from codebase - confirm before Execute. Run from the worktree root.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After every task with unit tests | `python3 -m unittest discover -s tests -p "test_pit_rebuild_<module>.py" -v` |
| Full | After CLI tasks (T17, T18) and the artifact task (T20) | `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py" -v` |
| Build | Last task of each phase | Full + `python3 -m compileall -q py/pit_rebuild` + `git diff --exit-code develop -- data snapshots baselines py/core api py/config.py py/pit_validation.py` + `$PY -m unittest discover -s tests -p "test_*.py"` (root suite, about 3 min; baseline 587 OK, 2 skipped) |

---

## Execution Plan

Phases run in order; tasks within a phase run in order. A heavy task counts as two.

| Phase | Tasks | Count (heavy = 2) | Sensitive path | Risk | Suggested tier |
| ----- | ----- | ----------------- | -------------- | ---- | -------------- |
| 1. Foundation and network boundary | T1-T6 | 6 | none | Low: plumbing, but SAFE-01 and the hash check live here. | mid |
| 2. Source parsers and mapping | T7-T11 | 5 | none | Medium: PIT selection and real file layouts. | high reasoning (T7, T8, T11), mid (T9, T10) |
| 3. Computation, comparison and report | T12-T16 | 5 | none | Medium: formulas, tolerance and verdict rules decide the result. | high reasoning |
| 4. CLI, docs and real run | T17-T20 | 5 (T20 heavy) | none | Medium: exit-code table; the real run downloads several GB of zips and parses COTAHIST. | mid (T17-T19), high (T20) |

No task touches a REVIEW.md sensitive path (`py/core/**`, `api/**`, `py/snapshot_*.py`, `data/allocation*/**`, `baselines/**`). Total 20 tasks, counted 21.

### Phase 1: Foundation and network boundary

```
T1 → T2
T3
T4
T1 → T5 → T6
```

### Phase 2: Source parsers and mapping

```
T7 → T8
T9
T10
T11
```

### Phase 3: Computation, comparison and report

```
T12
T13 → T14
T15
T14 → T16
T15 → T16
```

### Phase 4: CLI, docs and real run

```
T17 → T18 → T20
T19 → T20
```

---

## Task Breakdown

### T1: Create the package and its error classes

**What**: New package `py/pit_rebuild/` (`__init__.py` with the purpose and the exit-code table of CLI-01 as docstring) and `errors.py` with `RebuildError` (`exit_code`, `field_name`, `reason`), `ArgumentError`=2, `MissingPathError`=3, `IntegrityError`=4, `PITLeakError(IntegrityError)` (ticker, date, offending date in the message), `SourceError`=5.
**Where**: `py/pit_rebuild/errors.py`
**Depends on**: None
**Reuses**: `py/pit_validation.py:94-122` error pattern
**Requirement**: CLI-01, PIT-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_errors.py`: each class's `exit_code`; `PITLeakError` is an `IntegrityError` and its message names ticker, date and offending date; the `__init__` docstring lists codes 0, 1, 2, 3, 4, 5.
- [ ] Gate check passes: Quick for `test_pit_rebuild_errors.py`
- [ ] Test count: 3 tests pass

**Tests**: unit
**Gate**: quick

---

### T2: Guard protected trees and write atomically

**What**: `assert_writable_dir(path, label)` (resolve with symlinks, `ArgumentError` when inside `REPO_ROOT/{data,snapshots,baselines}`) and `write_files_atomically(files)` (temp siblings, then `os.replace`; on error temps removed, targets untouched).
**Where**: `py/pit_rebuild/safety.py`
**Depends on**: T1
**Reuses**: none
**Requirement**: SAFE-01, RPT-01, RPT-05
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_safety.py`: one test per forbidden root (`data`, `snapshots`, `baselines`, a nested child, a symlink pointing into `data`) raises exit 2; `docs/validation` and a temp dir pass; a failure injected on the second `os.replace` leaves both pre-existing targets byte-identical and no temp file behind; success writes both files.
- [ ] Gate check passes: Quick for `test_pit_rebuild_safety.py`
- [ ] Test count: ≥ 7 tests pass

**Tests**: unit
**Gate**: quick

---

### T3: Read the indicator list from `py/config.py` without importing it

**What**: `metric_groups(config_path)` via `ast` + `literal_eval`; `SCORE_FIELDS`, `EXTRA_FIELDS`, `ALL_FIELDS`, `PERCENT_FIELDS` (A-2), `tolerance_kind(field)`.
**Where**: `py/pit_rebuild/fields.py`
**Depends on**: None
**Reuses**: `py/config.py:116-142`
**Requirement**: IND-02, CMP-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_fields.py`: `SCORE_FIELDS` equals the 16 names of `METRIC_GROUPS` in group order and `ALL_FIELDS` has 19; a temp config with an extra metric yields 17 score fields (proves it is read, not copied); `config` is not in `sys.modules` after importing `pit_rebuild.fields` in a subprocess; `tolerance_kind` returns `pp` for exactly the 8 A-2 fields.
- [ ] Gate check passes: Quick for `test_pit_rebuild_fields.py`
- [ ] Test count: ≥ 4 tests pass

**Tests**: unit
**Gate**: quick

---

### T4: Load a Status Invest CSV read-only and select the eligible universe

**What**: `load(path)` (exit 3 absent; exit 4 on a missing required column or a duplicated ticker; blank → `None`; sha256 recorded) and `eligible(csv)` with `ELIGIBILITY = {"cap_min": 200_000_000, "liq_min": 50_000}` (A-1).
**Where**: `py/pit_rebuild/status_invest.py`
**Depends on**: None
**Reuses**: eligibility rule of `py/core/preprocessing.py:103-106` (restated)
**Requirement**: SAFE-02, MAP-01 (denominator), A-1
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_status_invest.py` with fixture `tests/fixtures/pit_rebuild/si_small.csv`: boundaries (cap exactly 200,000,000 and liq exactly 50,000 eligible; 199,999,999.99 or 49,999.99 not; blank not); missing column exit 4 naming it; duplicated ticker exit 4 naming it; absent path exit 3; file bytes and mtime unchanged after `load`; a text read of `api/app/services/premium_policy.py` finds the `arrojado` anchor `cap_min`/`liq_min` equal to `ELIGIBILITY`; on `data/raw/status_invest_fundamentals.csv` `eligible` returns 269 tickers.
- [ ] Gate check passes: Quick for `test_pit_rebuild_status_invest.py`
- [ ] Test count: ≥ 7 tests pass

**Tests**: unit
**Gate**: quick

---

### T5: Hash-checked download cache with offline mode

**What**: `Cache(root, offline, opener)` with `get`, `verify`, `head_size`, `used`, and `manifest.json` (`url`, `name`, `sha256`, `bytes`, `retrieved_at`; `sizes`). Adds `tests/pit_rebuild_support.py` (fake opener recording requests; zip builder from text fixtures), reused by later tests.
**Where**: `py/pit_rebuild/cache.py`
**Depends on**: T1
**Reuses**: none
**Requirement**: SRC-01, SRC-02, SRC-03, SRC-04, CLI-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cache.py`: miss downloads once and records the manifest entry; second `get` makes zero requests (SRC-02); a byte flipped in a cached file makes `verify` raise exit 4 naming the file (SRC-03); HTTP error, timeout and a non-zip body on a required `.zip` each raise exit 5 naming the URL and keep previously cached files (SRC-04); offline miss raises exit 5 when required and returns `None` when not; offline mode makes zero calls to the opener (CLI-02); `head_size` records `Content-Length`.
- [ ] Gate check passes: Quick for `test_pit_rebuild_cache.py`
- [ ] Test count: ≥ 8 tests pass

**Tests**: unit
**Gate**: quick

---

### T6: Source URLs and the per-date file plan

**What**: URL builders for DFP, ITR, FCA, cadastro, COTAHIST, B3 company (base64 JSON), Yahoo chart, CVM listings; `plan_files(dates)` with the SRC-01 year ranges, deduplicated and sorted.
**Where**: `py/pit_rebuild/sources.py`
**Depends on**: T5
**Reuses**: URLs verified in design.md "Sources"
**Requirement**: SRC-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_sources.py`: for `[2026-08-07, 2026-11-06]` the plan holds DFP 2019-2026, ITR 2024-2026, FCA 2010-2026, COTAHIST 2025-2026 and the cadastro, each once; for `[2020-03-31]` DFP starts at 2013; the B3 URL decodes to `{"issuingCompany":"PETR","language":"pt-br"}`; every URL is `https`.
- [ ] Gate check passes: Build (last task of Phase 1)
- [ ] Test count: ≥ 4 tests pass

**Tests**: unit
**Gate**: build

---

### T7: Parse CVM DFP/ITR zips and select filings point-in-time

**What**: `load_filings(zip_paths, kind)` (index with `DT_RECEB` per `VERSAO`; `BPA/BPP/DRE` `_con`/`_ind`; `composicao_capital`; `ESCALA_MOEDA` scaling; `ORDEM_EXERC`) and `usable(store, cnpj, d)` with the A-6 rule and the `refiled_after_d` flag; `load_cadastro(path)`.
**Where**: `py/pit_rebuild/cvm.py`
**Depends on**: None (Phase 1 done)
**Reuses**: `tests/pit_rebuild_support.py` zip builder
**Requirement**: PIT-01, PIT-03, A-6
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cvm.py` with text fixtures in `tests/fixtures/pit_rebuild/cvm/` (two companies, latin-1, `;`, real headers from design.md): a filing with `DT_RECEB == d` is usable, `d + 1` is not; a company whose 2026-Q2 ITR has v1 delivered before `d` and v2 after `d` falls back to Q1 and is flagged `refiled_after_d`; `MIL` values are ×1000; `_ind` is used when `_con` is absent and recorded as `statement: individual`; cadastro yields `CD_CVM` and `SETOR_ATIV`.
- [ ] Gate check passes: Quick for `test_pit_rebuild_cvm.py`
- [ ] Test count: ≥ 6 tests pass

**Tests**: unit
**Gate**: quick

---

### T8: Trailing-12-month, fiscal-year, balance and share accessors

**What**: `ttm`, `ttm_year_before`, `annual`, `balance`, `shares` (outstanding per class and total) on top of T7, each returning the value with its `dt_receb`.
**Where**: `py/pit_rebuild/cvm.py`
**Depends on**: T7
**Reuses**: T7 `FilingStore`
**Requirement**: IND-01, PIT-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cvm.py` extended: TTM after an ITR = `FY_prev + YTD − YTD_prev` on hand-computed fixture numbers; TTM after a DFP = annual value; missing `FY_prev` → `None`; `annual(..., years_back=5)` reads the DFP five fiscal years back and `None` when absent; `shares` subtracts treasury per class; every returned item carries a `dt_receb ≤ d`.
- [ ] Gate check passes: Quick for `test_pit_rebuild_cvm.py`
- [ ] Test count: ≥ 11 tests pass in the file (T7's plus ≥ 5)

**Tests**: unit
**Gate**: quick

---

### T9: Parse COTAHIST and read price, liquidity and ISIN at `d`

**What**: `load(zip_paths, tickers)` (fixed-width positions verified in design.md; `TIPREG 01`, `TPMERC 010`, `CODBDI 02`; `FATCOT`), `quote_at`, `avg_volume` (30 sessions), `isin`.
**Where**: `py/pit_rebuild/cotahist.py`
**Depends on**: None (Phase 1 done)
**Reuses**: `tests/pit_rebuild_support.py`
**Requirement**: PIT-01, IND-01 (`PRECO`, `LIQUIDEZ MEDIA DIARIA`), A-7
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cotahist.py` with `tests/fixtures/pit_rebuild/cotahist.txt` (real line layout, including the 2026-11-06 PETR4 line values 45.36 / `BRPETRACNPR6`): a session on `d` is used and one on `d + 1` is not; `TPMERC` other than `010` and `CODBDI` other than `02` are ignored; `avg_volume` over 30 sessions and over fewer when fewer exist; last session more than 30 calendar days before `d` → `None` (`no_quote`); `FATCOT 1000` divides the price.
- [ ] Gate check passes: Quick for `test_pit_rebuild_cotahist.py`
- [ ] Test count: ≥ 6 tests pass

**Tests**: unit
**Gate**: quick

---

### T10: Parse B3 and Yahoo dividends and sum the 12 months to `d`

**What**: `parse_b3`, `parse_yahoo`, `dps_12m` (window of design.md; leak guard), unit DPS from class DPS and composition.
**Where**: `py/pit_rebuild/dividends.py`
**Depends on**: None (Phase 1 done)
**Reuses**: T1 `PITLeakError`
**Requirement**: PIT-02, SRC-05 (parsers), IND-01 (`DY`)
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_dividends.py` with `tests/fixtures/pit_rebuild/b3_petr.json` (shape of the live response in design.md) and `yahoo_petr4.json`: `,` decimals and `dd/mm/yyyy` parsed; `lastDatePriorEx = d − 1` counts, `= d` does not, `= d − 365` does not; matched by ISIN per class; a record with ex-date after `d` passed straight to the sum raises exit 4; unit DPS = 1×ON + 2×PN for `1 ON / 2 PN`.
- [ ] Gate check passes: Quick for `test_pit_rebuild_dividends.py`
- [ ] Test count: ≥ 6 tests pass

**Tests**: unit
**Gate**: quick

---

### T11: Map tickers to CVM companies and classify the financial group

**What**: `load_fca`, `parse_composition`, `map_tickers` (FCA over years, cadastro, B3 `codeCVM` fallback; reasons of MAP-01), `classify` (A-10 rules).
**Where**: `py/pit_rebuild/mapping.py`
**Depends on**: None (Phase 1 done)
**Reuses**: T7 `load_cadastro`
**Requirement**: MAP-01, MAP-02, MAP-03
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_mapping.py` with `tests/fixtures/pit_rebuild/fca/` (two years; a ticker only in the older year; `TAEE11` `1 ON / 2 PN`; `KLBN11` `1 KLBN3 + 4 KLBN4`; an unparseable unit): mapping from an older FCA year; B3 fallback when absent from FCA; each reason `not_in_fca_or_b3`, `no_cvm_code`, `unit_composition_unparsed` produced by its own input; `parse_composition` returns `{ON:1, PN:2}` and `{KLBN3:1, KLBN4:4}`; `classify` by `SETOR_ATIV` (each of the 7 values) and by DRE `3.01` containing `Intermediação Financeira`, with the rule recorded; an unmapped ticker stays non-financial.
- [ ] Gate check passes: Build (last task of Phase 2)
- [ ] Test count: ≥ 8 tests pass

**Tests**: unit
**Gate**: build

---

### T12: Compute the 19 fields for one ticker at `d`

**What**: `FORMULAS` (design.md table) and `compute(inputs, d)` with per-field `None` + reason (IND-03), unit price per share (MAP-03), multi-class `VALOR DE MERCADO`, signed ratios, `short_history`, and the PIT leak guard.
**Where**: `py/pit_rebuild/indicators.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 `ALL_FIELDS`
**Requirement**: IND-01, IND-03, MAP-03, PIT-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_indicators.py`: one hand-computed expected value per field (19 asserts on one synthetic company), and a second ticker of the same company (PN) with the same `VALOR DE MERCADO` and `EV/EBIT` but a different `P/L` and `PSR`; unit `k = 3` divides `p` in `P/L`, `P/VP`, `PSR`; class without a quote uses the traded class's price in `VALOR DE MERCADO`; negative `LL` gives a negative `P/L`; zero `PC` leaves `LIQ. CORRENTE` empty with reason; CAGR with base ≤ 0 empty; missing FY−5 → `short_history`; an input with `dt_receb > d` raises exit 4; `set(FORMULAS) == set(ALL_FIELDS)`.
- [ ] Gate check passes: Quick for `test_pit_rebuild_indicators.py`
- [ ] Test count: ≥ 10 tests pass

**Tests**: unit
**Gate**: quick

---

### T13: Tolerance, difference and per-field statistics

**What**: `within`, `difference`, `p90` (nearest rank), `summarize(field, pairs)` → denominator, within, pct, `reference_blank`, `reference_blank_with_value`, median, p90, 10 worst (ties by ticker), missing with reasons.
**Where**: `py/pit_rebuild/compare.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 `tolerance_kind`
**Requirement**: CMP-01, CMP-02, A-3, A-4, A-12
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_compare.py`: relative at exactly 5.00 % within and 5.01 % outside; p.p. at exactly 0.50 within and 0.51 outside; rounding to 2 decimals before comparing (`10.004` vs `10.00`); zero reference within only when rounded calc is `0.00`; missing calc counted in the denominator and outside; blank reference outside the denominator and counted; p90 of `1..10` is 9 and of `1..11` is 10; worst list capped at 10 and tie-broken by ticker.
- [ ] Gate check passes: Quick for `test_pit_rebuild_compare.py`
- [ ] Test count: ≥ 9 tests pass

**Tests**: unit
**Gate**: quick

---

### T14: Verdicts per field, per group and for the class

**What**: `verdict(pct, denominator)`, `overall(v1, v2)`, `class_verdict(stats)` (16 score fields, non-financial; financial group apart; extras excluded).
**Where**: `py/pit_rebuild/compare.py`
**Depends on**: T13
**Reuses**: T3 `SCORE_FIELDS`
**Requirement**: VER-01, VER-02, VER-03, A-5
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_compare.py` extended: 90.0 → `reconstruível`, 89.99 and 70.0 → `com ressalvas`, 69.99 → `não reconstruível`, denominator 0 → `sem referência`; `overall` takes the worse date and ignores `sem referência`; class verdict = worst of the 16 non-financial fields; a `não reconstruível` financial field or a `não reconstruível` `LIQUIDEZ MEDIA DIARIA` does not change the class verdict but appears in its own field.
- [ ] Gate check passes: Quick for `test_pit_rebuild_compare.py`
- [ ] Test count: ≥ 14 tests pass in the file (T13's plus ≥ 5)

**Tests**: unit
**Gate**: quick

---

### T15: Earliest year per source and field, and the effort estimate

**What**: `earliest_years(listings, cotahist_probe)`, `LOOKBACK`, `field_earliest`, `effort(seconds_per_date, sizes)`.
**Where**: `py/pit_rebuild/coverage.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 field names
**Requirement**: COV-01, COV-02, COV-03, A-13
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_coverage.py` with `tests/fixtures/pit_rebuild/listing_dfp.html` (nginx autoindex shape, 2010-2026): earliest DFP 2010; with ITR 2011 the `ROE` earliest year is 2012 and `CAGR RECEITAS 5 ANOS` 2015, `PEG RATIO` 2013, `P/VP` (balance + price) the later of DFP/ITR and COTAHIST earliest; COTAHIST probe `{2005: False, 2006: True, …}` gives 2006; `effort(2.5, sizes)` gives 180 dates, 450.0 s and the byte sum.
- [ ] Gate check passes: Quick for `test_pit_rebuild_coverage.py`
- [ ] Test count: ≥ 5 tests pass

**Tests**: unit
**Gate**: quick

---

### T16: Build the report JSON and render the Markdown from it

**What**: `build(...)` with the data model of design.md (parameters, inputs, dates, verdicts, dividends incl. VER-04, coverage), `encode_json`, and pure `render_markdown(report)` in Portuguese.
**Where**: `py/pit_rebuild/report.py`
**Depends on**: T14, T15
**Reuses**: `docs/validation/class-allocation-oos.md` section style
**Requirement**: RPT-02, RPT-03, RPT-04, VER-04
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer` (Markdown wording)

**Done when**:

- [ ] `tests/test_pit_rebuild_report.py`: every raw file appears with URL, sha256 and bytes; both SI CSVs with path, sha256, date; parameters hold thresholds with their source lines, tolerances, `percent_fields`, `liquidity_sessions`, all 19 formulas; one Yahoo-sourced ticker sets `dividends.source = "secondary"` with counts and the caveat, zero sets `primary`; `render_markdown` is deterministic and changing one number in the dict changes the Markdown (proves it reads the dict, RPT-04); `encode_json` is sorted, indented, newline-terminated.
- [ ] Gate check passes: Build (last task of Phase 3)
- [ ] Test count: ≥ 6 tests pass

**Tests**: unit
**Gate**: build

---

### T17: `report` subcommand, argument checks and exit codes

**What**: `cli.main(argv, *, opener=None, clock=None)` with argparse (`fetch`, `report`; `--date`/`--si` pairs; `--cache-dir` default `.cache/pit-rebuild`; `--offline`; `--out-dir`), the docstring exit-code table, the check order of design.md, and the `report` orchestration (verify cache → SI load → CVM/COTAHIST/dividends from cache → mapping → indicators → compare → coverage → report → atomic write); `py/pit_rebuild/__main__.py`. Adds the end-to-end fixture cache builder to `tests/pit_rebuild_support.py`.
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: None (Phase 3 done)
**Reuses**: `py/pit_validation.py:1418-1432` `main`; `tests/test_pit_validation.py:355-374` out-dir snapshot
**Requirement**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, SAFE-01, SAFE-02, RPT-01, RPT-05, PIT-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cli.py`, all through `main([...])` on a fixture cache with two dates and two small SI CSVs, opener that fails the test if called: exit 0 writes both files with the hand-computed class verdict; exit 1 with an unwritable out-dir; exit 2 for a bad date, a repeated date, a `--date`/`--si` count mismatch and each forbidden out-dir/cache-dir root (one subtest each, nothing written); exit 3 for a missing `--si` and a missing cache dir; exit 4 for a tampered cached file, an SI CSV missing a column and a fixture filing whose `DT_RECEB` is forced past `d` into the computation; exit 5 for a required file absent with `--offline`; every failure leaves pre-existing report files byte-identical and prints one stderr line; SI CSV bytes unchanged; a hash snapshot of `data/`, `snapshots/`, `baselines/` is identical before and after the suite; `python3 py/pit_rebuild report --help` exits 0 in a subprocess.
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 10 tests pass in `test_pit_rebuild_cli.py`

**Tests**: integration
**Gate**: full

---

### T18: `fetch` subcommand with dividend fallback and coverage probes

**What**: `fetch` orchestration: plan files (T6) → download required files → SI load + mapping on cached files → B3 company JSON per mapped root, Yahoo per ticker on failure or empty → CVM listings and COTAHIST HEAD probes 2005-2012 and `Content-Length` of the 2007-2026 yearly files → manifest.
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: T17
**Reuses**: T5 `Cache`, T6 `plan_files`, T10 parsers, T11 `map_tickers`
**Requirement**: SRC-01, SRC-04, SRC-05, COV-01, CLI-05
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `tests/test_pit_rebuild_cli.py` extended with a fake opener serving the fixture files: `fetch` requests every planned URL once and writes the manifest; a second `fetch` makes no GET for cached files; a B3 company returning HTTP 500 or `[]` triggers one Yahoo request per eligible ticker of that company and the following `report --offline` records `yahoo` for them; a failing required COTAHIST URL exits 5 naming it and keeps already cached files; `fetch` then `report --offline` exits 0 end to end.
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 15 tests pass in `test_pit_rebuild_cli.py` (T17's plus ≥ 5)

**Tests**: integration
**Gate**: full

---

### T19: Document how to run the rebuild test

**What**: New section "Reconstrução PIT de ações (issue #59)" in `docs/validation/README.md`: question, sources, eligibility rule and its source lines, tolerance and verdict rules (D-1, D-2, A-1..A-7), commands for `fetch` and `report` (the 2026-11-06 CSV path is outside git, A-11), cache location, exit-code table. No result numbers (they live in the generated report).
**Where**: `docs/validation/README.md`
**Depends on**: None (Phase 3 done)
**Reuses**: the "Como regenerar" sections already in that README
**Requirement**: CLI-01, RPT-06
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer`

**Done when**:

- [ ] Section added; its exit-code table equals the `cli.py` docstring table (checked by eye and by T20's artifact test reading both)
- [ ] Gate check passes: Build
- [ ] Test count: unchanged

**Tests**: none
**Gate**: build

---

### T20: Real run and committed report (HEAVY, counts as 2)

**What**: Run `fetch` then `report` on the real sources for 2026-08-07 (`data/raw/status_invest_fundamentals.csv`) and 2026-11-06 (`/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv`, read only), `--cache-dir .cache/pit-rebuild`, `--out-dir docs/validation`; commit the generated `pit-rebuild-stocks.json` and `.md` unedited; add `tests/test_pit_rebuild_artifact.py`. If a real layout differs from the fixtures (for example ITR member names, flagged in design.md), fix the parser in its own module with a new fixture-backed test, commit that first as its own task-scoped commit, then rerun.
**Where**: `docs/validation/pit-rebuild-stocks.json`
**Depends on**: T18, T19
**Reuses**: the whole package
**Requirement**: RPT-06, RPT-04, VER-03, COV-03
**Sensitive area**: no (writes only `docs/validation/` and the gitignored `.cache/`)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `report` exits 0; downloaded bytes and wall time are noted in the commit body
- [ ] `tests/test_pit_rebuild_artifact.py`: `render_markdown(json.load(committed json))` equals the committed `.md` byte for byte; the JSON has both dates, a class verdict in the three-value set, a financial-group verdict, ≥ 1 raw file with a 64-hex sha256 per source kind (DFP, ITR, FCA, cadastro, COTAHIST, dividends), and the SI sha256 of `data/raw/status_invest_fundamentals.csv` equals the file's current hash; the exit-code table in `docs/validation/README.md` equals the one in `cli.py`'s docstring
- [ ] `git status --porcelain -- data snapshots baselines` empty and the 2026-11-06 CSV's sha256 unchanged after the run
- [ ] Gate check passes: Build
- [ ] Test count: ≥ 4 tests pass in `test_pit_rebuild_artifact.py`

**Tests**: integration
**Gate**: build

**Commit**: `docs(validation): commit the PIT rebuild feasibility report (#59)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4

Phase 1:  T1 → T2
          T3
          T4
          T1 → T5 → T6
Phase 2:  T7 → T8
          T9
          T10
          T11
Phase 3:  T12
          T13 → T14 → T16
          T15 → T16
Phase 4:  T17 → T18 → T20
          T19 → T20
```

Execution is strictly sequential, task by task in number order.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | error classes (+ package marker) | ✅ Granular |
| T2 | one module: guard + atomic write | ✅ Granular (2 cohesive functions) |
| T3 | one module: field list | ✅ Granular |
| T4 | one module: SI load + eligibility | ✅ Granular |
| T5 | one module: cache | ✅ Granular (test helper is test code) |
| T6 | one module: URLs + plan | ✅ Granular |
| T7 | `cvm.py` parse + PIT selection | ✅ Granular |
| T8 | `cvm.py` accessors | ✅ Granular |
| T9 | one module: COTAHIST | ✅ Granular |
| T10 | one module: dividends | ✅ Granular |
| T11 | one module: mapping | ✅ Granular |
| T12 | one function family: indicators | ✅ Granular |
| T13 | `compare.py` stats | ✅ Granular |
| T14 | `compare.py` verdicts | ✅ Granular |
| T15 | one module: coverage | ✅ Granular |
| T16 | one module: report | ✅ Granular |
| T17 | `cli.py` report path + exit codes | ⚠️ Largest task; cohesive (one entry point), kept whole so each exit code is tested through the real path in the task that creates it |
| T18 | `cli.py` fetch path | ✅ Granular |
| T19 | one doc section | ✅ Granular |
| T20 | real run + artifact test | ✅ Granular (heavy) |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | none | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | None | none | ✅ Match |
| T4 | None | none | ✅ Match |
| T5 | T1 | T1 → T5 | ✅ Match |
| T6 | T5 | T5 → T6 | ✅ Match |
| T7 | None (Phase 1 done) | none | ✅ Match |
| T8 | T7 | T7 → T8 | ✅ Match |
| T9 | None (Phase 1 done) | none | ✅ Match |
| T10 | None (Phase 1 done) | none | ✅ Match |
| T11 | None (Phase 1 done) | none | ✅ Match |
| T12 | None (Phase 2 done) | none | ✅ Match |
| T13 | None (Phase 2 done) | none | ✅ Match |
| T14 | T13 | T13 → T14 | ✅ Match |
| T15 | None (Phase 2 done) | none | ✅ Match |
| T16 | T14, T15 | T14 → T16, T15 → T16 | ✅ Match |
| T17 | None (Phase 3 done) | none | ✅ Match |
| T18 | T17 | T17 → T18 | ✅ Match |
| T19 | None (Phase 3 done) | none | ✅ Match |
| T20 | T18, T19 | T18 → T20, T19 → T20 | ✅ Match |

No task depends on a later phase.

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | pure rules (errors) | unit | unit | ✅ OK |
| T2 | pure rules (safety) | unit | unit | ✅ OK |
| T3 | pure rules (fields) | unit | unit | ✅ OK |
| T4 | pure rules (status_invest) | unit | unit | ✅ OK |
| T5 | network boundary (cache) | unit | unit | ✅ OK |
| T6 | network boundary (sources) | unit | unit | ✅ OK |
| T7 | parser (cvm) | unit | unit | ✅ OK |
| T8 | parser (cvm) | unit | unit | ✅ OK |
| T9 | parser (cotahist) | unit | unit | ✅ OK |
| T10 | parser (dividends) | unit | unit | ✅ OK |
| T11 | parser (mapping) | unit | unit | ✅ OK |
| T12 | pure rules (indicators) | unit | unit | ✅ OK |
| T13 | pure rules (compare) | unit | unit | ✅ OK |
| T14 | pure rules (compare) | unit | unit | ✅ OK |
| T15 | pure rules (coverage) | unit | unit | ✅ OK |
| T16 | pure rules (report) | unit | unit | ✅ OK |
| T17 | CLI | integration | integration | ✅ OK |
| T18 | CLI | integration | integration | ✅ OK |
| T19 | docs | none | none | ✅ OK |
| T20 | committed report | artifact test (integration) | integration | ✅ OK |

## Tools

No MCP is needed. Skills: `docs-writer` for T16 (Markdown wording) and T19. Pending human confirmation, as the skill's Tasks step asks.
