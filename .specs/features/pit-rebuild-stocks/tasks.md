# PIT Rebuild Feasibility for Stocks Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

---

**Spec**: `.specs/features/pit-rebuild-stocks/spec.md`
**Design**: `.specs/features/pit-rebuild-stocks/design.md`
**Decision record**: AD-031 and AD-032 in `.specs/STATE.md` (both accepted by the human)
**Status**: r5 (2026-11-10). r4 approved 2026-11-09; T1-T19 done; T20 superseded by T32. Phases 5 and 6 (T21-T32) implement the r5 amendment the human approved on 2026-11-10 (spec "Amendment r5", design "Amendment r5", AD-032).

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
| Full | After CLI tasks (T17, T18, T29, T30) and the artifact task (T32) | `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py" -v` |
| Build | Last task of each phase | Full + `python3 -m compileall -q py/pit_rebuild` + `git diff --exit-code develop -- data snapshots baselines py/core api py/config.py py/pit_validation.py` + `$PY -m unittest discover -s tests -p "test_*.py"` (root suite, about 3 min; baseline 587 OK, 2 skipped at r4 planning; 799 OK, 2 skipped at `4ec8665`) |

---

## Execution Plan

Phases run in order; tasks within a phase run in order. A heavy task counts as two.

| Phase | Tasks | Count (heavy = 2) | Sensitive path | Risk | Suggested tier |
| ----- | ----- | ----------------- | -------------- | ---- | -------------- |
| 1. Foundation and network boundary | T1-T6 | 6 | none | Low: plumbing, but SAFE-01 and the hash check live here. | mid |
| 2. Source parsers and mapping | T7-T11 | 5 | none | Medium: PIT selection, ITR year-to-date rows, dividend source rule and real file layouts. | high reasoning (T7, T8, T10, T11), mid (T9) |
| 3. Computation, comparison and report | T12-T16 | 5 | none | Medium: formulas, tolerance and verdict rules decide the result. | high reasoning |
| 4. CLI, docs and real run | T17-T20 | 7 (T17, T18 and T20 heavy) | none | Medium: exit-code table; fetch order (B3 records before mapping); the real run downloads several GB of zips and parses COTAHIST. T20 is superseded by T32. | high (T17, T18, T20), mid (T19) |
| 5. r5 rules in the pure modules | T21-T27 | 7 | none | Medium: each task changes one approved rule and flips the r4 test expectations it names; every r4 test it does not name must still pass. | high reasoning (T23, T24), mid (T21, T22, T25, T26, T27) |
| 6. r5 report, CLI wiring and the real run | T28-T32 | 9 (T28, T29, T30 and T32 heavy) | none | Medium-high: no sensitivity may reach the official verdict; the real run makes a few online requests (normalized B3 names, one live probe). | high (T28, T29, T30, T32), mid (T31) |
| 7. Verifier fixes (r1) | T33-T36 | 4 | none | Low: test-only fixes for the Verifier's blocking gaps G1 and G2 and nonblocking N1, plus one docstring line (N3); no production behaviour changes. | mid |

No task touches a REVIEW.md sensitive path (`py/core/**`, `api/**`, `py/snapshot_*.py`, `data/allocation*/**`, `baselines/**`). Total 36 tasks, counted 43 (r4: 20 tasks, counted 23, T20 superseded; r5: 12 tasks, counted 16; Verifier r1: 4 tasks, counted 4).

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
T7 → T11
T9
T10
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

### Phase 5: r5 rules in the pure modules

```
T21
T22
T23
T24
T25
T26
T27
```

### Phase 6: r5 report, CLI wiring and the real run

```
T28 → T29 → T30 → T31 → T32
```

### Phase 7: Verifier fixes (r1)

```
T33
T34
T35
T36
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

- [x] `tests/test_pit_rebuild_errors.py`: each class's `exit_code`; `PITLeakError` is an `IntegrityError` and its message names ticker, date and offending date; the `__init__` docstring lists codes 0, 1, 2, 3, 4, 5.
- [x] Gate check passes: Quick for `test_pit_rebuild_errors.py`
- [x] Test count: 3 tests pass

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

- [x] `tests/test_pit_rebuild_safety.py`: one test per forbidden root (`data`, `snapshots`, `baselines`, a nested child, a symlink pointing into `data`) raises exit 2; `docs/validation` and a temp dir pass; a failure injected on the second `os.replace` leaves both pre-existing targets byte-identical and no temp file behind; success writes both files.
- [x] Gate check passes: Quick for `test_pit_rebuild_safety.py`
- [x] Test count: ≥ 7 tests pass

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

- [x] `tests/test_pit_rebuild_fields.py`: `SCORE_FIELDS` equals the 16 names of `METRIC_GROUPS` in group order and `ALL_FIELDS` has 19; a temp config with an extra metric yields 17 score fields (proves it is read, not copied); `config` is not in `sys.modules` after importing `pit_rebuild.fields` in a subprocess; `tolerance_kind` returns `pp` for exactly the 8 A-2 fields.
- [x] Gate check passes: Quick for `test_pit_rebuild_fields.py`
- [x] Test count: ≥ 4 tests pass

**Tests**: unit
**Gate**: quick

---

### T4: Load a Status Invest CSV read-only and select the eligible universe

**What**: `REQUIRED_COLUMNS` (`TICKER`, `SETOR`, the 19 fields) and `load(path)` (exit 3 absent; exit 4 on a missing required column or a duplicated ticker; blank → `None`; `SETOR` kept per ticker; sha256 recorded) and `eligible(csv)` with `ELIGIBILITY = {"cap_min": 200_000_000, "liq_min": 50_000}` (A-1).
**Where**: `py/pit_rebuild/status_invest.py`
**Depends on**: None
**Reuses**: eligibility rule of `py/core/preprocessing.py:103-106` (restated)
**Requirement**: SAFE-02, MAP-01 (denominator), MAP-02 (`SETOR` input), A-1
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_status_invest.py` with fixture `tests/fixtures/pit_rebuild/si_small.csv`: boundaries (cap exactly 200,000,000 and liq exactly 50,000 eligible; 199,999,999.99 or 49,999.99 not; blank not); missing column exit 4 naming it (one subtest for `TICKER`, one for `SETOR`, one for `VALOR DE MERCADO`); duplicated ticker exit 4 naming it; absent path exit 3; file bytes and mtime unchanged after `load`; a text read of `api/app/services/premium_policy.py` finds the `arrojado` anchor `cap_min`/`liq_min` equal to `ELIGIBILITY`; on `data/raw/status_invest_fundamentals.csv` `eligible` returns 269 tickers, of which 66 have a blank `DY` loaded as `None` (the raw blank is kept; A-4's DY-as-zero rule lives in `compare.reference`, T13).
- [x] Gate check passes: Quick for `test_pit_rebuild_status_invest.py`
- [x] Test count: ≥ 8 tests pass

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

- [x] `tests/test_pit_rebuild_cache.py`: miss downloads once and records the manifest entry; second `get` makes zero requests (SRC-02); a byte flipped in a cached file makes `verify` raise exit 4 naming the file (SRC-03); HTTP error, timeout and a non-zip body on a required `.zip` each raise exit 5 naming the URL and keep previously cached files (SRC-04); offline miss raises exit 5 when required and returns `None` when not; offline mode makes zero calls to the opener (CLI-02); `head_size` records `Content-Length`.
- [x] Gate check passes: Quick for `test_pit_rebuild_cache.py`
- [x] Test count: ≥ 8 tests pass

**Tests**: unit
**Gate**: quick

---

### T6: Source URLs and the per-date file plan

**What**: URL builders for DFP, ITR, FCA, cadastro, COTAHIST, B3 company and B3 cash-dividend pages (base64 JSON), Yahoo chart (`period1`/`period2`), CVM listings; `plan_files(dates)` with the SRC-01 year ranges, deduplicated and sorted.
**Where**: `py/pit_rebuild/sources.py`
**Depends on**: T5
**Reuses**: URLs verified in design.md "Sources"
**Requirement**: SRC-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_sources.py`: for `[2026-08-07, 2026-11-06]` the plan holds DFP 2019-2026, ITR 2024-2026, FCA 2010-2026, COTAHIST 2025-2026 and the cadastro, each once; for `[2020-03-31]` DFP starts at 2013; the B3 company URL decodes to `{"issuingCompany":"PETR","language":"pt-br"}` and `b3_cash_url("PETROBRAS", 2)` to `{"language":"pt-br","pageNumber":2,"pageSize":120,"tradingName":"PETROBRAS"}`; the Yahoo URL has `period1=946684800`, `interval=1d`, `events=div` and no `range`; every URL is `https`.
- [x] Gate check passes: Build (last task of Phase 1)
- [x] Test count: ≥ 4 tests pass

**Tests**: unit
**Gate**: build

---

### T7: Parse CVM DFP/ITR zips and select filings point-in-time

**What**: `load_filings(zip_paths, kind)` (index with `DT_RECEB` per `VERSAO`; `BPA/BPP/DRE` `_con`/`_ind`; `composicao_capital`; `ESCALA_MOEDA` scaling; `ORDEM_EXERC`) and `usable(store, cnpj, d)` with the A-6 rule and the `refiled_after_d` flag; `load_cadastro(path)`.
**Where**: `py/pit_rebuild/cvm.py`
**Depends on**: None (Phase 1 done)
**Reuses**: `tests/pit_rebuild_support.py` zip builder
**Requirement**: PIT-01, PIT-03, A-6, IND-01 (year-to-date row selection)
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cvm.py` with text fixtures in `tests/fixtures/pit_rebuild/cvm/` (two companies, latin-1, `;`, real headers from design.md; ITR members named `itr_cia_aberta_*` as verified): a filing with `DT_RECEB == d` is usable, `d + 1` is not; a Q2 ITR `DRE_con` with two `ÚLTIMO` rows for `3.01` (year-to-date `DT_INI_EXERC` 2025-01-01 and three-month 2025-04-01, the three-month row written last) loads the year-to-date value, and the same for `PENÚLTIMO`; a company with fiscal year starting 2025-06-01 keeps its `DT_INI_EXERC` 2025-06-01 row for `DT_REFER` 2025-11-30; a company whose 2026-Q2 ITR has v1 delivered before `d` and v2 after `d` falls back to Q1 and is flagged `refiled_after_d`; `MIL` values are ×1000; `_ind` is used when `_con` is absent and recorded as `statement: individual`; cadastro yields `CD_CVM` and `SETOR_ATIV`.
- [x] Gate check passes: Quick for `test_pit_rebuild_cvm.py`
- [x] Test count: ≥ 8 tests pass (8 tests pass)

**Tests**: unit
**Gate**: quick

---

### T8: Trailing-12-month, fiscal-year, balance and share accessors

**What**: `ttm`, `ttm_year_before`, `annual`, `balance`, `shares` (issued per class and total, treasury included, A-17) and `share_scale` (A-18) on top of T7, each value returned with its `dt_receb`.
**Where**: `py/pit_rebuild/cvm.py`
**Depends on**: T7
**Reuses**: T7 `FilingStore`
**Requirement**: IND-01, IND-04, PIT-01
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cvm.py` extended: TTM after an ITR = `FY_prev + YTD − YTD_prev` on hand-computed fixture numbers, using the Q2 fixture whose three-month rows are written last (the expected value differs from the one the three-month rows would give); TTM for the June-start fiscal year uses the DFP whose period ends 2025-05-31; TTM after a DFP = annual value; missing `FY_prev` → `None`; `annual(..., years_back=5)` reads the DFP five fiscal years back and `None` when absent; `shares` returns `CAP_INTEGR` per class with treasury **not** subtracted (fixture with ON treasury 2,526,656 of 705,260,684 returns 705,260,684) from the newest usable filing, an ITR when it is newer than the DFP; `share_scale(31_258+61_361, 92_619_256)` → `(1000, True)`, `share_scale(1_253_079_108, 1_253_079_108)` → `(1, True)`, `share_scale(n, None)` and `share_scale(n, 0)` → `(1, False)`; `shares` returns `None`, without raising, when the newest usable filing's `composicao_capital` total is `0` (COBRASMA-shaped row `0;0;0`) and when that filing has no `composicao_capital` row for the company, even if an older usable filing has a positive count (no fallback); every returned item carries a `dt_receb ≤ d`.
- [x] Gate check passes: Quick for `test_pit_rebuild_cvm.py`
- [x] Test count: ≥ 18 tests pass in the file (T7's plus ≥ 10) (22 tests pass: 8 from T7, 14 new)

**Tests**: unit
**Gate**: quick

---

### T9: Parse COTAHIST and read price, liquidity and ISIN at `d`

**What**: `load(zip_paths, tickers)` (fixed-width positions verified in design.md; `TIPREG 01`, `TPMERC 010`, `CODBDI 02`; `FATCOT`), `quote_at`, `avg_volume` (30 sessions), `isin`, `share_class`, `next_session` (lower and upper horizon, A-16) and `class_prices` (A-17).
**Where**: `py/pit_rebuild/cotahist.py`
**Depends on**: None (Phase 1 done)
**Reuses**: `tests/pit_rebuild_support.py`
**Requirement**: PIT-01, IND-01 (`PRECO`, `LIQUIDEZ MEDIA DIARIA`), IND-04, A-7, A-15 (`ESPECI`), A-16 (session calendar), A-17
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cotahist.py` with `tests/fixtures/pit_rebuild/cotahist.txt` (real line layout, including the 2026-11-06 PETR4 line values 45.36 / `BRPETRACNPR6`): a session on `d` is used and one on `d + 1` is not; `TPMERC` other than `010` and `CODBDI` other than `02` are ignored; `avg_volume` over 30 sessions and over fewer when fewer exist; last session more than 30 calendar days before `d` → `None` (`no_quote`); `FATCOT 1000` divides the price; `share_class` returns the first `ESPECI` token (`PN` for PETR4, `UNT` for TAEE11, `PNA` for a `PNA     N2` line); `next_session` of a Friday returns the following Monday session from the calendar of all tickers, `None` after the last loaded session, and `None` for a day before the first loaded session (a 2020 day with the calendar starting 2025-01-02, A-16); `load` with `{EUCA3, EUCA4}` keeps both tickers' lines and drops a third ticker's; `class_prices` on a company with `ON` (`USIM3`), `PNA` (`USIM5`, high volume, 7.15) and `PNB` (`USIM6`, low volume, 8.65) returns `ON` → `USIM3` and `PREF` → `USIM5` at 7.15, skips a `UNT` line, breaks a volume tie by ticker, and returns `PREF` → `None` when the only preferred ticker's last session is more than 30 days before `d`.
- [x] Gate check passes: Quick for `test_pit_rebuild_cotahist.py`
- [x] Test count: ≥ 11 tests pass (13 tests pass)

**Tests**: unit
**Gate**: quick

---

### T10: Parse B3 and Yahoo dividends and sum the 12 months to `d`

**What**: `parse_b3_company` (`tradingName`, `codeCVM`, `totalNumberShares`), `parse_b3_cash` (pages, completeness), `parse_yahoo`, `choose_source` (A-15), `resolve` (Yahoo failure after `b3_empty`), `b3_ex_dates`, `in_window` and `dps_12m` (A-16; leak guard), class matching by `typeStock`, unit DPS from `UNT` records or from class DPS and composition.
**Where**: `py/pit_rebuild/dividends.py`
**Depends on**: None (Phase 1 done)
**Reuses**: T1 `PITLeakError`
**Requirement**: PIT-02, SRC-05, IND-01 (`DY`), A-15, A-16
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_dividends.py` with fixtures trimmed from real cached responses (keys exactly as in design.md "Sources"): `b3_company_petr.json` (`GetListedSupplementCompany`: `tradingName` with trailing spaces, `codeCVM`), `b3_cash_petrobras_p1.json` and `_p2.json` (`GetListedCashDividends`: `page` + `results` with `typeStock`, `valueCash`, `lastDatePriorEx`), `yahoo_petr4.json` (`events.dividends`). Asserts: `tradingName` stripped, `totalNumberShares` `"92.619.256"` → 92619256, an empty company list → `None`; `,` decimals and `dd/mm/yyyy` parsed; both pages present → `complete`, page 2 missing or `totalRecords` larger → not complete; `in_window`: ex-date `= d` counts, `= d − 364` counts, `= d − 365` does not, `= d + 1` does not; B3 ex-date = next session (`lastDatePriorEx` Friday `d − 3` → Monday session `d` → counts; `lastDatePriorEx = d` → next session after `d` → does not count); Yahoo ex-date from the epoch; `choose_source` returns `b3_failed`, `b3_empty`, `b3_incomplete` and `b3_short_history` each from its own input, and `b3` when the earliest record is exactly `d − 365 days`; records matched by `typeStock` (`PN` records do not enter `ON`); a `UNT` ticker uses `UNT` records when present and `1×ON + 2×PN` for `1 ON / 2 PN` when absent; a record with ex-date after `d` passed straight to the sum raises exit 4; `resolve` after `b3_empty` with Yahoo failed returns source `b3`, reason `b3_empty_yahoo_failed` and DPS 0 (AZUL-shaped fixture `b3_cash_azul_empty.json`, `totalRecords: 0`), after `b3_failed` with Yahoo failed returns `none`/`no_dividend_source`, and with Yahoo parsed returns `yahoo` and the original reason; a B3 record with `lastDatePriorEx` 2020-03-02 and a calendar starting 2025-01-02 gets no ex-date and adds nothing to `dps_12m` at `d = 2026-01-01` (A-16 lower cut).
- [x] Gate check passes: Quick for `test_pit_rebuild_dividends.py`
- [x] Test count: ≥ 14 tests pass (18 tests pass)

**Tests**: unit
**Gate**: quick

---

### T11: Map tickers to CVM companies and classify the financial group

**What**: `load_fca`, `parse_composition`, `map_tickers` (FCA over years, cadastro, B3 `codeCVM` fallback from the records of every eligible root; reasons of MAP-01), `quote_tickers` (A-17), `classify` (A-10 rules, including `si_setor` and `unclassified` for unmapped tickers).
**Where**: `py/pit_rebuild/mapping.py`
**Depends on**: T7
**Reuses**: T7 `load_cadastro`
**Requirement**: MAP-01, MAP-02, MAP-03, IND-04
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_mapping.py` with `tests/fixtures/pit_rebuild/fca/` (two years; a ticker only in the older year; `TAEE11` `1 ON / 2 PN`; `KLBN11` `1 KLBN3 + 4 KLBN4`; an unparseable unit): mapping from an older FCA year; B3 fallback when absent from every FCA year (the B3 record of the ticker's root resolves `codeCVM`, cadastro gives the CNPJ); each reason `not_in_fca_or_b3`, `no_cvm_code`, `unit_composition_unparsed` produced by its own input; `parse_composition` returns `{ON:1, PN:2}` and `{KLBN3:1, KLBN4:4}`; `classify` by `SETOR_ATIV` (each of the 7 values) and by DRE `3.01` containing `Intermediação Financeira`, with the rule recorded; an unmapped ticker with Status Invest `SETOR` `Consumo Cíclico` is `non_financial` with rule `si_setor`; an unmapped ticker with `SETOR` `Financeiro e Outros` is `unclassified`; a mapped ticker with `SETOR` `Financeiro e Outros` but a non-financial `SETOR_ATIV` and DRE `3.01` stays `non_financial` (SETOR never puts a mapped ticker in the financial group); `quote_tickers` with eligible `{EUCA4, KLBN11, USIM5}`, FCA listing `EUCA3`/`EUCA4` for EUCATEX, only `KLBN11` (composition `1 KLBN3 + 4 KLBN4`) for KLABIN and only `USIM5` for USIMINAS, and SI tickers (both dates) `{EUCA4, KLBN11, USIM5, USIM6, PETR4}` returns `{EUCA3, EUCA4, KLBN3, KLBN4, KLBN11, USIM5, USIM6}`; each source supplies a ticker no other source has (`EUCA3` only from FCA, `KLBN3`/`KLBN4` only from the unit composition, `USIM6` only from the same-root SI rule), so dropping any one source fails the test; the unrelated `PETR4` stays out.
- [x] Gate check passes: Build (last task of Phase 2)
- [x] Test count: ≥ 12 tests pass (15 tests pass)

**Tests**: unit
**Gate**: build

---

### T12: Compute the 19 fields for one ticker at `d`

**What**: `FORMULAS` (design.md table) and `compute(inputs, d)` with per-field `None` + reason (IND-03), unit price per share (MAP-03), `VALOR DE MERCADO` and the per-share divisor by A-17/A-18 (with the `vm_price_fallback` flag), signed ratios, `short_history`, and the PIT leak guard.
**Where**: `py/pit_rebuild/indicators.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 `ALL_FIELDS`
**Requirement**: IND-01, IND-03, IND-04, MAP-03, PIT-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_indicators.py`: one hand-computed expected value per field (19 asserts on one synthetic company), and a second ticker of the same company (PN) with the same `VALOR DE MERCADO` and `EV/EBIT` but a different `P/L` and `PSR`; unit `k = 3` divides `p` in `P/L`, `P/VP`, `PSR`; `VALOR DE MERCADO` with `N_ON` 705,260,684, `N_PREF` 547,818,424, `p_ON` 6.25 and class prices `PREF` → `USIM5` 7.15 equals 8,324,781,006.60 (the SI value of 2026-08-07; pricing the preferred shares at `USIM6` 8.65 would give 9,146,508,642.60, so the test fails if the wrong preferred ticker is used); a company with `PREF` `None` and `N_PREF > 0` prices the preferred shares at `p_ON` and sets `vm_price_fallback`; both classes `None` → no value, reason `no_quote`; `shares` `None` → `VALOR DE MERCADO`, `P/L`, `P/VP`, `PSR`, `EV/EBIT` and `PEG RATIO` have no value with reason `no_share_count` while `PRECO`, `ROE` and `DY` keep their values; counts with share scale 1000 multiply `VALOR DE MERCADO` and divide `P/L`'s per-share earnings accordingly; negative `LL` gives a negative `P/L`; zero `PC` leaves `LIQ. CORRENTE` empty with reason; CAGR with base ≤ 0 empty; missing FY−5 → `short_history`; an input with `dt_receb > d` raises exit 4; `set(FORMULAS) == set(ALL_FIELDS)`.
- [x] Gate check passes: Quick for `test_pit_rebuild_indicators.py`
- [x] Test count: ≥ 14 tests pass (18 tests pass)

**Tests**: unit
**Gate**: quick

---

### T13: Tolerance, difference and per-field statistics

**What**: `reference` (A-4, blank DY as `0.00`), `within`, `difference`, `p90` (nearest rank), `summarize(field, pairs)` → denominator, within, pct, `reference_blank`, `reference_blank_with_value`, `reference_blank_as_zero`, median, p90, 10 worst (ties by ticker), missing with reasons.
**Where**: `py/pit_rebuild/compare.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 `tolerance_kind`
**Requirement**: CMP-01, CMP-02, A-3, A-4, A-12
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_compare.py`: relative at exactly 5.00 % within and 5.01 % outside; p.p. at exactly 0.50 within and 0.51 outside; rounding to 2 decimals before comparing (`10.004` vs `10.00`); zero reference within only when rounded calc is `0.00`; missing calc counted in the denominator and outside; blank reference of a non-DY field (e.g. `P/L`) outside the denominator and counted in `reference_blank`; blank `DY` reference with rebuilt DY `3.0` is in the denominator and outside tolerance; blank `DY` with rebuilt `0.50` within and `0.51` outside; blank `DY` with no rebuilt value in the denominator and outside; `reference_blank_as_zero` counts the blank-DY tickers; p90 of `1..10` is 9 and of `1..11` is 10; worst list capped at 10 and tie-broken by ticker.
- [x] Gate check passes: Quick for `test_pit_rebuild_compare.py`
- [x] Test count: ≥ 12 tests pass (13 tests pass)

**Tests**: unit
**Gate**: quick

---

### T14: Verdicts per field, per group and for the class

**What**: `verdict(pct, denominator)`, `overall(v1, v2)`, `class_verdict(stats)` (16 score fields, non-financial incl. unclassified; the same without unclassified and whether they differ; financial group apart; extras excluded).
**Where**: `py/pit_rebuild/compare.py`
**Depends on**: T13
**Reuses**: T3 `SCORE_FIELDS`
**Requirement**: VER-01, VER-02, VER-03, A-5
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_compare.py` extended: 90.0 → `reconstruível`, 89.99 and 70.0 → `com ressalvas`, 69.99 → `não reconstruível`, denominator 0 → `sem referência`; `overall` takes the worse date and ignores `sem referência`; class verdict = worst of the 16 non-financial fields; a `não reconstruível` financial field or a `não reconstruível` `LIQUIDEZ MEDIA DIARIA` does not change the class verdict but appears in its own field; an unclassified ticker missing every field lowers `class` but not `class_without_unclassified`, and `differs` is `True` exactly then.
- [x] Gate check passes: Quick for `test_pit_rebuild_compare.py`
- [x] Test count: ≥ 18 tests pass in the file (T13's plus ≥ 6) (19 tests pass: 13 from T13, 6 new)

**Tests**: unit
**Gate**: quick

---

### T15: Earliest year per source and field, and the effort estimate

**What**: `earliest_years(listings, cotahist_probe)` (probe floor), `LOOKBACK`, `field_earliest`, `dy_coverage`, `full_rebuild_years`, `effort(load, compute_seconds, n_dates, sizes, kind_of_url)` (A-13).
**Where**: `py/pit_rebuild/coverage.py`
**Depends on**: None (Phase 2 done)
**Reuses**: T3 field names
**Requirement**: COV-01, COV-02, COV-03, A-13
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_coverage.py` with `tests/fixtures/pit_rebuild/listing_dfp.html` (nginx autoindex shape, 2010-2026): earliest DFP 2010; with ITR 2011 the `ROE` earliest year is 2012 and `CAGR RECEITAS 5 ANOS` 2015, `PEG RATIO` 2013, `P/VP` (balance + price) the later of DFP/ITR and COTAHIST earliest; COTAHIST probe `{2005: False, 2006: True, …}` gives `{year: 2006, at_probe_floor: False}` and `{2005: True, …}` gives `{year: 2005, at_probe_floor: True}` (never an earlier year); `full_rebuild_years` with DFP 2010 / ITR 2011 / FCA 2010 / COTAHIST 2005 gives DFP 2010-2026, ITR 2011-2026, FCA 2010-2026, COTAHIST 2011-2026; `effort` with load `{COTAHIST: (10 s, 100 MB)}`, full COTAHIST bytes 1,600 MB, compute 5 s over 2 dates gives load 160 s, compute 2.5 s/date × 180 = 450 s, total 610 s (the load is not multiplied by 180); a kind with 0 measured bytes gives total `None`; `dy_coverage` with B3 used by 1 ticker takes DY's earliest year from B3 and with B3 used by none from Yahoo, each with its retention text.
- [x] Gate check passes: Quick for `test_pit_rebuild_coverage.py`
- [x] Test count: ≥ 8 tests pass (11 tests pass)

**Tests**: unit
**Gate**: quick

---

### T16: Build the report JSON and render the Markdown from it

**What**: `build(...)` with the data model of design.md (parameters, inputs, dates, verdicts, dividends incl. VER-04, coverage), `encode_json`, and pure `render_markdown(report)` in Portuguese.
**Where**: `py/pit_rebuild/report.py`
**Depends on**: T14, T15
**Reuses**: `docs/validation/class-allocation-oos.md` section style
**Requirement**: RPT-02, RPT-03, RPT-04, VER-03, VER-04, MAP-02, COV-02, IND-04
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer` (Markdown wording)

**Done when**:

- [x] `tests/test_pit_rebuild_report.py`: every raw file appears with URL, sha256 and bytes; both SI CSVs with path, sha256, date; parameters hold thresholds with their source lines, tolerances, `percent_fields`, `liquidity_sessions`, all 19 formulas, `share_scale_rule` (text naming the B3 count and the 31.6× limit) and `non_pit_inputs` with exactly the seven inputs of spec A-19, each with its role (`unit_only`, `identity` ×3, `classification`, `source_selection`, `dated_value_current_amount`), use and full-rebuild check; the Markdown renders the rule and one table row per non-PIT input (removing either from the dict changes the Markdown); one Yahoo-sourced ticker sets `dividends.source = "secondary"` with counts per date and per reason and the caveat, zero sets `primary`; `mapping.mapped_via_b3`, `mapping.vm_price_fallback` (count and companies), `mapping.share_scale` (`x1000`, `unverified`) and `dividends.b3_empty_yahoo_failed` are in the JSON per date and rendered in the Markdown; the unclassified list, `class_without_unclassified` and `class_differs` are in the JSON and the Markdown verdict line shows both class verdicts when they differ; COTAHIST at the probe floor renders `≤ 2005 (limite da sondagem)`; DY coverage shows each source with its retention; `render_markdown` is deterministic and changing one number in the dict changes the Markdown (proves it reads the dict, RPT-04); `encode_json` is sorted, indented, newline-terminated.
- [x] Gate check passes: Build (last task of Phase 3)
- [x] Test count: ≥ 10 tests pass (15 tests pass)

**Tests**: unit
**Gate**: build

---

### T17: `report` subcommand, argument checks and exit codes (HEAVY, counts as 2)

**What**: `cli.main(argv, *, opener=None, clock=None)` with argparse (`fetch`, `report`; `--date`/`--si` pairs; `--cache-dir` default `.cache/pit-rebuild`; `--offline`; `--out-dir`), the docstring exit-code table, the check order of design.md, and the `report` orchestration (verify cache → SI load → CVM/COTAHIST/dividends from cache, timed per source kind as load → per date: mapping, classification, dividend source choice, indicators, compare, timed as compute → coverage and effort → report → atomic write); `py/pit_rebuild/__main__.py`. Adds the end-to-end fixture cache builder to `tests/pit_rebuild_support.py`.
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: None (Phase 3 done)
**Reuses**: `py/pit_validation.py:1418-1432` `main`; `tests/test_pit_validation.py:355-374` out-dir snapshot
**Requirement**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, SAFE-01, SAFE-02, RPT-01, RPT-05, PIT-01, COV-03, IND-04
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cli.py`, all through `main([...])` on a fixture cache with two dates and two small SI CSVs, opener that fails the test if called: exit 0 writes both files with the hand-computed class verdict; exit 1 with an unwritable out-dir; exit 2 for a bad date, a repeated date, a `--date`/`--si` count mismatch and each forbidden out-dir/cache-dir root (one subtest each, nothing written); exit 3 for a missing `--si` and a missing cache dir; exit 4 for a tampered cached file, an SI CSV missing a column and a fixture filing whose `DT_RECEB` is forced past `d` into the computation; exit 5 for a required file absent with `--offline`; every failure leaves pre-existing report files byte-identical and prints one stderr line; SI CSV bytes unchanged; a hash snapshot of `data/`, `snapshots/`, `baselines/` is identical before and after the suite; `python3 py/pit_rebuild report --help` exits 0 in a subprocess; with an injected `clock` that advances a known step per call, the JSON's `effort.load` and `compute_seconds_per_date` equal the hand-computed values; the fixture cache has a company whose only eligible ticker is its PN (`EUCA4`-shaped) with a non-eligible ON ticker quoted at a different price, and the exit-0 JSON gives that company's `VALOR DE MERCADO` from the ON ticker's own close (fails if the COTAHIST load keeps only eligible tickers), plus a company whose preferred ticker has no session in the 30 days before `d`, counted once in `mapping.vm_price_fallback`; a company whose `composicao_capital` is filed in thousands (`EUCATEX`-shaped `31258;61361;92619`) with a B3 record in the cache (`totalNumberShares` 92.619.256) appears in `mapping.share_scale.x1000` and its exit-0 `VALOR DE MERCADO` equals the hand-computed `31,258,000 × p_ON + 61,361,000 × p_PREF` (fails if the CLI does not carry the scale factor into the computation), and an FCA-mapped company with no B3 record in the cache appears in `mapping.share_scale.unverified` with its counts used as filed; a company whose `composicao_capital` total is 0 (`COBRASMA`-shaped `0;0;0`) still lets `report` exit 0, gets reason `no_share_count` on `VALOR DE MERCADO`, `P/L`, `P/VP`, `PSR`, `EV/EBIT` and `PEG RATIO`, and is listed in neither `share_scale.x1000` nor `share_scale.unverified` (fails if the CLI calls `share_scale` or reads the total of a missing share count).
- [x] Gate check passes: Full (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`: 196 tests OK)
- [x] Test count: ≥ 14 tests pass in `test_pit_rebuild_cli.py` (20 tests pass)

**Tests**: integration
**Gate**: full

---

### T18: `fetch` subcommand with dividend fallback and coverage probes (HEAVY, counts as 2)

**What**: `fetch` orchestration in design.md's order: plan files (T6) → download required files → SI load → the B3 company record for the root of **every** eligible ticker of every date → CVM/FCA parse and `map_tickers` with those records → `quote_tickers` and the COTAHIST parse → every page of `GetListedCashDividends` for each `tradingName` resolved for a mapped company → Yahoo (`period1`/`period2`) for each eligible ticker that `choose_source` sends to Yahoo at any requested date → CVM listings, COTAHIST HEAD probes 2005-2012 and `Content-Length` of every yearly file in `full_rebuild_years` → manifest.
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: T17
**Reuses**: T5 `Cache`, T6 `plan_files`, T10 parsers, T11 `map_tickers`
**Requirement**: SRC-01, SRC-04, SRC-05, MAP-01, COV-01, COV-03, CLI-05
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cli.py` extended with a fake opener serving the fixture files: `fetch` requests every planned URL once and writes the manifest; a second `fetch` makes no GET for cached files; a B3 cash-dividend history spread over two pages fetches both pages; a B3 company record returning HTTP 500, a cash history with `totalRecords: 0`, and a history whose page 2 fails each trigger one Yahoo request per eligible ticker of that company, with reasons `b3_failed`, `b3_empty`, `b3_incomplete`; a complete history whose earliest `lastDatePriorEx` is after `d − 365 days` triggers the Yahoo request and the following `report --offline` takes that ticker's DY from Yahoo with reason `b3_short_history` and marks DY secondary; a complete history reaching `d − 365` makes no Yahoo request and records `b3`; a failing required COTAHIST URL exits 5 naming it and keeps already cached files; `fetch` then `report --offline` exits 0 end to end; a ticker absent from every FCA fixture year but present in a B3 company fixture (its root's `codeCVM` in the cadastro fixture) gets its B3 record requested by `fetch` and, after `report --offline`, is mapped (listed in `mapping.mapped_via_b3`, not unmapped with `not_in_fca_or_b3`) and has rebuilt fields; an AZUL-shaped company (B3 record resolved, `GetListedCashDividends` `totalRecords: 0`, Yahoo answering 404) gets DY `0.00` from `report --offline` with source `b3`, reason `b3_empty_yahoo_failed`, counted under `b3` in `dividends.by_source` and not under `none`.
- [x] Gate check passes: Full (pit suite: 210 tests OK)
- [x] Test count: ≥ 22 tests pass in `test_pit_rebuild_cli.py` (T17's plus ≥ 10; 34 tests pass)

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

- [x] Section added; its exit-code table equals the `cli.py` docstring table (checked by eye and by T20's artifact test reading both)
- [x] Gate check passes: Build (pit suite 210 OK; root suite 797 OK, 2 skipped; protected paths unchanged against origin/develop)
- [x] Test count: unchanged (210)

**Tests**: none
**Gate**: build

---

### T20: Real run and committed report (HEAVY, counts as 2) - SUPERSEDED by T32 (2026-11-10)

> Stopped on 2026-11-10 before committing the report: the real run exposed spec defects (STATE Handoff of 2026-11-10). The human approved the r5 amendment on 2026-11-10; T32 redoes this run on the amended rules. Its boxes stay unchecked on purpose.

**What**: Run `fetch` then `report` on the real sources for 2026-08-07 (`data/raw/status_invest_fundamentals.csv`) and 2026-11-06 (`/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv`, read only), `--cache-dir .cache/pit-rebuild`, `--out-dir docs/validation`; commit the generated `pit-rebuild-stocks.json` and `.md` unedited; add `tests/test_pit_rebuild_artifact.py`. If a real layout differs from the fixtures (for example a B3 `typeStock` label with no matching `ESPECI` token), fix the parser in its own module with a new fixture-backed test, commit that first as its own task-scoped commit, then rerun.
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

### T21: Quote codes, 21-session liquidity, stale class prices and stale references in COTAHIST

**What**: In `cotahist.py`: `QUOTE_CODBDI = ("02", "07", "08")`, `LOT_CODBDI` and the same-session rule in `load` (F1); `LIQUIDITY_SESSIONS = 21` (F5); `quote_at(..., max_age_days=None)` and `last_trade` (F14); `class_prices` returning `(ticker, close, session_day)` with the any-age fallback (F14); `reference_stale` (D-A). Design "Amendment r5" › `cotahist.py`.
**Where**: `py/pit_rebuild/cotahist.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T9 fixture `tests/fixtures/pit_rebuild/cotahist.txt` (lines appended in the real layout)
**Requirement**: IND-01 (`PRECO`, `LIQUIDEZ MEDIA DIARIA`), IND-04, VER-05, A-7, A-17, A-22
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cotahist.py`: T9's `test_only_standard_lot_cash_market_records_are_kept` is replaced by `test_lot_and_recovery_records_are_kept_and_the_lot_wins_the_session` (approved r5 expectation change, F1): `CODBDI` 07 and 08 lines of a requested ticker are kept; 10, 12 and 58 lines and a `TPMERC` other than `010` are dropped; a ticker with only `08` lines gets its `quote_at` from them; on a session with a `02` line and an `08` line of the same ticker at different closes, the `02` close is used both when the `08` line comes first and when it comes last.
- [x] T9's `test_avg_volume_uses_the_last_30_sessions_and_fewer_when_fewer_exist` becomes the 21-session test: on 25 sessions of distinct `VOLTOT` the default averages the last 21 (the expected mean differs from the 20- and 30-session means), fewer when fewer exist, and `LIQUIDITY_SESSIONS == 21`.
- [x] `quote_at(..., max_age_days=None)` and `last_trade` return a session 400 days before `d`; `quote_at` with the default still returns `None` at 31 days.
- [x] `class_prices`: an EMAE-shaped company whose only `ON` ticker last traded 2025-08-18 at 150.00 returns `ON` → that ticker, 150.00, 2025-08-18 at `d` 2026-11-05; a preferred ticker with a session within 30 days and lower volume beats a stale one with higher volume; a session after `d` is never used; T9's `test_class_prices_give_no_preferred_price_when_its_last_session_is_too_old` changes (approved, F14): the stale preferred ticker is now returned with its session day; the USIM case still returns `PREF` → `USIM5` at 7.15, now with its session day.
- [x] `reference_stale`: last session at `d − 30` → `False`; at `d − 31` → `True`; only a session at `d + 1` → `True`; no session → `True`.
- [x] Gate check passes: Quick for `test_pit_rebuild_cotahist.py` (19 OK; pit suite 218 OK; compileall OK)
- [x] Test count: ≥ 17 tests pass in the file (T9's 13 with one replaced, plus ≥ 4) - 19 pass

> Done (T21): `cli.py` reads class prices through `_r4_class_price`, which keeps the r4 rule (a stale price is no price) until T29 wires `vm_stale_class_price`. `tests/test_pit_rebuild_report.py` expects `liquidity_sessions` 21 (F5; T28 asserts the same value). Both were needed to keep the r4 CLI and report tests green: F14 moved two CLI tests and F5 one report test.

**Tests**: unit
**Gate**: quick

---

### T22: Parse unit compositions with the r5 grammar

**What**: `parse_composition(text)` per spec A-9's unit grammar (F2): split on `/`, `+` or a standalone `e`/`E`; classes `ON`, `PN`, `PNA`, `PNB`, `PNs` → `PN`, the Portuguese word forms, tickers; anything else → `None`. Design "Amendment r5" › `mapping.py`.
**Where**: `py/pit_rebuild/mapping.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T11 tests and FCA fixtures
**Requirement**: MAP-03, MAP-01 (`unit_composition_unparsed`), A-9
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_mapping.py`: T11's `test_class_and_ticker_compositions_are_read_and_anything_else_is_none` changes its `2 ON e 1 PN` expectation from `None` to `{"ON": 2, "PN": 1}` (approved r5 expectation change, F2) and keeps `""` and `None` → `None`.
- [x] `test_real_fca_unit_strings_parse`, one subtest each: `1 ON E 2 PN` → `{ON: 1, PN: 2}`; `1 ON e 4 PN` → `{ON: 1, PN: 4}`; `1 ON e 2 PN` → `{ON: 1, PN: 2}`; `2 ações preferenciais e 1 ação ordinária` → `{PN: 2, ON: 1}`; `1 ação ordinária e 4 ações preferenciais` → `{ON: 1, PN: 4}`; `1 ENG3 e 4 ENGI4` → `{ENG3: 1, ENGI4: 4}`; `1 ON+ 2PN` → `{ON: 1, PN: 2}`; `1 ON e 2PNs` → `{ON: 1, PN: 2}` with no `PNs` key; `1 ação ON + 2 ações PN` → `{ON: 1, PN: 2}`; `1 ON E 2 PNA` → `{ON: 1, PNA: 2}`; `1 ON + 1 ON` → `{ON: 2}`; `1 ON / 2 PN` and `1 KLBN3 + 4 KLBN4` as in r4.
- [x] `test_any_other_part_leaves_the_composition_unparsed`, one subtest each → `None`: `1 ON e duas PN`, `1 BDR`, `ON + PN`, `1 ON - 2 PN`, `1 ação especial`, `1 ON e`, `1 ONe 2 PN`.
- [x] `map_tickers` maps a unit whose FCA composition is `1 ON e 2 PN` with composition `{ON: 1, PN: 2}`, not `unit_composition_unparsed`.
- [x] Gate check passes: Quick for `test_pit_rebuild_mapping.py` (18 OK; pit suite 221 OK; compileall OK)
- [x] Test count: ≥ 18 tests pass in the file (T11's 15 plus ≥ 3) - 18 pass

> Done (T22). Two changes beyond the named expectations. (1) The ticker key takes 3 or 4 letters, `[A-Z]{3,4}\d{1,2}`: A-9's pattern `[A-Z]{4}\d{1,2}` cannot read the real string `1 ENG3 e 4 ENGI4`, which this task expects as `{ENG3: 1, ENGI4: 4}`. The spec pattern needs the human's correction. (2) The ZETA11 row of both FCA fixtures changes from `2 ON e 1 PN` to `1 ON e duas PN`, which r5 still leaves unparsed, so `test_reason_unit_composition_unparsed...` keeps its meaning. `cli.py` keeps its own 4-letter `_TICKER` (line 57) until T29 wires the unit dividends.

**Tests**: unit
**Gate**: quick

---

### T23: Filter B3 dividend records and adjust them for splits, bonus shares and groupings

**What**: In `dividends.py`: F16 in `parse_b3_cash` (records identical in every field once across pages, `REST CAP DIN` dropped, `complete` on the raw count, `raw_records`/`duplicates`/`capital_returns`); F17: `StockEvent`, `B3Company.stock_events` from `stockDividends` in `parse_b3_company`, `split_divisor`, and the keyword-only `events`/`isin`/`isins`/`d` of `b3_ex_dates` and `unit_dividends`; F3: `b3_earliest_raw`. Design "Amendment r5" › `dividends.py`.
**Where**: `py/pit_rebuild/dividends.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T10 fixtures; new fixtures trimmed from the cached B3 responses
**Requirement**: SRC-06, IND-05, COV-02 (raw date), PIT-02, A-15, A-20
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_dividends.py` with new fixtures `b3_company_sbsp.json` (`stockDividends` as cached: `DESDOBRAMENTO` `400,00000000000` 28/04/2026, `BONIFICACAO` `0,16098032200` 19/03/2026 and `2,96469750000` 23/12/2025, `GRUPAMENTO` `0,00800000000` 01/06/2007, all ISIN `BRSBSPACNOR5`), `b3_company_espa.json` (`GRUPAMENTO` `0,10000000000` 12/06/2026) and `b3_cash_dupes_p1.json`/`_p2.json` (a record repeated on its page and again on page 2, one `REST CAP DIN` record, `totalRecords` = the raw count).
- [x] `parse_b3_company`: SBSP multipliers 5.0, 1.0016098032, 1.0296469750 and 0.008; ESPA 0.1; `CIS RED CAP`, `RESG TOTAL RV` and `INCORPORACAO` entries skipped; two identical entries give one event; a record without `stockDividends` → `()`.
- [x] `parse_b3_cash`: the repeated record counts once (same page and across pages), `duplicates` and `capital_returns` count them, no `REST CAP DIN` record in `records`, and `complete` holds because the raw count equals `totalRecords` though `len(records)` is smaller; T10's `test_page_two_missing_or_totals_announcing_more_records_is_not_complete` still passes.
- [x] `split_divisor` and `b3_ex_dates(..., events=, isin=, d=)`, on a calendar of every weekday: an SBSP dividend with ex-date 2026-03-02 at `d` 2026-08-07 is divided by `5 × 1.0016098032` (the 2025-12-23 bonus is before it; hand-computed); the same dividend at `d` 2026-04-27 is divided by 1.0016098032 only; a dividend with ex-date after the split is not divided; an event whose ex-date equals the dividend's is not applied; an event of another ISIN is not applied; `isin=None` leaves amounts unchanged; an ESPA dividend before the 2026-06-12 grouping at `d` 2026-08-07 is multiplied by 10.
- [x] `unit_dividends(..., events=, isins=, d=)` adjusts `UNT` records by the unit's ISIN and class records by each class's ISIN (ENGI-shaped 10 % bonus on the ON, PN and unit ISINs, `1 ON e 4 PN`).
- [x] `b3_earliest_raw`: `lastDatePriorEx` 1996-03-21 and 2025-12-31 → 1996-03-22; only 2020-12-31 → 2021-01-01; `[]` → `None`.
- [x] Every r4 call without the new keywords gives the r4 result (T10's tests unchanged and passing).
- [x] Gate check passes: Quick for `test_pit_rebuild_dividends.py` (38 OK: T10's 18 unchanged and 20 new; pit suite 241 OK; compileall OK)
- [x] Test count: ≥ 25 tests pass in the file (T10's 18 plus ≥ 7) - 38 pass

> Done (T23). Two choices the spec leaves open. (1) A record identical to one already received is a duplicate, and the duplicate test runs before the REST CAP DIN filter, so a repeated capital return counts once as a capital return and again as a duplicate. (2) A complete history whose records are all capital returns is `b3_short_history`: no record reaches d, and `min` had no value to compare. `cli.py` still calls the parser without the events, ISINs and `d`, so the split adjustment reaches real runs only with T29.

**Tests**: unit
**Gate**: quick

---

### T24: Statement choice, per-row net income and the power-of-ten share scale in CVM

**What**: In `cvm.py`: F18 statement choice in `load_filings`; F7 synthetic DRE account `NET_INCOME = "LL"` per (filing, `ORDEM_EXERC`), with `3.01`, `3.11` and `3.11.01` added to a given `accounts` filter; F13 `SHARE_SCALE_FACTORS` and `share_scale` returning a float factor. Design "Amendment r5" › `cvm.py`.
**Where**: `py/pit_rebuild/cvm.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T7/T8 fixtures in `tests/fixtures/pit_rebuild/cvm/`
**Requirement**: IND-06, IND-04, IND-01, A-18, A-21
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cvm.py`: a TIMS-shaped filing (`DRE_con` `3.01` and `3.11` `ÚLTIMO` 0, `DRE_ind` non-zero) is `statement: individual` and its balances and DRE come from `_ind`; a filing with `DRE_con` `3.01` non-zero and `3.11` 0 stays consolidated; a filing with all-zero `DRE_con` and no `_ind` stays consolidated; T7's `test_company_without_consolidated_income_statement_is_recorded_as_individual` still passes.
- [x] `NET_INCOME`: a DFP with `3.11.01 = 0` and `3.11 = −120` gives `annual(..., cvm.NET_INCOME)` −120; one with `3.11.01 = 80` and `3.11 = 100` gives 80; an ITR whose `ÚLTIMO` row has `3.11.01 = 50` and whose `PENÚLTIMO` row has `3.11.01 = 0` and `3.11 = 30` gives `ttm` = `FY_prev` + 50 − 30 (hand-computed; the r4 per-company choice gives a different number); `load_filings(..., accounts={"3.11"})` still yields `NET_INCOME`.
- [x] `share_scale` with `b3_total / cvm_total` = 10 → 10, 0.1 → 0.1 (ESPA-shaped), 0.01 → 0.01, 100 → 100, 1000 → 1000 (EUCATEX), 1 → 1, 3.1 → 1, 3.2 → 10 (the √10 boundary), 5 → 10 (a 1:5 split), 50,000 → 1000 (never 10⁴), 0.0005 → 0.01 (never 10⁻³), each `verified`; `None` and 0 → `(1, False)`; T8's `test_share_scale_picks_the_factor_that_matches_b3_or_is_unverified` keeps its 1000 and unverified cases.
- [x] `balance(..., "2.03.09")` returns a `2.03.09` BPP row of the fixture.
- [x] Gate check passes: Quick for `test_pit_rebuild_cvm.py` (36 OK: 24 unchanged, 12 new; pit suite 253 OK; compileall OK)
- [x] Test count: ≥ 31 tests pass in the file (24 plus ≥ 7) - 36 pass

> Done (T24). Two choices and one consequence. (1) The statement choice records an individual DRE before the account filter drops its rows, so a filtered load classifies a filing as the full load does. (2) `load_filings` keeps 3.01, 3.11 and 3.11.01 under any `accounts` filter, as the task requires. (3) `share_scale` now returns the float factor of the powers of ten (F13). `cli.py` multiplies its counts by it, so a verified factor other than 1 or 1000 applies from this commit; T29 rounds the counts and lists them in `share_scale.scaled`.

**Tests**: unit
**Gate**: quick

---

### T25: r5 formulas for ROE, P/VP, DIV. LIQ. / PATRI., ROIC and CAGR LUCROS

**What**: In `indicators.py`: `TickerInputs.pl_ctrl` (F8), `TickerInputs.ll_fy` removed, `ROE = 100 × LL / abs(PL_ctrl)` (F9), `P/VP` and `DIV. LIQ. / PATRI.` on `PL_ctrl` (F8), `ROIC = 100 × (EBIT − abs(IR)) / (PL + DB)` (F11), `CAGR LUCROS 5 ANOS` from `LL_TTM` over `LL_FY−5` (F10), and the r5 `FORMULAS` text (design "Indicator formulas").
**Where**: `py/pit_rebuild/indicators.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T12 synthetic company
**Requirement**: IND-06, IND-01, RPT-03 (formula text), A-21
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_indicators.py`: T12's `test_each_of_the_19_fields_matches_its_hand_computed_value` gets new expected values for `ROE`, `P/VP`, `DIV. LIQ. / PATRI.`, `ROIC` and `CAGR LUCROS 5 ANOS` (approved r5 expectation change) on a synthetic company with `PL` 1000, `PL_ctrl` 800, `LL` (TTM) 120, `LL_FY−5` 60, `IR` −30, each hand-computed in the test's comment; the other 14 expected values do not change.
- [x] `PL_ctrl` −400 with `LL` 120 → `ROE` 30.00 (positive); `LL` −120 with `PL_ctrl` −400 → −30.00.
- [x] `IR` +30 and `IR` −30 give the same `ROIC`, both `100 × (EBIT − 30) / (PL + DB)`.
- [x] `pl_ctrl` `None` → `ROE`, `P/VP` and `DIV. LIQ. / PATRI.` have no value with a `missing_input` reason while `ROIC` keeps its value.
- [x] `CAGR LUCROS 5 ANOS` = `100 × ((120 / 60)^(1/5) − 1)` and changes when only `LL` (TTM) changes; `LL` (TTM) ≤ 0 → `non_positive_base`; no `LL_FY−5` → `short_history`.
- [x] `FORMULAS["ROE"]` names `PL_ctrl` and `abs`, `FORMULAS["ROIC"]` names `abs(IR)`, `FORMULAS["CAGR LUCROS 5 ANOS"]` names `LL_TTM`, `FORMULAS["LIQUIDEZ MEDIA DIARIA"]` names 21 sessions; `TickerInputs` has no `ll_fy` field.
- [x] Gate check passes: Quick for `test_pit_rebuild_indicators.py` (25 OK: 18 r4 tests kept, 7 new; pit suite 260 OK; compileall OK)
- [x] Test count: ≥ 23 tests pass in the file (T12's 18 plus ≥ 5) - 25 pass

> Done (T25). On T12's company ROIC and CAGR LUCROS keep their numbers: IR is negative there, and LL_TTM over LL_FY−5 has the r4 ratio 2. Their r5 changes are tested by the IR ±30 and CAGR tests. The base company keeps PL_ctrl equal to PL (no non-controlling interests), so the other r4 tests keep their values; the 19-field test sets PL_ctrl to 6e6, the 80 % of PL that the task's 800/1000 example gives. `cli.py` builds `pl_ctrl` as 2.03 less 2.03.09 (and loads 2.03.09), and no longer passes `ll_fy`; its CAGR LUCROS still reads the r4 per-company account until T29.

**Tests**: unit
**Gate**: quick

---

### T26: DY coverage from raw dividend dates

**What**: `coverage.dy_coverage(by_source: {source: {ticker: earliest_raw_date}})` with `tickers`, `earliest_raw_date`, `earliest_year`, `median_ticker_year` and the retention text per source; DY's year from B3 when any ticker used B3, else Yahoo (F3). Design "Amendment r5" › `coverage.py`.
**Where**: `py/pit_rebuild/coverage.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T15 `RETENTION`
**Requirement**: COV-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_coverage.py`: T15's `test_dy_takes_its_earliest_year_from_b3_when_any_ticker_used_b3_and_from_yahoo_otherwise` moves to the new input shape (approved r5 change, F3) and keeps its rule.
- [x] B3 `{PETR4: 1996-03-22, VALE3: 2003-05-02, TAEE11: 2006-07-01}` gives `tickers` 3, `earliest_raw_date` 1996-03-22, `earliest_year` 1996 and `median_ticker_year` 2003; two tickers of 2003 and 2006 give `median_ticker_year` 2004.5; a source with no ticker gives `tickers` 0 and `None` dates and median; a 1996 date is reported as given (the function takes no session calendar and no `d`).
- [x] Gate check passes: Quick for `test_pit_rebuild_coverage.py` (14 OK; pit suite 263 OK; compileall OK)
- [x] Test count: ≥ 13 tests pass in the file (T15's 11 plus ≥ 2) - 14 pass

> Done (T26). `dy_coverage` takes each ticker's earliest raw date. Until T29 `cli.py` passes each ticker's earliest ex-date, because the B3 raw date needs the company's records in `_dividends_of`; the two differ by the days to the next session, so a date at a year boundary can land in the other year until T29 wires the raw dates. The report's DY table reads `earliest_raw_date` under the header `primeira data bruta`. `tests/test_pit_rebuild_report.py` gives its fixture the per-ticker shape; its assertions do not change.

**Tests**: unit
**Gate**: quick

---

### T27: Query B3 cash dividends without "/" in the trading name

**What**: `sources.b3_query_name(trading_name)` (strip, remove every `/`, collapse inner whitespace) and `b3_cash_url` applying it to the payload's `tradingName` (F15).
**Where**: `py/pit_rebuild/sources.py`
**Depends on**: None (Phase 4 code done)
**Reuses**: T6 URL tests
**Requirement**: SRC-07, A-15
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_sources.py`: `b3_query_name("AMBEV S/A   ")` → `AMBEV SA`; `KLABIN S/A` → `KLABIN SA`; `PETROBRAS   ` → `PETROBRAS`; `A  /  B` → `A B`; `b3_cash_url("AMBEV S/A", 1)` decodes to `{"language":"pt-br","pageNumber":1,"pageSize":120,"tradingName":"AMBEV SA"}`; T6's PETROBRAS page-2 URL is unchanged.
- [x] Gate check passes: Build (last task of Phase 5): pit suite 265 OK; root suite 852 OK, 2 skipped (799 at 4ec8665 plus the 53 new pit tests); compileall OK; protected paths unchanged against origin/develop 9f65467
- [x] Test count: ≥ 9 tests pass in the file (T6's 7 plus ≥ 2) - 9 pass

**Tests**: unit
**Gate**: build

---

### T28: Report schema v2 with the r5 fields and the sensitivities (HEAVY, counts as 2)

**What**: In `report.py`: `SCHEMA = "pit-rebuild-stocks/v2"`; the r5 `SHARE_SCALE_RULE`; the eighth `NON_PIT_INPUTS` row (`B3 stockDividends`, `dated_event_current_record`); `build(..., sensitivities=None)` computing `verdicts` from the official data first and then the `sensitivities` block (`official: false`, `without_reference_stale`, `filings_cutoff`, `class_without_roic_peg` with `ROIC_PEG_NOTE`); `parameters.quote_codbdi`, `parameters.filings_cutoffs`, `liquidity_sessions` 21; `dividends.b3_filters`, `split_adjust_no_isin`, `slash_names` carried when given; `status_invest` entries carrying `collected_on` and `as_of_check`; `render_markdown` with the sensitivity section, the as-of table, and the r5 mapping, dividend and DY-coverage keys. Design "Amendment r5" › `report.py` and "Report JSON additions (r5)".
**Where**: `py/pit_rebuild/report.py`
**Depends on**: None (Phase 5 done)
**Reuses**: T14 `compare.verdict`/`overall`/`class_verdict`; T16 renderer
**Requirement**: RPT-02, RPT-03, RPT-04, IND-04, PIT-04, COV-02, VER-05, VER-06, VER-07, VER-08, SRC-07
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer` (Portuguese wording of the new sections)

**Done when**:

- [x] `tests/test_pit_rebuild_report.py`: T16's `test_the_non_pit_inputs_are_the_seven_inputs_of_a19_with_their_roles` becomes the eight-input test (approved r5 change): eight rows, the eighth with input naming `stockDividends` and role `dated_event_current_record`; the Markdown has one row per input.
- [x] `parameters.share_scale_rule` names `10^-2`, `10^3`, `√10` and the residual and no longer `31,6`; `parameters.liquidity_sessions == 21`; `parameters.quote_codbdi == ["02", "07", "08"]`; `parameters.filings_cutoffs` echoes its input; `schema == "pit-rebuild-stocks/v2"`.
- [x] The test fixture's `share_scale` becomes `{"scaled": [{"cd_cvm": "1603", "factor": 1000, "residual": 1.0}, {"cd_cvm": "1605", "factor": 0.1, "residual": 1.02}], "unverified": ["1604"]}` (T16 change) and the Markdown shows each factor and residual; `reference_stale` and `vm_stale_class_price` are rendered, and deleting either from the dict changes the Markdown.
- [x] `status_invest` entries keep `collected_on` and `as_of_check`; the Markdown shows the as-of table and a warning line exactly when `best_session` differs from `date`.
- [x] Official isolation (VER-08): `build` with `sensitivities` whose stats would make every verdict `não reconstruível` returns `verdicts` equal (`==`) to `build` without `sensitivities`; `sensitivities["official"] is False`.
- [x] `class_without_roic_peg`: official verdicts where only `ROIC` is `não reconstruível` give `class` `não reconstruível` and `class_without_roic_peg.class` `reconstruível`, with the note from `ROIC_PEG_NOTE`.
- [x] `filings_cutoff`: on hand-built stats where the recomputed second date lifts one field from `com ressalvas` to `reconstruível`, the sensitivity's class verdict is the worst of `overall(official first date, recomputed second date)` (hand-computed).
- [x] `without_reference_stale`: verdicts recomputed from the given stats, `excluded` listed per date.
- [x] Markdown: the verdict line holds the official class verdict and none of the sensitivity verdicts (fixture with distinct values); the section titled `Sensibilidades (não são o veredito oficial)` follows the verdict section and holds each sensitivity's class verdict and the D-C note; without `sensitivities` the section is absent; changing one sensitivity verdict in the dict changes the Markdown (RPT-04).
- [x] `dividends.b3_filters`, `split_adjust_no_isin` and `slash_names`, and the r5 DY coverage (`earliest_raw_date`, `median_ticker_year`), are in the JSON and the Markdown.
- [x] Gate check passes: Quick for `test_pit_rebuild_report.py` - 29 OK
- [x] Test count: ≥ 25 tests pass in the file (T16's 15 plus ≥ 10) - 29 pass

**Tests**: unit
**Gate**: quick

---

### T29: Wire the r5 rules into `report` and `fetch` (HEAVY, counts as 2)

**What**: In `cli.py`, using T21-T28: `STATEMENT_ACCOUNTS` with `2.03.09`; `_company_inputs` from `cvm.NET_INCOME` with `pl_ctrl`; `_company` with rounded scaled counts, `share_scale.scaled` entries `{cd_cvm, factor, residual}`, class prices with session days, `vm_stale_class_price` and the r5 `vm_price_fallback`; B3 cash pages requested and cached under `sources.b3_query_name`; `_choice` sending a `/` name with an empty complete history to Yahoo as `b3_failed`; `_b3_dividends(..., day)` with the company's `stock_events`, the class ISINs and `split_adjust_no_isin`; `dividends.b3_filters` and `slash_names`; DY coverage from `dividends.b3_earliest_raw` and the raw Yahoo dates. Design "Amendment r5" › `cli.py` (the `--collected-on` and sensitivity items are T30).
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: T28
**Reuses**: T17/T18 fixture cache builder in `tests/pit_rebuild_support.py` (extended)
**Requirement**: IND-01, IND-04, IND-05, IND-06, SRC-06, SRC-07, MAP-03, COV-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cli.py`, through `main([...])` on the fixture cache: T17's `test_share_scale_lists_the_thousands_company_and_the_one_without_b3_record` reads `share_scale.scaled` (EUCX factor 1000 with its hand-computed residual) instead of `x1000` (approved r5 change); EUCX's `VALOR DE MERCADO` is unchanged. Any other r4 expected value that the r5 formulas change is updated in the same commit with its hand computation in the test comment, and the commit body lists each one.
- [x] A ticker whose fixture COTAHIST lines are all `CODBDI 08` is mapped with a `PRECO` (not `no_quote`) and its `LIQUIDEZ MEDIA DIARIA` is the mean of its last 21 sessions.
- [x] A unit with FCA composition `1 ON e 2 PN` is mapped and its `P/L` uses `k = 3`.
- [x] A company with a `2.03.09` row gets `ROE` and `P/VP` from `PL_ctrl` (hand-computed); a company whose DFP has `3.11.01 = 0` and `3.11` non-zero gets `P/L` with `3.11`'s sign; a TIMS-shaped company (zero consolidated DRE) gets its values from `_ind`.
- [x] An ESPA-shaped company (CVM total 10× the B3 count) is in `share_scale.scaled` with factor 0.1 and its `VALOR DE MERCADO` uses the rounded scaled counts (hand-computed).
- [x] An EMAE-shaped company whose `ON` ticker's last session is 2025-08-18 with `d` 2026-08-07 is in `vm_stale_class_price` with that ticker and session, its `VALOR DE MERCADO` uses that close, and it is not in `vm_price_fallback`; a company with a class that has no session at all `≤ d` is still in `vm_price_fallback`.
- [x] An SBSP-shaped company (B3 company record with a `DESDOBRAMENTO` `400` between a dividend's ex-date and `d`) gets `DY` from the amount ÷ 5 (hand-computed).
- [x] A history page with a repeated record and a `REST CAP DIN` record: `DY` counts the repeated record once and not the capital return, and `dividends.b3_filters` counts both.
- [x] A company whose `tradingName` is `FOO S/A`: `fetch` requests the cash URL whose decoded `tradingName` is `FOO SA` and no requested payload carries `/`; with that history at `totalRecords: 0` and Yahoo answering 404, `report --offline` gives its ticker source `none`, reason `no_dividend_source` and Yahoo reason `b3_failed` (not `b3_empty_yahoo_failed`), and `dividends.slash_names` lists it; the AZUL-shaped case (no `/`) still gives `b3_empty_yahoo_failed`.
- [x] `coverage.dy.by_source.b3.earliest_raw_date` is the earliest fixture `lastDatePriorEx` + 1 day, also when that date is before the first loaded COTAHIST session.
- [x] Gate check passes: Full (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`) - 293 OK
- [x] Test count: ≥ 45 tests pass in `test_pit_rebuild_cli.py` (34 plus ≥ 11) - 48 pass

**Tests**: integration
**Gate**: full

---

### T30: As-of dates, collection dates and the sensitivities in the CLI (HEAVY, counts as 2)

**What**: In `cli.py`: `--collected-on` on both subcommands and `--filings-cutoff DATE:CUTOFF` on `report`, with CLI-03's exit-2 checks; `_Request.collected_on` and `_Request.filings_cutoffs`; Yahoo `period2 = max(collected_on) + 1 day`; `_as_of_check`; `reference_stale` per date; `_evaluate_day(acq, day, filings_day=None)`; the untimed sensitivity evaluations passed to `report.build(sensitivities=…)`; `status_invest` entries with `collected_on` and `as_of_check`; `parameters.filings_cutoffs`. The docstring's exit codes do not change; its usage lines gain the two options. Design "Amendment r5" › `cli.py`.
**Where**: `py/pit_rebuild/cli.py`
**Depends on**: T29
**Reuses**: T17 argument checks and out-dir snapshot
**Requirement**: PIT-04, CLI-03, VER-05, VER-06, VER-07, VER-08, COV-03, RPT-02
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] `tests/test_pit_rebuild_cli.py`, exit 2 with one stderr line naming the argument and nothing written, one subtest each (`CollectedOnAndCutoffTests`): `--collected-on` malformed; given once for two dates; before its `--date`; `--filings-cutoff` without `:` or with a malformed side; naming a date not given with `--date`; repeating a date; with a cutoff after its date.
- [x] Without `--collected-on`, every `status_invest` entry has `collected_on` equal to its `date`.
- [x] A fixture CSV whose `PRECO` values equal the closes of the session before `d`: `as_of_check.best_session` is that session with hand-counted `matches`, and the Markdown warns; a CSV matching `d`'s closes gets no warning.
- [x] `fetch` with `--collected-on` one day after `--date` requests Yahoo with `period2` = `collected_on` + 1 day (decoded epoch), and `report --offline` with the same arguments finds every file (exit 0).
- [x] A fixture ticker whose last session is `d − 31` is in `mapping.reference_stale`, stays in the official denominator outside tolerance, and is out of the `sensitivities.without_reference_stale` denominator (hand-computed `pct_within` for one field).
- [x] `--filings-cutoff D2:C` with a fixture filing received between `C` and `D2`: the official JSON uses that filing and the sensitivity's fields use the older one; the `verdicts` and `dates` objects of the run with `--filings-cutoff` equal (`==`) those of the same run without it (VER-08).
- [x] `sensitivities.class_without_roic_peg` is present on every exit-0 run.
- [x] With an injected clock, `effort.compute_seconds_per_date` is the same with and without `--filings-cutoff`.
- [x] Gate check passes: Full (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`) - 302 OK
- [x] Test count: ≥ 53 tests pass in `test_pit_rebuild_cli.py` (T29's 45 plus ≥ 8) - 57 pass

**Tests**: integration
**Gate**: full

---

### T31: Document the r5 rules and arguments

**What**: Update the section "Reconstrução PIT de ações (issue #59)" of `docs/validation/README.md`: the `fetch`/`report` commands with `--collected-on` and `--filings-cutoff` as the real run uses them (`--date 2026-11-05 … --collected-on 2026-11-06`, `--filings-cutoff 2026-11-05:2026-11-02`); the r5 rules in a short list (CODBDI 02/07/08, 21 sessions, as-of session, B3 filters and split adjustment, the sensitivities and that none of them is the verdict); the exit-code table unchanged and equal to the `cli.py` docstring. No result numbers.
**Where**: `docs/validation/README.md`
**Depends on**: T30
**Reuses**: T19's section
**Requirement**: CLI-01, RPT-06
**Sensitive area**: no

**Tools**:

- MCP: NONE
- Skill: `docs-writer`

**Done when**:

- [x] Section updated; its exit-code table equals the `cli.py` docstring table (checked by T32's artifact test). The table is unchanged: codes and meanings match the docstring, the wording is pt-BR
- [x] Gate check passes: Build (pit suite 302 OK; compileall OK; protected paths unchanged against origin/develop 9f65467; root suite 889 OK, 2 skipped)
- [x] Test count: unchanged (302 pit tests, no test added in T31)

**Tests**: none
**Gate**: build

---

### T32: Real run on the r5 rules and committed report (HEAVY, counts as 2) - supersedes T20

**What**: (1) Record the sha256 and size of `data/raw/status_invest_fundamentals.csv` and of `/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv` (`87c15cb5…`, 115,847 B before T20), and check `git status --porcelain -- data snapshots baselines` is empty. (2) **F15 live probe, exactly one request:** `GetListedCashDividends` page 1 for `sources.b3_query_name("AMBEV S/A")`, built with `sources.b3_cash_url` and sent once outside the cache; record `totalRecords` and whether the records are AMBEV's in the commit body and the STATE Handoff. The code does not change with the result: if B3 returns 0 records, the `b3_failed` rule keeps the six `/` companies on Yahoo. (3) `fetch --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv --collected-on 2026-08-07 --date 2026-11-05 --si /home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv --collected-on 2026-11-06` online on the existing `.cache/pit-rebuild`. Expected new files: the B3 cash pages of the six normalized names and Yahoo files of tickers newly sent to Yahoo; nothing else, because `COTAHIST_A2026.ZIP` already holds 2026-11-05 (verified 2026-11-10), `stockDividends` are inside the cached B3 company records, and Yahoo's `period2` (2026-11-07) is the one cached. Record the count and bytes of new files; any other new file is a deviation to explain in the commit body. (4) `report --offline` with the same arguments plus `--filings-cutoff 2026-11-05:2026-11-02 --out-dir docs/validation`. (5) Commit `pit-rebuild-stocks.json` and `.md` unedited. (6) Add `tests/test_pit_rebuild_artifact.py`. If a real layout differs from the fixtures, fix the parser in its own module with a fixture-backed test in its own task-scoped commit first, then rerun.
**Where**: `docs/validation/pit-rebuild-stocks.json`
**Depends on**: T31
**Reuses**: the whole package
**Requirement**: RPT-06, RPT-04, VER-03, VER-08, COV-03, PIT-04, SRC-07
**Sensitive area**: no (writes only `docs/validation/`, `tests/` and the gitignored `.cache/`)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] The probe was sent once and its result is recorded (B3 answered `AMBEV SA` with 40 records, one page; see the commit body); `report` exits 0; downloaded files, bytes and wall times are in the commit body.
- [x] `tests/test_pit_rebuild_artifact.py`: `render_markdown(json.load(committed JSON))` equals the committed `.md` byte for byte; the JSON's dates are exactly `2026-08-07` and `2026-11-05`, with `collected_on` `2026-08-07` and `2026-11-06`, and each `as_of_check.best_session` equals its `date`; a class verdict in the three-value set and a financial-group verdict; `sensitivities.official` is `false` and `parameters.filings_cutoffs == {"2026-11-05": "2026-11-02"}`; `parameters.liquidity_sessions == 21` and `parameters.quote_codbdi == ["02", "07", "08"]`; ≥ 1 raw file with a 64-hex sha256 per source kind (DFP, ITR, FCA, cadastro, COTAHIST, B3 company, B3 dividends); the SI sha256 of `data/raw/status_invest_fundamentals.csv` equals the file's current hash; the exit-code table in `docs/validation/README.md` equals the one in `cli.py`'s docstring.
- [x] `git status --porcelain -- data snapshots baselines` is empty and both CSV hashes are unchanged after the run.
- [x] Gate check passes: Build (pit suite 311 OK; compileall OK; protected paths unchanged against origin/develop; root suite run in the build gate, see the commit body)
- [x] Test count: ≥ 7 tests pass in `test_pit_rebuild_artifact.py` - 8 pass

**Tests**: integration
**Gate**: build
**Commit**: `docs(validation): commit the PIT rebuild feasibility report (#59)`

---

### T33: Pin the receipt-date cut of the older periods in `annual`, `ttm_year_before` and the TTM read (G1, blocking)

**What**: Add `PeriodReceiptBoundaryTests` to `tests/test_pit_rebuild_cvm.py`. Each helper that reads a period other than the newest one (`annual(years_back=5)`, `ttm_year_before`, and the fiscal-year DFP that `ttm` reads for an ITR) gets two cases built from inline DFP and ITR stores: the older filing's latest version received on d + 1 gives no value; received on d it is used. No production code changes.
**Where**: `tests/test_pit_rebuild_cvm.py`
**Depends on**: None (Phase 6 done; Verifier report, gap G1)
**Reuses**: `inline_store`, `dre_line` (same module)
**Requirement**: PIT-01 (`DT_RECEB ≤ d` for every value), A-6
**Sensitive area**: no (tests only)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Removing the cut at `py/pit_rebuild/cvm.py:291` fails the three d + 1 tests (`FAILED (failures=3)`); the original passes all six (evidence in the commit body)
- [x] Changing `<=` to `<` at `py/pit_rebuild/cvm.py:291` fails the three d tests; the original passes all six
- [x] Pit suite passes: 317 OK (311 + 6)

**Tests**: unit
**Gate**: full
**Commit**: `test(pit-rebuild): pin the receipt date of the older periods (#59)`

---

### T34: Pin the receipt-date cut of the annual base (G2, blocking)

**What**: Add `AnnualBaseReceiptTests` to `tests/test_pit_rebuild_cvm.py`. A DFP received after d does not become the base of `annual`: `annual(years_back=0)` is the year before's value and `years_back=5` counts back from that year. A DFP received on d is the base.
**Where**: `tests/test_pit_rebuild_cvm.py`
**Depends on**: None (Phase 6 done; Verifier report, gap G2)
**Reuses**: `inline_store`, `dre_line` (same module)
**Requirement**: PIT-01, PIT-03 (the newest period received after d gives the previous filing), A-6
**Sensitive area**: no (tests only)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [x] Removing the cut at `py/pit_rebuild/cvm.py:208` fails the after-d test (`FAILED (failures=1)`); the original passes it
- [x] Changing `<=` to `<` at `py/pit_rebuild/cvm.py:208` fails the on-d test (`FAILED (failures=1)`); the original passes it
- [x] Pit suite passes: 319 OK (317 + 2)

**Tests**: unit
**Gate**: full
**Commit**: `test(pit-rebuild): pin the receipt date of the annual base (#59)`

---

### T35: Round scaled share counts to whole shares (N1, nonblocking)

**What**: Add `ScaledShareCountTests` to `tests/test_pit_rebuild_cli.py`. It calls `cli._company` on inline inputs: one DFP with ON 30 000 007 and PREF 10 000 003 shares, and one B3 record of 4 000 001 shares. The factor is 0.1, so the scaled counts are 3 000 000.7 and 1 000 000.3, and the counts must be 3 000 001 and 1 000 000. Removing `round` at `py/pit_rebuild/cli.py:443` fails it, and so does truncating with `int`.
**Where**: `tests/test_pit_rebuild_cli.py`
**Depends on**: None (Phase 6 done; Verifier report, gap N1)
**Reuses**: `make_zip` (`tests/pit_rebuild_support.py`), `cvm.load_filings`
**Requirement**: A-18 (counts rounded after the factor; AD-032 (4))
**Sensitive area**: no (tests only)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Removing `round` at `py/pit_rebuild/cli.py:443` fails the test; replacing it with `int` fails it too; the original passes
- [ ] Pit suite passes: 320 OK

**Tests**: unit (one function of the CLI; the integration tests of the shape universe are unchanged)
**Gate**: full
**Commit**: `test(pit-rebuild): round scaled share counts in the company unit (#59)`

---

### T36: The first docstring example of `cli.py` uses the approved as-of date (N3, nonblocking)

**What**: In the `report` docstring, change the first example's `--date 2026-11-06` to `--date 2026-11-05`, the as-of date of the 2026-11-06 snapshot (A-11, AD-032 (1)). The exit-code table and the second example stay as they are.
**Where**: `py/pit_rebuild/cli.py` (docstring line 6 only)
**Depends on**: None (Phase 6 done; Verifier report, gap N3)
**Reuses**: none
**Requirement**: CLI-01 (the docstring and the README agree; the exit-code table is compared by code)
**Sensitive area**: no (docstring only)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] The diff of `py/pit_rebuild/cli.py` is that one docstring line; no code changes
- [ ] The exit-code table still parses: `tests/test_pit_rebuild_errors.py` and `tests/test_pit_rebuild_artifact.py` pass
- [ ] Build gate: pit suite OK, `compileall` OK, protected paths unchanged against `origin/develop`, root suite OK

**Tests**: none (docstring; the exit-code table keeps its tests)
**Gate**: build
**Commit**: `docs(pit-rebuild): use the as-of date in the first example (#59)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

Phase 1:  T1 → T2
          T3
          T4
          T1 → T5 → T6
Phase 2:  T7 → T8
          T7 → T11
          T9
          T10
Phase 3:  T12
          T13 → T14 → T16
          T15 → T16
Phase 4:  T17 → T18 → T20
          T19 → T20
Phase 5:  T21
          T22
          T23
          T24
          T25
          T26
          T27
Phase 6:  T28 → T29 → T30 → T31 → T32
Phase 7:  T33
          T34
          T35
          T36
```

Execution is strictly sequential, task by task in number order. T20 is superseded: T32 replaces it, after Phase 6's code tasks. Phase 7 (Verifier r1) follows T32; its four tasks are independent.

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
| T17 | `cli.py` report path + exit codes + timing | ⚠️ Heavy (counts 2); cohesive (one entry point), kept whole so each exit code is tested through the real path in the task that creates it |
| T18 | `cli.py` fetch path (B3 records before mapping, quote set, dividends, probes) | ⚠️ Heavy (counts 2); cohesive (one subcommand), kept whole so the fetch order is tested end to end with `report --offline` |
| T19 | one doc section | ✅ Granular |
| T20 | real run + artifact test | ✅ Granular (heavy) |
| T21 | `cotahist.py`: codes, window, stale prices, stale references | ✅ Granular (one module, one concern: quotes) |
| T22 | one function: `parse_composition` | ✅ Granular |
| T23 | `dividends.py`: record filters and split adjustment | ✅ Granular (one module; both change what a B3 record contributes) |
| T24 | `cvm.py`: statement, net income, share scale | ⚠️ Three rules in one module; cohesive (all decide which filed number is read) |
| T25 | `indicators.py`: five formulas | ✅ Granular |
| T26 | one function: `dy_coverage` | ✅ Granular |
| T27 | one function: `b3_query_name` | ✅ Granular |
| T28 | `report.py` v2 | ⚠️ Heavy (counts 2); one module, kept whole so the official/sensitivity isolation is tested where it is built |
| T29 | `cli.py` wiring of the r5 rules | ⚠️ Heavy (counts 2); one module, end-to-end checks of each rule |
| T30 | `cli.py` arguments, as-of check, sensitivities | ⚠️ Heavy (counts 2); one module, kept whole so each new exit-2 case is tested through the real path |
| T31 | one doc section | ✅ Granular |
| T32 | real run + artifact test | ✅ Granular (heavy) |
| T33 | one test class: the three helpers that read the older period | ✅ Granular |
| T34 | one test class: the annual base | ✅ Granular |
| T35 | one test class: one CLI function (`_company`) | ✅ Granular |
| T36 | one docstring line | ✅ Granular |

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
| T11 | T7 | T7 → T11 | ✅ Match |
| T12 | None (Phase 2 done) | none | ✅ Match |
| T13 | None (Phase 2 done) | none | ✅ Match |
| T14 | T13 | T13 → T14 | ✅ Match |
| T15 | None (Phase 2 done) | none | ✅ Match |
| T16 | T14, T15 | T14 → T16, T15 → T16 | ✅ Match |
| T17 | None (Phase 3 done) | none | ✅ Match |
| T18 | T17 | T17 → T18 | ✅ Match |
| T19 | None (Phase 3 done) | none | ✅ Match |
| T20 | T18, T19 | T18 → T20, T19 → T20 | ✅ Match |
| T21 | None (Phase 4 done) | none | ✅ Match |
| T22 | None (Phase 4 done) | none | ✅ Match |
| T23 | None (Phase 4 done) | none | ✅ Match |
| T24 | None (Phase 4 done) | none | ✅ Match |
| T25 | None (Phase 4 done) | none | ✅ Match |
| T26 | None (Phase 4 done) | none | ✅ Match |
| T27 | None (Phase 4 done) | none | ✅ Match |
| T28 | None (Phase 5 done) | none | ✅ Match |
| T29 | T28 | T28 → T29 | ✅ Match |
| T30 | T29 | T29 → T30 | ✅ Match |
| T31 | T30 | T30 → T31 | ✅ Match |
| T32 | T31 | T31 → T32 | ✅ Match |
| T33 | None (Phase 6 done) | none | ✅ Match |
| T34 | None (Phase 6 done) | none | ✅ Match |
| T35 | None (Phase 6 done) | none | ✅ Match |
| T36 | None (Phase 6 done) | none | ✅ Match |

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
| T21 | parser (cotahist) | unit | unit | ✅ OK |
| T22 | parser (mapping) | unit | unit | ✅ OK |
| T23 | parser (dividends) | unit | unit | ✅ OK |
| T24 | parser (cvm) | unit | unit | ✅ OK |
| T25 | pure rules (indicators) | unit | unit | ✅ OK |
| T26 | pure rules (coverage) | unit | unit | ✅ OK |
| T27 | network boundary (sources) | unit | unit | ✅ OK |
| T28 | pure rules (report) | unit | unit | ✅ OK |
| T29 | CLI | integration | integration | ✅ OK |
| T30 | CLI | integration | integration | ✅ OK |
| T31 | docs | none | none | ✅ OK |
| T32 | committed report | artifact test (integration) | integration | ✅ OK |
| T33 | parser (cvm) | unit | unit | ✅ OK |
| T34 | parser (cvm) | unit | unit | ✅ OK |
| T35 | CLI (one function) | integration | unit | ⚠️ Smallest unit that exercises `cli.py:443`, as the Verifier allowed; the CLI integration tests are unchanged |
| T36 | docs | none | none | ✅ OK |

## Tools

No MCP is needed. Skills: `docs-writer` for T16, T19, T28 (Markdown wording) and T31. r4 tools confirmed with the r4 approval; r5 adds no new tool.
