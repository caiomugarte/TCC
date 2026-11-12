# PIT Rebuild Fixes for Stocks Tasks (issue #64)

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user - do not proceed without it.**

Execution is delegated by the issue-workflow orchestrator in batches of whole phases. Each batch worker implements its phase's tasks in order, one atomic commit per task, and reports: tasks done, commit SHAs, test counts, deviations. Workers start no sub-agents and do not push.

---

**Spec**: `.specs/features/pit-rebuild-stocks-fixes/spec.md`
**Design**: `.specs/features/pit-rebuild-stocks-fixes/design.md`
**Decision record**: AD-033 in `.specs/STATE.md` (proposed with this design); AD-031 and AD-032 still hold.
**Status**: Draft (2026-11-12) - waiting for spec review.
**Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`). Gates compare against `origin/develop`.

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec - confirm before Execute. Guidelines found: `REVIEW.md` (new CLI exit codes in the documented table with a test through the real path; every number in `docs/validation/` equals the committed artifact; a new test must fail when the behaviour it names is removed), `.specs/codebase/TESTING.md` (deterministic, no network), the #59 matrix (`.specs/features/pit-rebuild-stocks/tasks.md`). Runner: stdlib `unittest`; no pytest, CI, coverage threshold, linter or formatter. Strong defaults: every AC and edge case of the spec maps to at least one test; every boundary (270 days, 20 %, `DT_REFER` and `d` for events, 70 %) is tested on both sides.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Pure rules (`fields`, `indicators`, `compare`, `shares`) | unit | 1:1 to the ACs they own; each boundary on both sides; each guard with its own input | `tests/test_pit_rebuild_<module>.py` | `python3 -m unittest discover -s tests -p "test_pit_rebuild_<module>.py" -v` |
| Parsers and readers (`cvm`, `cotahist`, `dividends`) | unit on inline stores, temp zips or committed fixtures | PIT boundaries (`DT_RECEB == d`, `d + 1`; session on `d`, after `d`); fallback paths; None paths | `tests/test_pit_rebuild_<module>.py`, `tests/fixtures/pit_rebuild/` | same |
| Report object and Markdown (`report`) | unit on `build` / `render_markdown` with hand-made statistics | Every new key and section; official `verdicts` unchanged with the new blocks (ADJ-07); render is a pure function of the JSON | `tests/test_pit_rebuild_report.py` | same |
| CLI (`cli`) | integration through `main([...])` on a fixture cache | Each new mapping list and the adjusted block through the real `report` path; exit codes unchanged | `tests/test_pit_rebuild_cli.py` | `python3 -m unittest discover -s tests -p "test_pit_rebuild_cli.py" -v` |
| Committed report and audit | artifact test on the committed files | Markdown equals the render of the JSON; v3 keys; audit rows equal the JSON's worst rows (AUD-01..AUD-04); exit-code table of README equals `cli.py`'s | `tests/test_pit_rebuild_artifact.py` | same pattern |
| Docs (`docs/validation/README.md`) | none | build gate; its numbers are checked by the artifact test where they cite the report | - | build gate |

**Test environment**: the tool and its tests are stdlib-only; system `python3` runs them. `$PY` = `/home/caio/Documents/projetos/TCC/.venv/bin/python` (numpy, pandas) runs the root suite. Baseline at `ebaf1fd`: pit suite 322 OK.

## Gate Check Commands

> Generated from codebase - confirm before Execute. Run from the worktree root.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After a task with unit tests only | `python3 -m unittest discover -s tests -p "test_pit_rebuild_<module>.py" -v` |
| Full | After a task that changes `cli.py` or a signature used by it, and after the artifact tasks | `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py" -v` |
| Build | Last task of each phase | Full + `python3 -m compileall -q py/pit_rebuild` + `git diff --exit-code origin/develop -- data snapshots baselines py/core api py/config.py py/pit_validation.py` + `$PY -m unittest discover -s tests -p "test_*.py"` (root suite, about 3 min; 909 OK, 2 skipped at `53fde43`) |

---

## Execution Plan

Phases run in order; tasks within a phase run in order. A heavy task (real data, report regeneration, large fixture work) counts as two.

| Phase | Tasks | Weighted count | Network / real data | Risk | Suggested tier |
| ----- | ----- | -------------- | ------------------- | ---- | -------------- |
| 1. Readers and events | T1-T7 | 7 | none | Medium: T1 changes a signature used by `cli.py`; T2 changes the TTM rule. | high (T2, T7), mid (others) |
| 2. Pure rules and the adjusted block | T8-T13 | 7 (T12 heavy) | none | Medium: the adjusted block must never reach the official verdict. | high (T12), mid (others) |
| 3. Report lists and CLI wiring | T14-T18 | 6 (T15 heavy) | none | Medium: fixture companies for events, stale filings and ISIN changes. | high (T15), mid (others) |
| 4. `si_lag`, MBRF3, regeneration, README | T19-T22 | 7 (T19, T20, T21 heavy) | T20 reads the real cache; T21 regenerates from the real cache offline | Medium-high: the regenerated numbers are the result. | high |
| 5. Audit of the worst cases | T23-T24 | 4 (both heavy) | reads the real cache and the committed JSON | Medium: hand classification of up to 280 rows. | high |

No task touches a REVIEW.md sensitive path (`py/core/**`, `api/**`, `py/snapshot_*.py`, `data/allocation*/**`, `baselines/**`). No task needs the network; T20 stops and asks the orchestrator if its evidence needs a request outside the cache.

### Phase 1: Readers and events

```
T1 → T2 → T3 → T4 → T5 → T6 → T7
```

### Phase 2: Pure rules and the adjusted block

```
T8 → T9 → T10 → T11 → T12 → T13
```

### Phase 3: Report lists and CLI wiring

```
T14 → T15 → T16 → T17 → T18
```

### Phase 4: `si_lag`, MBRF3, regeneration, README

```
T19 → T20 → T21 → T22
```

### Phase 5: Audit of the worst cases

```
T23 → T24
```

---

## Task Breakdown

### T1: Read the ISIN and the share class at `d`

**What**: `cotahist.isin(q, ticker, d)` and `cotahist.share_class(q, ticker, d)` return the value of the row at `_last_index(q, ticker, d)`, None when there is none; `_price_class` and `class_prices` pass `d`; every call site in `cli.py` (`_class_isin`, `_b3_dividends`) passes the decision date `day` (`_class_isin` gains a `d` parameter).
**Where**: `py/pit_rebuild/cotahist.py`
**Files**: `py/pit_rebuild/cotahist.py`, `py/pit_rebuild/cli.py` (call sites only), `tests/test_pit_rebuild_cotahist.py` (and any existing test calling the old signature)
**Heavy**: no
**Depends on**: None
**Reuses**: `_last_index`; COTAHIST lines built in a temp zip (do not add sessions to `tests/fixtures/pit_rebuild/cotahist.txt`, #59 pitfall)
**Requirement**: ISN-01, ISN-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] A ticker whose ISIN and ESPECI change on a session after `d`: `isin(q, t, d)` and `share_class(q, t, d)` give the old values; at the later session they give the new ones (ISN-01)
- [ ] A ticker with no session on or before `d` gives None for both (ISN-02)
- [ ] `class_prices` classifies by the ESPECI at `d` (a ticker that becomes PN after `d` is ON at `d`)
- [ ] `grep -n "isin(\|share_class(" py/pit_rebuild/*.py` shows `d` passed at every call
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 325 pit tests pass (≥ 3 new; no test deleted)

**Tests**: unit
**Gate**: full
**Commit**: `fix(pit-rebuild): read the ISIN and share class at d (#64)`

---

### T2: 12-month flow from the year-earlier ITR when the comparative is zero or absent

**What**: `cvm._ttm_of(dfp, itr, cnpj, d, account, period)`: for an ITR, when the `PENÚLTIMO` row is None or 0, the prior year-to-date is the `ÚLTIMO` row of the usable ITR whose period ends `_shift_years(dt_refer, −1)`; without it, None. `Dated` gains `prior_from_itr: bool = False`, set when the fallback was used. `ttm` and `ttm_year_before` pass `itr`.
**Where**: `py/pit_rebuild/cvm.py`
**Files**: `py/pit_rebuild/cvm.py`, `tests/test_pit_rebuild_cvm.py`
**Heavy**: no
**Depends on**: T1
**Reuses**: `inline_store`, `dre_line` in `tests/test_pit_rebuild_cvm.py`; `_usable_period`, `_shift_years`
**Requirement**: TTM-01, TTM-02, TTM-03

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] DFP 2025 = 1000, ITR 2026-06-30 `ÚLTIMO` 300 / `PENÚLTIMO` 0, ITR 2025-06-30 `ÚLTIMO` 200: `ttm` = 1100, `prior_from_itr` true, dated by the latest `DT_RECEB` of the three (TTM-01)
- [ ] Same with `PENÚLTIMO` absent: 1100 (TTM-01)
- [ ] Year-earlier ITR received on `d + 1`, or absent, or without the account: None (TTM-02); received on `d`: 1100
- [ ] `PENÚLTIMO` 150 (non-zero) with the year-earlier ITR present: 1000 + 300 − 150 = 1150, `prior_from_itr` false (TTM-03)
- [ ] Newest usable filing a DFP: annual value, no fallback
- [ ] `ttm_year_before` uses the same fallback on its ITR
- [ ] Gate check passes: Full (the CLI suite reads `ttm`)
- [ ] Test count: ≥ 6 new tests in `test_pit_rebuild_cvm.py`; no existing expectation changes unless it fed a zero `PENÚLTIMO` (name it in the commit body)

**Tests**: unit
**Gate**: full
**Commit**: `fix(pit-rebuild): take the prior YTD from last year's ITR (#64)`

---

### T3: Newest period end, newest filing and the 270-day constant

**What**: In `cvm.py`: `STALE_FILING_DAYS = 270`; `newest_period_end(dfp, itr, cnpj, d) -> date | None`; `newest_filing(dfp, itr, cnpj, d) -> Filing | None`, both through `_newest_usable`.
**Where**: `py/pit_rebuild/cvm.py`
**Files**: `py/pit_rebuild/cvm.py`, `tests/test_pit_rebuild_cvm.py`
**Heavy**: no
**Depends on**: T2
**Reuses**: `_newest_usable`
**Requirement**: STL-01 (age input), ADJ-04 (filing of a worst row)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] DFP 2025-12-31 and ITR 2026-03-31 both usable: period end 2026-03-31 and the ITR's `Filing`; ITR received on `d + 1`: 2025-12-31 and the DFP's
- [ ] No usable filing: None for both
- [ ] `STALE_FILING_DAYS == 270`
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 3 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): expose the newest usable period and filing (#64)`

---

### T4: Period end of the share count and the previous count

**What**: `Shares` gains `dt_refer`; `cvm.shares` fills it; `previous_shares(dfp, itr, cnpj, d) -> Shares | None` gives the newest usable period older than the newest usable one whose composition total is > 0 (B-4).
**Where**: `py/pit_rebuild/cvm.py`
**Files**: `py/pit_rebuild/cvm.py`, `tests/test_pit_rebuild_cvm.py`
**Heavy**: no
**Depends on**: T3
**Reuses**: `_newest_usable`, `_Period.composition`
**Requirement**: SHR-01 (period end), SHR-05 (previous filing)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `shares(...).dt_refer` is the newest usable period end
- [ ] Three usable periods, the middle one without composition: `previous_shares` is the oldest one
- [ ] Only one usable period with composition: None
- [ ] A previous period received after `d` is skipped
- [ ] Gate check passes: Full (`Shares` is built in `cli.py`)
- [ ] Test count: ≥ 4 new tests

**Tests**: unit
**Gate**: full
**Commit**: `feat(pit-rebuild): date the share count and read the previous one (#64)`

---

### T5: A filing store without one period of one company

**What**: `FilingStore.without(cnpj, dt_refer) -> FilingStore`: answers as the original except that the company has no period `dt_refer`; the original is not changed; other companies share their period dicts.
**Where**: `py/pit_rebuild/cvm.py`
**Files**: `py/pit_rebuild/cvm.py`, `tests/test_pit_rebuild_cvm.py`
**Heavy**: no
**Depends on**: T4
**Reuses**: `FilingStore`
**Requirement**: LAG-01 (B-9 recomputation)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] On the view, `ttm`, `balance`, `shares` and `newest_period_end` of the company answer from the next newest period
- [ ] The original store still has the period; another company's answers are identical on both
- [ ] `without` on a period the company does not have returns an equivalent store
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 3 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): view a filing store without one period (#64)`

---

### T6: Event ex-dates and the events of a class in a window

**What**: In `dividends.py`: `event_ex_date(event, next_session)` (B-1: first session after `lastDatePrior`, else the calendar day after it) and `class_events(events, isin, after, until, next_session)` (events of `isin` with `after < ex <= until`; `until` None is unbounded; sorted by ex-date). `split_divisor` unchanged.
**Where**: `py/pit_rebuild/dividends.py`
**Files**: `py/pit_rebuild/dividends.py`, `tests/test_pit_rebuild_dividends.py`
**Heavy**: no
**Depends on**: T5
**Reuses**: `StockEvent`; `b3_company_sbsp.json`, `b3_company_espa.json` fixtures
**Requirement**: SHR-01, SHR-02 (window), B-1

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Ex-date on `after`: out; on `until`: in; one day after `until`: out (SHR-02)
- [ ] An event of another ISIN is out
- [ ] `next_session` None (outside the calendar): ex-date = `lastDatePrior + 1 day`
- [ ] `until=None` keeps every later event
- [ ] `split_divisor` results of the existing tests unchanged
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 5 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): select the stock events of a class in a window (#64)`

---

### T7: Share count at `d` from the filing, the events and the scale

**What**: New `py/pit_rebuild/shares.py` with `Adjusted` and `adjust(filed, isins, events, b3_total, d, next_session)` (design `shares.py`, B-2, B-3): scale chosen on the count adjusted by every event after the period end; value = filed × factor × events in (`DT_REFER`, `d`]; a class with shares and no ISIN is not adjusted and is in `no_isin`.
**Where**: `py/pit_rebuild/shares.py`
**Files**: `py/pit_rebuild/shares.py` (new), `tests/test_pit_rebuild_shares.py` (new)
**Heavy**: no
**Depends on**: T6
**Reuses**: `cvm.share_scale`, `dividends.class_events`
**Requirement**: SHR-01, SHR-02, SHR-04, B-3

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] ON 1,000,000 filed at 2026-06-30, 100 % bonus (multiplier 2) ex 2026-07-15, `d` 2026-08-07, B3 total 2,000,000: `n_on` 2,000,000, factor 1, applied `[{class ON, factor 2, ...}]` (SHR-01)
- [ ] Same with `d` 2026-07-14: `n_on` 1,000,000 and nothing applied (SHR-02)
- [ ] 1:5 split after `d`, B3 total 5,000,000: factor 1 (not 10), `n_on` 1,000,000 (B-3, the double-count guard)
- [ ] Two events in the window multiply; a grouping (multiplier 0.1) divides
- [ ] PREF with shares and ISIN None: not adjusted, `no_isin == ["PREF"]` (SHR-04)
- [ ] No B3 total: factor 1, `verified` false, events still applied
- [ ] Gate check passes: Build (end of Phase 1)
- [ ] Test count: ≥ 7 tests in `test_pit_rebuild_shares.py`

**Tests**: unit
**Gate**: build
**Commit**: `feat(pit-rebuild): adjust share counts by the B3 events up to d (#64)`

---

### T8: Suspect share-count change

**What**: `shares.suspect(prev, prev_factor, current, current_factor, events, isins, next_session) -> dict | None` and `SUSPECT_CHANGE = 0.20` (design, B-4).
**Where**: `py/pit_rebuild/shares.py`
**Files**: `py/pit_rebuild/shares.py`, `tests/test_pit_rebuild_shares.py`
**Heavy**: no
**Depends on**: None (Phase 1 done)
**Reuses**: `dividends.class_events`
**Requirement**: SHR-05, SHR-06, SHR-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] 236,000,000 → 82,000,000 with no event: row with `ratio` 0.3475 (SHR-05, the Vivara shape)
- [ ] Change exactly 20 % (100 → 120 and 100 → 80): None (SHR-07)
- [ ] 20.01 %: row
- [ ] Same change with an event ex-date in (prev `DT_REFER`, current `DT_REFER`]: None; event on prev `DT_REFER`: row (SHR-07)
- [ ] Totals compared after their own factors: 1,000 (×1000) → 1,000,000 (×1): None
- [ ] `suspect` never changes `Adjusted` (SHR-06: it takes no count to return)
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 6 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): flag share counts that jump without a B3 event (#64)`

---

### T9: Accounting and adjusted field lists

**What**: `fields.ACCOUNTING_FIELDS` (15, B-7) and `fields.ADJUSTED_FIELDS` (14, B-8), both in `ALL_FIELDS` order, derived from `SCORE_FIELDS`/`ALL_FIELDS`, not typed twice.
**Where**: `py/pit_rebuild/fields.py`
**Files**: `py/pit_rebuild/fields.py`, `tests/test_pit_rebuild_fields.py`
**Heavy**: no
**Depends on**: T8
**Reuses**: `SCORE_FIELDS`, `ALL_FIELDS`
**Requirement**: STL-01, STL-02, ADJ-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `set(ALL_FIELDS) - set(ACCOUNTING_FIELDS) == {PRECO, VALOR DE MERCADO, LIQUIDEZ MEDIA DIARIA, DY}` and `len == 15`
- [ ] `ADJUSTED_FIELDS` = `SCORE_FIELDS` without `ROIC` and `PEG RATIO`, `len == 14`
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 2 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): name the accounting and adjusted field sets (#64)`

---

### T10: Mask the accounting fields of a stale filing

**What**: `indicators.STALE_FILING_REASON = "stale_filing"` and `mask_accounting(values, reason)`; the docstring's reason list names it.
**Where**: `py/pit_rebuild/indicators.py`
**Files**: `py/pit_rebuild/indicators.py`, `tests/test_pit_rebuild_indicators.py`
**Heavy**: no
**Depends on**: T9
**Reuses**: `Value`, `fields.ACCOUNTING_FIELDS`
**Requirement**: STL-01, STL-02

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Every `ACCOUNTING_FIELDS` value becomes `Value(None, "stale_filing")`
- [ ] `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA`, `DY` keep value, reason and note
- [ ] The input dict is not mutated
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 2 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): blank the accounting fields of a stale filing (#64)`

---

### T11: `si_lag` predicate

**What**: `compare.is_si_lag(field, raw, calc, calc_prev) -> bool` (design `compare.py`, B-9).
**Where**: `py/pit_rebuild/compare.py`
**Files**: `py/pit_rebuild/compare.py`, `tests/test_pit_rebuild_compare.py`
**Heavy**: no
**Depends on**: T10
**Reuses**: `reference`, `within`
**Requirement**: LAG-01, LAG-03, LAG-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] P/L ref 10.00, calc 15, prev 10.4: true; prev 10.6: false (5 % edge, LAG-01)
- [ ] calc within: false even when prev is within (LAG-03)
- [ ] calc None (missing), prev within: true; prev None: false
- [ ] Blank non-DY reference: false (out of denominator); blank DY (ref 0) follows the A-4 zero rule
- [ ] Percent field: 0.5 p.p. edge on both sides
- [ ] `summarize` output unchanged for the same pairs (LAG-04)
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 6 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): detect misses that the previous filing explains (#64)`

---

### T12: Adjusted verdict block in the report object (HEAVY, counts as 2)

**What**: `report.build(..., adjusted=None)` and `_adjusted(given, iso_dates)` (design `report.py`): scope text, fields, excluded, per date and field the statistics with `verdict` and `si_lag {count, tickers}`, worst rows extended with `si_lag`, `filing`, `previous_value`; per field `overall`; `class`; `criterion` (70 %, `met`, `failing`). Built after `verdicts`, never read by it.
**Where**: `py/pit_rebuild/report.py`
**Files**: `py/pit_rebuild/report.py`, `tests/test_pit_rebuild_report.py`
**Heavy**: yes (new block with its own verdict and criterion rules; many hand-computed cases)
**Depends on**: T11
**Reuses**: `compare.verdict`, `overall`, `worst`; `_without_reference_stale` pattern; `fields.ADJUSTED_FIELDS`
**Requirement**: ADJ-02, ADJ-03, ADJ-04, ADJ-05 (JSON), ADJ-06, ADJ-07, LAG-02 (JSON), LAG-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] The block lists exactly the 14 `ADJUSTED_FIELDS`; `ROIC` and `PEG RATIO` absent (ADJ-02)
- [ ] Per field, `overall` is the worse of the two dates; `class` the worst field (ADJ-03)
- [ ] Worst rows carry `si_lag`, `filing` (or null) and `previous_value` (or null) as given (ADJ-04)
- [ ] `criterion.met` true at 70.0 % on every field and date; false at 69.99 % on one, with that `{field, date, pct_within}` in `failing` (ADJ-05)
- [ ] Denominator 0 on a date: the field fails with `pct_within` null (ADJ-06)
- [ ] `verdicts` byte-identical (via `encode_json`) with and without `adjusted` (ADJ-07)
- [ ] `si_lag.count` equals `len(tickers)`; within counts unchanged by `si_lag` (LAG-02, LAG-04)
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 8 new tests in `test_pit_rebuild_report.py`

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): add the adjusted verdict and release criterion (#64)`

---

### T13: Adjusted verdict section in the Markdown

**What**: `render_markdown` prints "## Veredito ajustado (#64)" after "## Veredito": scope sentence, the table per field (share within and denominator per date, `si_lag` per date, verdict), the criterion line ("Critério de liberação (#64): cumprido" / "não cumprido" with the failing rows, plus the pointer to `pit-rebuild-stocks-audit.md`), and the worst rows per field and date. Every number is read from the object.
**Where**: `py/pit_rebuild/report.py`
**Files**: `py/pit_rebuild/report.py`, `tests/test_pit_rebuild_report.py`
**Heavy**: no
**Depends on**: T12
**Reuses**: `_table`, `_pct`, `_delta`, `_canonical`
**Requirement**: ADJ-01, ADJ-05 (Markdown), LAG-02 (Markdown)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] "## Veredito" comes before "## Veredito ajustado (#64)", and the official section text is unchanged (ADJ-01)
- [ ] Both criterion wordings render from `met`; failing rows listed when not met (ADJ-05)
- [ ] The `si_lag` count of a field and date appears in its row (LAG-02)
- [ ] Without the `adjusted_verdict` key, the Markdown has no such section (v2 objects still render)
- [ ] Gate check passes: Build (end of Phase 2)
- [ ] Test count: ≥ 4 new tests

**Tests**: unit
**Gate**: build
**Commit**: `feat(pit-rebuild): render the adjusted verdict section (#64)`

---

### T14: Schema v3, investigations and the new mapping lists in the Markdown

**What**: `SCHEMA = "pit-rebuild-stocks/v3"`, `AMENDED_BY = [64]` (header "(issue #59, correções da #64)"), `INVESTIGATIONS = ()` stored as `investigations` and rendered under "## Investigações" ("Nenhuma." when empty); the mapping section of each date renders `share_events.adjusted`, `share_events.no_isin`, `share_count_suspect`, `ttm_prior_from_itr`, `stale_filing` ("nenhum" when empty; absent keys render nothing, so v2 objects still render).
**Where**: `py/pit_rebuild/report.py`
**Files**: `py/pit_rebuild/report.py`, `tests/test_pit_rebuild_report.py` (the v2 schema assertion becomes v3)
**Heavy**: no
**Depends on**: None (Phase 2 done)
**Reuses**: `_mapping_section`, `_table`, `_list_or_none`
**Requirement**: SHR-03, SHR-04, SHR-05 (Markdown), STL-03 (Markdown), TTM-04 (Markdown), MBR-02 (slot), B-12

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `schema == "pit-rebuild-stocks/v3"`, `amended_by == [64]`, header line names both issues
- [ ] Each of the five lists renders one row per entry, in the object's order, and "nenhum" when empty
- [ ] One `INVESTIGATIONS` entry given through a patched constant renders ticker, date, field, finding, evidence
- [ ] Gate check passes: Quick
- [ ] Test count: ≥ 5 new tests

**Tests**: unit
**Gate**: quick
**Commit**: `feat(pit-rebuild): report schema v3 with the new mapping lists (#64)`

---

### T15: Wire the event-adjusted share count and the suspect flag into the CLI (HEAVY, counts as 2)

**What**: `cli._company` uses `shares.adjust` with the ISINs of the A-17 class tickers at `day` (B-2) and the company's B3 events; `previous_shares` + a second `adjust` feed `shares.suspect`. `_evaluate_day` adds `share_events {adjusted, no_isin}` and `share_count_suspect` to the mapping block (sorted as in SHR-03, SHR-05); `share_scale.scaled`/`unverified` keep their keys, residual on the event-adjusted total.
**Where**: `py/pit_rebuild/cli.py`
**Files**: `py/pit_rebuild/cli.py`, `tests/test_pit_rebuild_cli.py`, fixture additions under `tests/fixtures/pit_rebuild/` or the temp shape universe (`write_shape_universe`)
**Heavy**: yes (fixture company with a CVM composition, a B3 record with `stockDividends` and COTAHIST ISINs; expectations of `ScaledShareCountTests` may move and each move is hand-computed)
**Depends on**: T14
**Reuses**: `write_shape_universe`, `ScaledShareCountTests`, `b3_company_sbsp.json`
**Requirement**: SHR-01, SHR-02, SHR-03, SHR-04, SHR-05, SHR-06, SHR-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Through `main(["report", ...])`: a fixture company with an event between period end and the first date has `N` × multiplier in `VALOR DE MERCADO` and is in `share_events.adjusted` with that factor (SHR-01, SHR-03)
- [ ] The same company at a date before the event: not listed, count as filed (SHR-02)
- [ ] A fixture company whose count jumps > 20 % between two filings with no event is in `share_count_suspect` and its `VALOR DE MERCADO` uses the jumped count (SHR-05, SHR-06)
- [ ] A company with a PREF class and no PREF ISIN is in `share_events.no_isin` (SHR-04)
- [ ] Any changed #59 expectation (SBSP, ESPA, scaled residuals) is recomputed by hand in the test and named in the commit body
- [ ] Exit code 0 on the fixture run; the exit-code table is unchanged (RPT-05)
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 4 new CLI tests; pit suite passes

**Tests**: integration
**Gate**: full
**Commit**: `feat(pit-rebuild): use event-adjusted share counts in the rebuild (#64)`

---

### T16: Wire the stale-filing rule into the CLI

**What**: In `_evaluate_day`, after `compute`, a company whose `(day − newest_period_end).days > cvm.STALE_FILING_DAYS` (age from the decision date, B-6) gets `indicators.mask_accounting` on each ticker; the mapping block gains `stale_filing` rows `{ticker, cd_cvm, dt_refer, age_days}` sorted by ticker.
**Where**: `py/pit_rebuild/cli.py`
**Files**: `py/pit_rebuild/cli.py`, `tests/test_pit_rebuild_cli.py`, fixture additions (a company whose newest filing ends 2025-10-31)
**Heavy**: no
**Depends on**: T15
**Reuses**: T3 `newest_period_end`, T10 `mask_accounting`
**Requirement**: STL-01, STL-02, STL-03, STL-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] At 2026-08-07 (age 280): the 15 accounting fields are missing with reason `stale_filing` in the field statistics, and the ticker is in `stale_filing` (STL-01, STL-03)
- [ ] The ticker stays in `mapped` and in each denominator; `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA`, `DY` have values (STL-02)
- [ ] With the filing ending 270 days before the date: not stale (STL-04)
- [ ] The `--filings-cutoff` evaluation applies the same rule
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 3 new CLI tests

**Tests**: integration
**Gate**: full
**Commit**: `feat(pit-rebuild): blank indicators of filings older than 270 days (#64)`

---

### T17: List the companies whose 12-month flow used the year-earlier ITR

**What**: `_evaluate_day` adds `ttm_prior_from_itr` rows `{cd_cvm, dt_refer, accounts}` for the mapped companies whose `LL`, `3.01`, `3.05` or `3.08` flow had `prior_from_itr`, sorted by `cd_cvm`.
**Where**: `py/pit_rebuild/cli.py`
**Files**: `py/pit_rebuild/cli.py`, `tests/test_pit_rebuild_cli.py`, fixture additions (an ITR with a zero `PENÚLTIMO` and the year-earlier ITR)
**Heavy**: no
**Depends on**: T16
**Reuses**: T2 `Dated.prior_from_itr`
**Requirement**: TTM-01 (through the CLI), TTM-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Through `main`: the fixture company's `P/L` uses the fallback flow (hand-computed) and the company is listed with the accounts that used it (TTM-04)
- [ ] A company with a non-zero `PENÚLTIMO` is not listed
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 2 new CLI tests

**Tests**: integration
**Gate**: full
**Commit**: `feat(pit-rebuild): report the flows rebuilt from last year's ITR (#64)`

---

### T18: ISIN at `d` through the CLI

**What**: An integration test (and any wiring fix it exposes) that a ticker whose ISIN changes between the two decision dates has its dividends at the first date divided by the old ISIN's event, and its share count adjusted by the old ISIN's events.
**Where**: `tests/test_pit_rebuild_cli.py`
**Files**: `tests/test_pit_rebuild_cli.py`, fixture additions in a temp universe (COTAHIST lines built inline; no new session in the committed `cotahist.txt` after 2026-11-06 or between 2026-08-21 and 2026-08-24); `py/pit_rebuild/cli.py` only if the test exposes a call site T1 missed
**Heavy**: no
**Depends on**: T17
**Reuses**: T1, `write_shape_universe`
**Requirement**: ISN-03

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Through `main`: at the first date the old ISIN's split divides the dividend (hand-computed DY); at the second date the new ISIN's does
- [ ] The test fails when `cli.py` passes the last loaded session instead of `day` (mutation checked in a scratch copy, noted in the commit body)
- [ ] Gate check passes: Build (end of Phase 3)
- [ ] Test count: ≥ 1 new CLI test

**Tests**: integration
**Gate**: build
**Commit**: `test(pit-rebuild): pin the ISIN at d through the report command (#64)`

---

### T19: Previous-filing pass, `si_lag` and the adjusted block through the CLI (HEAVY, counts as 2)

**What**: In the official evaluation only: per computed company, the B-9 recomputation on `acq` with `dfp`/`itr` replaced by `without(cnpj, newest_period_end)`, same quotes and dividends, same stale rule; `_DayResult` carries `previous` values and each ticker's newest `Filing`; `_report_object` builds `adjusted` from the stale-free statistics of `_sensitivities` (`non_financial` group, `ADJUSTED_FIELDS`), `compare.is_si_lag` over the pairs, and the worst rows' `filing` and `previous_value`, and passes it to `report.build`.
**Where**: `py/pit_rebuild/cli.py`
**Files**: `py/pit_rebuild/cli.py`, `tests/test_pit_rebuild_cli.py`, fixture additions (a company with two ITRs and a Status Invest value matching the older one)
**Heavy**: yes (second evaluation path and the end-to-end adjusted block)
**Depends on**: None (Phase 3 done)
**Reuses**: T5 `without`, T11 `is_si_lag`, T12 `build(adjusted=...)`, `_sensitivities`
**Requirement**: LAG-01, LAG-02, LAG-03, LAG-04, ADJ-02, ADJ-04, ADJ-05, ADJ-07

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Through `main`: the fixture ticker whose SI `P/L` matches the older ITR is outside tolerance and in `adjusted_verdict.dates[d].fields["P/L"].si_lag.tickers` (LAG-01, LAG-02)
- [ ] A ticker of a company with one usable filing is never `si_lag` (LAG-03)
- [ ] `adjusted_verdict.excluded` equals each date's `reference_stale`; `ROIC` and `PEG RATIO` are absent; `class` and `criterion` follow the fixture's numbers (ADJ-02, ADJ-05)
- [ ] A worst row carries the filing kind, `DT_REFER`, `VERSAO`, `DT_RECEB` of the fixture filing and its previous value (ADJ-04)
- [ ] The official `verdicts` and `dates[d].fields` are identical to a run with the adjusted wiring removed (compare through a patched `_report_object` or a stored expectation) (ADJ-07, LAG-04)
- [ ] The `--filings-cutoff` block has no `si_lag`
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 5 new CLI tests

**Tests**: integration
**Gate**: full
**Commit**: `feat(pit-rebuild): count Status Invest lag in the adjusted verdict (#64)`

---

### T20: Investigate MBRF3's DY on 2026-11-05 (HEAVY, counts as 2; reads the real cache)

**What**: Find why MBRF3 has DY 0.00 against Status Invest's 18.41 % on 2026-11-05 (committed v2 JSON) using only the cached files: `b3_company_MBRF.json` (`tradingName` `MARFRIG`), `b3_cash_MARFRIG_p1.json`, the COTAHIST records of MBRF3, the Status Invest CSV row. Test the lookup hypothesis first (B-11). If the cause is the lookup after the name change, fix it in `cli.py`/`sources.py` with a fixture test (MBR-01). Otherwise put the cause in `report.INVESTIGATIONS` with the evidence (MBR-02).
**Where**: `py/pit_rebuild/report.py`
**Files**: `py/pit_rebuild/report.py` (`INVESTIGATIONS`), `tests/test_pit_rebuild_report.py`; or, on the MBR-01 branch, `py/pit_rebuild/cli.py` or `py/pit_rebuild/sources.py` with `tests/test_pit_rebuild_cli.py` and a fixture
**Heavy**: yes (real-data investigation)
**Depends on**: T19
**Reuses**: the #59 cache (precondition: the copy of T21 step 1 can be made here first; read-only)
**Requirement**: MBR-01, MBR-02

**Preconditions**: `.cache/pit-rebuild` present in this worktree (T21 step 1). **No network**: if the evidence needs a request outside the cache (for example B3 cash under another trading name), stop and report to the orchestrator with the exact URL and why; the request needs an explicit go-ahead and would be its own step.

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] The commit body states the cause with the evidence (file name, record, dates, window `d − 365 < ex ≤ d`)
- [ ] MBR-01 branch: a fixture test reproduces the miss and passes after the fix; or MBR-02 branch: `INVESTIGATIONS` holds `{ticker: "MBRF3", date: "2026-11-05", field: "DY", finding, evidence}` and a report test pins its render
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 1 new test

**Tests**: unit
**Gate**: full
**Commit**: `docs(pit-rebuild): record the cause of the MBRF3 DY gap (#64)` (or `fix(pit-rebuild): ... (#64)` on the MBR-01 branch)

---

### T21: Regenerate the report from the cache, offline (HEAVY, counts as 2; real data, no network)

**What**: Regenerate `docs/validation/pit-rebuild-stocks.json` and `.md` with the committed code and commit them unedited; extend `tests/test_pit_rebuild_artifact.py` for v3.
**Where**: `docs/validation/pit-rebuild-stocks.json`
**Files**: `docs/validation/pit-rebuild-stocks.json`, `docs/validation/pit-rebuild-stocks.md`, `tests/test_pit_rebuild_artifact.py`
**Heavy**: yes (about 360 MB of zips parsed; the run takes minutes)
**Depends on**: T20
**Reuses**: the #59 cache and run arguments (`.specs/features/pit-rebuild-stocks/tasks.md` T32)
**Requirement**: RPT-01, RPT-02, RPT-03, RPT-05

**Steps** (from the worktree root, clean tree at the commit of T20):

1. Cache: if `.cache/pit-rebuild/manifest.json` is absent, `mkdir -p .cache && cp -al /home/caio/.t3/worktrees/TCC/t3-issue-59-d1f86a5f2296a99f/.cache/pit-rebuild .cache/pit-rebuild` (hard links; `cp -a` if the link fails). `.cache/` is gitignored. Check `python3 -c "import json;print(len(json.load(open('.cache/pit-rebuild/manifest.json'))['files']))"` prints 585 or more.
2. Inputs: record `sha256sum data/raw/status_invest_fundamentals.csv /home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv`; the first must equal `inputs.status_invest[0].sha256` of the committed v2 JSON, the second must start with `87c15cb5`. `git status --porcelain -- data snapshots baselines` must be empty.
3. Cache check against v2: `git archive origin/develop py/pit_rebuild py/config.py | tar -x -C <scratch>/v2` and run `python3 <scratch>/v2/py/pit_rebuild report --offline <args of step 4> --cache-dir .cache/pit-rebuild --out-dir <scratch>/v2-out`; its JSON must equal the committed v2 JSON except `tool_commit`, the timings in `coverage.effort` and `inputs.raw_files[*].retrieved_at` if any. A difference elsewhere means the cache is not the #59 one: stop and report.
4. Run:

   ```sh
   python3 py/pit_rebuild report --offline \
     --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv --collected-on 2026-08-07 \
     --date 2026-11-05 --si /home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv \
     --collected-on 2026-11-06 --filings-cutoff 2026-11-05:2026-11-02 \
     --cache-dir .cache/pit-rebuild --out-dir docs/validation
   ```

   Exit 0 expected. Exit 5 means a required file is missing from the cache: stop and report (a download needs a go-ahead and becomes its own step with `fetch`). Exit 4: stop and report.
5. Commit the two files unedited, with the test changes. The commit body lists: wall time, the official verdict (unchanged or changed, and why), the adjusted class and `criterion.met`, the counts of `share_events.adjusted`, `share_count_suspect`, `ttm_prior_from_itr`, `stale_filing` per date, the companies whose A-18 factor changed against v2, and whether TIM's `LL` TTM on 2026 dates moved toward R$ 4.3 bi (issue AC3 example) and Ambipar is `stale_filing` (AC4 example) and Vivara in `share_count_suspect` (AC2 example). An example that does not hold is reported, not forced.

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] `report --offline` exits 0; no network request (offline)
- [ ] `tests/test_pit_rebuild_artifact.py`: schema v3 and `amended_by == [64]`; `adjusted_verdict` has the 14 fields, both dates, `criterion` with `threshold_pct` 70 and `met` equal to the rule recomputed from its own `pct_within` values; each date's mapping has the five new keys; the Markdown equals the render of the JSON byte for byte (RPT-02); the existing #59 artifact tests still pass (exit-code table, SI hash, dates, as-of) (RPT-05)
- [ ] `git status --porcelain -- data snapshots baselines` empty; both CSV hashes unchanged
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 3 new tests in `test_pit_rebuild_artifact.py`

**Tests**: integration
**Gate**: full
**Commit**: `docs(validation): regenerate the PIT report with the #64 fixes (#64)`

---

### T22: README section for the fixes

**What**: Update the "Reconstrução PIT de ações" section of `docs/validation/README.md`: the share-count rule with events (B-1..B-3), the suspect flag, the TTM fallback, `stale_filing`, the ISIN at `d`, the adjusted verdict and the release criterion, `si_lag`, the audit document, and the cache copy precondition of the regeneration. The exit-code table stays 0-5. Every number cited from the report equals the committed JSON (RPT-04).
**Where**: `docs/validation/README.md`
**Files**: `docs/validation/README.md`
**Heavy**: no
**Depends on**: T21
**Reuses**: the #59 section and its "Como regenerar"
**Requirement**: RPT-04, RPT-05

**Tools**:

- MCP: NONE
- Skill: `docs-writer`

**Done when**:

- [ ] The section names each new rule and list with its JSON key
- [ ] Any number about the report in the README is checked against the committed JSON (cite the key in the commit body)
- [ ] The artifact test of the exit-code table passes
- [ ] Gate check passes: Build (end of Phase 4)

**Tests**: none
**Gate**: build
**Commit**: `docs(validation): document the #64 rules of the PIT rebuild (#64)`

---

### T23: Audit the worst cases of 2026-08-07 (HEAVY, counts as 2; reads the real cache)

**What**: Create `docs/validation/pit-rebuild-stocks-audit.md` with its header (purpose, classes, how a row is read), the MBRF3 finding of T20 (MBR-02 repeat), and the section of 2026-08-07: one table per adjusted field with each worst row of `adjusted_verdict.dates["2026-08-07"]` (ticker, SI value, rebuilt value as in the JSON; class; evidence: filing kind, `DT_REFER`, `VERSAO`, `DT_RECEB` and the filed number), and the class counts per field. Add `AuditDocumentTests` to `tests/test_pit_rebuild_artifact.py`: for each date section present, rows equal the JSON worst rows (no missing, extra or different), each class in the set, evidence names a filing, counts equal the rows.
**Where**: `docs/validation/pit-rebuild-stocks-audit.md`
**Files**: `docs/validation/pit-rebuild-stocks-audit.md` (new), `tests/test_pit_rebuild_artifact.py`
**Heavy**: yes (up to 140 rows classified from the filings in the cache)
**Depends on**: None (Phase 4 done)
**Reuses**: the worst rows' `filing`, `previous_value` and `si_lag` (ADJ-04); `.cache/pit-rebuild` zips read with `cvm.load_filings` in a scratch script outside the repo
**Requirement**: AUD-01, AUD-02, AUD-03, AUD-04 (2026-08-07)

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Every worst row of 2026-08-07 in the committed JSON has one audit row, and no other row exists for that date (AUD-01, AUD-03)
- [ ] Each row has one class of the three and evidence with kind, `DT_REFER`, `VERSAO`, `DT_RECEB` and a number (AUD-02)
- [ ] Counts per field equal the rows (AUD-04)
- [ ] The test fails when one row is removed, one SI value is changed, or a class is misspelled (checked in a scratch copy, noted in the commit body)
- [ ] Gate check passes: Full
- [ ] Test count: ≥ 3 new tests

**Tests**: integration
**Gate**: full
**Commit**: `docs(validation): audit the 2026-08-07 worst cases of the adjusted verdict (#64)`

---

### T24: Audit the worst cases of 2026-11-05 (HEAVY, counts as 2; reads the real cache)

**What**: Add the 2026-11-05 section in the same format, a summary table (class counts per field and date) and the release-criterion line copied from the JSON (`criterion.met`, failing rows). `AuditDocumentTests` now requires both dates and the summary to equal the per-date counts.
**Where**: `docs/validation/pit-rebuild-stocks-audit.md`
**Files**: `docs/validation/pit-rebuild-stocks-audit.md`, `tests/test_pit_rebuild_artifact.py`
**Heavy**: yes (up to 140 rows)
**Depends on**: T23
**Reuses**: T23 format and test
**Requirement**: AUD-01, AUD-02, AUD-03, AUD-04, ADJ-05 (audit half of DEC-2), RPT-04

**Tools**:

- MCP: NONE
- Skill: NONE

**Done when**:

- [ ] Both dates present; every worst row of both dates audited, nothing extra (AUD-01, AUD-03)
- [ ] Summary counts equal the per-date counts; the criterion line equals the JSON (AUD-04, RPT-04)
- [ ] Gate check passes: Build (end of Phase 5)
- [ ] Test count: ≥ 1 new test (both dates required)

**Tests**: integration
**Gate**: build
**Commit**: `docs(validation): audit the 2026-11-05 worst cases and summarize (#64)`

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

Phase 1:  T1 → T2 → T3 → T4 → T5 → T6 → T7
Phase 2:  T8 → T9 → T10 → T11 → T12 → T13
Phase 3:  T14 → T15 → T16 → T17 → T18
Phase 4:  T19 → T20 → T21 → T22
Phase 5:  T23 → T24
```

Execution is sequential. One batch per phase fits the ~7 budget; Phases 4 and 5 hold the real-data work.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1 | 2 functions + call sites (one signature) | ✅ Cohesive (a split would leave the suite red) |
| T2 | 1 function + 1 field | ✅ Granular |
| T3 | 2 functions + 1 constant | ✅ Cohesive |
| T4 | 1 field + 1 function | ✅ Granular |
| T5 | 1 method | ✅ Granular |
| T6 | 2 functions | ✅ Cohesive |
| T7 | 1 new module function + dataclass | ✅ Granular |
| T8 | 1 function | ✅ Granular |
| T9 | 2 constants | ✅ Granular |
| T10 | 1 function | ✅ Granular |
| T11 | 1 function | ✅ Granular |
| T12 | 1 block (`_adjusted`) | ✅ Granular (heavy) |
| T13 | 1 Markdown section | ✅ Granular |
| T14 | schema + 1 section + mapping rows | ⚠️ Cohesive in one file (render of new keys) |
| T15 | `_company` wiring | ✅ Granular (heavy) |
| T16 | stale wiring | ✅ Granular |
| T17 | 1 list | ✅ Granular |
| T18 | 1 test | ✅ Granular |
| T19 | previous pass + adjusted wiring | ✅ Cohesive (heavy) |
| T20 | 1 investigation | ✅ Granular (heavy) |
| T21 | 1 regeneration | ✅ Granular (heavy) |
| T22 | 1 README section | ✅ Granular |
| T23 | 1 audit section + test | ✅ Granular (heavy) |
| T24 | 1 audit section + summary | ✅ Granular (heavy) |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | start of Phase 1 | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | T2 | T2 → T3 | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T4 | T4 → T5 | ✅ Match |
| T6 | T5 | T5 → T6 | ✅ Match |
| T7 | T6 | T6 → T7 | ✅ Match |
| T8 | None (Phase 1 done) | start of Phase 2 | ✅ Match |
| T9 | T8 | T8 → T9 | ✅ Match |
| T10 | T9 | T9 → T10 | ✅ Match |
| T11 | T10 | T10 → T11 | ✅ Match |
| T12 | T11 | T11 → T12 | ✅ Match |
| T13 | T12 | T12 → T13 | ✅ Match |
| T14 | None (Phase 2 done) | start of Phase 3 | ✅ Match |
| T15 | T14 | T14 → T15 | ✅ Match |
| T16 | T15 | T15 → T16 | ✅ Match |
| T17 | T16 | T16 → T17 | ✅ Match |
| T18 | T17 | T17 → T18 | ✅ Match |
| T19 | None (Phase 3 done) | start of Phase 4 | ✅ Match |
| T20 | T19 | T19 → T20 | ✅ Match |
| T21 | T20 | T20 → T21 | ✅ Match |
| T22 | T21 | T21 → T22 | ✅ Match |
| T23 | None (Phase 4 done) | start of Phase 5 | ✅ Match |
| T24 | T23 | T23 → T24 | ✅ Match |

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | reader (`cotahist`) + CLI call sites | unit (+ full gate) | unit, gate full | ✅ OK (CLI behaviour pinned in T18) |
| T2-T5 | reader (`cvm`) | unit | unit | ✅ OK |
| T6 | reader (`dividends`) | unit | unit | ✅ OK |
| T7, T8 | pure rule (`shares`) | unit | unit | ✅ OK |
| T9 | pure rule (`fields`) | unit | unit | ✅ OK |
| T10 | pure rule (`indicators`) | unit | unit | ✅ OK |
| T11 | pure rule (`compare`) | unit | unit | ✅ OK |
| T12-T14 | report | unit | unit | ✅ OK |
| T15-T19 | CLI | integration | integration | ✅ OK |
| T20 | report constant (or CLI on MBR-01) | unit (integration on MBR-01) | unit | ✅ OK (the MBR-01 branch adds a CLI test) |
| T21, T23, T24 | committed artifacts | artifact test | integration | ✅ OK |
| T22 | docs | none | none | ✅ OK |

## Requirement → Task Traceability

| Requirement | Tasks |
| ----------- | ----- |
| SHR-01 | T4, T6, T7, T15 |
| SHR-02 | T6, T7, T15 |
| SHR-03 | T14, T15 |
| SHR-04 | T7, T14, T15 |
| SHR-05 | T4, T8, T14, T15 |
| SHR-06 | T8, T15 |
| SHR-07 | T8, T15 |
| TTM-01 | T2, T17 |
| TTM-02 | T2 |
| TTM-03 | T2 |
| TTM-04 | T14, T17 |
| STL-01 | T3, T9, T10, T16 |
| STL-02 | T9, T10, T16 |
| STL-03 | T14, T16 |
| STL-04 | T16 |
| ISN-01 | T1 |
| ISN-02 | T1 |
| ISN-03 | T18 |
| MBR-01 | T20 |
| MBR-02 | T14, T20, T23 |
| ADJ-01 | T13 |
| ADJ-02 | T9, T12, T19 |
| ADJ-03 | T12 |
| ADJ-04 | T3, T12, T19 |
| ADJ-05 | T12, T13, T19, T24 |
| ADJ-06 | T12 |
| ADJ-07 | T12, T19 |
| LAG-01 | T5, T11, T19 |
| LAG-02 | T12, T13, T19 |
| LAG-03 | T11, T19 |
| LAG-04 | T11, T12, T19 |
| AUD-01 | T23, T24 |
| AUD-02 | T23, T24 |
| AUD-03 | T23, T24 |
| AUD-04 | T23, T24 |
| RPT-01 | T21 |
| RPT-02 | T21 |
| RPT-03 | every test task (fixtures only, no network) |
| RPT-04 | T22, T24 |
| RPT-05 | T15, T21, T22 |

40 of 40 requirements mapped.

## Tasks needing real data or network

| Task | Real data | Network |
| ---- | --------- | ------- |
| T20 | `.cache/pit-rebuild` (read-only) | none; stops and asks if a request outside the cache is needed |
| T21 | `.cache/pit-rebuild`, both Status Invest CSVs (one outside the worktree) | none (`--offline`) |
| T23, T24 | committed JSON; `.cache/pit-rebuild` zips for the filed numbers | none |

## Weighted count per phase (heavy = 2)

| Phase | Tasks | Task count | Heavy | Weighted count |
| ----- | ----- | ---------- | ----- | -------------- |
| 1 | T1-T7 | 7 | none | 7 |
| 2 | T8-T13 | 6 | T12 | 7 |
| 3 | T14-T18 | 5 | T15 | 6 |
| 4 | T19-T22 | 4 | T19, T20, T21 | 7 |
| 5 | T23-T24 | 2 | T23, T24 | 4 |
| **Total** | T1-T24 | 24 | 7 | 31 |
