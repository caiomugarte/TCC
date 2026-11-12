# PIT Rebuild Fixes for Stocks Specification (issue #64)

**Issue**: https://github.com/caiomugarte/TCC/issues/64
**Builds on**: `.specs/features/pit-rebuild-stocks/` (issue #59, PR #62, merged in `develop` at `ebaf1fd`). Every rule of that spec (A-1..A-22, AD-031, AD-032) holds unless a requirement below changes it, and the change is named.
**Status**: Draft r2 (2026-11-12) - fixes of the spec review iw-64-judge-spec-r1 (S1..S10), waiting for re-review.

## Problem Statement

The #59 report rated stocks "não reconstruível". Most of the worst cases come from Status Invest (late ITRs, untraded tickers, profits with no basis in the filing), but three are ours: the CVM share count misses splits and bonuses after the period end, the 12-month flow is wrong when the ITR's comparative column is zero, and a filing more than a year old still feeds the indicators. The production data source will move to CVM and B3 (issue `## Decisões`), so the tool needs these fixes and a measurement that separates our errors from the reference's before that move.

## Goals

- [ ] Rebuilt values follow the corrected rules for share counts, 12-month flows, stale filings and the ISIN at `d` (AC1-AC5 of the issue).
- [ ] The report states the adjusted verdict and whether part 1 of the release criterion holds: every adjusted indicator ≥ 70 % within tolerance on both dates (AC7, AC8). Part 2 (the 10 worst checked and explained) is stated by the audit document, which gives the whole criterion (ADJ-05, AUD-06).
- [ ] Each of the 10 worst cases of every adjusted indicator, on both dates, is classified in `docs/validation/pit-rebuild-stocks-audit.md`, with the source record that supports the classification: the filing for accounting indicators, the dividend record and the close for DY (AC9).
- [ ] `docs/validation/pit-rebuild-stocks.{json,md}` are regenerated from the committed code, every number in `docs/validation/` equals the committed artifact, and every rule text the report prints states the #64 rules (AC10, REVIEW.md).

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Own rules for banks and insurers | Issue `## Fora de escopo`. The financial group is still measured and reported as in #59, and stays out of the adjusted verdict. |
| Capital increases between filings (no B3 event) | Issue `## Fora de escopo`. They are flagged as suspect (SHR-05), never corrected. |
| Switching the production data source to CVM and B3 | Issue `## Fora de escopo`: another issue, after the release criterion holds. This feature only reports whether it holds (ADJ-05). |
| FIIs | Issue #60. |
| N4 and N5 of the PR #62 review | Issue `## Fora de escopo`. |
| Fixing the "erro nosso" cases the audit finds | The audit classifies (AUD-01..AUD-04); each fix it motivates is a new issue. Fixing in place would move the numbers the audit is classifying. |
| ROIC and PEG RATIO in the adjusted verdict | Issue `## Decisões`: their formulas become ours. They stay in the official verdict and in the per-date statistics. |
| New CLI arguments or exit codes | Nothing in the issue needs one. The exit-code table of `cli.py` and `docs/validation/README.md` stays 0-5 (RPT-05). |

---

## Decisions (closed in the issue, cited, not rediscussed)

| ID | Decision (issue `## Decisões`) | Where it lands |
| -- | ------------------------------ | -------------- |
| DEC-1 | Source of each field: CVM for accounting data; B3 for price, liquidity, dividends and corporate events; share count = CVM adjusted by the B3 events in the past, B3's current count today; one source per field. Status Invest is dropped as a source. | SHR-01..SHR-04 |
| DEC-2 | Release criterion: every indicator of the adjusted verdict with at least 70 % of the tickers within tolerance on both dates, and the 10 worst of each checked and explained. Rejected: 90 %; no fixed criterion. | ADJ-05, AUD-01 |
| DEC-3 | Stale filing: 270 days. Rejected: 200 and 365 days. | STL-01 |
| DEC-4 | Suspect share count: flag only, no automatic correction. | SHR-05..SHR-07 |
| DEC-5 | ROIC and PEG RATIO stay out of the adjusted verdict. | ADJ-02 |

---

## Assumptions & Open Questions

Each row is a choice the issue does not make. The chosen default is applied unless the spec review changes it.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| B-1 Event ex-date | An event's ex-date is the first market session after its `lastDatePrior` in the loaded COTAHIST calendar (A-16, A-20); when the calendar does not reach that day (before its first or after its last session), it is the calendar day after `lastDatePrior`. | A-20 already dates events this way for dividends. The fallback keeps an event outside the loaded calendar from being dropped from the share count; `split_divisor` (dividends, A-20) is not changed. | n |
| B-2 Which events adjust which count | `N_ON` takes the events of the ON price ticker; `N_PREF` those of the PREF price ticker (the A-17 choice of `class_prices` at `d`). The ISINs of a class at `d` are `I_d(cls)`: the ISIN of the ticker at `d` (ISN-01) plus every ISIN the ticker had in a loaded session in (`DT_REFER`, `d`]. Events are read for every ISIN of the set and deduplicated on (`label`, `multiplier`, `lastDatePrior`), so an event B3 lists under the old and the new ISIN counts once. A class with shares and an empty `I_d` (no session on or before `d`) is not adjusted and is listed in `share_events.no_isin`. | B3 lists one `stockDividends` entry per ISIN. The A-17 ticker is the one the market value already uses, so one class is priced and adjusted from the same ticker. The ISIN at `d` alone would drop an event of the old ISIN when the ISIN changed between `DT_REFER` and `d`. Dividends (A-20) keep the ISIN at `d` alone (`split_divisor` unchanged). | n |
| B-3 Share scale (A-18) after the fix | The A-18 power of ten is chosen on the filed count adjusted by `M_all`: the deduplicated events of `I_all(cls)` with ex-date in (`DT_REFER`, `R`], where `R` is the retrieval date of the company's B3 record (`retrieved_at` of `b3_company_<root>.json` in the cache manifest, read in São Paulo time, UTC−3) and `I_all(cls)` is `I_d(cls)` plus every ISIN the ticker had in a loaded session after `d`. Without `R` the window has no upper bound. The value is then the filed count × factor × `M_d` (events of `I_d` with ex-date in (`DT_REFER`, `d`], SHR-01). Events after `d` only pick the power of ten. AD-033 records the change to AD-032 (4). | Without it the A-18 factor absorbs a split between filing and the B3 retrieval (a 1:5 split became ×10) and the event factor applies it again: 50× instead of 5×. `totalNumberShares` is the count on the retrieval day (2026-11-10 for the #59 cache), so an event with ex-date after `R` is not in it yet and must not enter `M_all`. The ISIN can change after `d` (e.g. AZUL's `BRAZULA01OR8`, review S4), so the ISIN at `d` alone would miss events B3 lists under the new one. The B3 count still only picks the power of ten and never enters a value (AD-031 input 1). Limit: an ISIN change after the last loaded COTAHIST session (2026-11-06) and before `R` is not seen (AD-033). | n |
| B-4 Suspect change baseline | "Previous filing" is the newest usable filing (DFP or ITR) older than the one the count comes from whose composition total is > 0. Both totals are compared after their own A-18 factor (B-3). Change = `abs(S_new / S_prev − 1)`; strictly above 0.20 is suspect. "Event in the interval" is any deduplicated event of the ISINs the company's class tickers had in a loaded session from the previous period end to `d`, plus their ISINs at `d`, with ex-date in (previous period end, current period end]. | Comparing raw totals would flag every company that files one statement in thousands and the next in units. 0.20 itself is not suspect ("mais de 20 %"). | n |
| B-5 12-month flow fallback | When the ITR's `PENÚLTIMO` row of the account is absent or exactly 0, the prior year-to-date is the `ÚLTIMO` row of the same account in the ITR whose period ends one year before (29 Feb → 28 Feb), if that ITR is usable at `d`. Without it the flow has no value (it no longer becomes `FY_prev + YTD − 0`). The fallback applies wherever `ttm` reads an ITR, `ttm_year_before` included. | The issue's rule, applied in the one function (`cvm._ttm_of`) all four flows and PEG go through. A zero comparative that was real gives the same result through the fallback when last year's ITR also says 0. | n |
| B-6 Stale filing reference date | Age = `d − DT_REFER` of the newest usable filing, in days, with `d` the decision date even in the `--filings-cutoff` sensitivity. Stale when age > 270. Exactly 270 is not stale. | "a mais de 270 dias de `d`". The cutoff sensitivity only moves which filings are received, not `d`. | n |
| B-7 Accounting indicators | The fields that read a CVM statement value: `P/L`, `P/VP`, `PSR`, `EV/EBIT`, `ROE`, `ROA`, `ROIC`, `MARG. LIQUIDA`, `MARGEM EBIT`, `LIQ. CORRENTE`, `DIVIDA LIQUIDA / EBIT`, `DIV. LIQ. / PATRI.`, `CAGR RECEITAS 5 ANOS`, `CAGR LUCROS 5 ANOS`, `PEG RATIO` (15). `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA` and `DY` keep their values. | VALOR DE MERCADO is a market field: shares × price; the share count is no statement value. A stale ticker stays mapped and in every denominator, with no value and the reason `stale_filing` (A-4). | n |
| B-8 Adjusted verdict scope | Fields: the 16 `METRIC_GROUPS` fields less `ROIC` and `PEG RATIO` (14). Group: `non_financial` (unclassified tickers included, as the official class). Tickers: each date's eligible tickers less its `reference_stale` tickers (D-A). Per field and date: the A-12 statistics and `compare.verdict`; per field: the worse of the two dates; class: the worst field. `si_lag` tickers stay in the denominator as outside tolerance. | The issue lists exactly these cuts: "sem os papéis sem cotação recente, sem ROIC e PEG, grupo não financeiro, nas duas datas". It does not remove `si_lag` tickers, only counts them apart. | n |
| B-9 `si_lag` | A ticker, field and date is `si_lag` when the ticker is in that field's denominator, is outside tolerance (a missing value counts as outside), and the value recomputed as if the company's newest usable filing did not exist is within tolerance (CMP-01) of the same Status Invest value. The recomputation repeats every CVM read (flows, balances, share count and its events, stale rule) on the next newest usable filing, with the same quotes and dividends. Computed for the official evaluation only, not for the `--filings-cutoff` one. | "o valor recalculado com o demonstrativo anterior bate com o Status Invest". Rerunning the whole company read keeps one rule path; a field that reads no filing (PRECO, LIQUIDEZ, DY) can never be `si_lag`, which is correct. | n |
| B-10 Audit document | Hand-written Markdown, one table per adjusted field and date with the 10 worst cases of the committed JSON (`adjusted_verdict.dates[d].fields[f].worst`), each with: ticker, Status Invest value, rebuilt value (both as in the JSON), class (`erro nosso`, `erro do Status Invest` or `diferença de método`), and the evidence for that indicator. Evidence items are written with a fixed prefix so a test can read them: `CVM <DFP\|ITR> <DT_REFER> v<VERSAO> rec <DT_RECEB>: <conta> = <valor>` (a filing and the filed number); `B3 evento <ISIN> <label> <lastDatePrior> ×<multiplicador>` (a stock event); `COTAHIST <ticker> <sessão> <fechamento>` (a close); `B3 proventos <arquivo>: <data-ex> <valueCash>` or `Yahoo <arquivo>: <data-ex> <valor>` (a dividend record), or `B3 proventos <arquivo>: nenhum registro com d − 365 < data-ex ≤ d` (the file read and found empty in the window). A row of an accounting field carries a `CVM` item, or a `B3 evento` or `COTAHIST` item when the gap comes from the share count or the price. A `DY` row carries a dividend item (`B3 proventos` or `Yahoo`) and a `COTAHIST` item, and no `CVM` item is required (DY reads no filing). A worst list shorter than 10 is audited whole. The worst list holds only tickers with a rebuilt value and a non-zero reference on relative fields (`compare.summarize`): tickers with no value (`missing`, e.g. `stale_filing`, `missing_input`) and, on relative fields, tickers with a zero reference are not in it. The audit says so and summarizes `missing` per date, field and reason with the counts of the JSON (AUD-05); zero-reference tickers are not counted in the JSON and are named as a limit. | "classifica os 10 piores casos de cada indicador do veredito ajustado, nas duas datas ... com o número do demonstrativo". DY has no statement number: its number is the dividend record and the close, the two inputs of its formula. A test reads both files, so the audit cannot drift from the JSON. | n |
| B-11 MBRF3 outcome | The cause is investigated on the cached files first. If it is the B3 history lookup after the name change, the lookup is fixed with a fixture test. Otherwise the cause is written in the report through a fixed note in `report.py` (`INVESTIGATIONS`), rendered in the Markdown and stored in the JSON, and repeated in the audit document. A request outside the cache needs the orchestrator's go-ahead first. | "Se for a busca do histórico depois da troca de nome, ela é corrigida; se não, a causa fica escrita no relatório." A note in the code keeps the generated report regenerable byte for byte. | n |
| B-12 Report identity | `schema` becomes `pit-rebuild-stocks/v3`; `issue` stays 59 and a new key `amended_by: [64]` is rendered next to it. A new key `previous_report = {"schema": "pit-rebuild-stocks/v2", "commit": "ebaf1fd"}` names where the v2 numbers are. The official verdict keeps the #59 verdict rule (thresholds 90/70, class = worst field, same groups) and its position; its numbers are that rule applied to the values the #64 rules give, so they may differ from v2. The section "## Veredito" says so in one line rendered from `amended_by` and `previous_report`: "Regra de veredito da #59, aplicada aos valores com as correções da #64; os números da v2 estão no commit `ebaf1fd`." | The report is still #59's tool; the schema bump tells readers the object gained keys. "Unchanged in content" would read as "same numbers", which the share, TTM and stale fixes change. | n |

**Open questions:** none - all resolved by the issue's `## Decisões` or logged above as B-1..B-12 for the spec review. Revised in r2 after the review iw-64-judge-spec-r1: B-2, B-3, B-4 (S3, S4), B-10 (S2, S6), B-12 (S9).

---

## User Stories

### P1: Share count at `d` follows the B3 events ⭐ MVP

**User Story**: As the owner of the stock data move, I want the CVM share count adjusted by the B3 splits, groupings and bonuses up to `d` so that per-share indicators are not off by a split.

**Why P1**: Issue AC1 and AC2; the count enters `VALOR DE MERCADO`, `P/L`, `P/VP`, `PSR` and `EV/EBIT`.

**Acceptance Criteria**:

1. WHEN a company's newest usable filing at `d` has a composition total > 0 THEN the rebuild SHALL set `N_class = round(filed_class_count × f × M_class)`, where `f` is the A-18 factor of B-3 (SHR-08) and `M_class` (`M_d` of B-3) is the product of the multipliers of the deduplicated events of the class ISINs `I_d(cls)` (B-2) with ex-date (B-1) in (`DT_REFER`, `d`]. <!-- SHR-01 -->
2. WHEN an event's ex-date equals the filing's `DT_REFER` or is after `d` THEN the rebuild SHALL leave it out of `M_class`. <!-- SHR-02 -->
3. WHEN `abs(M_class − 1) > 1e-9` for a class of a company at `d` THEN the report SHALL list it in `dates[d].mapping.share_events.adjusted` as `{cd_cvm, class, isins, dt_refer, factor, events: [{isin, label, ex_date, multiplier}]}` (`isins` = `I_d(cls)` sorted), sorted by `cd_cvm` then class, and the Markdown SHALL print the same rows. A product that returns to 1 within 1e-9 (a 1:2 split and a 2:1 grouping) SHALL NOT be listed. <!-- SHR-03 -->
4. IF a class has shares > 0 and an empty `I_d(cls)` THEN the rebuild SHALL apply no event to that class and SHALL list `{cd_cvm, class}` in `dates[d].mapping.share_events.no_isin`. <!-- SHR-04 -->
5. WHEN the scaled composition total of the newest usable filing differs from the scaled total of the previous filing (B-4) by more than 20 % and no event of the company has an ex-date in (previous `DT_REFER`, current `DT_REFER`] THEN the report SHALL list `{cd_cvm, dt_refer_prev, total_prev, dt_refer, total, ratio}` in `dates[d].mapping.share_count_suspect`, and the Markdown SHALL print it. <!-- SHR-05 -->
6. WHEN a count is listed in `share_count_suspect` THEN the rebuild SHALL use it unchanged (SHR-01 still applies). <!-- SHR-06 -->
7. WHEN the change is exactly 20 %, or an event of the company falls in the interval THEN the rebuild SHALL NOT list the company in `share_count_suspect`. <!-- SHR-07 -->
8. WHEN the rebuild chooses the A-18 factor `f` of a class count THEN it SHALL compare with the B3 `totalNumberShares` the filed count × `M_all`, the product of the deduplicated events of `I_all(cls)` with ex-date in (`DT_REFER`, `R`] (B-3); an event with ex-date after `R` SHALL NOT enter `M_all`, and an event of an ISIN the ticker took after `d` SHALL enter it. <!-- SHR-08 -->

**Independent Test**: on a fixture company with one ON ISIN, a filing ending 2026-06-30 and a 100 % bonus with `lastDatePrior` 2026-07-14, the rebuilt `N_ON` at 2026-08-07 is twice the filed count and the company is listed with factor 2; at 2026-07-14 it is not. With the B3 record retrieved on 2026-07-20 and a 1:5 split with ex-date 2026-07-25, the factor is chosen without the split (SHR-08).

---

### P1: 12-month flows survive a zeroed ITR comparative ⭐ MVP

**User Story**: As the owner of the stock data move, I want the trailing 12-month profit, revenue, EBIT and tax right when an ITR's comparative column is zero or missing.

**Why P1**: Issue AC3 (TIM 2026: R$ 6.1 bi → R$ 4.3 bi).

**Acceptance Criteria**:

1. WHEN the newest usable filing is an ITR whose `PENÚLTIMO` row of the account is absent or 0 and the ITR of the same period one year earlier is usable at `d` THEN `cvm.ttm` SHALL return `FY_prev + YTD − YTD_prior`, where `YTD_prior` is that earlier ITR's `ÚLTIMO` row of the account, dated with the latest `DT_RECEB` of the three filings used. <!-- TTM-01 -->
2. IF the `PENÚLTIMO` row is absent or 0 and the earlier ITR is not usable at `d` (missing, or `DT_RECEB` > `d`) or has no `ÚLTIMO` row for the account THEN `cvm.ttm` SHALL return no value. <!-- TTM-02 -->
3. WHEN the `PENÚLTIMO` row is present and non-zero THEN `cvm.ttm` SHALL use it, as in #59, even when the earlier ITR exists. <!-- TTM-03 -->
4. WHEN any of the flows `LL`, `3.01`, `3.05`, `3.08` of a mapped company used the earlier ITR at `d` THEN the report SHALL list `{cd_cvm, dt_refer, accounts}` in `dates[d].mapping.ttm_prior_from_itr`, sorted by `cd_cvm`. <!-- TTM-04 -->

**Independent Test**: an inline ITR store where the 2026-06-30 ITR has `3.11` `ÚLTIMO` 300 and `PENÚLTIMO` 0, the 2025-06-30 ITR has `ÚLTIMO` 200 and the 2025 DFP has 1000: `ttm` gives 1000 + 300 − 200 = 1100; with the 2025 ITR's `DT_RECEB` at `d + 1` it gives no value.

---

### P1: A stale filing gives no accounting indicator ⭐ MVP

**User Story**: As the owner of the stock data move, I want a company whose newest filing is more than 270 days old to have no accounting indicator, so that a year-old profit never looks current.

**Why P1**: Issue AC4 (Ambipar); DEC-3.

**Acceptance Criteria**:

1. WHEN the newest usable filing of a mapped company ends more than 270 days before `d` (B-6) THEN the rebuild SHALL give each of its tickers no value with the reason `stale_filing` for the 15 accounting fields of B-7. <!-- STL-01 -->
2. WHILE a ticker is `stale_filing` the rebuild SHALL keep it mapped and in every denominator, and SHALL compute `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA` and `DY` as for any ticker. <!-- STL-02 -->
3. WHEN a ticker is `stale_filing` at `d` THEN the report SHALL list `{ticker, cd_cvm, dt_refer, age_days}` in `dates[d].mapping.stale_filing`, sorted by ticker, and the Markdown SHALL print it. <!-- STL-03 -->
4. WHEN the age is exactly 270 days THEN the rebuild SHALL NOT mark the ticker `stale_filing`. <!-- STL-04 -->

**Independent Test**: a fixture company whose only usable filing ends 2025-10-31 has `stale_filing` on every accounting field at 2026-08-07 (age 280 days) and values at 2026-07-28 (age 270).

---

### P1: ISIN and share class are read at `d` ⭐ MVP

**User Story**: As the reviewer of PR #62 (N3), I want the ISIN and the share class of a ticker read from its last session on or before `d`, not from the last session loaded.

**Why P1**: Issue AC5. Both choose the split events of dividends (A-20) and now of share counts (B-2), and the class of a price (A-17).

**Acceptance Criteria**:

1. WHEN a ticker's ISIN or ESPECI differs between its last session on or before `d` and a later loaded session THEN `cotahist.isin(q, ticker, d)` and `cotahist.share_class(q, ticker, d)` SHALL return the value of the session on or before `d`. <!-- ISN-01 -->
2. IF a ticker has no session on or before `d` THEN `cotahist.isin` and `cotahist.share_class` SHALL return None. <!-- ISN-02 -->
3. The rebuild SHALL pass the decision date `d` to every call of `isin`, `share_class` and `class_prices` in `cli.py`, so that on a fixture ticker whose ISIN changes after the first of two dates the first date's split adjustment uses the old ISIN. <!-- ISN-03 -->

**Independent Test**: a two-session COTAHIST fixture where a ticker's ISIN changes after `d`: `isin(q, t, d)` is the old one; through `main([...])` the dividend at the first date is divided by the old ISIN's event.

---

### P2: The MBRF3 DY gap has a stated cause

**User Story**: As the auditor of the report, I want the cause of MBRF3's DY 0 against Status Invest's 18.41 % on 2026-11-05 found and either fixed or written down.

**Why P2**: Issue AC6. One ticker; it does not change a rule unless the cause is the lookup.

**Acceptance Criteria**:

1. WHEN the investigation finds that the B3 cash history is looked up under a name that misses MBRF's records after its name change THEN the rebuild SHALL look the history up under a name that returns them, with a fixture test of the lookup (B-11). <!-- MBR-01 -->
2. IF the cause is anything else THEN the report SHALL state it in `investigations` (JSON) and in the Markdown section "Investigações", with the evidence (the cached file and the record that shows it), and the audit document SHALL repeat it. <!-- MBR-02 -->

**Independent Test**: either the fixture test of MBR-01 passes, or the committed JSON has `investigations` with an `MBRF3` entry and the Markdown renders it.

---

### P1: Adjusted verdict and release criterion ⭐ MVP

**User Story**: As the owner of the stock data move, I want a verdict that leaves out what the issue says does not measure us (stale references, ROIC, PEG, the financial group), next to the official one, and a line that says whether the release criterion holds.

**Why P1**: Issue AC7 and DEC-2; it is the number the production move waits on.

**Acceptance Criteria**:

1. The report SHALL keep the official `verdicts` block and its Markdown section "Veredito" with the verdict rule of #59, printed before the adjusted verdict, and the section SHALL carry the B-12 line saying the rule is #59's applied to the corrected values and the v2 numbers are at `ebaf1fd`. <!-- ADJ-01 -->
2. The report SHALL add a top-level `adjusted_verdict` block with the scope of B-8: `fields` (the 14 fields), `excluded` (`reference_stale` tickers per date), and per date and field the A-12 statistics of the `non_financial` group without those tickers and their `compare.verdict`. <!-- ADJ-02 -->
3. The `adjusted_verdict` block SHALL hold per field the worse verdict of the two dates (`compare.overall`) and a `class` verdict, the worst field. <!-- ADJ-03 -->
4. The worst-case rows of `adjusted_verdict` (the 10 of A-12 per field and date) SHALL each carry `si_lag` (bool), the filing the rebuilt value came from (`{kind, dt_refer, versao, dt_receb}`, or null without one; always null on `DY` rows, which read no filing) and `previous_value` (the B-9 recomputation, rounded to 2 decimals, or null; always null on `DY` rows). <!-- ADJ-04 -->
5. The `adjusted_verdict` block SHALL hold `criterion = {"part": 1, "threshold_pct": 70, "met": bool, "failing": [{field, date, pct_within}], "part2": "docs/validation/pit-rebuild-stocks-audit.md"}`, where `met` is true exactly when every one of the 14 fields has `pct_within` ≥ 70 on both dates. The Markdown SHALL print "Critério de liberação (#64), parte 1 (≥ 70 % nas duas datas): cumprida" or "... não cumprida" with the failing rows, then "Parte 2 (10 piores conferidos e explicados): ver `pit-rebuild-stocks-audit.md`", and SHALL NOT print the criterion as a whole as met. <!-- ADJ-05 -->
6. IF a field has denominator 0 on a date in the adjusted scope THEN the criterion SHALL count that field as failing on that date (`pct_within` null). <!-- ADJ-06 -->
7. The official `verdicts` block SHALL be identical with and without the adjusted block computed (no adjusted rule reaches it). <!-- ADJ-07 -->

**Independent Test**: on the shape universe of the #59 tests, the adjusted block leaves out one stale ticker and ROIC/PEG, and `criterion.met` flips when one field's within count crosses 70 %.

---

### P1: Status Invest lag is counted apart ⭐ MVP

**User Story**: As the auditor, I want to see how many misses disappear when we use the filing before the newest one, so that a late reference is not read as our error.

**Why P1**: Issue AC8.

**Acceptance Criteria**:

1. WHEN a ticker is outside tolerance on a field and date and its B-9 recomputation is within tolerance THEN the rebuild SHALL mark it `si_lag` for that field and date. <!-- LAG-01 -->
2. The `adjusted_verdict` block SHALL hold per field and date `si_lag = {count, tickers}` over the adjusted scope, and the Markdown SHALL print the count beside the share within tolerance. <!-- LAG-02 -->
3. IF the company has no usable filing older than its newest one, or the ticker is within tolerance with the newest filing THEN the rebuild SHALL NOT mark it `si_lag`. <!-- LAG-03 -->
4. The `si_lag` mark SHALL NOT change any denominator, `within` count, `pct_within` or verdict. <!-- LAG-04 -->

**Independent Test**: a fixture company with two ITRs where the Status Invest P/L matches the older one: the ticker is outside tolerance and `si_lag`; with a Status Invest value matching neither it is not.

---

### P1: Worst cases audited ⭐ MVP

**User Story**: As the owner of the stock data move, I want each of the 10 worst cases of every adjusted indicator classified with the filing that proves it, so that the release criterion's second half ("conferidos e explicados") is a document, not a claim.

**Why P1**: Issue AC9; DEC-2.

**Acceptance Criteria**:

1. The repository SHALL have `docs/validation/pit-rebuild-stocks-audit.md` with one row per worst case of `adjusted_verdict` in the committed JSON, for every adjusted field and both dates (B-10). <!-- AUD-01 -->
2. Each audit row SHALL carry exactly one class from {`erro nosso`, `erro do Status Invest`, `diferença de método`} and evidence for its indicator (B-10): on an accounting field, a `CVM` item (kind, `DT_REFER`, `VERSAO`, `DT_RECEB`, account and filed number), or a `B3 evento` or `COTAHIST` item when the gap comes from the share count or the price; on `DY`, a dividend item (`B3 proventos` or `Yahoo`: file, ex-date and value, or the file with no record in the window) and a `COTAHIST` close. <!-- AUD-02 -->
3. Each audit row's ticker, Status Invest value and rebuilt value SHALL equal the JSON's worst row, and a test SHALL fail when a row is missing, extra or differs. <!-- AUD-03 -->
4. The audit document SHALL give per field and date the count of each class, and those counts SHALL equal the rows. <!-- AUD-04 -->
5. The audit document SHALL say that tickers with no rebuilt value and, on relative fields, tickers with a zero reference are not in the worst list, and SHALL give per date, field and reason the count of the `missing` rows of `adjusted_verdict.dates[d].fields[f]`, equal to the JSON. <!-- AUD-05 -->
6. The audit document SHALL state the whole release criterion of DEC-2: "cumprido" exactly when `criterion.met` is true and every worst row of both dates is audited (part 2), else "não cumprido" with the failing part; it SHALL give the count of `erro nosso` rows beside it. <!-- AUD-06 -->

**Independent Test**: `tests/test_pit_rebuild_artifact.py` reads the committed JSON and the audit document and checks AUD-01..AUD-06: a `DY` row with only a `CVM` item fails, a `DY` row with a `B3 proventos` and a `COTAHIST` item passes.

---

### P1: Regenerated, consistent report ⭐ MVP

**User Story**: As the reviewer, I want the committed report to be the output of the committed code, with the tests on small fixtures and no network.

**Why P1**: Issue AC10; REVIEW.md "Todo número ... bate com o artefato commitado".

**Acceptance Criteria**:

1. The repository SHALL have `docs/validation/pit-rebuild-stocks.json` and `.md` written by `report --offline` from the commit that carries the code of this feature, unedited, with `schema` `pit-rebuild-stocks/v3` and `amended_by` `[64]` (B-12). <!-- RPT-01 -->
2. The committed Markdown SHALL equal `render_markdown` of the committed JSON byte for byte. <!-- RPT-02 -->
3. Every test of the feature SHALL run on fixtures under `tests/fixtures/pit_rebuild/` or built in a temp dir, with no network request and no read of `.cache/`. <!-- RPT-03 -->
4. Every number about the report in `docs/validation/README.md` and in the audit document SHALL equal the committed JSON. <!-- RPT-04 -->
5. The CLI SHALL keep the exit codes 0-5 of #59, and the table in `cli.py`'s docstring SHALL equal the one in `docs/validation/README.md`. <!-- RPT-05 -->
6. The rule texts the report prints SHALL state the AD-033 share count: `parameters.share_scale_rule` SHALL say that N = filed count × `f` × the B3 events of the class with ex-date in (`DT_REFER`, `d`], that `f` is chosen on the count adjusted by the events up to the B3 retrieval, that events after `d` only pick the power of ten, and that the √10 limit applies only to a change with no B3 event; the `stockDividends` row of `parameters.non_pit_inputs` SHALL say it adjusts dividends (A-20) and share counts (AD-033); `parameters.formulas["VALOR DE MERCADO"]` SHALL name the events in (`DT_REFER`, `d`]. None of them SHALL keep the #59 sentences "só eventos com data-ex ≤ d" for the share count or "um desdobramento de 1:5 dá ×10" as the rule for evented changes. <!-- RPT-06 -->

**Independent Test**: `tests/test_pit_rebuild_artifact.py` (existing tests plus the v3 keys and the RPT-06 texts) passes on the committed files.

---

## Edge Cases

- IF two events of one class fall in (`DT_REFER`, `d`] THEN the rebuild SHALL multiply both multipliers (SHR-01).
- IF a `stockDividends` entry is repeated in every field THEN it SHALL count once (A-20 parsing, unchanged).
- IF a company has no B3 record THEN the A-18 factor SHALL be 1 and unverified as in #59, and `M_class` SHALL be 1 (no events are known).
- IF the B3 record has no `retrieved_at` in the manifest THEN `M_all` SHALL have no upper bound (B-3).
- IF B3 lists the same event (label, multiplier, `lastDatePrior`) under the old and the new ISIN of a class THEN it SHALL count once in `M_d` and `M_all` (B-2).
- WHEN the suspect check finds no previous filing with a composition THEN the rebuild SHALL NOT flag the company (SHR-05).
- IF the newest usable filing is a DFP THEN the TTM fallback SHALL NOT apply (TTM-03; a DFP is annual).
- WHEN a ticker is `stale_filing` with the newest filing and not with the previous one THEN B-9 still recomputes with the previous filing, which is older, so it is stale too and cannot be `si_lag` (LAG-03).
- IF a ticker is unmapped or has no quote THEN it has no `si_lag` (no value to recompute) and is outside tolerance as in #59.
- WHEN the `--filings-cutoff` sensitivity runs THEN the new rules SHALL apply to it, and `si_lag` SHALL NOT be computed for it (B-9).

---

## Requirement Traceability

Issue acceptance criteria (AC1..AC10, in the issue's order) → requirements.

| Requirement ID | Story | Issue AC | Phase | Status |
| -------------- | ----- | -------- | ----- | ------ |
| SHR-01 | P1: Share count at d | AC1 | Tasks | Partial (T1 sets, T4 date; T7, T15) |
| SHR-02 | P1: Share count at d | AC1 | Tasks | Partial (T6 window; T7, T15) |
| SHR-03 | P1: Share count at d | AC1 | Tasks | Pending |
| SHR-04 | P1: Share count at d | AC1 | Tasks | Pending |
| SHR-05 | P1: Share count at d | AC2 | Tasks | Partial (T4 previous count; T8 flag) |
| SHR-06 | P1: Share count at d | AC2 | Tasks | Pending |
| SHR-07 | P1: Share count at d | AC2 | Tasks | Pending |
| SHR-08 | P1: Share count at d | AC1 | Tasks | Partial (T1, T6 sets and R; T7, T15) |
| TTM-01 | P1: 12-month flows | AC3 | Tasks | Unit (T2); CLI T17 |
| TTM-02 | P1: 12-month flows | AC3 | Tasks | Done (T2) |
| TTM-03 | P1: 12-month flows | AC3 | Tasks | Done (T2) |
| TTM-04 | P1: 12-month flows | AC3 | Tasks | Pending |
| STL-01 | P1: Stale filing | AC4 | Tasks | Partial (T3 age; T16 mask) |
| STL-02 | P1: Stale filing | AC4 | Tasks | Pending |
| STL-03 | P1: Stale filing | AC4 | Tasks | Pending |
| STL-04 | P1: Stale filing | AC4 | Tasks | Pending |
| ISN-01 | P1: ISIN at d | AC5 | Tasks | Done (T1) |
| ISN-02 | P1: ISIN at d | AC5 | Tasks | Done (T1) |
| ISN-03 | P1: ISIN at d | AC5 | Tasks | Partial (T1 call sites; T18 test) |
| MBR-01 | P2: MBRF3 | AC6 | Tasks | Pending |
| MBR-02 | P2: MBRF3 | AC6 | Tasks | Pending |
| ADJ-01 | P1: Adjusted verdict | AC7 | Tasks | Pending |
| ADJ-02 | P1: Adjusted verdict | AC7 | Tasks | Pending |
| ADJ-03 | P1: Adjusted verdict | AC7 | Tasks | Pending |
| ADJ-04 | P1: Adjusted verdict | AC7, AC9 | Tasks | Partial (T3 filing; T19 rows) |
| ADJ-05 | P1: Adjusted verdict | AC7 (DEC-2) | Tasks | Pending |
| ADJ-06 | P1: Adjusted verdict | AC7 | Tasks | Pending |
| ADJ-07 | P1: Adjusted verdict | AC7 | Tasks | Pending |
| LAG-01 | P1: SI lag | AC8 | Tasks | Partial (T5 view; T11, T19) |
| LAG-02 | P1: SI lag | AC8 | Tasks | Pending |
| LAG-03 | P1: SI lag | AC8 | Tasks | Pending |
| LAG-04 | P1: SI lag | AC8 | Tasks | Pending |
| AUD-01 | P1: Audit | AC9 | Tasks | Pending |
| AUD-02 | P1: Audit | AC9 | Tasks | Pending |
| AUD-03 | P1: Audit | AC9 | Tasks | Pending |
| AUD-04 | P1: Audit | AC9 | Tasks | Pending |
| AUD-05 | P1: Audit | AC9 | Tasks | Pending |
| AUD-06 | P1: Audit | AC7, AC9 (DEC-2) | Tasks | Pending |
| RPT-01 | P1: Regenerated report | AC10 | Tasks | Pending |
| RPT-02 | P1: Regenerated report | AC10 | Tasks | Pending |
| RPT-03 | P1: Regenerated report | AC10 | Tasks | Pending |
| RPT-04 | P1: Regenerated report | AC10 | Tasks | Pending |
| RPT-05 | P1: Regenerated report | AC10 (REVIEW.md) | Tasks | Pending |
| RPT-06 | P1: Regenerated report | AC1, AC10 (REVIEW.md) | Tasks | Pending |

**Coverage:** 44 requirements; every issue AC (AC1..AC10) maps to at least one; all 44 mapped to tasks in `tasks.md` (see its traceability table).

---

## Implicit-requirement sweep

| Dimension | Resolution |
| --------- | ---------- |
| Input validation & bounds | 270-day and 20 % bounds are exact (STL-04, SHR-07); events outside (`DT_REFER`, `d`] ignored (SHR-02). |
| Failure / partial-failure states | No new failure: a missing ISIN, previous filing or previous-year ITR gives a listed case or no value (SHR-04, TTM-02, LAG-03), never an exception. The report is still written atomically after every computation (#59 SAFE rules). |
| Idempotency / retry | N/A because the report is a pure function of the cache and the CSVs; two `--offline` runs give the same JSON except timings and `tool_commit` (RPT-01). |
| Auth boundaries & rate limits | N/A because the tool is local research code; the regeneration runs `--offline` with no request. |
| Concurrency / ordering | N/A because the CLI is single-process; list orders are fixed by sort keys (SHR-03, STL-03, TTM-04). |
| Data lifecycle / expiry | The cache is reused read-only; no file in `data/`, `snapshots/` or `baselines/` is written (#59 SAFE-01, unchanged). |
| Observability | Every new rule leaves a list in the report: `share_events`, `share_count_suspect`, `ttm_prior_from_itr`, `stale_filing`, `si_lag`, `investigations`; the rule texts in `parameters` state the #64 rules (RPT-06). |
| External-dependency failure | N/A because the regeneration is offline; a missing required file still exits 5 (RPT-05). |
| State-transition integrity | N/A because there is no state machine. |

## Success Criteria

- [ ] The regenerated JSON has `adjusted_verdict.criterion.met` (part 1) set by the rule of ADJ-05, the Markdown prints it as part 1, and the audit document states the whole criterion (AUD-06).
- [ ] `tests/test_pit_rebuild_artifact.py` passes on the committed JSON, Markdown and audit document.
- [ ] The pit suite and the root suite pass; protected paths are unchanged against the merge base with `origin/develop` (`ebaf1fd`).
