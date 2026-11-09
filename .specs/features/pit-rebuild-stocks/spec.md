# PIT Rebuild Feasibility for Stocks Specification

Issue: [#59](https://github.com/caiomugarte/TCC/issues/59) "Mede se dá para reconstruir os indicadores de ações do seletor com dados da CVM e da B3". Route: DRIVEN (decisions closed in the issue's `## Decisões`). Depends on: nothing.

## Problem Statement

The Premium stock selection has no historical validation: the point-in-time audit (#10, #49) marked `stock_selection` `UNVERIFIABLE` because Status Invest keeps no history, and the only forward test (2026-08-07 → 2026-11-06) is one point. The way out is to rebuild the selector's indicators in the past from primary sources (CVM DFP/ITR/FCA/cadastro, B3 COTAHIST and B3 cash dividends) and run the selector month by month (2012–2026). Before that large project, this feature measures whether the rebuild matches the real Status Invest on the two dates we hold. If it does not match, the large project is not worth doing.

## Goals

- [ ] A research CLI rebuilds, for every ticker of the eligible universe, the 16 `METRIC_GROUPS` indicators (`py/config.py:116-142`) plus `PRECO`, `VALOR DE MERCADO` and `LIQUIDEZ MEDIA DIARIA`, point-in-time, on 2026-08-07 and 2026-11-06.
- [ ] The committed report `docs/validation/pit-rebuild-stocks.json` and `.md` gives, per indicator and date, the share of tickers within tolerance, the median and p90 difference, the 10 worst cases, and a verdict per indicator and for the class (banks and insurers apart).
- [ ] The report states the earliest year each source covers each indicator and the effort of the full 2012–2026 monthly rebuild.
- [ ] Every raw file the report used is identified by URL and sha256, so the run is reproducible from the CVM and B3 sources.

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| FII feasibility test | Issue `## Fora de escopo` (next issue). |
| Full 2012–2026 rebuild and selection backtest | Issue `## Fora de escopo`; only if this verdict allows, in another issue. |
| Changing the selector, the score or the Premium policy | Issue `## Fora de escopo`. `py/core/**`, `api/**` and `py/config.py` stay byte-identical. |
| Growing `py/pit_validation.py` | Issue `## Áreas afetadas` (#50): the tool is its own package under `py/`. |
| Committing raw CVM/B3 files | Issue `## Decisões`: hundreds of MB; URL + sha256 suffice. |
| Recovering superseded filing versions from CVM RAD (`LINK_DOC`) | The yearly CVM files carry only each filing's latest version (A-6); fetching every past version per filing is part of the full rebuild, not of this feasibility test. |
| Tuning formulas until they match | The tool measures one declared formula per indicator; a mismatch is a result. Iterating formulas is follow-up work the report may recommend. |

---

## Assumptions & Open Questions

Decisions D-1..D-4 come from the issue's `## Decisões` and are settled (confirmed by the issue author). A-n rows are this spec's defaults for points the issue leaves open.

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| D-1 Match criterion | Within tolerance: relative difference ≤ 5%, or ≤ 0.5 p.p. for percent indicators. Indicator verdict: ≥ 90% of the denominator within tolerance → `reconstruível`; ≥ 70% and < 90% → `com ressalvas`; < 70% → `não reconstruível`. Class verdict = worst verdict among the indicators the score uses. | Issue `## Decisões`: the rebuild will carry 14 years of backtest, so it must be strict. | y (issue) |
| D-2 Banks and insurers | Measured in their own group with their own verdict; their verdict never changes the class verdict. | Issue `## Decisões`: different CVM chart of accounts (no EBIT, no net debt in the common format). | y (issue) |
| D-3 Raw data | Cached under `.cache/pit-rebuild/` (gitignored via `.cache/`); the committed report records URL + sha256 of each raw file. | Issue `## Decisões`. | y (issue) |
| D-4 Dividends for DY | B3 first (B3 listed-company cash dividends); Yahoo Finance as fallback per ticker; the report marks DY as secondary-source when any DY value came from Yahoo and records the caveat that Yahoo lacks delisted companies. A-15 and A-16 fix the source rule and the window. | Issue `## Decisões`: dividend per share is not ready-made in CVM or COTAHIST. | y (issue) |
| A-1 Eligible universe (denominator) | Rows of the Status Invest CSV of that date with `VALOR DE MERCADO ≥ 200,000,000` and `LIQUIDEZ MEDIA DIARIA ≥ 50,000` (blank or non-numeric → not eligible). | Production applies `VALOR DE MERCADO ≥ cap_min` and `LIQUIDEZ MEDIA DIARIA ≥ liq_min` (`py/core/preprocessing.py:103-106`); Premium interpolates `cap_min`/`liq_min` between the `STOCK_RULES` anchors (`api/app/services/premium_policy.py:274-280`), whose lowest values are the `arrojado` anchor 200,000,000 / 50,000 (`premium_policy.py:56`). Every profile's universe is therefore a subset of this one. Eligibility uses Status Invest's own values, the ones production used that day. | n |
| A-2 Percent indicators | `DY`, `ROE`, `ROA`, `ROIC`, `MARG. LIQUIDA`, `MARGEM EBIT`, `CAGR RECEITAS 5 ANOS`, `CAGR LUCROS 5 ANOS` use the 0.5 p.p. tolerance (values in percent, as in the CSV). All other 11 indicators use the 5% relative tolerance. | The issue names "DY, ROE e margens" as examples; these eight are the ones Status Invest publishes in percent. | n |
| A-3 Rounding and zero reference | The rebuilt value is rounded to 2 decimals (the CSV's precision) before comparison. With reference 0.00 and a relative-tolerance indicator, the ticker is within tolerance only when the rounded rebuilt value is 0.00. | Avoids division by zero without loosening the criterion. | n |
| A-4 Missing values | Denominator per indicator and date = eligible tickers whose Status Invest value is non-blank. A ticker in the denominator with no rebuilt value (unmapped, missing statement, missing quote, zero divisor) counts as outside tolerance. Eligible tickers whose Status Invest value is blank are reported as a count (`reference_blank`), with the subset where the rebuild has a value, and stay out of the denominator. **Exception, `DY` only:** a blank Status Invest `DY` is the reference value `0.00` and stays in the denominator, so it is within tolerance only when the rounded rebuilt DY is `≤ 0.50` (0.5 p.p., A-2, A-3); these tickers are counted as `reference_blank_as_zero`. | Keeps the denominator faithful to "eligible universe" while not inventing a reference. Rebuild gaps must hurt the score, or a tool that maps nothing would pass. The DY exception follows production: `py/core/preprocessing.py:280-285` fills a blank DY with `0.0` because Status Invest leaves DY blank for non-payers (the export has no zero DY). Blank DY is 66 of 269 eligible tickers on 2026-08-07 and 69 of 274 on 2026-11-06 (measured), so excluding them would drop a quarter of the universe the score actually sees. | n |
| A-5 Class verdict scope | The class verdict is the worst of the 16 `METRIC_GROUPS` indicators, across both dates, for the non-financial group. `PRECO`, `VALOR DE MERCADO` and `LIQUIDEZ MEDIA DIARIA` get indicator verdicts but do not enter the class verdict. | D-1 says "indicators the score uses"; the three filter/price fields are not scored (`py/config.py:205-206`). The report shows their verdicts so a failure in eligibility inputs is still visible. | n |
| A-6 Filing versions | CVM yearly zips carry only each filing's latest version, while the index file lists every version with its `DT_RECEB` (verified on `dfp_cia_aberta_2025.zip`: 76 companies with >1 version in the index, 0 with >1 version in `DRE_con`). A filing is usable at date `d` only when its latest version's `DT_RECEB ≤ d`; otherwise the tool falls back to the company's previous filing (an older period) and counts the ticker as `refiled_after_d`. | Satisfies the issue's PIT rule strictly (no byte delivered after `d` enters) at the cost of staleness, which the report quantifies. | n |
| A-7 Price and liquidity window | `PRECO` = COTAHIST last close (`PREULT`) of the ticker's last trading session ≤ `d`. `LIQUIDEZ MEDIA DIARIA` = mean of `VOLTOT` over the ticker's last 30 sessions ≤ `d` (fewer if fewer exist). The window is a module constant recorded in the report. | Status Invest does not publish its window; 30 sessions is a stated guess. `LIQUIDEZ MEDIA DIARIA` is outside the class verdict (A-5), so a wrong window cannot fail the class. | n |
| A-8 Indicator formulas | One formula per indicator, listed in `design.md` "Indicator formulas" and copied into the report's `formulas` field. | Status Invest does not publish exact formulas; the report measures this declared reading. | n |
| A-9 Ticker → company mapping | Primary: CVM FCA `valor_mobiliario` (`Codigo_Negociacao` → `CNPJ_Companhia`) over all FCA years ≤ year of `d`, latest `Data_Referencia` wins; CNPJ → `CD_CVM` via the cadastro. Fallback: B3 company data by ticker root (`codeCVM`). Units (`Valor_Mobiliario = Units`) map to their company with the share count parsed from `Composicao_BDR_Unit`. The B3 company record is fetched for the root (first four characters) of **every** eligible ticker of every date, before mapping, so the fallback has its data for tickers that no FCA year lists. | FCA 2026 alone misses 31 of the 269 eligible tickers on 2026-08-07 (measured); union of years plus B3 closes most gaps; each remaining gap is reported with its reason. Fetching B3 records only for companies FCA already mapped would starve the fallback (review r2, S10). | n |
| A-10 Financial group | A ticker is in the financial group when its company's cadastro `SETOR_ATIV` is one of `Bancos`, `Emp. Adm. Part. - Bancos`, `Seguradoras e Corretoras`, `Emp. Adm. Part. - Seguradoras e Corretoras`, `Intermediação Financeira`, `Emp. Adm. Part. - Intermediação Financeira`, `Arrendamento Mercantil`, OR its usable DRE line `3.01` description contains `Intermediação Financeira`. The report lists each such ticker with the rule that classified it. An unmapped ticker whose Status Invest `SETOR` is not `Financeiro e Outros` is non-financial (rule `si_setor`). An unmapped ticker whose `SETOR` is `Financeiro e Outros` is `unclassified`: it stays in the non-financial group for the class verdict, and the report also gives the class verdict computed without the unclassified tickers (VER-03). | D-2's reason is the chart of accounts; the second rule catches a bank filed under another sector label. Values verified in the live cadastro. Status Invest's `SETOR` is too coarse to put a ticker in the financial group: on 2026-08-07 the 44 eligible `Financeiro e Outros` tickers include banks and insurers but also `ALOS3`, `MULT3`, `IGTI11`, `LOGG3`, `B3SA3` and `SIMH3`. It is precise enough to say a ticker outside that sector is not a bank or insurer. Reporting both class verdicts keeps an unmapped bank from silently failing the class against D-2 without loosening the official denominator. | n |
| A-11 Status Invest dates | The 2026-08-07 CSV is `data/raw/status_invest_fundamentals.csv` (tracked). The 2026-11-06 CSV is `snapshots/data/selectors-2026-11-06/stocks.csv`, present only in the main checkout and not in git; the CLI takes both paths as arguments and only reads them. The report records each CSV's path as given and its sha256. | Triage note; the tool must not depend on an untracked path being present. | n |
| A-12 Statistics | Difference per ticker: relative `|calc − ref| / |ref|` (in %) for relative indicators, `|calc − ref|` in p.p. for percent indicators, over tickers where both exist (and ref ≠ 0 for relative). Median = `statistics.median`; p90 = nearest-rank (the `ceil(0.9·n)`-th smallest). 10 worst = largest differences, ties by ticker ascending; tickers with no rebuilt value are listed apart with their reason. | Fixed definitions make the numbers testable. | n |
| A-13 Effort estimate | The run times two phases apart. **Load** (parsing raw files, done once per file in a full rebuild): seconds and bytes parsed per yearly-file source kind (DFP, ITR, FCA, cadastro, COTAHIST); dividend parsing is timed and reported apart, not projected. **Compute** (mapping, indicators and comparison for one date): seconds over the run's dates ÷ number of dates. Projection: load = Σ over kinds of (measured seconds ÷ measured bytes) × full-rebuild bytes of that kind; compute = seconds per date × 180 monthly decision dates (2012-01 … 2026-12); total = load + compute. Full-rebuild bytes = Σ HTTP `Content-Length` of every yearly file SRC-01's rule names for some month of 2012-01 … 2026-12 (DFP from 2012 − 7, ITR from 2012 − 2, FCA from 2010, COTAHIST from 2012 − 1, each through 2026), clipped to the earliest year the source offers (COV-01), plus the cadastro; the report lists the clipped ranges. Per-company dividend files are counted, not sized or projected. | Load scales with bytes and happens once; compute scales with dates. Multiplying one per-date figure that mixes both by 180 would count the load 180 times. Scaling by bytes, not years, absorbs COTAHIST's uneven yearly sizes. The compute projection assumes today's universe size, which the report states. | n |
| A-14 Tool runtime | Standard library only (`csv`, `zipfile`, `urllib`, `hashlib`, `statistics`), so tests run under the system `python3` without pandas. | `py/pit_validation.py` is stdlib-only; keeps the gate simple. | n |
| A-15 Dividend source per ticker and date | B3 source: the company record `GetListedSupplementCompany` (gives `tradingName`, `codeCVM`) and the paginated history `GetListedCashDividends` by `tradingName` (records `typeStock`, `valueCash`, `lastDatePriorEx`, `corporateAction`), every page fetched. B3 is used for a ticker at `d` only when its company's history is complete (records fetched = `page.totalRecords`), non-empty, and its earliest `lastDatePriorEx ≤ d − 365 days`; otherwise the ticker uses Yahoo (`period1`/`period2` daily chart with `events=div`) and the reason is recorded (`b3_failed`, `b3_empty`, `b3_incomplete`, `b3_short_history`). **Confirmed non-payer:** when the reason is `b3_empty` (company record resolved, history complete with `totalRecords = 0`) and Yahoo fails too, DY's dividends are 0 with source `b3` and reason `b3_empty_yahoo_failed`, not `no_dividend_source`. A ticker's B3 records are those whose `typeStock` equals the first token of the ticker's COTAHIST `ESPECI` (`ON`, `PN`, `PNA`, `PNB`, `UNT`, …); a unit with no `UNT` records sums its composition's class records. Every `corporateAction` counts (dividends, JCP, `RENDIMENTO`). | AZUL (verified 2026-11-09): `GetListedCashDividends` for `AZUL` returns `totalRecords: 0` and Yahoo `AZUL4.SA`/`AZUL53.SA` answer "No data found, symbol may be delisted"; a complete empty B3 history is evidence of no dividends, so turning it into a miss would fail DY for a known non-payer. Verified 2026-11-09: the company record's embedded `cashDividends` keeps only recent records (PETR: 24, `lastDatePrior` 2025-12-22 … 2026-08-21), which leaves the window of `d = 2026-08-07` (from 2025-08-08) uncovered and would make DY silently low while labelled primary. `GetListedCashDividends` returns the full history (PETROBRAS: 343 records, 1996-03-21 … 2026-08-21; the window holds 24 records including 2025-08-21) and carries `UNT`/`PNA` classes (TAESA, BTGP BANCO). The earliest-record check guards against any retention-limited response. Yahoo `range=max` collapses dividends to month buckets, so explicit `period1`/`period2` is required (verified: PETR4 ex-dates 2025-08-22, 2026-08-24). | n |
| A-16 Dividend window | One rule for both sources: a dividend counts for `d` when `d − 365 days < ex_date ≤ d`. Yahoo gives `ex_date` directly. For B3, `ex_date` = the first market session strictly after `lastDatePriorEx`, where market sessions = every session date in the loaded COTAHIST files; a record with no such session in the loaded data has its ex-date after the data horizon and never counts. A record whose `lastDatePriorEx` is before the first loaded session has no ex-date either and never counts (its true ex-date is unknown; mapping it onto the first loaded session would put a 2020 dividend at 2025-01-02). | A single definition, testable at both ends. Using the session calendar instead of `lastDatePriorEx + 1 day` is exact when `lastDatePriorEx` is a Friday (PETR: Friday 2026-08-21 → Yahoo ex-date Monday 2026-08-24). COTAHIST starts at January of `year(d) − 1` (SRC-01), so the lower cut can only drop a dividend whose `lastDatePriorEx` is the last session of `year(d) − 2` and whose ex-date falls inside the window, which needs `d` in the first days of January; neither run date is. | n |
| A-17 Share counts and `VALOR DE MERCADO` | `N_ON`, `N_PREF` = issued shares `QT_ACAO_ORDIN_CAP_INTEGR`, `QT_ACAO_PREF_CAP_INTEGR` (treasury **included**) of the newest usable filing's `composicao_capital` (DFP or ITR), times the share scale of A-18; `N_total = N_ON + N_PREF` is the per-share divisor of every per-share field. `VALOR DE MERCADO` = `N_ON × p_ON + N_PREF × p_PREF`, the same for every ticker of the company, where `p_ON` is the `PRECO` (A-7) of the company's `ON` ticker (COTAHIST `ESPECI` first token `ON`) and `p_PREF` the `PRECO` of its preferred ticker (`ESPECI` first token starting with `PN`: `PN`, `PNA`, `PNB`, …) with the highest A-7 average volume, ties by ticker; units never enter the sum. A class with `N > 0` and no quoted ticker uses the other class's price, and the company is counted in `vm_price_fallback`; with neither class quoted the field has no value (`no_quote`). COTAHIST is loaded for every eligible ticker, every FCA `Codigo_Negociacao` of a mapped company, every Status Invest ticker sharing a root with an eligible ticker, and every ticker named in a unit composition, so a non-eligible class (`EUCA3` next to eligible `EUCA4`) is priced from its own quote. | Status Invest's own number, checked 2026-11-09 on the 2026-11-06 CSV with B3's current `numberCommonShares`/`numberPreferredShares` (issued, treasury included: USIMINAS ON 705,260,684 = CVM `CAP_INTEGR`) and SI's `PRECO`: `ON × p_ON + PREF × p_(most liquid preferred)` reproduces `VALOR DE MERCADO` within 10⁻⁷ (cents) for USIM (USIM5, not USIM6), UNIP (UNIP6, not UNIP5), BRKM (BRKM5), BRSR (BRSR6), CPLE (CPLE5), EQPA, CMIG, EUCA, TASA, PETR, ITUB, VALE, WEGE, VIVT and RANI; pricing the preferred shares at the other preferred ticker misses by up to 16 % (USIM6 on 2026-11-06). Exceptions: AXIA (SI used AXIA6, not the more liquid AXIA7), unit companies (KLBN, TAEE within 0.5 %; ENGI 5.9 %, SULA 8.8 %), and on 2026-08-07 companies whose count changed since (AXIA, BBDC). SI's `VPA` also divides by issued shares: on 2026-08-07 against Q1-2026 ITR `PL`, of the 13 checked companies with ≥ 3 % treasury, 10 match issued shares better (SUZB, 22 % treasury: issued within 0.3 %, outstanding off by 29 %; HAPV, SBFG, COGN, LWSA likewise) and 3 match outstanding better (MULT, TRIS, PRIO). On 2026-08-07, 30 companies have an eligible and a non-eligible ticker (EUCA3/EUCA4, TASA3/TASA4, RANI3/RANI4, USIM6, BRKM6, UNIP5, CGAS3, …) and 10 eligible companies trade two or more preferred tickers (USIM, UNIP, BRKM, BRSR, CPLE, EQPA, AXIA, COCE, CRPG, CEBR); 30 and 11 on 2026-11-06. Counts come from the CVM filing (point-in-time), not from B3's current record. | n |
| A-18 Share scale | CVM `composicao_capital` carries no scale column, and some companies file counts in thousands. Per filing, the scale factor `f ∈ {1, 1000}` is the one that brings `N_total × f` closest, in `|log10|` distance, to `totalNumberShares` of the company's B3 record (A-9); the B3 count picks the unit only and never enters a value. No B3 record or a zero count → `f = 1` and the company is counted under `share_scale_unverified`. The report lists, per date, the companies with `f = 1000` and the unverified ones. | Measured 2026-11-09: matching Q1-2026 ITR `PL / N_total` against SI's `VPA` on 2026-08-07, 39 of 252 mapped companies only match with counts × 1000 (YDUQ, TRIS, KEPL, LREN, …); CVM vs B3: EUCATEX 31,258 vs 31,257,700, CEMIG 956,602 vs 956,601,911, IRANI 230,501 vs 230,501,219. Left as filed, every per-share field and `VALOR DE MERCADO` of 15 % of the universe would be off by 1000×. The B3 record is current, not dated `≤ d`; using it for the unit only is exact unless the count changed by ≥ 31.6× between the filing and the B3 retrieval (stated in the report; AD-031). | n |

**Open questions:** none - all resolved or logged above. Human checkpoint: rows A-1..A-18 are agent defaults pending human approval of this spec.

---

## User Stories

### P1: Rebuild and compare on two real dates ⭐ MVP

**User Story**: As the project owner, I want the selector's 19 stock fields rebuilt point-in-time from CVM and B3 and compared with the real Status Invest on 2026-08-07 and 2026-11-06, so that I know whether a 2012–2026 backtest of the stock selection can rest on rebuilt data.

**Why P1**: It is the whole question of the issue.

**Acceptance Criteria**:

1. WHEN `fetch` runs for dates `d1, d2` THEN the tool SHALL download into the cache directory every raw file the rebuild of those dates needs: CVM cadastro, DFP yearly zips for every year from `year(d) − 7` to `year(d)`, ITR yearly zips for every year from `year(d) − 2` to `year(d)`, FCA yearly zips for every year from 2010 to `year(d)`, B3 COTAHIST yearly zips for `year(d) − 1` and `year(d)`, the B3 company record for the root of every eligible ticker of every date (fetched before mapping, A-9), every page of the B3 cash-dividend history for each `tradingName` those records resolve for a mapped company (A-15), and Yahoo dividends for each ticker that SRC-05 sends to Yahoo, and SHALL record each file's URL, sha256, byte size and retrieval timestamp in `<cache>/manifest.json`. (SRC-01)
2. WHEN `fetch` finds a file already in the cache whose sha256 matches the cache manifest THEN the tool SHALL not download it again. (SRC-02)
3. IF a cached file's sha256 differs from the one recorded in the cache manifest THEN the tool SHALL exit 4 naming the file and SHALL write no report. (SRC-03)
4. IF a required CVM or COTAHIST download fails (HTTP error, timeout, or a zip that does not open) THEN the tool SHALL exit 5 naming the URL and SHALL keep the files already cached. (SRC-04)
5. WHEN, for an eligible ticker and a date `d`, its company's B3 cash-dividend history failed to download, is empty, is incomplete (records fetched ≠ `totalRecords`), or has its earliest `lastDatePriorEx` after `d − 365 days` THEN the tool SHALL compute that ticker's DY at `d` from Yahoo Finance dividends and SHALL record, per ticker and date, the source `yahoo` and the reason from `{b3_failed, b3_empty, b3_incomplete, b3_short_history}`; otherwise the source SHALL be `b3`. WHEN the reason is `b3_empty` and Yahoo fails too THEN the ticker's 12-month dividends SHALL be 0 with source `b3` and reason `b3_empty_yahoo_failed`. (SRC-05)
6. WHEN `report` builds date `d` THEN the tool SHALL use only filings whose latest-version `DT_RECEB ≤ d` and only COTAHIST sessions dated `≤ d`. (PIT-01)
7. IF a value dated after `d` (a filing with `DT_RECEB > d`, a session after `d`, or a dividend with ex-date after `d`) reaches the indicator computation for date `d` THEN the tool SHALL exit 4 naming the ticker, the date and the offending date. (PIT-02)
8. WHEN a company's latest filing for the most recent period has `DT_RECEB > d` THEN the tool SHALL use its previous usable filing and SHALL count the ticker under `refiled_after_d` for date `d`. (PIT-03)
9. WHEN `report` runs THEN the tool SHALL map every eligible ticker to a CVM company by A-9 and SHALL report, per date, the counts of mapped and unmapped tickers and, for each unmapped ticker, one reason from `{not_in_fca_or_b3, no_cvm_code, no_usable_filing, unit_composition_unparsed, no_quote}`. (MAP-01)
10. The tool SHALL classify each eligible ticker as financial, non-financial or unclassified by A-10 and SHALL list every financial and every unclassified ticker with the rule that classified it. (MAP-02)
11. WHEN a ticker is a unit THEN the tool SHALL compute its per-share indicators with the unit price divided by the number of shares in its composition. (MAP-03)
12. WHEN `report` builds date `d` THEN the tool SHALL compute, for each mapped eligible ticker, the 19 fields (the 16 `METRIC_GROUPS` indicators plus `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA`) with the formulas of A-8, using trailing-12-month flows and the latest usable balance sheet. (IND-01)
13. The tool's indicator list SHALL equal the flattened `METRIC_GROUPS` of `py/config.py` plus `PRECO`, `VALOR DE MERCADO`, `LIQUIDEZ MEDIA DIARIA`, read from `config.py` at run time, not copied. (IND-02)
14. IF a formula's divisor is zero or an input is missing THEN the tool SHALL leave that field without a value and SHALL record the reason for that ticker and field. (IND-03)
15. WHEN `report` builds date `d` THEN the tool SHALL compute `VALOR DE MERCADO` and the per-share divisor by A-17 with the share scale of A-18, using a quote of every ticker A-17 lists (eligible or not), and SHALL report per date the companies that used the class-price fallback (`vm_price_fallback`), the companies with share scale 1000, and the `share_scale_unverified` companies. (IND-04)
16. WHEN comparing a field THEN the tool SHALL apply the tolerance of D-1 with the indicator classes of A-2 and the rounding and zero rule of A-3. (CMP-01)
17. WHEN comparing a date THEN the tool SHALL report, per indicator, per date and per group (non-financial, financial): denominator size (A-4, with a blank `DY` reference counted as `0.00` inside the denominator), count and % within tolerance, `reference_blank` count, `reference_blank_as_zero` count (DY only), median and p90 difference (A-12), and the 10 worst cases with ticker, reference, rebuilt value and difference. (CMP-02)
18. WHEN an indicator's % within tolerance is ≥ 90 THEN its verdict for that date and group SHALL be `reconstruível`; WHEN ≥ 70 and < 90 THEN `com ressalvas`; WHEN < 70 THEN `não reconstruível`; WHEN its denominator is 0 THEN `sem referência`. (VER-01)
19. The tool SHALL give each indicator and group an overall verdict equal to the worse of its two date verdicts, with the order `reconstruível` > `com ressalvas` > `não reconstruível` and `sem referência` ignored when the other date has a verdict. (VER-02)
20. The tool SHALL set the class verdict to the worst overall verdict among the 16 `METRIC_GROUPS` indicators of the non-financial group (A-5), unclassified tickers included (A-10); SHALL report the class verdict recomputed without the unclassified tickers, the list of those tickers, and whether the two verdicts differ; and SHALL report the financial group's verdict, computed the same way, as a separate field that does not change the class verdict. (VER-03)
21. WHERE any eligible ticker's DY came from Yahoo Finance the report SHALL mark DY `source: secondary`, give the count of tickers per dividend source, and state that Yahoo lacks delisted companies. (VER-04)

**Independent Test**: Run `report` against a committed fixture cache with two dates and two small Status Invest CSVs; the JSON gives the hand-computed % within tolerance, verdicts and class verdict.

---

### P1: Committed, reproducible report

**User Story**: As a reviewer, I want the result committed as JSON and Markdown with the provenance of every input, so that the numbers can be checked and the run reproduced.

**Why P1**: The issue's AC requires the committed report; REVIEW.md treats docs numbers that do not match the artifact as Important.

**Acceptance Criteria**:

1. WHEN `report` succeeds THEN the tool SHALL write `pit-rebuild-stocks.json` and `pit-rebuild-stocks.md` in `--out-dir`, both written only after every computation finished. (RPT-01)
2. The JSON SHALL record, for every raw file used, its URL, sha256 and byte size, and for each Status Invest CSV its path as given, sha256 and date. (RPT-02)
3. The JSON SHALL record the parameters it used: dates, eligibility thresholds and their source lines, tolerances, indicator classes, liquidity window, formulas, and the tool's git commit when available. (RPT-03)
4. Every number in the Markdown SHALL be rendered from the JSON object written in the same run. (RPT-04)
5. IF any step fails THEN the tool SHALL leave any existing report files in `--out-dir` byte-identical. (RPT-05)
6. WHEN `docs/validation/pit-rebuild-stocks.json` is committed THEN it SHALL be the output of a real `fetch` + `report` run on 2026-08-07 and 2026-11-06, and the Markdown committed next to it SHALL be the same run's output. (RPT-06)

**Independent Test**: Fixture run writes both files; a test re-renders the Markdown from the JSON and gets identical bytes; a failing run leaves pre-existing report files unchanged.

---

### P1: Safe research CLI

**User Story**: As the maintainer, I want a CLI with a documented exit-code table that never writes into frozen or production data trees.

**Why P1**: REVIEW.md "Sempre verificar" (exit codes in the documented table, each tested through the real path; tools never write into `data/`, `snapshots/`, `baselines/` or the registry); issue AC.

**Acceptance Criteria**:

1. The CLI docstring SHALL hold the exit-code table: 0 written; 1 the report files could not be written (`OSError` on output, as `py/pit_validation.py:1427-1429`); 2 invalid arguments; 3 a given path does not exist; 4 integrity violation (cache hash mismatch, PIT leak, Status Invest CSV missing a required column); 5 a required download failed or a required file is absent from the cache in `--offline` mode. (CLI-01)
2. IF `--out-dir` or `--cache-dir` resolves inside `<repo>/data`, `<repo>/snapshots` or `<repo>/baselines` THEN the tool SHALL exit 2 before reading or writing anything. (SAFE-01)
3. The tool SHALL open each Status Invest CSV read-only, and their bytes SHALL be identical before and after a run. (SAFE-02)
4. WHEN `report` runs with `--offline` THEN the tool SHALL make no network request. (CLI-02)
5. IF a date argument is not `YYYY-MM-DD`, a date repeats, or the number of `--si` CSVs differs from the number of dates THEN the tool SHALL exit 2 with one stderr line naming the argument. (CLI-03)
6. IF a `--si` CSV or `--cache-dir` (for `report`) does not exist THEN the tool SHALL exit 3 with one stderr line naming the path. (CLI-04)
7. Each exit code 0, 1, 2, 3, 4 and 5 SHALL have a test that drives `main([...])` with the real subcommand path and a fixture cache, with no network (downloads mocked at the HTTP-open function). (CLI-05)

**Independent Test**: Unit tests drive `main()` for each exit code; a test tree snapshot of `data/`, `snapshots/`, `baselines/` stays identical.

---

### P2: Historical coverage and effort

**User Story**: As the project owner, I want the report to estimate from which year each indicator can be rebuilt and what the full rebuild costs, so that I can size the follow-up issue.

**Why P2**: Required by the issue, but the verdict (P1) decides whether the follow-up exists.

**Acceptance Criteria**:

1. WHEN `fetch` runs THEN the tool SHALL read the CVM directory listings of DFP, ITR and FCA and probe B3 COTAHIST yearly files from 2005 to 2012, record the earliest year each source offers, and store the listings in the cache with sha256; WHEN the 2005 COTAHIST file exists THEN the report SHALL state COTAHIST's earliest year as `≤ 2005 (probe floor)` and never an earlier year. (COV-01)
2. The report SHALL give, per indicator, the earliest year it can be rebuilt = the latest of its sources' earliest years plus its lookback (5 fiscal years for the two CAGR indicators; 2 years for `PEG RATIO`, which compares two trailing-12-month net incomes one year apart; 1 year for the other trailing-12-month flows; 0 for balance-sheet-only and price-only fields), and, for DY, the earliest ex-date per dividend source actually used in the run (B3, Yahoo) with the count of tickers that used it and that source's retention as requested (B3 `GetListedCashDividends`: full company history; Yahoo: from the request's `period1`); DY's earliest year comes from B3 when any ticker used B3, else from Yahoo. (COV-02)
3. The report SHALL estimate the full rebuild's effort per A-13: decision-date count; measured load seconds and bytes per source kind; measured compute seconds per date; projected load, compute and total time; the clipped year range per source; and the full-rebuild bytes with each URL's `Content-Length`. (COV-03)

**Independent Test**: Fixture listings with known earliest years give the expected per-indicator earliest year; with an injected clock, the effort block scales the load by bytes and multiplies only the per-date compute by 180.

---

## Edge Cases

- IF an eligible ticker's company files no consolidated statement for the period THEN the tool SHALL use the individual statement (`_ind`) of the same filing and SHALL record `statement: individual` for that ticker.
- IF a ticker has no COTAHIST session within 30 calendar days before `d` THEN the tool SHALL mark it unmapped with reason `no_quote`.
- IF the Status Invest CSV has a duplicated ticker THEN the tool SHALL exit 4 naming the ticker.
- IF the Status Invest CSV lacks `TICKER`, `SETOR` (read by A-10), any of the 19 fields, `VALOR DE MERCADO` or `LIQUIDEZ MEDIA DIARIA` THEN the tool SHALL exit 4 naming the column.
- WHEN a company has two classes traded (e.g. ON and PN) THEN `VALOR DE MERCADO` SHALL be `N_ON × p_ON + N_PREF × p_PREF` with issued shares (A-17), using the other class's price for a class with no quote (counted in `vm_price_fallback`), and SHALL be the same value for every ticker of the company.
- WHEN a company trades two preferred tickers (e.g. `USIM5` with high volume and `USIM6` with almost none) THEN every preferred share SHALL be priced at the preferred ticker with the highest A-7 average volume.
- WHEN only the preferred ticker of a company is eligible (e.g. `EUCA4`) THEN `p_ON` SHALL come from the non-eligible `ON` ticker's own quote (`EUCA3`), never from the preferred price.
- WHEN a company's `composicao_capital` total is about 1/1000 of its B3 `totalNumberShares` THEN its counts SHALL be multiplied by 1000 (A-18).
- WHEN a B3 record's `lastDatePriorEx` is before the first loaded COTAHIST session THEN that record SHALL have no ex-date and SHALL not count for any `d`.
- WHEN the B3 and Yahoo dividend sources both fail for a ticker THEN its DY SHALL have no value with reason `no_dividend_source` (counted outside tolerance per A-4, also when its Status Invest DY is blank), except when the B3 failure is `b3_empty` (SRC-05: dividends 0, source `b3`, reason `b3_empty_yahoo_failed`).
- WHEN an eligible ticker's Status Invest `DY` is blank and its rebuilt DY is `3.00` THEN that ticker SHALL count in DY's denominator and outside tolerance; WHEN its rebuilt DY is `0.50` THEN within.
- WHEN an ITR carries two rows for the same (company, `DT_REFER`, account, `ORDEM_EXERC`) — the fiscal year-to-date and the three-month quarter (Q2 and Q3 filings) — THEN the trailing-12-month flow SHALL use the row with the earliest `DT_INI_EXERC` (year-to-date), whatever the row order in the file; this also holds for fiscal years that do not start in January.
- WHEN a ticker's company lacks the usable fiscal year five years before its latest usable fiscal year THEN the two CAGR fields SHALL have no value with reason `short_history`.
- WHEN a ticker's company lacks a usable trailing-12-month net income ending one year before the latest one THEN `PEG RATIO` SHALL have no value with reason `short_history`.
- WHEN the trailing-12-month net income or EBIT is negative THEN ratios SHALL still be computed with sign (Status Invest publishes negative `P/L` and `EV/EBIT`), and only a zero divisor leaves the field empty.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| SRC-01 | P1 rebuild | Tasks (T5, T6, T18) | In Tasks |
| SRC-02 | P1 rebuild | Tasks (T5) | In Tasks |
| SRC-03 | P1 rebuild | Tasks (T5, T17) | In Tasks |
| SRC-04 | P1 rebuild | Tasks (T5, T18) | In Tasks |
| SRC-05 | P1 rebuild | Tasks (T10, T18) | In Tasks |
| PIT-01 | P1 rebuild | Tasks (T7, T8, T9, T17) | In Tasks |
| PIT-02 | P1 rebuild | Tasks (T1, T10, T12, T17) | In Tasks |
| PIT-03 | P1 rebuild | Tasks (T7) | In Tasks |
| MAP-01 | P1 rebuild | Tasks (T4, T11, T18) | In Tasks |
| MAP-02 | P1 rebuild | Tasks (T11, T16) | In Tasks |
| MAP-03 | P1 rebuild | Tasks (T11, T12) | In Tasks |
| IND-01 | P1 rebuild | Tasks (T8, T9, T10, T12) | In Tasks |
| IND-02 | P1 rebuild | Tasks (T3) | In Tasks |
| IND-03 | P1 rebuild | Tasks (T12) | In Tasks |
| IND-04 | P1 rebuild | Tasks (T8, T9, T11, T12, T16, T17) | In Tasks |
| CMP-01 | P1 rebuild | Tasks (T3, T13) | In Tasks |
| CMP-02 | P1 rebuild | Tasks (T4, T13) | In Tasks |
| VER-01 | P1 rebuild | Tasks (T14) | In Tasks |
| VER-02 | P1 rebuild | Tasks (T14) | In Tasks |
| VER-03 | P1 rebuild | Tasks (T14, T16, T20) | In Tasks |
| VER-04 | P1 rebuild | Tasks (T16) | In Tasks |
| RPT-01 | P1 report | Tasks (T2, T17) | In Tasks |
| RPT-02 | P1 report | Tasks (T16) | In Tasks |
| RPT-03 | P1 report | Tasks (T16) | In Tasks |
| RPT-04 | P1 report | Tasks (T16, T20) | In Tasks |
| RPT-05 | P1 report | Tasks (T2, T17) | In Tasks |
| RPT-06 | P1 report | Tasks (T19, T20) | In Tasks |
| CLI-01 | P1 CLI | Tasks (T1, T17, T19) | In Tasks |
| CLI-02 | P1 CLI | Tasks (T5, T17) | In Tasks |
| CLI-03 | P1 CLI | Tasks (T17) | In Tasks |
| CLI-04 | P1 CLI | Tasks (T17) | In Tasks |
| CLI-05 | P1 CLI | Tasks (T17, T18) | In Tasks |
| SAFE-01 | P1 CLI | Tasks (T2, T17) | In Tasks |
| SAFE-02 | P1 CLI | Tasks (T4, T17) | In Tasks |
| COV-01 | P2 coverage | Tasks (T15, T18) | In Tasks |
| COV-02 | P2 coverage | Tasks (T15, T16) | In Tasks |
| COV-03 | P2 coverage | Tasks (T15, T17, T20) | In Tasks |

**ID format:** `[CATEGORY]-[NUMBER]`.

**Coverage:** 37 total, 37 mapped to tasks, 0 unmapped.

---

## Success Criteria

- [ ] `docs/validation/pit-rebuild-stocks.json` and `.md` are committed from a real run and give a class verdict in `{reconstruível, com ressalvas, não reconstruível}` plus a separate financial-group verdict.
- [ ] Every number in the committed Markdown equals the committed JSON (re-render test).
- [ ] Each CLI exit code 0, 1, 2, 3, 4, 5 has a passing test through `main()`; the suite runs with no network.
- [ ] `git diff develop -- data/ snapshots/ baselines/ py/core/ api/ py/config.py py/pit_validation.py` is empty.
