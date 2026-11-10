# PIT Rebuild Feasibility for Stocks Validation

## Validation: FAIL — 2 blocking, 3 nonblocking

| Field | Value |
| --- | --- |
| Date | 2026-11-10 |
| Feature | `pit-rebuild-stocks` (issue #59), delegation `iw-59-verify-r1` |
| Spec | `.specs/features/pit-rebuild-stocks/spec.md` (amended r5, AD-032) |
| Diff range | `origin/develop` (`9f65467`)`..f4fb761`, 56 commits on `t3/issue-59` |
| Verifier | independent sub-agent (did not write the spec, the design, the tasks or the code) |
| Worktree state | `git status --porcelain` empty before and after verification. Mutants ran only in a scratch `git worktree` under the session scratchpad, which was removed afterwards. `.cache/pit-rebuild/` (path, size and mtime of each of its 586 files) and both Status Invest CSVs (sha256) are unchanged after every run. |

The implementation meets the spec on every criterion I could check, and the committed report reproduces. The verdict is FAIL because of two surviving mutants in the PIT filing selection, a path the brief treats as critical. In both cases the code is correct, but no test catches the fault, and the fault changes the real output (G1, G2).

---

## Task Completion

| Task | Status | Notes |
| ---- | ------ | ----- |
| T1–T19, T21–T32 | ✅ Done | Every box is checked in `tasks.md`. |
| T20 | ⏭️ Superseded | Left unchecked on purpose (`tasks.md:592-594`). T32 redid the real run on the r5 rules. |

---

## Gate Check

- **Full gate**: `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py" -v` gives `Ran 311 tests … OK` (exit 0).
- **Root suite**: `$PY -m unittest discover -s tests -p "test_*.py"` with `$PY=/home/caio/Documents/projetos/TCC/.venv/bin/python` gives `OK` (exit 0). The Handoff at `1f21839` recorded 898 OK and 2 skipped. The count before the feature was 799 OK and 2 skipped at `4ec8665` (`tasks.md:41`), and no test was deleted.
- **Protected trees**: `git diff --stat origin/develop -- data snapshots baselines py/core api py/config.py py/pit_validation.py` is empty. Every file the branch changes is under `py/pit_rebuild/`, `tests/test_pit_rebuild*`, `tests/pit_rebuild_support.py`, `tests/fixtures/pit_rebuild/`, `.specs/`, `docs/validation/pit-rebuild-stocks.*` or `docs/validation/README.md`.
- **Skipped tests**: none in the pit suite.

---

## Reproducibility (RPT-04, RPT-06)

I re-ran the documented command against the real cache with output in the scratchpad:

`report --offline --date 2026-08-07 --si data/raw/status_invest_fundamentals.csv --collected-on 2026-08-07 --date 2026-11-05 --si <main>/snapshots/data/selectors-2026-11-06/stocks.csv --collected-on 2026-11-06 --filings-cutoff 2026-11-05:2026-11-02`

- It exits 0 in 47.8 s.
- A recursive JSON diff against `docs/validation/pit-rebuild-stocks.json` finds 16 differing leaves: 15 under `coverage.effort` (measured seconds and the projections that come from them) and `tool_commit` (`611e839…` against `f4fb761…`). Both kinds are run-dependent by design (A-13, RPT-03). Every statistic, verdict, list, hash and parameter is identical.
- The Markdown diff shows only the same fields: line 4 (commit) and lines 2206–2215 (effort).
- The committed `.md` is the byte-for-byte render of the committed `.json`: `tests/test_pit_rebuild_artifact.py:57` – `assertEqual(MD_PATH.read_bytes(), report.render_markdown(self.report).encode("utf-8"))`. The test passes.
- `report` changed no cache file, and the two CSV hashes stayed `ceeed0b1…` and `87c15cb5…`.

## Verdict recomputed from the JSON (VER-01..VER-03, D-1, D-2)

I recomputed the verdicts from the committed JSON with my own code, not with `compare.py`:

- For each date, field and group I took `within/denominator`, checked it against `pct_within`, and applied ≥ 90 / ≥ 70 / < 70, with `sem referência` when the denominator is 0. That gives 0 mismatches against `dates.*.verdicts`.
- Taking the worse of the two dates, with `sem referência` ignored, gives 0 mismatches against `verdicts.fields`.
- The class is the worst of the 16 `METRIC_GROUPS` fields of `non_financial`: `não reconstruível`, equal to `verdicts.class`. The failing fields are ROIC, P/L, EV/EBIT, PEG RATIO and DY.
- The class without unclassified tickers is `não reconstruível`, equal to the JSON. The financial group is `não reconstruível`, equal to the JSON, and it is a separate field.
- For every entry in the ten-worst lists, `diff` recomputed as `|reb−ref|/|ref|·100` (relative) or `|reb−ref|` (p.p., the eight A-2 fields) matches. `diff_unit` follows A-2 for all 19 fields.
- The parameters hold tolerance 5 % and 0.5 p.p., thresholds 90/70, `liquidity_sessions` 21, `quote_codbdi` 02/07/08, the filings cutoff `2026-11-05 → 2026-11-02`, and 8 `non_pit_inputs`.

## Hand spot-checks from the raw cache

I recomputed these values with my own parsers (COTAHIST fixed-width, CVM CSV and B3 JSON), not the tool's:

| Value in the JSON | Hand computation | Match |
| --- | --- | --- |
| LIQUIDEZ MTSA4, 2026-08-07: 55,671.48 (`dates.2026-08-07.fields.LIQUIDEZ MEDIA DIARIA.non_financial.worst`) | mean `VOLTOT` over the last 21 sessions ≤ d, CODBDI 02/07/08, `02` wins its session = 55,671.48 (with 30 sessions it would be 64,095.93) | ✅ |
| LIQUIDEZ ALPA3 101,211.81 and NGRD3 1,186,237.48 (same list) | 101,211.81 and 1,186,237.48 | ✅ |
| DY TGMA3, 2026-08-07: 14.87 | B3 `TEGMA` history: 83 records, complete; type ON, `REST CAP DIN` dropped, duplicates dropped. Ex-date is the next session after `lastDatePriorEx`. The window (2025-08-07, 2026-08-07] holds 0.85 + 0.29 + 1.52 + 0.79 + 0.18 + 1.21 + 0.14 = 4.98, with no stock events. 100 × 4.98 / 33.48 = 14.87 | ✅ |
| P/L CAML3, 2026-08-07: 14.57 (fiscal year March–February) | Newest usable filing: ITR 2026-05-31 (received 2026-07-14). LL per row from `3.11.01`: FY to 2026-02-28 148,369 + YTD 27,909 − prior YTD 66,002 = 110,276 thousand. Issued shares 350,000,000 (ITR `composicao_capital`; B3 350.000.000, so f = 1). Price 4.59 on 2026-08-07. 4.59 / (110,276,000 / 350,000,000) = 14.568, which rounds to 14.57 | ✅ |

---

## Spec-Anchored Acceptance Criteria

### P1: Rebuild and compare on two real dates

| Req | Spec-defined outcome | `file:line` – assertion | Result |
| --- | --- | --- | --- |
| SRC-01 | `fetch` downloads every planned file, B3 company before mapping, and writes a manifest with URL, sha256, bytes and timestamp | `tests/test_pit_rebuild_cli.py:442` fetch requests every planned URL once and writes the manifest; `tests/test_pit_rebuild_sources.py:35`, `:45` year ranges; `tests/test_pit_rebuild_cli.py:543` B3 record for a ticker missing from FCA; `tests/test_pit_rebuild_cache.py:44` manifest entry | ✅ |
| SRC-02 | a cache hit makes no download | `tests/test_pit_rebuild_cache.py:59` zero requests; `tests/test_pit_rebuild_cli.py:464` | ✅ |
| SRC-03 | hash mismatch → exit 4 naming the file, no report | `tests/test_pit_rebuild_cache.py:77`; `tests/test_pit_rebuild_cli.py:320` `assertEqual(code, 4)`, `assertIn("dfp_cia_aberta_2025.zip", err)` | ✅ |
| SRC-04 | failed download or bad zip → exit 5 naming the URL, cached files kept | `tests/test_pit_rebuild_cache.py:111`, `:116`, `:119`; `tests/test_pit_rebuild_cli.py:519` | ✅ |
| SRC-05 | Yahoo with a reason ∈ {b3_failed, b3_empty, b3_incomplete, b3_short_history}; `b3_empty` + Yahoo failed → 0, source `b3`, reason `b3_empty_yahoo_failed` | `tests/test_pit_rebuild_dividends.py:109` each reason from its own input; `:144`, `:151`; `tests/test_pit_rebuild_cli.py:477`, `:487`, `:500`, `:511`, `:515` | ✅ |
| PIT-01 | filings `DT_RECEB ≤ d`, sessions ≤ d | `tests/test_pit_rebuild_cvm.py:101` receipt on d usable, d+1 not; `tests/test_pit_rebuild_cotahist.py:48`; `tests/test_pit_rebuild_cvm.py:378` | ✅, but see G1 and G2 for the helpers `_usable_period` and `annual` |
| PIT-02 | a value dated after d → exit 4 naming ticker and dates | `tests/test_pit_rebuild_cli.py:341` `assertEqual(code, 4)`, `assertIn("2026-12-01", err)`; `tests/test_pit_rebuild_indicators.py:224`, `:231`; `tests/test_pit_rebuild_dividends.py:138` | ✅ |
| PIT-03 | newest period received after d → previous filing, counted in `refiled_after_d` | `tests/test_pit_rebuild_cvm.py:119`; `tests/test_pit_rebuild_cli.py:209` | ✅ (N2) |
| MAP-01 | mapped and unmapped counts per date, one reason per unmapped ticker | `tests/test_pit_rebuild_mapping.py:65`, `:68`, `:71`; `tests/test_pit_rebuild_cli.py:209`; JSON 246 + 23 = 269 and 257 + 17 = 274 | ✅ |
| MAP-02 | financial, non-financial or unclassified, with the rule | `tests/test_pit_rebuild_mapping.py:139`, `:155`, `:164`, `:168`, `:172` | ✅ |
| MAP-03 | r5 unit grammar; price divided by k | `tests/test_pit_rebuild_mapping.py:104` real FCA strings; `:124`; `tests/test_pit_rebuild_cli.py:835` unit `1 ON e 2 PN`, k = 3 | ✅ |
| IND-01 | the 19 fields by A-8 with TTM flows and the latest balance sheet | `tests/test_pit_rebuild_indicators.py:78` each field equals its hand-computed value; `tests/test_pit_rebuild_cvm.py:301`, `:308`; hand checks above | ✅ |
| IND-02 | list read from `config.py` at run time | `tests/test_pit_rebuild_fields.py:39`, `:50` (an extra metric gives 17 fields), `:62` | ✅ |
| IND-03 | zero divisor or missing input → no value with a reason | `tests/test_pit_rebuild_indicators.py:195`, `:199`, `:217` | ✅ |
| IND-04 | VM by A-17 and A-18; `vm_price_fallback`, `vm_stale_class_price`, `scaled`, `unverified` | `tests/test_pit_rebuild_cli.py:173`, `:185`, `:853`, `:862`, `:869`; `tests/test_pit_rebuild_cotahist.py:124`, `:138` | ✅ (N1) |
| CMP-01 | 5 % / 0.5 p.p. inclusive; rounding; zero reference | `tests/test_pit_rebuild_compare.py:20`, `:26`, `:32`, `:37` | ✅ |
| CMP-02 | denominator, within, blank counts, median and p90 (nearest rank), 10 worst | `tests/test_pit_rebuild_compare.py:55`, `:60`, `:69`, `:84`, `:89`, `:97` | ✅ |
| VER-01 | ≥ 90 / ≥ 70 / < 70 / `sem referência` | `tests/test_pit_rebuild_compare.py:122` boundaries inclusive from below; `:128` | ✅ |
| VER-02 | worse date; `sem referência` ignored | `tests/test_pit_rebuild_compare.py:132` | ✅ |
| VER-03 | class is the worst of the 16 non-financial fields; without unclassified; financial apart | `tests/test_pit_rebuild_compare.py:138`, `:146`, `:158`; `tests/test_pit_rebuild_report.py:276` | ✅ |
| VER-04 | DY `secondary`, counts per source, caveat | `tests/test_pit_rebuild_report.py:242`; JSON `dividends.source = "secondary"`, `by_source`, `caveat` | ✅ |
| PIT-04 | `--date` is d; `collected_on`; `as_of_check` | `tests/test_pit_rebuild_cli.py:907`, `:1028`, `:1072`; `tests/test_pit_rebuild_artifact.py:61` | ✅ |
| SRC-06 | dedup, `REST CAP DIN` dropped, `complete` on the raw count | `tests/test_pit_rebuild_dividends.py:214`, `:223`, `:228`, `:233`; `tests/test_pit_rebuild_cli.py:882` | ✅ |
| SRC-07 | `/` removed and whitespace collapsed; `/` name with 0 records → `b3_failed` | `tests/test_pit_rebuild_sources.py:77`, `:83`; `tests/test_pit_rebuild_cli.py:890`, `:975` | ✅ |
| IND-05 | split, bonus and grouping adjustment by ISIN, events by d only | `tests/test_pit_rebuild_dividends.py:250`, `:256`, `:275`, `:283`, `:292`, `:301`; `tests/test_pit_rebuild_cli.py:876` | ✅ |
| IND-06 | F18 statement choice; LL per row; PL_ctrl; ROE with abs; ROIC with abs(IR); CAGR LL from TTM | `tests/test_pit_rebuild_cvm.py:179`, `:216`, `:220`, `:228`; `tests/test_pit_rebuild_indicators.py:111`, `:116`, `:130`; `tests/test_pit_rebuild_cli.py:840`, `:845`, `:849` | ✅ |
| VER-05 | `reference_stale` listed; non-official sensitivity without them | `tests/test_pit_rebuild_report.py:457`; `tests/test_pit_rebuild_cli.py:951` | ✅ |
| VER-06 | filings cutoff sensitivity, non-official, combined with the official verdict of the other date | `tests/test_pit_rebuild_report.py:434`; `tests/test_pit_rebuild_cli.py:1032` | ✅ |
| VER-07 | class without ROIC and PEG, with its note | `tests/test_pit_rebuild_report.py:417`; `tests/test_pit_rebuild_cli.py:1051` | ✅ |
| VER-08 | official verdicts never read a sensitivity; `official: false`; Markdown verdict line official only | `tests/test_pit_rebuild_report.py:401`, `:494`, `:477`; `tests/test_pit_rebuild_artifact.py:80` | ✅ |

### P1: Committed, reproducible report

| Req | Spec-defined outcome | `file:line` – assertion | Result |
| --- | --- | --- | --- |
| RPT-01 | both files written only after every computation | `tests/test_pit_rebuild_safety.py:73`, `:80`, `:100`; `tests/test_pit_rebuild_cli.py:144` | ✅ |
| RPT-02 | URL, sha256 and bytes per raw file; SI path, sha256, date, `collected_on`, `as_of_check` | `tests/test_pit_rebuild_cli.py:248`; `tests/test_pit_rebuild_artifact.py:90`, `:98`; JSON has 571 raw files | ✅ |
| RPT-03 | parameters (thresholds, tolerances, 21 sessions, codes, formulas, √10 rule, 8 non-PIT inputs, cutoffs, commit) | `tests/test_pit_rebuild_report.py:181`, `:194`, `:215`, `:223` | ✅ |
| RPT-04 | Markdown rendered from the JSON | `tests/test_pit_rebuild_artifact.py:57`; `tests/test_pit_rebuild_report.py:305`, `:308` | ✅ |
| RPT-05 | a failure leaves earlier report files byte-identical | `tests/test_pit_rebuild_cli.py:361` | ✅ |
| RPT-06 | committed JSON is the real run with both dates and the cutoff | `tests/test_pit_rebuild_artifact.py:61`, `:80`; reproduced above | ✅ |

### P1: Safe research CLI

| Req | Spec-defined outcome | `file:line` – assertion | Result |
| --- | --- | --- | --- |
| CLI-01 | exit-code table 0–5 in the docstring, and the README agrees | `py/pit_rebuild/cli.py:17-24`; `docs/validation/README.md` "Códigos de saída"; `tests/test_pit_rebuild_artifact.py:106`, `tests/test_pit_rebuild_errors.py:45`. I compared the wording by hand: each code has the same meaning in both, and the duplicate-ticker case of code 4 is in both. | ✅ (N3) |
| SAFE-01 | out or cache dir inside data, snapshots or baselines → exit 2 | `tests/test_pit_rebuild_safety.py:24`, `:32`, `:38`; `tests/test_pit_rebuild_cli.py:286`, `:559` | ✅ |
| SAFE-02 | SI CSVs read-only and byte-identical | `tests/test_pit_rebuild_status_invest.py:125`; `tests/test_pit_rebuild_cli.py:379` (also hashes the `data/`, `snapshots/` and `baselines/` trees) | ✅ |
| CLI-02 | `--offline` makes no request | `tests/test_pit_rebuild_cli.py:168` (the `no_network` opener raises on any request); `tests/test_pit_rebuild_cache.py:132` | ✅ |
| CLI-03 | invalid date, collected-on or cutoff argument → exit 2 with one line | `tests/test_pit_rebuild_cli.py:286`, `:1006`, `:1016` | ✅ |
| CLI-04 | missing `--si` or cache → exit 3 naming the path | `tests/test_pit_rebuild_cli.py:310`, `:559` | ✅ |
| CLI-05 | each exit code through `main([...])` on a fixture cache, no network | 0: `tests/test_pit_rebuild_cli.py:144`; 1: `:277`; 2: `:286`; 3: `:310`; 4: `:320`, `:330`, `:341`; 5: `:351`, `:519`. All go through `cli.main(args, opener=no_network)` (`tests/test_pit_rebuild_cli.py:82-86`) | ✅ |

### P2: Historical coverage and effort

| Req | Spec-defined outcome | `file:line` – assertion | Result |
| --- | --- | --- | --- |
| COV-01 | earliest year per source; COTAHIST `≤ 2005 (probe floor)` | `tests/test_pit_rebuild_coverage.py:35`, `:44`, `:47`; `tests/test_pit_rebuild_report.py:292` | ✅ |
| COV-02 | earliest year per field plus lookback; DY from raw dates with the per-ticker median | `tests/test_pit_rebuild_coverage.py:50`, `:53`, `:65`, `:77`, `:89`; `tests/test_pit_rebuild_cli.py:1108` | ✅ |
| COV-03 | load projected by bytes, compute × 180, clipped ranges, Content-Length | `tests/test_pit_rebuild_coverage.py:102`, `:112`, `:131`; `tests/test_pit_rebuild_cli.py:263` | ✅ |

**Status**: every AC has evidence and the asserted values match the spec outcomes. There are no spec-precision gaps. The two blocking gaps below are discrimination gaps inside PIT-01's helpers, not missing ACs.

### Issue acceptance criteria (#59)

| Issue AC | Evidence |
| --- | --- |
| Downloads or reads from cache DFP, ITR, cadastro and COTAHIST, and rebuilds 16 + 3 fields per ticker | SRC-01, IND-01, IND-02 rows above; JSON `dates.*.fields` has 19 keys |
| Point-in-time by delivery date and sessions | PIT-01..PIT-03 (G1 and G2 apply) |
| Compared with the two real CSVs; denominator is the eligible universe | `tests/test_pit_rebuild_status_invest.py:80` (269 eligible); JSON eligible 269 and 274; A-11 as-of 2026-11-05 approved in AD-032 |
| Ticker → CVM mapping with coverage and reasons | MAP-01 |
| % within, median, p90 and 10 worst per indicator and date | CMP-02 |
| Verdict per indicator and class, banks and insurers apart | VER-01..VER-03, recomputed above |
| Earliest year per source and full-rebuild effort | COV-01..COV-03 |
| `docs/validation/pit-rebuild-stocks.json` and `.md` with URL and sha256 | RPT-02, RPT-06 |
| Exit-code table in the docstring, each code tested through the real path, no network | CLI-01, CLI-05 |
| No writes to `data/`, `snapshots/`, `baselines/` or the registry | SAFE-01, SAFE-02, protected-tree diff empty |

---

## Edge Cases

- [x] Individual statement when there is no consolidated one: `tests/test_pit_rebuild_cvm.py:130`
- [x] No session within 30 days → `no_quote`, listed in `reference_stale`, kept in the official denominator: `tests/test_pit_rebuild_cli.py:951`
- [x] Duplicated ticker → exit 4: `tests/test_pit_rebuild_status_invest.py:102`
- [x] Missing column → exit 4: `tests/test_pit_rebuild_cli.py:330`
- [x] Two classes, most liquid preferred ticker, non-eligible ON quote: `tests/test_pit_rebuild_indicators.py:155`; `tests/test_pit_rebuild_cli.py:173`
- [x] Scale ×1000, ×0.1 and a residual: `tests/test_pit_rebuild_cvm.py:249`; `tests/test_pit_rebuild_cli.py:853`. The rounding part is weak (N1).
- [x] No share count → `no_share_count`: `tests/test_pit_rebuild_cvm.py:349`, `:353`; `tests/test_pit_rebuild_indicators.py:176`
- [x] B3 record before the first session never counts: `tests/test_pit_rebuild_dividends.py:101`
- [x] Both dividend sources fail → `no_dividend_source`, except `b3_empty`: `tests/test_pit_rebuild_dividends.py:144`, `:151`
- [x] Blank DY with rebuilt 3.00 is out, with 0.50 is in: `tests/test_pit_rebuild_compare.py:69`
- [x] ITR YTD row against the 3-month row, including a June fiscal year: `tests/test_pit_rebuild_cvm.py:109`, `:114`
- [x] `short_history` for CAGR and PEG: `tests/test_pit_rebuild_indicators.py:208`, `:213`
- [x] Negative LL or EBIT keeps its sign: `tests/test_pit_rebuild_indicators.py:192`
- [x] CODBDI 08 alone, and `02` wins its session: `tests/test_pit_rebuild_cotahist.py:52`; `tests/test_pit_rebuild_cli.py:827`
- [x] `1 ON e 2PNs` parses, `duas PN` does not: `tests/test_pit_rebuild_mapping.py:104`, `:124`
- [x] Zero `3.11.01` row with a non-zero one: `tests/test_pit_rebuild_cvm.py:216`, `:220`
- [x] Negative PL_ctrl gives a positive ROE: `tests/test_pit_rebuild_indicators.py:111`
- [x] Zero consolidated DRE → individual: `tests/test_pit_rebuild_cvm.py:179`
- [x] Stale ON ticker → `vm_stale_class_price`, not fallback: `tests/test_pit_rebuild_cli.py:862`; `tests/test_pit_rebuild_cotahist.py:146`
- [x] Identical B3 records count once, `REST CAP DIN` dropped: `tests/test_pit_rebuild_dividends.py:214`, `:223`
- [x] 1:5 split before or after d: `tests/test_pit_rebuild_dividends.py:250`, `:256`
- [x] `/` name with an empty history → `b3_failed`, then `no_dividend_source`: `tests/test_pit_rebuild_cli.py:890`
- [x] `--filings-cutoff` leaves the official verdicts identical: `tests/test_pit_rebuild_cli.py:1032`

---

## Discrimination Sensor

**Sensor depth**: P0-full (the verdict aggregation and PIT selection are critical per the brief). I applied 63 manual behavior-level mutations, one at a time, in a scratch `git worktree` of `f4fb761`, and ran the full pit suite after each one. Script: `mutants.py` in the session scratchpad.

| # | File | Mutation | Killed? |
| --- | --- | --- | --- |
| M01 | `py/pit_rebuild/cvm.py:285` | `_newest_usable` drops the `DT_RECEB ≤ d` filter | ✅ (42 failures) |
| M02 | `py/pit_rebuild/cvm.py:285` | `≤ d` → `< d` | ✅ |
| **M03** | `py/pit_rebuild/cvm.py:291` | `_usable_period` drops its `DT_RECEB ≤ d` filter | ❌ **Survived** (G1) |
| M04 | `py/pit_rebuild/cvm.py:338` | DT_RECEB of the earliest version, not the latest | ✅ |
| **M05** | `py/pit_rebuild/cvm.py:208` | `annual()` base DFP without the `DT_RECEB ≤ d` filter | ❌ **Survived** (G2) |
| M06 | `py/pit_rebuild/cvm.py:172` | `refiled_after_d` always False | ✅ |
| M07 | `py/pit_rebuild/indicators.py:133` | PIT-02 leak guard removed | ✅ |
| M08, M09 | `py/pit_rebuild/cvm.py:151` | YTD row: latest `DT_INI` kept / last row in file wins | ✅ ✅ |
| M10, M11 | `py/pit_rebuild/cotahist.py:25` | CODBDI `02` only / `08` dropped | ✅ ✅ |
| M12 | `py/pit_rebuild/cotahist.py:74` | `02` precedence on a session removed | ✅ |
| M13, M14 | `py/pit_rebuild/cotahist.py:24` | liquidity window 30 / 20 | ✅ ✅ |
| M15 | `py/pit_rebuild/cotahist.py:23` | PRECO max age 31 | ✅ |
| M16, M17 | `py/pit_rebuild/cvm.py:397` | LL = `3.11.01` even when zero / always `3.11` | ✅ ✅ |
| M18 | `py/pit_rebuild/cli.py:388` | PL_ctrl = PL | ✅ |
| M19 | `py/pit_rebuild/indicators.py:114` | ROE without abs | ✅ |
| M20 | `py/pit_rebuild/indicators.py:116` | ROIC with `+ IR` | ✅ |
| M21 | `py/pit_rebuild/cvm.py:384` | F18: consolidated even when zero | ✅ |
| M22, M23 | `py/pit_rebuild/cvm.py:39` | scale factors {1, 1000} / without 0.1 | ✅ ✅ |
| **M24** | `py/pit_rebuild/cli.py:443` | scaled counts not rounded | ❌ Survived (N1, nonblocking) |
| M25 | `py/pit_rebuild/dividends.py:134` | B3 dedup removed | ✅ |
| M26 | `py/pit_rebuild/dividends.py:138` | `REST CAP DIN` kept | ✅ |
| M27 | `py/pit_rebuild/dividends.py:150` | `complete` on the filtered count | ✅ |
| M28 | `py/pit_rebuild/dividends.py:279` | split adjustment removed | ✅ |
| M29 | `py/pit_rebuild/dividends.py:278` | split events after d applied | ✅ |
| M30 | `py/pit_rebuild/dividends.py:96` | GRUPAMENTO multiplier as 1 + f/100 | ✅ |
| M31 | `py/pit_rebuild/dividends.py:89` | stock events not deduped | ✅ |
| M32 | `py/pit_rebuild/sources.py:50` | `/` not removed from the query name | ✅ |
| M33 | `py/pit_rebuild/cli.py:349` | `/` name with an empty history → `b3_empty` | ✅ |
| M34 | `py/pit_rebuild/dividends.py:293` | DY window lower bound inclusive | ✅ |
| M35 | `py/pit_rebuild/cotahist.py:133` | ex-date = same day (`bisect_left`) | ✅ |
| M36–M39 | `py/pit_rebuild/compare.py:20-21`, `:68`, `:65` | tolerance 5.5 % / `<` at 5 % / 0.6 p.p. / `<` at 0.5 p.p. | ✅ ✅ ✅ ✅ |
| M40, M41 | `py/pit_rebuild/compare.py:63`, `:67` | no rounding / zero reference always within | ✅ ✅ |
| M42 | `py/pit_rebuild/compare.py:54` | blank DY not read as 0 | ✅ |
| M43 | `py/pit_rebuild/compare.py:88` | p90 floor rank | ✅ |
| M44–M46 | `py/pit_rebuild/compare.py:129-131` | `> 90` / `> 70` / threshold 80 | ✅ ✅ ✅ |
| M47 | `py/pit_rebuild/compare.py:141` | worst → best | ✅ |
| M48 | `py/pit_rebuild/compare.py:138` | `sem referência` not ignored | ✅ |
| M49–M52 | `py/pit_rebuild/compare.py:156` | class from the financial group / including it / over 19 fields / without unclassified | ✅ ✅ ✅ ✅ |
| M53 | `py/pit_rebuild/report.py:232` | overall = last date only | ✅ |
| M54 | `py/pit_rebuild/cli.py:759` | official evaluation uses the filings cutoff | ✅ |
| M55 | `py/pit_rebuild/report.py:157` | official class = D-C (without ROIC and PEG) | ✅ |
| M56 | `py/pit_rebuild/cli.py:761` | official stats exclude `reference_stale` | ✅ |
| M57 | `py/pit_rebuild/report.py:269` | sensitivities `official: true` | ✅ |
| M58 | `py/pit_rebuild/report.py:322` | D-B recomputes every date | ✅ |
| M59 | `py/pit_rebuild/cli.py:732` | D-A keeps the stale tickers | ✅ |
| M60, M61 | `py/pit_rebuild/mapping.py:40`, `:44` | unit grammar without `e`/`E` / without `PNs` | ✅ ✅ |
| M62 | `py/pit_rebuild/indicators.py:253` | CAGR exponent 1/4 | ✅ |
| M63 | `py/pit_rebuild/cvm.py:316` | TTM without the prior YTD | ✅ |

**Sensor result**: 60 of 63 mutants killed (FAIL). Every mutant in the verdict aggregation and the sensitivity isolation (M44–M59) is killed. Two mutants in the PIT filing selection survive (M03, M05), and so does one in share-count rounding (M24).

**The survivors change real output.** I ran the real `report --offline` (the same arguments as above, output in the scratchpad) with each PIT mutant:

- **M03**: exit 4, `pit_rebuild: EQPA3: decision 2026-08-07 received a value dated 2026-10-02`. Without the `_usable_period` filter, an older filing whose latest version was received after d reaches CAGR, PEG or TTM inputs. The PIT-02 guard stops the run, but no unit test pins the filter.
- **M05**: exit 0, and the JSON changes. For 2026-08-07, `CAGR RECEITAS 5 ANOS` `non_financial` `missing` grows from 21 to 23, and reasons and tickers shift. When the newest DFP was received after d, `annual()` anchors the five years on it, so the field goes missing or uses the wrong fiscal years.

---

## Code Quality

| Principle | Status |
| --- | --- |
| Minimum code / no scope creep | ✅ Own package `py/pit_rebuild/`, stdlib only (A-14); `py/pit_validation.py` untouched |
| Surgical changes | ✅ Only feature paths changed (see Gate Check) |
| Matches patterns | ✅ Follows `py/pit_validation.py`: exit-code table, `.json` + `.md`, sha256 provenance |
| Spec-anchored outcome check | ✅ Asserted values are hand-computed spec outcomes (for example `tests/test_pit_rebuild_cli.py:144`: 15 / 11 / `com ressalvas`) |
| Per-layer coverage | ⚠️ Pure rules and the CLI are covered 1:1. The parser layer misses two PIT boundaries in `cvm.py` helpers (G1, G2). |
| Every test maps to a requirement | ✅ Test names cite their AC or edge; no unclaimed tests found in the files read |
| Guidelines followed | ✅ `REVIEW.md` "Sempre verificar": exit codes in the table and tested through `main`; docs numbers match the artifact; no writes to protected trees. One item is partly missed: "Um teste novo falha quando o comportamento que ele nomeia é removido" (G1, G2, N1). |

## Docs against the artifact

- `docs/validation/pit-rebuild-stocks.md` is a pure render of the JSON (artifact test above). Its verdict line holds only official verdicts (`:9-11`). The sensitivities sit in `## Sensibilidades (não são o veredito oficial)` (`:39-49`), after the verdict, with the text "Estas linhas não são o veredito oficial".
- The README section (`docs/validation/README.md:360-622`) quotes no result numbers. Its rules (5 %, 0.5 p.p., 90/70, 21 sessions, CODBDI 02/07/08, √10 and 10⁻²…10³) match `parameters` in the JSON. Its claims about the as-of warning, `dividends.slash_names` and the B3 filter counts all match fields that exist in the JSON.
- STATE Handoff numbers checked against the JSON: P/L 187/236 = 79.2 % on 2026-08-07 non-financial; D-A excludes 23 and 17 tickers. Both match.

---

## Gaps

| ID | Severity | Gap | Evidence | Suggested fix |
| --- | --- | --- | --- | --- |
| G1 | **blocking** | `cvm._usable_period` (`py/pit_rebuild/cvm.py:289-291`) filters by `DT_RECEB ≤ d`, and no test fails when that filter is removed (M03). The filter keeps a refiled older DFP out of `annual(years_back=5)`, `ttm_year_before` and TTM's `fy_prev`. | M03 survives the suite; the real run with M03 exits 4 on EQPA3 (value dated 2026-10-02 at d = 2026-08-07) | Add `tests/test_pit_rebuild_cvm.py` cases where the referenced older DFP's latest version has `DT_RECEB = d + 1`. `annual(..., years_back=5)` should be `None` (and the indicator `short_history`); `ttm` after an ITR whose `fy_prev` arrives at d + 1 should be `None`; with `DT_RECEB = d` each one is used. |
| G2 | **blocking** | `cvm.annual` (`py/pit_rebuild/cvm.py:208`) picks its base DFP among filings received by d, and no test fails without that filter (M05). | M05 survives; the real run with M05 changes CAGR RECEITAS 5 ANOS on 2026-08-07 (missing 21 → 23) | Add a test with DFP N received after d and DFP N−1 before: `annual(years_back=0)` is DFP N−1's value and `annual(years_back=5)` reads DFP N−6. |
| N1 | nonblocking | Scaled share counts are rounded (A-18, `py/pit_rebuild/cli.py:443`), but the only scale fixture uses counts that × 0.1 gives an integer, so removing `round` survives (M24). Outside the critical paths; the code is correct. | `tests/test_pit_rebuild_cli.py:853` (50 M × 0.1) | Give one fixture a count that is not a multiple of 10 at f = 0.1 (or 100 at f = 0.01), and assert the rounded count in the VM value. |
| N2 | nonblocking | `refiled_after_d` holds 217 of 246 mapped tickers on 2026-08-07, because most Q2-2026 ITRs arrived after that date. That follows PIT-03's text and the Markdown wording is accurate ("o mais recente chegou depois de d"), but the key name reads like refiling. | `docs/validation/pit-rebuild-stocks.json` `dates.2026-08-07.mapping.refiled_after_d` | Optional: say in the report or the README that this counts on-schedule filings received after d, not only refilings. |
| N3 | nonblocking | The first usage example in the `cli.py` docstring still passes the 2026-11-06 snapshot as `--date 2026-11-06`, which contradicts A-11's as-of 2026-11-05. The second example and the README are right. | `py/pit_rebuild/cli.py:5-6` | Drop the first example or change its date to `2026-11-05`. |

## Fix Plans

### Fix 1 (G1 + G2): pin the PIT filter of the `cvm.py` helpers

- **Root cause**: PIT-01's receipt-date boundary is tested only through `_newest_usable` (`tests/test_pit_rebuild_cvm.py:101`). The two other entry points that read filings by period have no boundary test.
- **Fix task**: test-only. In `tests/test_pit_rebuild_cvm.py`, build a store where the period referenced by `annual` (years back), `ttm_year_before`, or TTM's `fy_prev` has its latest version received at d + 1, then at d. Assert `None` and then the value. Add one case where DFP N arrives after d.
- **Done when**: the suite still passes, and re-injecting M03 and M05 each makes at least one test fail.
- **Priority**: Blocker, per the brief's mutation rule.

## Requirement Traceability Update

The brief forbids editing `spec.md`, so the table there stays as it is. Proposed update, to apply after Fix 1 and a re-verify: every requirement → ✅ Verified. Until then, PIT-01 → ❌ Needs Fix (discrimination); the rest → ✅ Verified.

## Lessons

The brief limits writes to this file, so `lessons.py` was not run. Proposed lesson for the orchestrator to record: "When a PIT rule (receipt date ≤ d) is applied in several helpers, test the boundary through each helper that selects a filing, not only the main selector. One test on the main selector leaves the other filters mutable."

## Summary

**Overall**: ❌ Not ready. The fix is two test-only gaps; there is no code defect.

- **Spec-anchored check**: 46/46 requirements and 10/10 issue ACs have evidence, with no spec-precision gaps.
- **Sensor**: 60/63 mutants killed. M03 and M05 (PIT selection) are blocking; M24 is nonblocking.
- **Gate**: pit suite 311 OK, root suite OK, protected-tree diff empty.
- **Reproducibility**: the re-run JSON equals the committed JSON except the run-dependent effort timings and `tool_commit`. The committed Markdown equals the render of the committed JSON.

**What works**: the verdict aggregation (recomputed by hand from the JSON, 0 mismatches), the sensitivity isolation (every mutant killed), the exit-code table and its tests through `main`, the safety guards, the r5 rules, and the hand spot-checks of LIQUIDEZ, DY and P/L.

**Next steps**: an implementer adds the Fix 1 tests (test-only, one commit referencing #59), then the Verifier re-runs M03, M05 and the full suite.
