# Validation: pit-rebuild-stocks-fixes (issue #64) - FAIL

**Date**: 2026-11-12
**Spec**: `.specs/features/pit-rebuild-stocks-fixes/spec.md` (r2, 44 requirements)
**Diff range**: `ebaf1fd..59529dd` (merge base with `origin/develop` .. HEAD of `t3/issue-64`, 41 commits)
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-64-verify-r1`

**Verdict: FAIL.** The code passes the gates, the sensor kills 23 of 24 non-equivalent mutants, the committed report regenerates from the committed code, and every number sampled in `docs/validation/` matches the JSON. Two blocking gaps remain: issue AC9 is not met as written (V1), and two audit rows attribute a Status Invest value to a filing received three months after the collection when the cache shows an earlier one (V2).

---

## Task Completion

| Task | Status | Notes |
| ---- | ------ | ----- |
| T1-T24 | ✅ Done | All `Done when` boxes in `tasks.md` are checked (0 unchecked). One commit per task, `1764831` .. `996490d`; handoff `59529dd`. |

---

## Gate Check

- **Full (pit) gate**: `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"` → `Ran 471 tests`, `OK`.
- **Build gate**: `python3 -m compileall -q py/pit_rebuild` ok; `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` empty; `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → `Ran 1058 tests in 189.772s`, `OK (skipped=2)`.
- **Test count before the feature** (pit suite at `ebaf1fd`, run on a `git archive` copy): 322. **After**: 471. **Delta**: +149. No test deleted.
- **Skipped**: 2 in the root suite, both outside this feature (unchanged from the baseline the handoff records).
- **Regeneration (RPT-01)**: `python3 py/pit_rebuild report --offline` with the README arguments, written to a scratch directory, gives a JSON that differs from the committed one only in `tool_commit` and the 14 timing keys of `coverage.effort`. `py/` is unchanged between the generating commit `550c89f` and HEAD.

---

## Issue Acceptance Criteria (AC1..AC10)

| Issue AC | Evidence | Status |
| -------- | -------- | ------ |
| AC1 count × B3 events in (period end, d], adjusted companies listed with factor | `tests/test_pit_rebuild_shares.py:58` (bonus ×2 → `n_on` 2 000 000, listed with factor 2.0), `:71` (ex-date after d not applied), `:75` (ex-date = DT_REFER not applied), `:101`-`:128` (SHR-08 scale window); `tests/test_pit_rebuild_cli.py:1060` (BONU3 through `main`, `share_events.adjusted` row and VALOR DE MERCADO 2 000 000 000); JSON `dates[d].mapping.share_events.adjusted`: 6 and 3 classes. Mutants M1-M6 killed. | ✅ |
| AC2 change > 20 % with no B3 event flagged, value not corrected | `tests/test_pit_rebuild_shares.py:207`, `:214` (exactly 20 % not listed), `:219` (20.01 % listed), `:223` (event explains), `:238` (count used as filed); `tests/test_pit_rebuild_cli.py:1090` (JUMP: ratio 1.5 listed, VALOR DE MERCADO uses 150 M). JSON `share_count_suspect`: 5 and 8. Mutants M7, M8 killed. | ✅ |
| AC3 12-month flows from last year's ITR when PENÚLTIMO is 0/absent | `tests/test_pit_rebuild_cvm.py:548` (`Dated(1100.0, 2026-08-06, prior_from_itr=True)` = 1000 + 300 − 200), `:555` (absent), `:562` (year-earlier ITR received d + 1 → None), `:568`, `:574` (non-zero comparative kept), `:589` (`ttm_year_before`); `tests/test_pit_rebuild_cli.py:1123` (TTMX through `main`, listed in `ttm_prior_from_itr`). Mutants M9, M10, M22 killed. TIMS3: JSON lists cd 24929 in `ttm_prior_from_itr`; the audit records the R$ 4,33 bi LL. | ✅ |
| AC4 stale filing > 270 days → no accounting value, `stale_filing` | `tests/test_pit_rebuild_cli.py:1103` (OLDF3, age 280, 15 fields `stale_filing`, market fields kept, listed), `:1579` (270 not stale, 271 stale), `:1569` (cutoff evaluation). JSON: AMBP3 age 403 at 2026-08-07; AALR3, AMBP3, BSLI3 at 2026-11-05. Mutants M11, M12, M18 killed. | ✅ |
| AC5 ISIN and class from the last session ≤ d (N3) | `tests/test_pit_rebuild_cotahist.py:238` (old ISIN/ESPECI until the change session), `tests/test_pit_rebuild_cli.py:1616` (dividend divided by the old ISIN's event at the first date). Mutants M13, M16 killed. | ✅ |
| AC6 MBRF3 DY cause investigated and written | JSON `investigations[0]` (MBRF3, DY, 2026-11-05); `tests/test_pit_rebuild_artifact.py:180`, `tests/test_pit_rebuild_report.py:752`, `:762`. Verified against the cache: `b3_cash_MARFRIG_p1.json` holds 2,8105011104 with ex-date 2025-09-19 (412 days before 2026-11-05); COTAHIST MBRF3 closes 16,07 on 2026-08-07 and 15,63 on 2026-11-05; 100 × 2,8105 / 16,07 = 17,49 = SI's DY in `data/raw/status_invest_fundamentals.csv`. | ✅ (MBR-02 branch) |
| AC7 adjusted verdict beside the official one | `tests/test_pit_rebuild_report.py:536`, `:545`, `:576`, `:583`, `:589`, `:616`, `:633`, `:732`; `tests/test_pit_rebuild_cli.py:1833` and the ADJ-07 test after `:1859`; `tests/test_pit_rebuild_artifact.py:131`, `:152`. JSON: `adjusted_verdict.class` `com ressalvas`, `criterion.met` true, `failing` []; official `verdicts.class` `não reconstruível`. Mutants M15, M19, M20, M24, M25 killed. | ✅ |
| AC8 `si_lag` counted apart | `tests/test_pit_rebuild_compare.py:185`, `:193`, `:197`; `tests/test_pit_rebuild_cli.py:1803`, `:1808`, `:1816`, `:1827`. JSON sums: 32 (2026-08-07), 350 (2026-11-05). Mutant M14 killed; M17 survived (V3). | ✅ (V3 nonblocking) |
| AC9 the 10 worst per adjusted indicator and date classified as erro nosso / erro do SI / diferença de método, with the filing number | `docs/validation/pit-rebuild-stocks-audit.md` has all 280 rows, numbers equal to the JSON (`tests/test_pit_rebuild_artifact.py:246`, `:254`, independently re-checked: 0 numeric mismatches). **153 rows carry a fourth class, `não classificado`**: 121 outside tolerance with no cause, 32 inside tolerance. The test admits the fourth class (`tests/test_pit_rebuild_artifact.py:187`). | ❌ V1 |
| AC10 JSON and MD regenerated and committed; small fixtures, no network | Regeneration above (only `tool_commit` and timings differ); `tests/test_pit_rebuild_artifact.py:57` (MD = render of JSON byte for byte), `:125` (schema v3, `amended_by` [64], `previous_report`); CLI tests run on temp-dir fixtures with `opener=no_network` (`tests/test_pit_rebuild_cli.py:149`); no test reads `.cache/`. | ✅ |

---

## Spec Requirements (44)

| Req | Spec-defined outcome | Evidence (`file:line` + assertion) | Result |
| --- | -------------------- | ---------------------------------- | ------ |
| SHR-01 | N = round(filed × f × M_d) | `tests/test_pit_rebuild_shares.py:58` `(n_on, factor) == (2_000_000, 1)`; `tests/test_pit_rebuild_cli.py:1060` VALOR DE MERCADO 2e9; M6 killed | ✅ |
| SHR-02 | ex-date = DT_REFER or > d left out | `tests/test_pit_rebuild_shares.py:71`, `:75`; M1, M2 killed | ✅ |
| SHR-03 | listed when abs(M − 1) > 1e-9, row shape, cancel-out not listed | `tests/test_pit_rebuild_shares.py:58` (`applied` row), `:86` (split + grouping unlisted); `tests/test_pit_rebuild_cli.py:1060` (JSON row); M4 killed | ✅ |
| SHR-04 | class with shares and empty I_d: no event, listed in `no_isin` | `tests/test_pit_rebuild_shares.py:175`; `tests/test_pit_rebuild_cli.py:1098`; M23 killed | ✅ |
| SHR-05 | > 20 % with no event → `share_count_suspect` row | `tests/test_pit_rebuild_shares.py:207`, `:219`; `tests/test_pit_rebuild_cli.py:1090` (ratio 1.5 row) | ✅ |
| SHR-06 | suspect count used unchanged | `tests/test_pit_rebuild_shares.py:238`; `tests/test_pit_rebuild_cli.py:1090` (value uses 150 M) | ✅ |
| SHR-07 | exactly 20 % or an event → not listed | `tests/test_pit_rebuild_shares.py:214`, `:223`; `tests/test_pit_rebuild_cli.py:1498`; M7, M8 killed | ✅ |
| SHR-08 | f chosen on M_all up to R; event after R out; ISIN after d in | `tests/test_pit_rebuild_shares.py:101`, `:107`, `:113`, `:118`, `:128`; `tests/test_pit_rebuild_cli.py` LATE3 test after `:1060` (factor 1, ×5 applied); M5 killed | ✅ |
| TTM-01 | FY_prev + YTD − YTD_prior(ITR), dated by latest DT_RECEB | `tests/test_pit_rebuild_cvm.py:548` `Dated(1100.0, date(2026, 8, 6), prior_from_itr=True)`, `:555`; M9 killed | ✅ |
| TTM-02 | earlier ITR unusable/absent/no row → None | `tests/test_pit_rebuild_cvm.py:562` (`assertIsNone` at d + 1), `:568`; M10 killed | ✅ |
| TTM-03 | non-zero PENÚLTIMO used | `tests/test_pit_rebuild_cvm.py:574` (1150, `prior_from_itr` false); `tests/test_pit_rebuild_cli.py` TTMZ test after `:1123` | ✅ |
| TTM-04 | `ttm_prior_from_itr` rows sorted by cd_cvm | `tests/test_pit_rebuild_cli.py:1123` (`[{"cd_cvm": "2019", "dt_refer": "2026-06-30", "accounts": ["3.01", "LL"]}]`); M22 killed | ✅ |
| STL-01 | age > 270 → 15 fields `stale_filing` | `tests/test_pit_rebuild_cli.py:1103` (loop over ACCOUNTING_FIELDS); M12 killed | ✅ |
| STL-02 | stays mapped, market fields computed | `tests/test_pit_rebuild_cli.py:1103` (`assertNotIn("OLDF3", unmapped)`, VALOR DE MERCADO 1e9, PRECO/LIQUIDEZ/DY not `stale_filing`) | ✅ |
| STL-03 | `stale_filing` rows sorted by ticker | `tests/test_pit_rebuild_cli.py:1103` (ANTE3, OLDF3 rows with age 280) | ✅ |
| STL-04 | exactly 270 not stale | `tests/test_pit_rebuild_cli.py:1579`; M11 killed | ✅ |
| ISN-01 | value of the session ≤ d | `tests/test_pit_rebuild_cotahist.py:238`; M13 killed | ✅ |
| ISN-02 | no session ≤ d → None | `tests/test_pit_rebuild_cotahist.py:238` (task T1 box) | ✅ |
| ISN-03 | d passed at every call; old ISIN at first date through `main` | `tests/test_pit_rebuild_cli.py:1616`; M16 killed | ✅ |
| MBR-01 | fix the lookup if it is the cause | not taken: the cause is not the lookup (MBR-02 branch, verified on the cache) | ✅ n/a |
| MBR-02 | cause in `investigations`, Markdown and audit | `tests/test_pit_rebuild_artifact.py:180`; `tests/test_pit_rebuild_report.py:762`; audit row MBRF3 (`docs/validation/pit-rebuild-stocks-audit.md:534`) | ✅ |
| ADJ-01 | official section kept, before adjusted, with the B-12 line | `tests/test_pit_rebuild_report.py:633`, `:732` | ✅ |
| ADJ-02 | block with 14 fields, `excluded`, per-date stats | `tests/test_pit_rebuild_report.py:536`; `tests/test_pit_rebuild_cli.py:1833` (STLQ3 excluded, denominator 3) | ✅ |
| ADJ-03 | per-field worse of two dates; class = worst field | `tests/test_pit_rebuild_report.py:545`; M20, M24 killed | ✅ |
| ADJ-04 | worst rows with `si_lag`, filing, `previous_value`; null on DY | `tests/test_pit_rebuild_report.py:553`; `tests/test_pit_rebuild_cli.py` worst-row test after `:1833` (`assertIsNone(dy["filing"])`); M25 killed (M21 equivalent) | ✅ |
| ADJ-05 | criterion object; met iff every field ≥ 70 on both dates; MD prints part 1 only | `tests/test_pit_rebuild_report.py:576` (70 → met), `:583` (69.99 → fails); `tests/test_pit_rebuild_artifact.py:131`, `:152`; M15 killed | ✅ |
| ADJ-06 | denominator 0 → failing with null | `tests/test_pit_rebuild_report.py:589`; `tests/test_pit_rebuild_cli.py` criterion test (`{"field": "PSR", ..., "pct_within": None}`); M19 killed | ✅ |
| ADJ-07 | official `verdicts` identical with and without the block | `tests/test_pit_rebuild_report.py:616`; `tests/test_pit_rebuild_cli.py` ADJ-07 test (`without["verdicts"] == report["verdicts"]`) | ✅ |
| LAG-01 | outside + previous within → `si_lag` | `tests/test_pit_rebuild_compare.py:185`; `tests/test_pit_rebuild_cli.py:1803` | ✅ |
| LAG-02 | `si_lag = {count, tickers}` over the adjusted scope; MD count | `tests/test_pit_rebuild_report.py:604`, `:658`. The "over the adjusted scope" cut (`py/pit_rebuild/cli.py:932`) has no test: M17 survived | ⚠️ V3 |
| LAG-03 | no older filing, or within with the newest → not `si_lag` | `tests/test_pit_rebuild_compare.py:193`; `tests/test_pit_rebuild_cli.py:1808`, `:1816`; M14 killed | ✅ |
| LAG-04 | mark changes no statistic | `tests/test_pit_rebuild_compare.py:217`; `tests/test_pit_rebuild_cli.py:1827` | ✅ |
| AUD-01 | one row per worst case, every field, both dates | `tests/test_pit_rebuild_artifact.py:246`; re-checked: 28 sections, 280 rows | ✅ |
| AUD-02 | exactly one class of {erro nosso, erro do SI, diferença de método} + evidence | evidence part: `tests/test_pit_rebuild_artifact.py:275`, `:286` (0 evidence problems in an independent parse). Class part fails: 153 rows `não classificado`; `AUDIT_CLASSES` at `tests/test_pit_rebuild_artifact.py:187` admits it | ❌ V1, ⚠️ V5 |
| AUD-03 | ticker, SI and rebuilt equal the JSON | `tests/test_pit_rebuild_artifact.py:254`; independent re-check of 280 rows: 0 mismatches (reference, rebuilt, diff, si_lag) | ✅ |
| AUD-04 | class counts per field/date equal the rows | `tests/test_pit_rebuild_artifact.py:294`, `:333`; re-check: 17/28/6/65/24 and 7/66/3/56/8 | ✅ |
| AUD-05 | worst-list exclusions named; `missing` per date/field/reason = JSON | `tests/test_pit_rebuild_artifact.py:307`; re-check of both "Sem valor reconstruído" tables against the JSON: equal | ✅ |
| AUD-06 | whole criterion "cumprido" iff part 1 and part 2; count of erro nosso | `tests/test_pit_rebuild_artifact.py:346`; document says "não cumprido", part 1 cumprida (72,77 %, P/L, 2026-11-05), part 2 não cumprida (121 rows), erro nosso 24 — all equal to the data | ✅ |
| RPT-01 | report written by `report --offline` from the feature's code, v3, `amended_by` [64] | `tests/test_pit_rebuild_artifact.py:125`; scratch regeneration differs only in `tool_commit` and timings | ✅ |
| RPT-02 | MD = render(JSON) byte for byte | `tests/test_pit_rebuild_artifact.py:57` | ✅ |
| RPT-03 | fixtures only, no network, no `.cache/` read | `tests/test_pit_rebuild_cli.py:149` (`opener=no_network`); `grep '\.cache' tests/test_pit_rebuild*.py` finds no cache read | ✅ |
| RPT-04 | every number in README and audit equals the JSON | 22 README numbers and all audit numbers re-checked (see "Numbers sampled"): equal. README sentence at `docs/validation/README.md:725` is misleading (V4) | ✅ (V4 nonblocking) |
| RPT-05 | exit codes 0-5; docstring table = README table | `tests/test_pit_rebuild_artifact.py:107` (README codes = docstring codes = 0..5) | ✅ |
| RPT-06 | rule texts state AD-033, drop the #59 sentences | `tests/test_pit_rebuild_artifact.py:158`; `py/pit_rebuild/report.py:51` keeps "1:5 → ×10" only under "sem evento da B3" | ✅ |

**Status**: ❌ gaps present (AUD-02 / AC9). One spec-precision gap flagged (V5).

---

## Discrimination Sensor

Scratch: `git archive HEAD` extracted under the session scratchpad; each mutant on a fresh copy; pit suite run there. Real worktree `git status --porcelain` empty before and after (matches the baseline).

| # | File (HEAD) | Mutation | Result |
| - | ----------- | -------- | ------ |
| M1 | `py/pit_rebuild/dividends.py` `class_events` | `ex <= after` → `ex < after` (event on DT_REFER counts) | ✅ Killed (3) |
| M2 | same | drop the `ex > until` bound | ✅ Killed (8) |
| M3 | same | drop the (label, multiplier, lastDatePrior) dedup | ✅ Killed (2) |
| M4 | `py/pit_rebuild/shares.py` `adjust` | list every class (`abs(M−1) > ε` → `True`) | ✅ Killed (10) |
| M5 | same | `M_all` window without R | ✅ Killed (3) |
| M6 | same | `n_on` uses `M_all` instead of `M_d` | ✅ Killed (7) |
| M7 | `shares.suspect` | `<= 0.20` → `< 0.20` | ✅ Killed (2) |
| M8 | same | ignore events in the interval | ✅ Killed (1) |
| M9 | `py/pit_rebuild/cvm.py` `_ttm_of` | zero comparative no longer triggers the fallback | ✅ Killed (7) |
| M10 | same | year-earlier ITR read without the d cut | ✅ Killed (1) |
| M11 | `py/pit_rebuild/cli.py:551` | `> 270` → `>= 270` | ✅ Killed (1) |
| M12 | `py/pit_rebuild/cli.py:809` | no `stale_filing` mask | ✅ Killed (7) |
| M13 | `py/pit_rebuild/cotahist.py:118` | `isin` reads the last loaded session | ✅ Killed (5) |
| M14 | `py/pit_rebuild/compare.py:125` | drop "newest within tolerance" guard | ✅ Killed (2) |
| M15 | `py/pit_rebuild/report.py:425` | `< 70` → `<= 70` | ✅ Killed (3) |
| M16 | `py/pit_rebuild/cli.py:629` | `isin(q, ticker, date.max)` at the call site | ✅ Killed (1) |
| M17 | `py/pit_rebuild/cli.py:932` | `left_out = set()`: reference_stale tickers can be `si_lag` | ❌ Survived → V3 |
| M18 | `py/pit_rebuild/cli.py:819` | previous-filing value not masked when stale | ✅ Killed (2) |
| M19 | `py/pit_rebuild/report.py:425` | null `pct_within` not failing | ✅ Killed (2) |
| M20 | `py/pit_rebuild/report.py` `class` | class = one field instead of the worst | ✅ Killed (1) |
| M21 | `py/pit_rebuild/cli.py` `_adjusted_input` | DY row keeps its filing | ➖ Equivalent: `report._worst_row` nulls DY anyway (`py/pit_rebuild/report.py:446`) |
| M22 | `py/pit_rebuild/cli.py:544` | `ttm_prior_from_itr` never filled | ✅ Killed (1) |
| M23 | `shares.adjust` | `no_isin` never filled | ✅ Killed (2) |
| M24 | `py/pit_rebuild/report.py:420` | per-field overall from one date | ✅ Killed (2) |
| M25 | `py/pit_rebuild/report.py:446` | DY rows keep filing/previous value | ✅ Killed (2) |

**Sensor depth**: lightweight+ (no P0 path; 25 mutants over every new rule). **Result**: 23/24 non-equivalent killed - FAIL ❌ (one survivor). The survivor is off any critical path and its criterion's main behaviour has proof (V3, nonblocking).

---

## Numbers Sampled in docs/validation (point 4)

All equal to `docs/validation/pit-rebuild-stocks.json` unless noted.

- README "Resultado da regeneração": official class `não reconstruível`; LIQ. CORRENTE 2026-08-07 group without unclassified 90,21 % (v2 at `ebaf1fd`) → 89,79 % (211/235), verdict `reconstruível` → `com ressalvas`, AMBP3 in `missing` with `stale_filing`; adjusted class `com ressalvas`; part 1 met; minimum 72,77 % (P/L, 2026-11-05); `si_lag` sums 32 and 350; `share_events.adjusted` 6 and 3; `no_isin` 9 and 10; `share_count_suspect` 5 and 8; `ttm_prior_from_itr` 18 and 20; `stale_filing` 1 and 3; cd_cvm 11975 grouping ex-date 2026-06-26 (in `share_events.adjusted`).
- README "maior parte das marcas ... ITR de 2026-09-30, recebido entre 2026-11-03 e 2026-11-05": re-counted from the cache (FCA ticker → CNPJ, newest filing ≤ 2026-11-05): 315 of the 325 mappable marks (25 marks not mapped by this quick check). Holds.
- Audit summary 17/28/6/65/24/140, 7/66/3/56/8/140, totals 24/94/9/121/32/280; per-field count tables; both missing tables; 280 rows (reference, rebuilt, diff, si_lag): 0 mismatches. In-tolerance label matches `compare.within` on all 280 rows.
- Audit evidence spot-checked against the cache: VTRU3 ITR 2026-03-31 1.01 = 1 406 980 MIL and 2.01 = 660 877 MIL; ONCO3 ITR 2026-06-30 1.01 = 2 611 886 MIL, 2.01 = 5 892 598 MIL; HBRE3 0,874071 + 0,291357 = 1,1654 → 100 × 1,1654 / 2,22 = 52,50 (SI) and Yahoo 0,485595 → 21,87; EQPA3 0,316875 / 5,32 = 5,96 p.p. gap; MBRF3 17,49 / 17,98 / 412 days / 15,27.
- SIMH3 rows: the cited count is right but the cited filing is not the earliest that carries it (V2).

---

## Points Raised by the Implementer

1. **AC9 / AUD-02, 153 `não classificado` rows.** The document is honest: "não cumprido" equals the data (part 1 met at 72,77 %; part 2 fails on 121 rows: 65 + 56; erro nosso 24). That makes DEC-2 part 2 a legitimate *measurement* result. AC9, though, is a deliverable criterion: it asks that the 10 worst be classified into the three classes, and 121 rows outside tolerance are not. Under evidence-or-zero AC9 is not met → **V1, blocking**. Forcing a class without evidence would be false text, so the closure is a decision for the user, not more code.
2. **`share_scale` `verified=True` with residual 2,5 (VVEO, cd 25682) and 2,66 (AZEV, cd 11975).** Specified behaviour, not a defect. `cvm.share_scale` (`py/pit_rebuild/cvm.py:306`) is unchanged from #59: `verified` means "a positive B3 total chose f" (A-18), not "the count matches B3". Both residuals are under √10 after f, the A-18 limit for a change with no B3 event (VVEO: filed 322,8 M vs B3 1 286 M, ratio 3,98; AZEV: 29,6 M after the 20:1 grouping vs 111 M, ratio 3,75). The report never prints `verified`; it only fills `share_scale.unverified`. The audit's `erro nosso` for these rows is consistent; the fix is a new issue (spec Out of Scope).
3. **2026-08-07 CSV reproducing filings received up to 2026-08-14.** Supported by the cache: ORVR3 (cd 25550) ITR 2026-06-30 v1 rec 2026-08-13 and v2 rec 2026-08-14; ONCO3 ITR 2026-06-30 v1 rec 2026-08-14; BRST3 rec 2026-08-11; EQPA3 rec 2026-08-12; SIMH3's count 583 759 291 first appears in ITR 2026-06-30 v1 rec 2026-08-12 (`itr_cia_aberta_2026.zip`, composição). Two corrections: the audit cites SIMH3's ITR 2026-09-30 rec 2026-11-04 instead (V2), and the handoff's "EQPA3 DFP 2025 v2 em 10-02" has no audit row behind it (the EQPA3 row of 2026-08-07 uses ITR 2026-06-30 rec 2026-08-12); corrected in the feature `STATE.md`. Context for the user, not a gap: the JSON `as_of_check` matches 245 of 269 SI prices to the 2026-08-07 session, while `git log` shows the CSV last changed in `ce2bc1f` (author date 2026-09-01). The audit's own wording ("o cache não distingue as duas coisas") stays correct.
4. **Numbers in docs/validation vs JSON.** 22 README numbers, every audit number and 6 evidence values checked; all equal. One misleading sentence (V4).
5. **Gates.** Pit 471 OK; root 1058 OK, 2 skipped (foreground, 3 min 11 s).

---

## Gaps

| ID | Severity | Where | Gap | Suggested correction |
| -- | -------- | ----- | --- | -------------------- |
| V1 | blocking | `docs/validation/pit-rebuild-stocks-audit.md:28-43`; `tests/test_pit_rebuild_artifact.py:187` | Issue AC9 / AUD-02 not met: 121 rows outside tolerance (65 on 2026-08-07, 56 on 2026-11-05) have no class from the three; the test accepts a fourth class. | User decision, then one of: (a) amend AC9/AUD-02 to accept `não classificado` with a stated reason as a valid audit outcome (part 2 of DEC-2 stays "não cumprido"), record it in the issue/spec and keep the test; or (b) classify the 121 rows with evidence, e.g. a rule for "SI number with no basis in any cached filing", which the #59 conference already treated as an SI error. Do not assign classes without evidence. |
| V2 | blocking (REVIEW.md: docs/validation text claims more than the proof) | `docs/validation/pit-rebuild-stocks-audit.md:202`, `:229` | SIMH3 (P/VP and PSR rows of 2026-08-07) says SI "usa a composição de 2026-09-30 recebida em 2026-11-04". The same count (583 759 291) is in ITR 2026-06-30 v1 rec 2026-08-12. The text ties a 2026-08-07 collection to a filing received three months later; the cache supports 2026-08-12. | Cite `CVM ITR 2026-06-30 v1 rec 2026-08-12: QT_ACAO_ORDIN_CAP_INTEGR = 583759291, ...` and reword the reason to the 2026-06-30 filing; the class stays `erro do Status Invest`. |
| V3 | nonblocking | `py/pit_rebuild/cli.py:932`; no test | Mutant M17 survives: removing the reference_stale cut from the `si_lag` marks is undetected. LAG-02 asks the count over the adjusted scope; the code is correct today. | Add a CLI fixture where a `reference_stale` ticker would be `si_lag` and assert it is not in `adjusted_verdict.dates[d].fields[f].si_lag.tickers`. |
| V4 | nonblocking | `docs/validation/README.md:725-726` | "Referência zero ... O documento de auditoria dá a contagem por data, campo e motivo (AUD-05)" reads as if the audit counts zero-reference tickers; the audit says they are not counted (`pit-rebuild-stocks-audit.md:20`). | "... não entra na lista dos piores casos e não é contado no JSON. O documento de auditoria conta os tickers sem valor (`missing`) por data, campo e motivo (AUD-05)." |
| V5 | nonblocking (spec-precision) | `spec.md` AUD-02 / B-10 | The spec did not foresee worst lists padded with in-tolerance rows (32 rows with diff ≤ tolerance). `não classificado: dentro da tolerância` is a reasonable reading but not a spec outcome. | Amend AUD-02 together with V1: rows within tolerance carry no class (or are excluded from the audit) and do not count against part 2. |

---

## Code Quality

| Principle | Status |
| --------- | ------ |
| Minimum code / no scope creep | ✅ New module `shares.py` holds only the AD-033 rules; no new CLI arguments or exit codes |
| Surgical changes | ✅ Protected paths unchanged against `ebaf1fd` |
| Matches patterns | ✅ Same pure-function + CLI wiring style as #59 |
| Spec-anchored outcome check | ✅ Tests assert the spec's numbers (1100, age 280/270, factor 2, 20 %/20.01 %, 70/69.99) |
| Every test maps to a requirement | ✅ Test names and comments carry the IDs |
| Guidelines followed | `REVIEW.md` (numbers in docs/validation equal the artifact; exit-code table) |

---

## Requirement Traceability Update

| Requirement | Previous | New |
| ----------- | -------- | --- |
| AUD-02 | Partial | ❌ Needs fix (V1, V5) |
| LAG-02 | Done | ⚠️ Verified, one untested cut (V3) |
| RPT-04 | Partial | ✅ Verified (numbers equal); V4 wording |
| ADJ-03, ADJ-06 | Unit | ✅ Verified (mutants killed) |
| MBR-02, ADJ-05 | Partial | ✅ Verified |
| All other 36 | Done | ✅ Verified |

---

## Summary

**Overall**: ❌ Not ready.

**Spec-anchored check**: 42/44 requirements verified; AUD-02 fails (V1), LAG-02 has an untested cut (V3); 1 spec-precision gap (V5).
**Sensor**: 25 injected, 23 killed, 1 equivalent, 1 survived (V3).
**Gate**: pit 471 passed; root 1058 passed, 2 skipped, 0 failed.

**What works**: event-adjusted share counts and the scale window up to R, suspect counts, the 12-month fallback, the 270-day rule, the ISIN at d, the adjusted verdict and part 1 of the criterion, the `si_lag` marks, the MBRF3 note, a byte-for-byte regenerable report, and an audit whose numbers equal the JSON.

**Next steps**: user decides V1 (amend AC9/AUD-02 or classify the 121 rows); implementer fixes V2 (two audit rows) and optionally V3, V4; re-verify (iteration 2 of 3).
