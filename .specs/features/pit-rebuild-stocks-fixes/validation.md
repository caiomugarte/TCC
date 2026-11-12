# Validation: pit-rebuild-stocks-fixes (issue #64) - FAIL (iteration 2)

**Date**: 2026-11-12
**Spec**: `.specs/features/pit-rebuild-stocks-fixes/spec.md` (r3, 44 requirements; AUD-02, B-10 and AUD-06 amended)
**Diff range**: `ebaf1fd..a29381a` (feature). The fix round is `1def298..a29381a`: `3ee1da1` spec, `b1a0d62` test, `78b433b`, `07795f5`, `a29381a` docs, `86c3197` state. `py/` is unchanged since `1def298`.
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-64-verify-r2`. Iteration 1 report: commit `1def298`.

**Verdict: FAIL.** V1..V5 are closed: no row is left without a class, SIMH3 cites the right filing, M17 is an equivalent mutant, the README sentence is fixed, and the spec and tests carry the new classes. The gates pass. One new blocking gap (V6): the audit says, on 108 rows, that no cached demonstrativo reproduces the Status Invest value. The classification tested a narrower set than the text states. With the definition the document gives, 30 of those rows have a cached filing that reproduces the value, 3 of the 4 DY `sem base` rows have a reproducing subset, and 5 reproduced rows cite a filing that is not the oldest. No class changes, so the criterion stays "cumprido". The per-row text, the `sem base` counts and the oldest-filing citations are false as written.

---

## Gate Check

- **Pit suite**: `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"` gives `Ran 473 tests`, `OK` (471 + 2 new tests in the fix round).
- **Build gate**: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"` (foreground, exit 0) gives `Ran 1060 tests in 187.453s`, `OK (skipped=2)`. `python3 -m compileall -q py/pit_rebuild` passes. `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` is empty.
- **Test count**: 1058 at iteration 1, 1060 now. No test deleted. `AUDIT_CLASSES` grew from 4 to 5 entries, and the summary and criterion tests now read the fifth class (`tests/test_pit_rebuild_artifact.py:188`, `:352`, `:362`).
- **Skipped**: the same 2 root-suite skips as iteration 1, both outside this feature.

---

## Re-verification of V1..V5

| Gap | Status | Evidence |
| --- | ------ | -------- |
| V1 (blocking): AC9/AUD-02, rows outside tolerance with no class | ✅ Closed as a class gap; text defect moved to V6 | 280 rows parsed independently: 0 `não classificado`, 32 `dentro da tolerância`, 213 `erro do Status Invest`, 11 `diferença de método`, 24 `erro nosso`, equal to `docs/validation/pit-rebuild-stocks-audit.md:41-43` and to the per-field tables. The split at `:47-50` matches the rows' motives: 32 reproduced and 60 `sem base` on 2026-08-07 (59 accounting rows, 1 DY), and 69 and 52 on 2026-11-05 (49 accounting rows, 3 DY). Total `sem base`: 112, as stated at `:61`. The "Demonstrativos conferidos" tables (26 companies, 464 version rows, `:586`-EOF) equal the cache **row for row**: DT_RECEB, the 14 account values, the ON/PREF/TOTAL composition and the 80 "(só o índice no cache)" flags. No version received by 2026-11-09 is missing. All 108 `Conferidos: N` items and every "N demonstrativos / M versões só têm o índice" motive equal the cache counts. Criterion line `:56` "cumprido" equals the rows (0 `não classificado`) and `adjusted_verdict.criterion.met` true. The negative claim behind `sem base` does not hold as written (V6). |
| V2 (blocking): SIMH3 citation | ✅ Closed | `audit.md:220` (P/VP) and `:247` (PSR) cite `CVM ITR 2026-06-30 v1 rec 2026-08-12: QT_ACAO_ORDIN_CAP_INTEGR = 583759291`. Cache: `itr_cia_aberta_2026.zip`, cd 25003, ITR 2026-06-30 v1, DT_RECEB 2026-08-12, composition (583 759 291, 0, 583 759 291). ITR 2026-03-31 v1 had 426 797 757, so 2026-08-12 is the first filing with that count. The "o mais antigo é citado" part fails for these two rows under the document's own definition (V6). |
| V3 (nonblocking): M17 at `py/pit_rebuild/cli.py:932` | ✅ Closed: equivalent mutant, confirmed from the code | `cotahist.reference_stale` is `quote_at(q, ticker, d) is None` (`py/pit_rebuild/cotahist.py:104-106`). `_unmapped_reason` returns `no_quote` on the same `quote_at` at the same `day` (`py/pit_rebuild/cli.py:350`), or an earlier reason. Either way `_evaluate_day` hits `continue` before `previous_values[ticker]` is set (`cli.py:770-774`; the previous values are set only at `:816-820`). In `_adjusted_input` the call `result.previous.get(ticker, {}).get(name)` then gives `None`, and `compare.is_si_lag` returns False when `calc_prev is None` (`py/pit_rebuild/compare.py:133`). Removing `left_out` cannot change a mark. `tests/test_pit_rebuild_cli.py:1827` pins the invariant (STLQ3 is `reference_stale` and `no_quote`, with no mark). Whether to delete the redundant cut is a user decision, not a gap. |
| V4 (nonblocking): README zero-reference sentence | ✅ Closed | `docs/validation/README.md:725-727` now says that the audit counts only `missing` and that zero reference is a declared limit. This matches `audit.md:28`. |
| V5 (nonblocking, spec precision) | ✅ Closed | `spec.md` r3: AUD-02 (class set of five, `sem base` evidence), B-10 (fix-round paragraph) and AUD-06 (`dentro da tolerância` counts as audited). Tests: `tests/test_pit_rebuild_artifact.py:188` (`AUDIT_CLASSES`), `:268` (the class and the "dentro da tolerância" motive agree both ways; a `não classificado` row states a reason), `:277` (a `sem base` row carries a `CVM` item and a `Conferidos:` item; a DY `sem base` row carries a dividend item and a `COTAHIST` item). Sensor below: the mutants on these assertions are killed. |

---

## V1 Sample Against the Cache (rule of the user: "sem base" = no cached demonstrativo reproduces the SI value)

Method, independent of the implementer's scripts: every DFP and ITR version of the 26 companies was read from `.cache/pit-rebuild/{dfp,itr}_cia_aberta_*.zip`. That is 464 versions, with the statement choice F18, `cvm._value` scaling and the LL rule A-21 reused from `py/pit_rebuild/cvm.py`. Each version was re-read as the newest filing: the same TTM rule as `cvm._ttm_of`, the balance accounts, and the composition × A-18 scale × B3 events from `b3_company_<root>.json` up to d. Each version was tried whole (controller and total LL/PL) and by its count alone, and compared with `compare.within`. **Calibration**: for 139 of 147 rows, the emulation recomputes the rebuild's own value exactly as in the JSON. All 30 findings below are on calibrated rows.

Sample (24 rows: 20 recomputed, plus the 4 DY `sem base` rows checked by subset sum):

| Date | Field | Ticker | Audit says | Cache |
| ---- | ----- | ------ | ---------- | ----- |
| 08-07 | LIQ. CORRENTE | VTRU3 (`:91`) | sem base, 13 checked | ✅ none of 13 reproduces 4.48 |
| 08-07 | LIQ. CORRENTE | BHIA3 (`:94`) | sem base, 15 checked | ❌ ITR 2024-06-30 v1 (`audit.md:601`): 12 733 000 000 / 16 771 000 000 = **0.7592 → 0.76 = SI**. DFP 2019-12-31 v1 (`:594`) gives 0.7915 (4.1 %) |
| 08-07 | DIV. LIQ. / PATRI. | AALR3 (`:121`) | sem base, 13 checked | ❌ ITR 2025-09-30 v1 rec 2025-11-14 (`audit.md:840`): (236 911 000 + 250 040 000 − 123 751 000) / (1 099 291 000 − 32 363 000) = **0.3404 → 0.34 = SI** |
| 08-07 | DIVIDA LIQUIDA / EBIT | BHIA3 | sem base | ✅ none |
| 08-07 | ROE | TIMS3 (`:145`) | sem base | ❌ ITR 2025-09-30 v1 gives 15.98 (SI 15.91, 0.07 p.p.) |
| 08-07 | MARG. LIQUIDA | TPIS3 | sem base | ✅ none |
| 08-07 | MARGEM EBIT | ISAE3 | sem base | ✅ none |
| 08-07 | CAGR RECEITAS 5 ANOS | MLAS3 | sem base | ✅ none |
| 08-07 | P/L | TIMS3 | sem base | ✅ none |
| 08-07 | PSR | BRST3 (`:249`) | sem base | ❌ DFP 2024-12-31 v1 gives 0.86 (SI 0.89, 3.4 %) |
| 11-05 | ROE | EQPA3 | sem base | ✅ none |
| 11-05 | P/VP | TIMS3 | sem base | ✅ none |
| 11-05 | EV/EBIT | TFCO4 | sem base | ✅ none |
| 11-05 | CAGR LUCROS 5 ANOS | CAML3 | sem base | ✅ none |
| 08-07 | LIQ. CORRENTE | ONCO3 (`:92`) | reproduced, ITR 2026-03-31 v2 rec 2026-08-14, "2 reproduzem" | ✅ v2 gives 0.46. Two versions reproduce (v2 and ITR 2026-06-30 v1, 0.44) |
| 08-07 | MARGEM EBIT | BRST3 (`:188`) | reproduced, oldest = ITR 2026-06-30 v1 rec 2026-08-11 | ❌ ITR 2024-09-30 v1 (older) gives 14.99 (SI 14.96) |
| 11-05 | PSR | VIVA3 | reproduced by count, DFP 2020-12-31 v1 | ✅ the count of DFP 2020 v1 gives 1.35 (SI 1.36). It is the oldest |
| 11-05 | PSR | PMAM3 | reproduced by count, DFP 2020-12-31 v3 | ✅ 1.89 (SI 1.84). It is the oldest |
| 08-07 | PSR | EGIE3 (`:250`) | diferença de método: "nenhum demonstrativo do cache reproduz o valor do SI (3.08)" | Class ✅: the B3 total of R, 1 416 381 520 (`b3_company_EGIE.json`), gives 29.73 × 1 416 381 520 / 13 680 975 000 = 3.08. The text is ❌: DFP 2023-12-31 v1 (`audit.md:693`), 815 927 740 × 1.4 (BONIFICACAO 40 %, lastDatePrior 26/11/2025) = 1 142 298 836, gives 29.73 × 1 142 298 836 / 10 747 784 000 = **3.16** (2.6 %) |
| 11-05 | PSR | EGIE3 (`:516`) | same | Class ✅. Text ❌: DFP 2023-12-31 v1 gives 26.64 × 1 142 298 836 / 10 747 784 000 = **2.83** (SI 2.76, 2.5 %) |
| DY | JSLG3 11-05 (`:553`), EQTL3 11-05 (`:561`), MBRF3 11-05 (`:552`), HBOR3 08-07 (`:291`) | "nenhum subconjunto dos proventos da B3 do cache reproduz o DY do SI" | HBOR3 ✅: no subset of its 16 ON records. JSLG3, EQTL3 and MBRF3 ❌ as written: subsets of all cached ON records give the SI DY exactly to two decimals, with 3 708, 32 and 2 subsets at the close of d. None exists within ex-date (d − 730, d] |

Across all 110 rows whose text says no cached demonstrativo reproduces the value (108 `sem base` accounting rows and the 2 EGIE3 rows), **30 rows** (20 on 2026-08-07, 10 on 2026-11-05) have a cached version that reproduces the SI value under the document's own definition (`audit.md:14`). All 30 still use only the version's own inputs, read at max(d, DT_RECEB). The rows: BHIA3 (LIQ. CORRENTE, DIV. LIQ./PATRI. ×2, ROA ×2, MARG. LIQUIDA ×2, MARGEM EBIT, EV/EBIT ×2, CAGR RECEITAS ×2), BRST3 (DL/EBIT, DIV. LIQ./PATRI. ×2, ROE, ROA, MARG. LIQUIDA, PSR ×2), VTRU3 DL/EBIT, AALR3 DIV. LIQ./PATRI., CAML3 (ROE, P/L ×2), TIMS3 (ROE, ROA ×2) and EGIE3 PSR ×2. Five reproduced rows cite a filing that is not the oldest reproducer: BRST3 MARGEM EBIT 08-07, ORVR3 P/VP 08-07, SIMH3 P/VP and PSR 08-07, and CAML3 DIV. LIQ./PATRI. 11-05.

**Cause**: the implementer's classifier lives outside the repository (session scratchpad `3ec97855-…/scratchpad/`). Its `views.py:205-206` applies `mask_accounting(..., stale_filing)` whenever the version read as newest ends more than 270 days before d. Every version older than that cannot reproduce anything when read whole; only its count is tested. `classify2.py:136` limits DY candidates to ex-date in (d − 730, d]. The document states neither restriction. `audit.md:14` says all versions are tested whole and by count. `:13` and `:31` state the DY and `sem base` rules without a window. Each row says "nenhum dos N demonstrativos ... reproduz ..., testados o demonstrativo inteiro, ... e só a contagem de ações de cada um". Spec B-10 r3 says the same ("every version received up to R, each read as the newest filing").

**Impact**: no class changes. Under the rule order (`audit.md:13`), every one of the 30 rows stays `erro do Status Invest` (EGIE3 stays `diferença de método`). Part 2 and the "cumprido" criterion (AUD-06) hold either way. What is false is the per-row negative claim, the `sem base` counts (`audit.md:47-50`, `:61`: 112 as defined), and the "o mais antigo é citado" claim on 5 rows. REVIEW.md classes false text in `docs/validation/` as Important.

---

## Discrimination Sensor (fix-round surface)

Scratch: `git archive HEAD docs tests py` extracted under the session scratchpad. Each mutant was applied to the copy of `pit-rebuild-stocks-audit.md` and `tests.test_pit_rebuild_artifact` was run there. The copy has 1 baseline error: the hash test needs `data/raw/`, which was not copied. Real worktree `git status --porcelain`: empty before and after.

| # | Mutation (audit copy) | Result |
| - | --------------------- | ------ |
| A1 | drop the `Conferidos:` item of VTRU3 LIQ. CORRENTE (`sem base`) | ✅ Killed (`:277`) |
| A2 | in-tolerance row TIMS3 LIQ. CORRENTE relabelled `erro do Status Invest` | ✅ Killed (3 failures) |
| A3 | criterion line "cumprido" → "não cumprido" | ✅ Killed (`:362`) |
| A4 | `sem base` row BHIA3 relabelled `não classificado` | ✅ Killed (3 failures) |
| A5 | second Resumo table 32/60 → 33/59 (reproduced vs `sem base`) | ❌ Survived: no test reads that table (V8) |
| M17 (r1) | `cli.py:932` `left_out = set()` | ➖ Equivalent (V3 above) |

Iteration 1 code mutants (M1..M25) are not re-run: `py/` is unchanged since `1def298`.

---

## Gaps

| ID | Severity | Where | Gap | Suggested correction |
| -- | -------- | ----- | --- | -------------------- |
| V6 | **blocking** (REVIEW.md: false text in `docs/validation/` is Important) | `docs/validation/pit-rebuild-stocks-audit.md:13`, `:14`, `:31`, `:47-50`, `:61`; the motive of the 108 `sem base` rows (e.g. `:94`, `:121`, `:145`, `:249`) and of `:250`, `:516`; the "o mais antigo é citado" rows `:188`, `:220`, `:247`, and the ORVR3 P/VP 08-07 and CAML3 DIV. LIQ./PATRI. 11-05 rows; `spec.md` B-10 r3 | The document says every cached version was tested whole and by count, and that none reproduces the SI value. The classification blanked every version older than 270 days (STL-01) and limited DY to ex-date in (d − 730, d]. Under the stated definition, 30 of 110 rows have a reproducing version, 3 of 4 DY `sem base` rows have a reproducing subset, and 5 rows do not cite the oldest reproducer. | Pick one and apply it everywhere. (a) **Recommended**: state the two restrictions as part of the rule (`Reproduzir`, `Ordem das regras`, `sem base`, Limites, B-10 r3). Reword the row motive so it gives how many versions were tested whole (period end ≤ 270 days before d) and how many by count only, and the DY window. Then the counts stand as they are. (b) Re-run without the 270-day mask and the DY window, reclassify the 30 rows as reproduced (oldest cited), fix the 5 citations and the split table. Under (a) or (b), add a test that recounts the split table (V8). |
| V7 | nonblocking | `docs/validation/pit-rebuild-stocks-audit.md` as a whole; `:33` "168 das 1148 versões de 49 empresas" | The classification rules that produce `sem base` and the cited oldest filing exist only in a session scratchpad. B-10 makes the audit hand-written, and RPT-01 covers the generated report, not the audit, so no written rule is broken. But the negative claims cannot be re-run from the repository, and V6 shows the script had rules the text omits. The "168 of 1148 versions of 49 companies" figure covers more companies than the 26 listed and could not be checked from the repo (the 26 listed have 464 versions, 80 index-only, equal to the cache). | Commit the classifier (a research script next to the audit, read-only on `.cache/`), or describe its procedure fully in the audit (the V6 fix covers most of it). Tie the 168/1148/49 figure to the listed companies, or say which set it counts. |
| V8 | nonblocking | `docs/validation/pit-rebuild-stocks-audit.md:47-50`; `tests/test_pit_rebuild_artifact.py:336` (`summary_rows` skips the 3-cell table) | The table of reproduced vs `sem base` rows has no test. Mutant A5 survived. | Count the rows by motive prefix (`nenhum dos`, `nenhum subconjunto` vs the rest of `erro do Status Invest`) and assert the table. |

---

## Code Quality (fix round)

| Check | Status |
| ----- | ------ |
| Only touched files required for the gaps | ✅ spec, README, audit, two test files, STATE. No `py/` change |
| Tests map to the amended ACs | ✅ `:188`, `:268`, `:277`, `:352`, `:362` (AUD-02, AUD-04, AUD-06); `tests/test_pit_rebuild_cli.py:1827` (LAG-02 invariant) |
| Spec-anchored outcome | ⚠️ The tests check the shape of a `sem base` row (CVM and `Conferidos:` items). They cannot check the claim against the cache (`.cache/` is not read by tests, RPT-03). V6 is a text gap, found only by a cache re-check. |
| Guidelines | `REVIEW.md` ("documento que afirma mais do que a prova commitada mostra"; numbers in `docs/validation/` equal the artifact) |

---

## Summary

**Overall**: ❌ Not ready (iteration 2 of 3).

- **Closed**: V1 (as a class gap), V2, V3 (equivalent mutant), V4, V5.
- **New**: V6 blocking (the `sem base` claim and the oldest-filing claim are false as written; classes and criterion unaffected), V7 and V8 nonblocking.
- **Gates**: pit suite 473 OK; root suite 1060 OK, 2 skipped.
- **Sensor**: 5 audit mutants: 4 killed, 1 survived (V8). M17 equivalent.

**Next step**: implementer fixes V6, by stating the restrictions (recommended) or by re-running without them, plus V8 and optionally V7; then Verifier iteration 3 (the last before escalation).
