# Validation: pit-rebuild-stocks-fixes (issue #64) - FAIL (iteration 4, authorized by the user)

**Date**: 2026-11-12
**Spec**: `.specs/features/pit-rebuild-stocks-fixes/spec.md` (r5; B-10 and AUD-06 amended for V9, V10 and decisions 3 and 4)
**Diff range**: `ebaf1fd..93d3fb5` (feature). The fix round is `5e84eca..93d3fb5` (`932ed5d` … `4ccb098`, then `93d3fb5` state). `py/` is unchanged in the round.
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-64-verify-r4`. Iteration 3 report: commit `5e84eca`.

**Verdict: FAIL.** The code is right and the numbers are right. V11 and V12 are closed. Two sentences in the audit header and the TGMA3 row still say what the SI does, which breaks rule (b). The Citação paragraph and rule 9 also do not describe the code exactly. Both are text in `docs/validation/`, so both are blocking under REVIEW.md. The fixes are wording only: 4 sentences in the generator's HEADER plus the TGMA3 motive, then a regeneration. No class or count moves. The criterion stays "cumprido" with `erro nosso` 24.

The rest checks out. All 44 accounting citations follow rule (c). The 3 class changes follow rule (d). All 20 DY rows match an independent search of the B3 cache over contiguous windows. The document regenerates byte for byte from a clean checkout with no bytecode written. The gates pass.

---

## Gate Check

- **Pit suite**: `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`: `Ran 495 tests`, `OK` (480 in iteration 3; +15 from the new fixture and artifact tests).
- **Build gate** (foreground): `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → `Ran 1082 tests in 190.996s`, `OK (skipped=2)`, exit 0. No test was deleted (1067 → 1082). The 2 skips are the same root-suite skips as before, outside this feature.
- Real-tree `git status --porcelain` was empty before and after every check.

---

## Re-verification of V9..V12 (and V3)

| Gap | Status | Evidence |
| --- | ------ | -------- |
| V9 (blocking): text presents a tolerance match as what the SI did | ⚠️ **Mostly closed; residue in V13** | **Rows.** I normalized all 280 motives to templates. Every citation motive is a measure: `match_clause` (`scripts/pit_rebuild_audit.py:743`) writes "igual a duas casas a …" or "dentro da tolerância CMP-01 de … ; a coincidência não mostra a origem". The same holds for B-9 lag, post-d, the B3 count, LL/PL and DY windows. The only causal row text is TGMA3 (`audit.md:315`, "o corte do SI parece ser pelo lastDatePrior"). See V13. **Summary headers** (`audit.md:66`, `:79`) and limits `:49-52` are measures. **README** (`docs/validation/README.md:483-484`) only points to the audit. **Header**: `:31` ("porque pergunta de qual demonstrativo o SI tirou o valor") and `:33` ("como o SI as soma (casos MDNE3 e VLID3)") still state SI behavior. See V13. **Citation (rule c)**: I recomputed the expected citation from the generator's own search pool, dumped from a scratch run, for all 44 cited accounting rows. 0 mismatches: the exact match received first, else the smallest gap. 17 are exact and 27 within tolerance only, as the document says (`:51`). **Rule (d)**: EGIE3 PSR 08-07 (`:270`) and 11-05, and OPCT3 EV/EBIT 11-05, are `diferença de método`. In each, the B3 total count of R equals the SI at two decimals (3.08, 2.76, 14.12). The as-filed reproducers are all inexact: EGIE3 3.16/3.19/3.19/3.03 vs 3.08 and 2.83/2.86/2.86/2.71 vs 2.76; OPCT3 13.75 vs 14.12. No row uses rule "reproduzido (sobre a contagem da B3)" (rule histogram of the run). **Rule (e)**: an independent script reads COTAHIST for close and ESPECI and the raw `b3_cash_*` files, then searches contiguous windows, shortest first and then earliest, on the close of d and d − 1. It gives the same answer as the document for all 20 DY rows. Windows: EQPA3 08-07 (1 record), VLID3 ×2 (5 and 6, with 4 identical lines), PFRM3 ×2, PASS3, CLSC4 (1 record of 2019), EQTL3 (1 record of 2024). No window: RIAA3 ×2, MDNE3 ×2, HBOR3, MBRF3, JSLG3, EQPA3 11-05, COCE5 ×2. |
| V10 (nonblocking): Citação paragraph and rule 4 | ❌ **Not closed; residue in V14** | Spec B-10 r5 (`spec.md:60`) describes `citation` (`scripts/pit_rebuild_audit.py:139`) and the view order (`:848-849`, `as_filed` before `others`) exactly. The document's paragraph (`audit.md:32`) adds "Quando o mesmo demonstrativo reproduz por mais de uma dessas visões, vale a primeira na ordem: como arquivado, depois a contagem". The code ranks views of one filing by exactness and gap first. ORVR3 EV/EBIT 08-07 (`:255`) cites the count view of ITR 2026-06-30 v2 (31.8, exact), while the as-filed view of the same filing gives 33.2. Rule 9 (`:28`) still says "citado o mais antigo", but the code cites method views by `citation` too. No row is affected: each of the 4 rule-9 rows has a single filing. |
| V11 (nonblocking): scope, rule order and count sentences untested | ✅ Closed | `SearchScopeTests` (`tests/test_pit_rebuild_audit.py:121`), `RuleOrderTests` (`:148`, rule 6 exact-only at `:168`), `CitationTests` (`:42`), `MeasureWordingTests` (`:73`), `DyWindowTests` (`:89`), recount `tests/test_pit_rebuild_artifact.py:366`, cause list `:375`. Sensor below: 13 of 17 mutants killed. G2 and G3 survive the suite. They change only facts computed inside `classify_accounting`/`search_date`, which only the regeneration checks; see V15. |
| V12 (nonblocking): bytecode outside `--out` | ✅ Closed | `sys.dont_write_bytecode = True` at `scripts/pit_rebuild_audit.py:34`, before the `pit_rebuild` imports. On a `git archive HEAD` copy, with `PYTHONDONTWRITEBYTECODE` unset and only `.cache/pit-rebuild` linked, a full run (exit 0, 75.4 s) wrote no new `.pyc` (the list equals the 16 tracked `.pyc` of the archive) and no file newer than the run marker. The real tree stayed clean. The cache listing hash (names, sizes, mtimes of 586 files) was the same before and after. `GeneratorWritesNoBytecodeTests` (`tests/test_pit_rebuild_audit.py:176`) kills mutant G9. |
| V3 | accepted as equivalent (prior round) | unchanged; the user's pending item on `reference_stale` stays open |

### Manual DY readings (`DY_DECISIONS`, `scripts/pit_rebuild_audit.py:79-88`)

- **HBRE3 `erro nosso`, sustained.** The B3 file holds 2 ON lines of 30/12/2025, 0.874071 + 0.291357 = 1.165428. Over the close of d (2.22) that is 52.50 %, equal to the SI. The project's 21.87 % is the Yahoo record 0.485595, which matches neither B3 line. Our value contradicts the cached B3 record, so the class holds without assuming what the SI did.
- **TGMA3 `diferença de método`, sustained as a measure; the wording is not.** The 5 records with lastDatePrior in (d − 365, d] are indices 78..82 of the 83 ON lines. They sum to 3.63, and 3.63 / 33.48 = 10.84 %, equal to the SI. The project's window by data-ex in (d − 365, d] also takes 0.14 and 1.21 (lastDatePrior 07/08/2025, data-ex 08/08/2025), giving 14.87 %. So the SI value equals another window definition exactly, and ours does not, which is the same shape as rules 6 and 7. The automatic search would find the same 5-record window and call the row `reproduzido`, so this manual reading decides the class. Two defects: the motive states a cause ("o corte do SI parece ser pelo lastDatePrior"), and the class definition at `audit.md:16` lists only the B3 count, LL 3.11 and PL 2.03 as the other definitions. See V13.
- **COCE5 ×2, blank, sustained.** The SI cell is blank and enters as 0.00 (A-4). No window of the 58 PNA lines rounds to 0.00 on either close (independent search). `sem base`, `erro do Status Invest`.

### Implementer choices (judged)

- **"Mais próximo" = smallest absolute difference.** Sound. Within a row the SI value is fixed, so absolute and relative distance rank the reproducers the same way.
- **DY tie-break: shortest window, then earliest.** It matches rule (c) ("o mais antigo entre os exatos") among windows of equal length, and every DY window is exact. Shortest-first picks single old records: CLSC4 cites a 2019 record with no quote for its ex-date (`:314`), and EQTL3 a 2024 one (`:581`). The motive says "a coincidência não mostra a origem", and the choice moves no class and no reproduced/`sem base` count. Acceptable; no gap.

### Numbers and coherence (item 5)

- Regeneration from a clean archive: `cmp` is identical twice (`regen1`, exit 0; `regen2` through an in-process wrapper). The run prints `criterion cumprido; out_nc 0; erro nosso 24`.
- Independent parse of the 280 rows. 2026-08-07: 17 erro nosso, 92 erro do SI, 7 método, 24 dentro da tolerância. 2026-11-05: 7, 121, 4, 8. Origin: reproduzido 48 / 75, `sem base` 44 / 46 = 90. These equal the Resumo (`audit.md:61-70`), the sentence `:79` ("São 90 linhas `sem base`"), the "44 … 17" sentence (`:51`) and the STATE handoff. The criterion line `:74` ("cumprido", erro nosso 24) matches AUD-06 r5: part 1 from `adjusted_verdict.criterion`, part 2 with 0 `não classificado`.
- Against iteration 3: erro do SI 93/123 → 92/121 and método 6/2 → 7/4, exactly the 3 rule-(d) rows. `sem base` 41/40 → 44/46 adds the 9 DY rows that STATE lists. Accounting citations 47 → 44 lose the 3 rule-(d) rows. This matches AUD-06 r5 (`spec.md:206`).
- CVM evidence sample: 14 rows (BHIA3 LIQ. CORRENTE and MARG. LIQUIDA 08-07, VIVA3 and PMAM3 PSR 11-05, TIMS3 ROA, CAML3 P/L and DIV. LIQ./PATRI. 11-05, EMAE4 ROA, AALR3 DIV. LIQ./PATRI., SIMH3 P/VP, VTRU3 LIQ. CORRENTE, RAIZ4 P/VP, EGIE3 PSR, OPCT3 EV/EBIT). I read every `CVM` item's account value, `QT_ACAO_ORDIN`/`TOTAL` and DT_RECEB from the raw `dfp/itr_cia_aberta_*.zip`: 76 of 76 match. The 12 unmatched items are only the `QT_ACAO_PREF` column, which my reader did not index. BHIA3 LIQ. CORRENTE: 12 733 000 000 / 16 771 000 000 = 0.759, so 0.76, equal to the SI as cited.

---

## Discrimination Sensor (fix-round surface)

Scratch: a `git archive HEAD` copy with `.cache/pit-rebuild` symlinked read-only. Each mutant runs in a fresh copy of `scripts/ tests/ py/ docs/`. The control run has 3 environment errors (tests that need `data/raw` and `api/`, absent from the copy). "Survived" means no failure beyond those 3. The real-tree porcelain was empty before and after, and the cache listing is unchanged.

| # | Mutation | Result |
| - | -------- | ------ |
| G1 | `decide`: rule 6 yields to any cited reproducer (`exact_af` → `cited_af`) | ✅ Killed (`test_pit_rebuild_audit.py:148`, `:168`) |
| G2 | `classify_accounting:851`: `exact_af = best_af is not None` | ❌ Survived the suite; the regeneration differs in 14 lines: the 3 rule-(d) rows go back to `erro do Status Invest`, método 6/2 (V15) |
| G3 | `search_date:529`: every reproducer `exact = False` | ❌ Survived the suite; the regeneration differs in 17 lines: citation tails and the count sentence change, classes unchanged (V15) |
| G4 | `citation`: exact ones ranked by gap | ✅ Killed (`:57`) |
| G5 | `citation`: inexact ones ranked by age | ✅ Killed (`:42` class) |
| G6 | `dy_search`: equal-length tie goes to the latest window | ✅ Killed (`:111`) |
| G7 | `dy_search`: running sum not reset per start (non-window sums) | ✅ Killed (`:89` class) |
| G8 | `candidate_reads`: post-d versions read at d | ✅ Killed (`:121` class) |
| G9 | `dont_write_bytecode = False` | ✅ Killed (`:176`) |
| G10 | `match_clause`: never "igual a duas casas" | ✅ Killed (`:73` class) |
| G11 | DY class from the ticker suffix instead of ESPECI | ⚪ Survived; equivalent on this data: the only PNA ticker (COCE5) is in `DY_DECISIONS` and returns before the class is read |
| A1 | doc "São 90 linhas" → 91 | ✅ Killed (`test_pit_rebuild_artifact.py:366`) |
| A2 | doc "… 17 são iguais" → 18 | ✅ Killed (`:366`) |
| A3 | a row motive rewritten "o SI usa o ITR …" | ✅ Killed (`:375` and the citation recount) |
| A4 | header sentence "O SI soma janelas contíguas; …" added | ❌ Survived: `:375` reads only row motives, and its list misses "como o SI as soma" and "parece ser" (V13) |
| A5 | an exact citation relabelled as within tolerance | ✅ Killed (`:366`) |

**Result**: 17 mutants; 13 killed, 3 survived (G2, G3, A4), 1 equivalent (G11).

---

## Gaps

| ID | Severity | Where | Gap | Minimal correction |
| -- | -------- | ----- | --- | ------------------ |
| V13 | **blocking** (REVIEW.md: text in `docs/validation/` that claims more than the proof; rule (b)) | `docs/validation/pit-rebuild-stocks-audit.md:33`, `:31`, `:315`, `:16`; source `scripts/pit_rebuild_audit.py:1039`, `:1037`, `:84`, HEADER class list; test `tests/test_pit_rebuild_artifact.py:375` | Three sentences still state SI behavior. `:33` "linhas idênticas contam separadamente, como o SI as soma (casos MDNE3 e VLID3)" is causal and also false for MDNE3, where no window reproduces the SI on either date (`:308`, `:575`). `:31` justifies the search "porque pergunta de qual demonstrativo o SI tirou o valor". `:315` TGMA3: "o corte do SI parece ser pelo lastDatePrior, e não pela data-ex". The class definition `:16` does not list the DY window that TGMA3's `diferença de método` rests on. The no-cause test reads only row motives and lists neither phrase (mutant A4). | `:33`: "linhas idênticas contam separadamente (na VLID3, a janela que dá o DY do SI tem quatro linhas idênticas)". `:31`: "…porque mede quais demonstrativos ficam dentro da tolerância do valor do SI)". TGMA3: "o DY do SI (10,84 %) é igual, a duas casas, à janela de lastDatePrior em (d − 365, d] (cinco registros, 06/11/2025 a 06/08/2026); a janela do projeto, por data-ex em (d − 365, d], soma também 0,14 e 1,21 (14,87 %); a coincidência não mostra a origem". `:16`: add "ou a janela de proventos por lastDatePrior (TGMA3)". Extend `:375` to the whole document text, with "como o SI", "o corte do SI", "o SI tirou" and "parece ser". Regenerate. |
| V14 | **blocking** (text in `docs/validation/` contradicted by a row) | `docs/validation/pit-rebuild-stocks-audit.md:32` (last sentence), `:28` (rule 9); source `scripts/pit_rebuild_audit.py:1038`, `:1034` | `:32` says that when one filing reproduces by two views, "vale a primeira na ordem: como arquivado, depois a contagem". `citation` ranks by exactness and gap first, and ORVR3 EV/EBIT 08-07 (`:255`) cites the count view (31.8, exact) over the as-filed view (33.2) of the same ITR 2026-06-30 v2. Rule 9 says "citado o mais antigo", but method views are cited by the same rule (exact first, else closest). No row changes. | `:32`: "Quando o mesmo demonstrativo reproduz por mais de uma dessas visões, vale a regra acima (igual a duas casas, depois a mais próxima); empate: como arquivado, depois a contagem." `:28`: "citado pela regra de Citação". Regenerate. |
| V15 | nonblocking (off the criterion path; the code is correct, as the byte-for-byte regeneration and the 44/44 citation recount show) | `scripts/pit_rebuild_audit.py:529`, `:851` | The facts that feed `citation` and `decide` (`exact` per reproducer, `exact_af`) are computed inside `search_date`/`classify_accounting`, and no fixture reaches them (G2, G3). Only the regeneration detects a fault there, and the regeneration is not a gate. | Optional: extract the `exact`/`exact_af` computation into a small function and pin it with one fixture (SI 3.08, as-filed 3.16 and B3 3.08 → `diferença de método`). |

Nits (not counted): `scripts/pit_rebuild_audit.py:580` docstring and `tests/test_pit_rebuild_audit.py:105` test name say "as the SI sums them". The rows "igual a duas casas a patrimônio total" / "a contagem total" (`:270`, `:521`, AALR3 and RAIZ4 ROE) read "a" for "ao"/"à".

---

## Code Quality (fix round)

| Check | Status |
| ----- | ------ |
| Only touched files required for the gaps | ✅ generator, two test files, regenerated audit, spec r5, STATE. No `py/` change |
| Generator read-only on the cache, no bytecode | ✅ cache listing unchanged, no new `.pyc`, `--out` only |
| Rule = code | ✅ `decide` (`:784`) matches `Ordem das regras` 1..11. ❌ The Citação paragraph's view tie-break and rule 9's "mais antigo" (V14) |
| Tests map to ACs and discriminate | ✅ B-10 r5 citation, rule 6, windows, scope, wording, bytecode, recounts. ⚠️ V15, and A4 (V13) |
| Spec-anchored outcome | ✅ B-10 r5 and AUD-06 r5 match the code and the counts |
| Guidelines | `REVIEW.md` (claims beyond the committed proof; numbers equal the artifact; a new test fails when its behavior is removed) |

---

## Summary

**Overall**: ❌ Not ready. Two wording gaps remain in the document. Neither moves a number.

- **Closed**: V11, V12. V9 is closed for the rows, the citation, rules (d) and (e) and the numbers. Its residue is V13. V10's residue is V14.
- **New**: V13 blocking, V14 blocking, V15 nonblocking.
- **Criterion**: "cumprido", erro nosso 24. Unaffected.
- **Gates**: pit 495 OK; Build 1082 OK (2 skipped), exit 0.
- **Sensor**: 17 mutants; 13 killed, 3 survived (G2, G3, A4), 1 equivalent.
- **Lessons**: not recorded. The brief limits writes to `validation.md` and the feature `STATE.md`. Candidate lesson: "a no-cause guard over generated text must scan the whole document, headers included, not only the table cells".

**Next step**: an implementer applies the V13 and V14 wording in the generator's HEADER and in the TGMA3 motive, widens the `:375` guard, and regenerates. No user decision is needed.
