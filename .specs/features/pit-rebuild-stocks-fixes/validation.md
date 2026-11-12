# Validation: pit-rebuild-stocks-fixes (issue #64) - FAIL (iteration 3 of 3, escalate)

**Date**: 2026-11-12
**Spec**: `.specs/features/pit-rebuild-stocks-fixes/spec.md` (r4; B-10, AUD-02 and AUD-06 amended for V6..V8)
**Diff range**: `ebaf1fd..6ee005a` (feature). The fix round is `ad21302..6ee005a`: `b8bdfd3` generator and fixture tests, `44768f9` search without restrictions, regenerated audit and spec r4, `81912e4` origin-table test, `6ee005a` state. `py/` is unchanged since `1def298`.
**Verifier**: independent sub-agent (author ≠ verifier), clientRequestId `iw-64-verify-r3`. Iteration 2 report: commit `ad21302`.

**Verdict: FAIL.** V6, V7 and V8 are closed. The rule the audit states is the rule `scripts/pit_rebuild_audit.py` runs, with no hidden restriction. The 80 accounting `sem base` rows have no reproducing view, F0 included, and HBOR3 DY has no reproducing subset. The generator regenerates the document byte for byte without writing to the cache. The origin table is tested. One new blocking gap remains (V9). The per-row text of the 47 reproduced accounting rows says "o Status Invest reproduz o valor com <filing>", which presents a match within the 5 % tolerance as what the SI did. 38 of those 47 matches are not exact. 8 rows skip an exact reproducer to cite an older, inexact one, and BHIA3 LIQ. CORRENTE cites the DFP 2019. The 4 DY rows marked as reproduced cite subsets of records from 2007 to 2024. With 38 to 72 lines, such subsets reach about 98 % of all two-decimal DY values. The same tolerance matches change 3 classes from `diferença de método` to `erro do Status Invest` (EGIE3 PSR on both dates, OPCT3 EV/EBIT on 2026-11-05). In all 3 rows the B3 total count explains the SI value more closely. The criterion stays "cumprido": both classes count as explained, and `erro nosso` stays at 24.

---

## Gate Check

- **Pit suite**: `python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"` gives `Ran 480 tests`, `OK` (473 + 6 fixture tests in `tests/test_pit_rebuild_audit.py` + 1 origin-table test).
- **Build gate**: `/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"` (foreground, exit 0) gives `Ran 1067 tests in 181.568s`, `OK (skipped=2)`. No test was deleted (1060 → 1067).
- `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` is empty. `validate_spec.py` reports 0 errors and 0 warnings on spec r4.
- **Skipped**: the same 2 root-suite skips as in iterations 1 and 2, both outside this feature.

---

## Re-verification of V6..V8

| Gap | Status | Evidence |
| --- | ------ | -------- |
| V6 (blocking): the stated rule differs from the classifier | ✅ Closed | **Rule = code.** Version scope: `filings_of` (`scripts/pit_rebuild_audit.py:402-409`) takes every indexed version of every DFP and ITR with `DT_RECEB ≤ R`. The newest `DT_RECEB` among the 49 companies is 2026-11-05, so this is every cached version. There is no age filter and no STL-01 mask: `company_values` (`:464-499`) calls `cli._company` on the version store, and mutant S5 below shows that a 270-day filter would change 49 lines of the document. The views are the four `METHODS` (`:75`) plus `COUNT` when the version has a composition (`:519-524`), as stated in `audit.md:31` ("Reproduzir"). The tolerance is `compare.within` (CMP-01) (`:157-159`), as stated at `:31` and `:51`. DY: `b3_records` (`:556-575`) reads every page of the trading name's B3 cash file and keeps lines of the ticker's class (PN for 4/5/6, ON for 3, as stated at `audit.md:21`). It keeps duplicates and lines with no ex-date, applies no window, and tests the closes of d and d − 1 with an exact two-decimal check (`:609`). Oldest: `oldest` (`:152-154`) sorts by DT_RECEB, DT_REFER, VERSAO among as-filed and COUNT views, and method views are used only when no as-filed view reproduces (`:807-809`, `:860-869`). This matches spec B-10 r4 (`spec.md:60`) and rules 8 and 9 (`audit.md:27-28`); for the paragraph at `audit.md:32`, see V10. **`sem base` holds.** A dump of the generator's own search (all views, F0 included, scratchpad `dump.json`): 0 of 80 accounting `sem base` rows have any reproducer, and no index-only version reproduces anything (131 index-only versions). Independent check from the raw CVM zips, with no use of the generator (`indep.py`): 13 rows (VTRU3 LIQ. CORRENTE ×2, DIV. LIQ./PATRI. ×2, P/VP ×2; TIMS3 DIV. LIQ./PATRI. ×2, P/VP ×2; ONCO3 P/VP; BHIA3 P/VP ×2). For every version, the balance ratio as filed and with PL total, the count view, and the full view are each outside 5 %. Only the two BHIA3 P/VP rows match, at 0.24 vs 0.23 and 0.21 vs 0.22, and only when one version's PL is combined with the count of d. `audit.md:50` declares that cross-filing combination untested. The version counts in the motives match the zips: VTRU3 13 + 7 = 20, TIMS3 15 + 2 = 17, BHIA3 15 + 1 = 16. HBOR3 DY (`audit.md:311`): all 65 535 subsets of its 16 ON lines were enumerated, and none gives 1.05 % on the close of 2026-08-07 (1.92) or 2026-08-06 (1.93). **Oldest cited**: 47 of 47 cited reproducers equal the oldest as-filed reproducer other than F0 (method view in the 4 rule-9 rows). |
| V7 (nonblocking): the classifier was not in the repository | ✅ Closed | `python3 scripts/pit_rebuild_audit.py --out <scratchpad>/audit_regen.md` (exit 0, 72.6 s): `cmp` against `docs/validation/pit-rebuild-stocks-audit.md` is identical. The listing of `.cache/pit-rebuild` (586 files, with names, sizes and mtimes) is the same before and after the run. Real-tree `git status --porcelain` is empty, and no worktree file is newer than the run marker. The script also checks its own cache digest (`:946-953`, `:1216-1218`). Fixture tests: `tests/test_pit_rebuild_audit.py:24-37` (tolerance, oldest key) and `:40-59` (DY subsets, d − 1 close, identical lines). Mutants S1..S4 below are killed. The "131 das 876 versões de 49 empresas" figure (`audit.md:53`) equals the generator's count, and the per-company counts checked above equal the zips. See V11 and V12 for what the tests do not cover. |
| V8 (nonblocking): the origin table was not tested | ✅ Closed | `tests/test_pit_rebuild_artifact.py:390-392` recounts the table from the row motives. Mutant A5 (52/41 → 53/40) and mutant A7 (a reproduced row's motive given a `nenhum dos` prefix) are killed by that test. The table (`audit.md:69-70`: 52/41, 83/40) equals my own parse of the rows. |

### Numbers (item 5)

Independent parse of the 280 rows: 2026-08-07 has 17 `erro nosso`, 93 `erro do Status Invest`, 6 `diferença de método`, 24 `dentro da tolerância` and 0 `não classificado`. 2026-11-05 has 7, 123, 2, 8 and 0. These equal the Resumo (`audit.md:61-63`). The 81 `sem base` rows (`audit.md:81`) equal 41 + 40. The criterion line `:76` says "cumprido", erro nosso 24, with part 1 from the JSON and part 2 at 0 `não classificado`, matching AUD-06. "47 citações, 9 exatas" (`audit.md:51`) equals the dump (47 rows with a cited reproducer, 9 equal at two decimals). `docs/validation/README.md:483-484` points to the audit for part 2 and the whole criterion. That is coherent, though it does not name the generator.

---

## V9: per-row text presents a tolerance match as what the SI did (item 2)

**Accounting rows (47).** `reproduced_motive` (`scripts/pit_rebuild_audit.py:764-768`) writes "o Status Invest reproduz o valor com <filing>, recebido em …: com os demonstrativos desse filing (SI x, recalculado y; n demonstrativo(s) reproduzem, o mais antigo é citado)". The subject is the SI, so the sentence says the SI produced its value from that filing. The document's own definition (`audit.md:31`) is weaker: a filing reproduces the SI value when the recalculated value is within CMP-01. `audit.md:49` and `:51` say the evidence "não prova o que o SI fez", so the row text contradicts the limits. `:49` also says the SI number "não sai do cache", which is false for the 135 reproduced rows. Facts from the dump:

| What | Rows |
| ---- | ---- |
| Cited value equals the SI value at two decimals | 9 of 47 |
| Cited value only within 5 % / 0.5 p.p. | 38 of 47 |
| An exact reproducer exists, but an older inexact one is cited | 8: BHIA3 LIQ. CORRENTE 08-07 (`audit.md:114`, DFP 2019-12-31 v1 at 0.79 vs SI 0.76, while ITR 2024-06-30 v1 is exact); BHIA3 MARG. LIQUIDA 08-07 and 11-05 (`:191`, `:455`, DFP 2019, while ITR 2024-09-30 is exact); BHIA3 DIV. LIQ./PATRI. 08-07 and 11-05; CAML3 DIV. LIQ./PATRI. 11-05 (`:416`); ORVR3 EV/EBIT 08-07 (`:255`); SIMH3 P/VP 08-07 (`:240`) |
| Citations of a filing more than 5 years older than d | BHIA3 ×4 (DFP 2019), VIVA3 PSR (`:529`, DFP 2020), PMAM3 PSR (`:533`, DFP 2020 v3) |

A 2019 DFP within 4 % of a 2026 SI value is a coincidence of the search: up to 14 versions × 5 views per row are tested against a 5 % band, and the oldest match wins. It is not evidence of the SI's input. The text states it as the SI's input.

**Class changes from the same matches (3 rows).** Rule 6 (`audit.md:25`, code `:842-846`) moves a B3-total-count `diferença de método` to `erro do Status Invest` when any other filing reproduces as filed. Mutant S6 (rule disabled) shows exactly these 3 rows move. None of them is exact:

| Row | SI | Cited filing gives | B3 total count gives | Same company, same date |
| --- | -- | ------------------ | -------------------- | ----------------------- |
| EGIE3 PSR 08-07 (`audit.md:270`) | 3.08 | DFP 2023 v1: 3.16 (2.6 %) | 2.48 / 0.81 ≈ 3.06 (0.6 %) | — |
| EGIE3 PSR 11-05 (`:536`) | 2.76 | DFP 2023 v1: 2.83 (2.5 %) | 2.22 / 0.81 ≈ 2.74 (0.7 %) | — |
| OPCT3 EV/EBIT 11-05 (`:521`) | 14.12 | ITR 2025-06-30 v2: 13.75 (2.6 %) | reproduces (rule 6 applies) | OPCT3 P/VP and PSR of 11-05 stay `diferença de método` by the same B3 count |

So the "reproduction" changes classes. In the r2 document these 3 rows were `diferença de método`, with a mechanism that fits better and that explains the sibling fields of the same ticker. **The criterion does not change**: both classes are explained (AUD-06, part 2), `erro nosso` stays at 24, and part 1 is the JSON's.

**DY rows (4).** `classify_dy` (`:734-736`) writes "o DY do SI (x %) é exato com N registro(s) da B3 do cache (…)". As arithmetic it is true. I checked the sums: CLSC4 0.534024 / 147.99 = 0.36 %, MBRF3 2.877672 / 15.63 = 18.41 %, EQTL3 0.451381 / 33.18 = 1.36 %. But the subsets are CLSC4 2012 + 2022 (`:314`), MBRF3 2010 + 2024 (`:572`), JSLG3 15 records from 2011 to 2024 (`:573`) and EQTL3 2007 to 2018 (`:581`). The witness rule (`_witness`, `:615-623`) leaves the newest lines out first, which picks the least plausible subset. On a 0.0001 R$ grid, subsets of EQUATORIAL's 38 ON lines reach 1 956 of the 2 000 DY values from 0.01 % to 20.00 %. CELESC's 72 PN lines reach 1 982. MARFRIG's 13 lines reach 879. With that density, a reproducing subset says nothing about the source. The motive does not say so. These rows count as "reproduzido por um demonstrativo ou provento" (`audit.md:67-70`), and their evidence items are the 2007..2024 records. `audit.md:52` only says that the cited subset may not be the one the SI used.

**Classification**: Important under REVIEW.md ("Um documento que afirma mais do que a prova commitada mostra"; `docs/validation/` text is the result). It is **blocking**, as V6 was. It changes 3 classes (method → erro do SI) and the reproduced/`sem base` split. It does not change the criterion.

**Minimal correction (no user decision needed)**: (1) In `reproduced_motive`, state the measured fact: "o valor do SI (x) fica dentro da tolerância (CMP-01) do recalculado com <filing> (y)" or "é igual, a duas casas, ao recalculado com <filing>". Add "N demonstrativos ficam dentro da tolerância; cita-se o mais antigo; a coincidência não mostra que o SI usou esse demonstrativo". (2) In the DY motive, say "a soma de N registros (lastDatePrior A a B) dá o DY do SI a duas casas; com M registros, somas de subconjuntos alcançam quase todo DY de duas casas, então a soma não indica a origem". (3) At `audit.md:15`, define `reproduzido` as "um demonstrativo ou subconjunto fica dentro da tolerância", not as a source. At `:49`, drop "o número do SI não sai do cache" for reproduced rows. (4) Rows that depend on rule 6 need a decision, so the minimal option is to keep their class and use the same wording. Classes stay `erro do Status Invest` under the user's rule either way. **Recommended, and for the user (STATE decisions 1 and 3)**: cite an exact reproducer before an inexact one, or the closest one; let the rule 6 exception apply only to an exact match, which returns EGIE3 ×2 and OPCT3 to `diferença de método`; and for DY, accept only contiguous date windows of records, which would move CLSC4, JSLG3, EQTL3 and likely MBRF3 to `sem base`, still `erro do Status Invest`.

---

## Discrimination Sensor (fix-round surface)

Scratch: `git archive HEAD` extracted under the session scratchpad. Generator mutants that need the cache use a read-only symlink to `.cache/pit-rebuild`. The real worktree `git status --porcelain` was empty before and after, and the cache listing is unchanged.

| # | Mutation (copy) | Result |
| - | --------------- | ------ |
| A5 | origin table 52/41 → 53/40 | ✅ Killed by `test_pit_rebuild_artifact.py:390` |
| A6 | origin table 83/40 → 84/39 | ✅ Killed |
| A7 | a reproduced row's motive gets the `nenhum dos` prefix | ✅ Killed (`:390` and the `sem base` shape test) |
| A8 | "São 81 linhas `sem base`" → 82, and "47 … 9 exatas" → 12 | ❌ Survived (V11) |
| S1 | `reproduces` uses a 10 % band | ✅ Killed (`test_pit_rebuild_audit.py:24`) |
| S2 | `oldest` key DT_REFER before DT_RECEB | ✅ Killed (`:32`) |
| S3 | DY search ignores the d − 1 close | ✅ Killed (`:46`) |
| S4 | DY search drops identical lines | ✅ Killed (`:56`) |
| S5 | search skips versions with a period end > 270 days before d (the r2 restriction) | ❌ Survived the suite. The regenerated document differs in 49 lines (`sem base` 60/50) (V11) |
| S6 | rule 6 exception disabled | ❌ Survived the suite. The regenerated document differs in 14 lines (method 7/4) (V11) |

---

## Gaps

| ID | Severity | Where | Gap | Minimal correction |
| -- | -------- | ----- | --- | ------------------ |
| V9 | **blocking** (REVIEW.md: text in `docs/validation/` that claims more than the proof is Important) | `scripts/pit_rebuild_audit.py:764-768`, `:734-736`, `:997`, `:1031`; `docs/validation/pit-rebuild-stocks-audit.md:15`, `:49`, the 47 reproduced accounting rows (e.g. `:114`, `:191`, `:255`, `:270`, `:455`, `:521`, `:529`, `:536`) and the DY rows `:314`, `:572`, `:573`, `:581`; `spec.md:60` (B-10 r4 rule 6 and oldest citation) | The rows say the SI used a filing or records when the proof is a match within 5 % (38 of 47) or a subset sum in a set dense enough to hit almost any value. 3 classes change from method to erro do SI on such matches, while the B3 count fits better. The criterion is unchanged. | Reword the motives and `:15` and `:49` as measured facts ("dentro da tolerância de", "igual a duas casas a", "a coincidência não mostra a origem"), then regenerate. For the user: exact-first citation, rule 6 only on an exact match, DY only on contiguous windows. |
| V10 | nonblocking | `docs/validation/pit-rebuild-stocks-audit.md:32` ("Mais antigo"), `:23` (rule 4) | The paragraph says the oldest of all reproducers is cited and that LL/PL views come before the count view. The code and spec B-10 r4 cite the oldest as-filed or count view first and a method view only when there is none (`:807-809`). ORVR3 P/VP 08-07 (`:239`) cites ITR 2026-06-30 v2 (count), while ITR 2024-09-30 with PL total is older. Rules 8 and 9 (`:27-28`) state it correctly. | One sentence at `:32` and `:23`: "entre as visões como arquivadas e de contagem; a visão de método só quando nenhuma dessas reproduz". |
| V11 | nonblocking (off the criterion path; the code is correct, as shown by the regeneration and the dump) | `tests/test_pit_rebuild_audit.py`; `tests/test_pit_rebuild_artifact.py` | The search scope (S5) and the rule order (S6) have no test. Only the byte-for-byte regeneration detects them, and it is not part of a gate. The sentences "São 81 linhas `sem base`" (`audit.md:81`) and "47 … 9 exatas" (`:51`) are not tested (A8). | Add a fixture test of `search_date` and the rule order with a two-version store (an old version that reproduces must be found and must win over the B3 count only per the chosen rule 6), and recount the `:81` sentence from the row motives. |
| V12 | nonblocking | `scripts/pit_rebuild_audit.py:35-40`; `audit.md:5`; `spec.md:60` ("writes only its `--out` file") | On a checkout with no bytecode cache, the run writes `py/pit_rebuild/__pycache__/*.pyc` (checked on a fresh `git archive` copy: `--help` alone writes them). They are gitignored and harmless, but the claim is literal. | Set `sys.dont_write_bytecode = True` before the `pit_rebuild` imports (the AD-013 pattern of the test suite). |

---

## Code Quality (fix round)

| Check | Status |
| ----- | ------ |
| Only touched files required for the gaps | ✅ new generator and fixture tests, regenerated audit, spec r4, one artifact test, STATE. No `py/` change |
| Generator read-only on the cache | ✅ offline `Cache`, digest check, listing unchanged |
| Tests map to ACs | ✅ AUD-02/AUD-04 origin table `:390`; B-10 r4 tolerance, oldest key and DY subset fixtures. ⚠️ V11 |
| Spec-anchored outcome | ⚠️ The spec defines reproduction as a CMP-01 match. The tests assert that definition correctly. The defect is in what the document's sentences infer from it (V9). |
| Guidelines | `REVIEW.md` (claims beyond the committed proof; numbers equal the artifact) |

---

## Summary

**Overall**: ❌ Not ready. This is iteration 3 of 3, so the gaps go to the user.

- **Closed**: V6, V7, V8.
- **New**: V9 blocking; V10, V11, V12 nonblocking.
- **Classes and criterion**: the criterion is "cumprido" and is unaffected. 3 classes (EGIE3 PSR ×2, OPCT3 EV/EBIT) depend on the rule 6 exception and on within-tolerance matches.
- **Gates**: pit suite 480 OK; root suite 1067 OK, 2 skipped.
- **Sensor**: 10 mutants: 7 killed, 3 survived (A8, S5, S6 → V11).
- **Lessons**: not recorded in this round. The brief limits writes to `validation.md` and the feature `STATE.md`. Candidate lesson: "a reproduction search over many versions under a tolerance band finds coincidences. Cite exact matches first, and phrase a match as a measurement, not as the reference's source."

**Next step (user)**: decide STATE decisions 1 (exact-first or tolerance citation) and 3 (rule 6 exception), and the DY subset scope (decision 2: contiguous windows or all lines). Then an implementer applies the V9 wording, and V10 to V12 if wanted, and regenerates.
