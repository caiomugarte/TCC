# STATE - pit-rebuild-stocks-fixes (issue #64)

Feature-local state. Project decisions stay in `.specs/STATE.md` `## Decisions`; this feature proposes **AD-033** there (share scale chosen after the B3 events, superseding part of AD-032 (4)).

## Decisions

- Issue `## Decisões` (DEC-1..DEC-5 in `spec.md`) are closed with the user and are not reopened.
- B-1..B-12 (`spec.md` "Assumptions & Open Questions") are the author's defaults, waiting for the spec review. B-3 is the one with the widest effect: it changes `share_scale.scaled` for companies with a split between filing and B3 retrieval.
- r2 (2026-11-12), after the spec review iw-64-judge-spec-r1 (CHANGES_REQUIRED, blocking 2, non-blocking 8), all ten items taken:
  - S1: new RPT-06; T14 rewrites the rule texts the report prints (`report.py` `SHARE_SCALE_RULE`, `NON_PIT_INPUTS`; `indicators.py` `FORMULAS["VALOR DE MERCADO"]` and docstring) and is now heavy; T21 revises the √10 assertion of the artifact test (√10 stays: it still holds for a change with no B3 event).
  - S2: B-10, AUD-02, ADJ-04: evidence per indicator; DY rows cite the dividend record and the COTAHIST close, `filing` null.
  - S3, S4: B-3, new SHR-08, AD-033: `M_all` stops at the B3 retrieval date `R` and reads the ISINs the ticker had after `d`. B-2 goes further than the review asked: `M_d` also reads every ISIN of (`DT_REFER`, `d`], because the ISIN at `d` alone drops an old-ISIN event when the ISIN changed inside the window; events listed under two ISINs count once. Remaining limit (AD-033): an ISIN change between the last loaded session and `R`.
  - S5: ADJ-05 prints part 1 only; new AUD-06 puts the whole criterion in the audit (T24).
  - S6: B-10 and new AUD-05: the audit names what the worst list leaves out and summarizes `missing` per reason; zero references are not counted in the JSON and are named as a limit, not counted.
  - S7, S8: design references renumbered (T20 MBRF3, T21 cache copy); gates diff against the merge base, v2 read from `ebaf1fd`.
  - S9: B-12, ADJ-01: `previous_report` and one line under "## Veredito".
  - S10: SHR-03 lists a class when `abs(M − 1) > 1e-9`.

## Handoff

- **Feature**: pit-rebuild-stocks-fixes (issue #64) - `.specs/features/pit-rebuild-stocks-fixes/`
- **Phase / Task**: Specify, Design and Tasks done, r2 after the spec review iw-64-judge-spec-r1 (2026-11-12). 44 requirements, 24 tasks, 32 weighted. Next: spec re-review by the orchestrator, then Execute in five batches, one per phase (T1-T7: 7, T8-T13: 7, T14-T18: 7, T19-T22: 7, T23-T24: 4 weighted), then the Verifier (author != verifier).
- **Completed**: none (no production code or test written).
- **In-progress (file:line)**: none.
- **Next step**: re-review `spec.md` (B-2, B-3, B-4, B-10, B-12, SHR-03, SHR-08, ADJ-01, ADJ-04, ADJ-05, AUD-02, AUD-05, AUD-06, RPT-06) and AD-033; on approval, dispatch Phase 1 (T1-T7).
- **Blockers**: none. No task needs the network; T20 stops and asks if its evidence needs a request outside the cache.
- **Uncommitted files**: none.
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline**: pit suite 322 OK at `ebaf1fd` (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`). Root suite 909 OK, 2 skipped at `53fde43` (#59 Handoff), not rerun here.
- **Real data**: the #59 cache is at `/home/caio/.t3/worktrees/TCC/t3-issue-59-d1f86a5f2296a99f/.cache/pit-rebuild` (362 MB, 585 manifest entries). This worktree has no `.cache/pit-rebuild`; T21 step 1 copies it with hard links. The 2026-11-06 CSV is `/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv` (outside the worktree, read only).
- **Seen during design (evidence, not a conclusion)**: `b3_company_MBRF.json` has `tradingName` `MARFRIG` with no `stockDividends`; `b3_cash_MARFRIG_p1.json` has 13 records, the latest with `lastDatePriorEx` 18/09/2025, which is outside the 2026-11-05 window. MBRF3 is DY 0.00 against 18.41 only on 2026-11-05 in the committed v2 JSON. T20 checks this first.
- **Pitfalls carried from #59** (`.specs/STATE.md` Handoff of 2026-11-10):
  - Commit with `git commit -F <file>`: the shell hook rewrites a `-m` message containing the word "make".
  - `check_commit.py` warns above 72 characters.
  - `tests/fixtures/pit_rebuild/cotahist.txt`: no session between 2026-08-21 and 2026-08-24, none after 2026-11-06 (calendar tests).
  - `tests/test_pit_rebuild_cli.py` sets `sys.dont_write_bytecode`; the isolation guard reads docstrings, so `py/` must not contain the literal `baselines/`.
  - The Markdown is rendered in a fixed order (`_canonical`, sorted keys); keep new lists sorted by their keys.
  - In a gate, use `git diff --exit-code`, not `git diff --stat`.
  - A mutation copy of `py/` and `tests/` also needs `api/app/services/premium_policy.py`, `docs/validation/` and `data/raw/status_invest_fundamentals.csv`.
  - `mkstemp` writes the report files with mode 0600; git's file mode is unaffected.
