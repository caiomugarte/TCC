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
- **Phase / Task**: Execute. Lotes 1 e 2 de 5 concluídos (Phases 1 e 2, T1-T13). Próximo: lote 3 (Phase 3, T14-T18), depois os lotes 4 e 5 e o Verifier (autor != verificador).
- **Completed** (commits locais, um por tarefa, Conventional Commits com `(#64)`):
  - T1 `1764831`: `cotahist.isin`/`share_class` leem a sessão em `d`; `isins` (B-2, B-3); `class_isins` (I_d, S12); call sites de `cli.py` passam `d`.
  - T2 `dd08ff6`: TTM com o ITR do ano anterior quando `PENÚLTIMO` é ausente ou 0 (TTM-01..03); `Dated.prior_from_itr`.
  - T3 `0e203fe`: `newest_period_end`, `newest_filing`, `STALE_FILING_DAYS = 270`.
  - T4 `d0b8070`: `Shares.dt_refer`; `previous_shares` (B-4).
  - T5 `af7f5d0`: `FilingStore.without` (B-9); helper de teste `inline_filings`.
  - T6 `d51e425`: `event_ex_date`, `class_events` (B-1, B-2); `B3Company.retrieved_on`.
  - T7 `a39c8af`: `py/pit_rebuild/shares.py` com `adjust` (AD-033: M_all até `R`, M_d até `d`, sem-ISIN, epsilon).
  - T8 `1664eb9`: `shares.suspect` e `SUSPECT_CHANGE = 0.20` (SHR-05..SHR-07, B-4); 7 testes.
  - T9 `9d1f5d9`: `fields.ACCOUNTING_FIELDS` (15, B-7), `fields.ADJUSTED_FIELDS` (14, B-8) e `fields.NON_ACCOUNTING_FIELDS`.
  - T10 `816c8f5`: `indicators.STALE_FILING_REASON` e `mask_accounting` (STL-01, STL-02).
  - T11 `276b1c0`: `compare.is_si_lag` (B-9; LAG-01, LAG-03, LAG-04).
  - T12 `3f630f3`: `report.build(adjusted=...)` e `_adjusted`: bloco `adjusted_verdict` com critério parte 1 (ADJ-02..ADJ-07, AUD-05, LAG-02).
  - T13 `afb8111`: seção "## Veredito ajustado (#64)" no Markdown (ADJ-01, ADJ-05, LAG-02). Gate Build verde.
- **In-progress (file:line)**: none.
- **Next step**: despachar o lote 3 (T14-T18). T14 entra em `py/pit_rebuild/report.py` (SCHEMA v3, AMENDED_BY, PREVIOUS_REPORT, linha B-12, INVESTIGATIONS, as cinco listas do mapa e os textos de AD-033 em `SHARE_SCALE_RULE`, `NON_PIT_INPUTS`, e em `indicators.py`). T15 liga `shares.adjust` e `shares.suspect` em `cli.py`.
- **Blockers**: none.
- **Uncommitted files**: none.
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline**: pit suite 410 OK (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`; 371 no fim do lote 1). Suíte raiz 997 OK, 2 skipped (`/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"`, ~3 min, na árvore de T13).
- **Lote 2: escolhas que o lote 3 precisa conhecer** (fora da SPEC; cada uma com teste):
  - `shares.suspect(prev, prev_factor, current, current_factor, events, isins, next_session)` devolve `{dt_refer_prev, total_prev, dt_refer, total, ratio}` sem `cd_cvm`: o chamador acrescenta. Totais = `(on + pref) × fator`, arredondados a ação inteira; `ratio` com 4 casas, calculada antes do arredondamento. Total ≤ 0 não é comparado (SHR-05). Datas são `date`, como em `adjust`.
  - `report._adjusted`: `adjusted[iso]` = `{excluded, fields: {campo: FieldStats do grupo não financeiro}, si_lag: {campo: [ticker]}, worst_details: {campo: {ticker: {filing, previous_value, si_lag}}}}`. `filing` é um mapping `{kind, dt_refer, versao, dt_receb}` (datas `date` ou ISO); o chamador converte `cvm.Filing` (T19). `previous_value` é arredondado a 2 casas dentro de `_adjusted`. `_adjusted` passa o `si_lag` recebido; o chamador deve passar falso em DY, que não lê arquivamento (B-9).
  - `is_si_lag`: DY em branco tem referência 0,00 e a regra de meio ponto de DY, como `summarize` (A-4). É assim que o "zero rule" de T11 foi lido.
  - `SCHEMA` continua `pit-rebuild-stocks/v2` até T14.
- **S12 (fechado no lote 1)**: `cotahist.class_isins(q, t, DT_REFER, d)` é o I_d e inclui `isin(q, t, DT_REFER)`, porque `isins` deixa a sessão de DT_REFER de fora. Testes: `ClassAtDTests` (`test_pit_rebuild_cotahist.py`) e `IsinSetTests` (`test_pit_rebuild_shares.py`). T15 monta I_d com `class_isins` e I_all com `class_isins` mais `isins(q, t, d, None)`.
- **Real data**: the #59 cache is at `/home/caio/.t3/worktrees/TCC/t3-issue-59-d1f86a5f2296a99f/.cache/pit-rebuild` (362 MB, 585 manifest entries). This worktree has no `.cache/pit-rebuild`; T21 step 1 copies it with hard links. The 2026-11-06 CSV is `/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv` (outside the worktree, read only).
- **Seen during design (evidence, not a conclusion)**: `b3_company_MBRF.json` has `tradingName` `MARFRIG` with no `stockDividends`; `b3_cash_MARFRIG_p1.json` has 13 records, the latest with `lastDatePriorEx` 18/09/2025, which is outside the 2026-11-05 window. MBRF3 is DY 0.00 against 18.41 only on 2026-11-05 in the committed v2 JSON. T20 checks this first.
- **Pitfalls carried from #59** (`.specs/STATE.md` Handoff of 2026-11-10):
  - Commit with `git commit -F <file>`: the shell hook rewrites a `-m` message containing the word "make".
  - `check_commit.py` warns above 72 characters.
  - `tests/fixtures/pit_rebuild/cotahist.txt`: no session between 2026-08-21 and 2026-08-24, none after 2026-11-06 (calendar tests). Lote 1 monta sessões em zip temporário (`cotahist_line`), sem tocar na fixture.
  - `tests/test_pit_rebuild_cli.py` sets `sys.dont_write_bytecode`; the isolation guard reads docstrings, so `py/` must not contain the literal `baselines/`.
  - The Markdown is rendered in a fixed order (`_canonical`, sorted keys); keep new lists sorted by their keys.
  - In a gate, use `git diff --exit-code`, not `git diff --stat`.
  - A mutation copy of `py/` and `tests/` also needs `api/app/services/premium_policy.py`, `docs/validation/` and `data/raw/status_invest_fundamentals.csv`.
  - `mkstemp` writes the report files with mode 0600; git's file mode is unaffected.
  - pytest não está instalado no venv nem no sistema: o runner é `unittest` (o brief do lote 1 citava pytest; o gate do tasks.md é o correto).
  - `isins` (B-2) não inclui a sessão de `after`: quem monta I_d precisa de `isin(q, t, DT_REFER)` (S12, já no `class_isins`).
  - `unittest` mostra um erro de importação (SyntaxError) como `errors=1` do módulo: um mutante com sintaxe inválida "mata" o teste sem provar nada. Antes de contar uma morte, veja o traceback.
