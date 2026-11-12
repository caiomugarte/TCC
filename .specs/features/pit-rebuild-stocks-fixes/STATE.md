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
- **Phase / Task**: Execute. Lotes 1, 2 e 3 de 5 concluídos (Phases 1, 2 e 3, T1-T18). Próximo: lote 4 (Phase 4, T19-T22), depois o lote 5 (Phase 5, T23-T24) e o Verifier (autor != verificador).
- **Completed** (commits locais, um por tarefa, Conventional Commits com `(#64)`):
  - T1-T13 (lotes 1 e 2): `1764831` ... `afb8111`.
  - T14 `2339bfe`: schema v3 (`amended_by`, `previous_report`, linha B-12 em "## Veredito"), "## Investigações" (`INVESTIGATIONS`, vazio), as cinco listas do mapa e os textos AD-033 (RPT-06).
  - T15 `c76d95b`: `_company` usa `shares.adjust` (I_d e I_all dos tickers A-17, eventos da B3, R de `Cache.retrieved_at`); `shares.suspect` com `previous_shares`; `share_events` e `share_count_suspect` no mapa.
  - T16 `8b010fe`: filing com mais de 270 dias (idade a partir de d) zera os 15 campos contábeis com `stale_filing`; também no `--filings-cutoff`.
  - T17 `1240497`: `ttm_prior_from_itr` (LL, 3.01, 3.05, 3.08 com o ITR do ano anterior).
  - T18 `d0ccd8b`: teste ISN-03 por `main`; nenhuma correção de call site (T1 já cobria). Gate Build verde.
  - Fora das tarefas, achado no gate do T17: `e434054` fecha a resposta HTTP de erro que o cache descarta (ResourceWarning).
- **In-progress (file:line)**: none.
- **Next step**: despachar o lote 4 (T19-T22). T19 entra em `cli._company`/`_report_object` (passe do filing anterior para `si_lag`, B-9); T20 lê o cache real (MBRF3); T21 regenera `docs/validation/` do cache real; T22 README. Depois o lote 5 (T23-T24) e o Verifier.
- **Blockers**: none.
- **Uncommitted files**: none.
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline** (fim do lote 3): pit suite 444 OK (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`, ~2 s). Suíte raiz 1031 OK, 2 skipped (`/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"`, ~3 min). `compileall` ok. `git diff --exit-code $(git merge-base HEAD origin/develop) -- data snapshots baselines py/core api py/config.py py/pit_validation.py` vazio.
- **Lote 3: escolhas que o lote 4 precisa conhecer** (fora da SPEC; cada uma com teste):
  - Os artefatos commitados continuam v2 até T21. O teste de artefato passa porque cada seção nova só aparece quando a chave existe no objeto (um v2 renderiza igual). T21 regenera e RPT-01 passa a valer.
  - `_Company` ganhou `share_events`, `no_isin`, `suspect`, `period_end` e `ttm_prior`. A regra de contagem está em `_counts` (devolve `_Counts`). `_company` manteve a assinatura: T19 pode reusá-la para o passe do filing anterior.
  - `_is_stale(period_end, day)`: a idade usa `day` (d) e o filing usa `filings_day` (A-22 (b)).
  - `render_markdown`: "## Investigações" aparece sempre no objeto v3 ("Nenhuma." com `investigations: []`). T20 preenche `report.INVESTIGATIONS`.
  - `Cache.retrieved_at(name)` lê o manifest em UTC; `cli._retrieved_on` converte para o dia de São Paulo (UTC−3). O `populate_cache` de teste grava o relógio: fixe com `pin_retrieval` (`tests/test_pit_rebuild_cli.py`).
  - Universo de shape (`write_shape_universe`): ganhou BONU, LATE, JUMP, OLDF, ANTE, TTMX, TTMZ e ISNX. A B3 total do SBSP é 500 M (pós-desdobramento). `universe_routes` serve ITR de `universe/itr/<ano>/` (pasta ausente = zip vazio).
  - Expectativas da universe que a regra de 270 dias mudou (corpo do commit `8b010fe`): em 2026-08-07, REFL3 usa o filing de 2024 (584 dias) e o P/L vai para 10 de 15; em 2026-11-06, todo P/L fica em branco (0 de 15), e o grupo financeiro vira "não reconstruível" (BANC3, 310 dias). Respeite isso se T19-T21 mexerem na universe.
  - Os textos de regra e as contas de TTMX (LL 130 M, REC 580 M, P/L 7,69, PSR 1,72) e de ISNX (DY 5,00 em 2026-07-20 e 3,33 em 2026-08-07) estão em `tests/test_pit_rebuild_cli.py`.
- **Lote 2, ainda vale para T19**: `report._adjusted(given, iso_dates)` espera por data `{excluded, fields: {campo: FieldStats do grupo não financeiro}, si_lag: {campo: [ticker]}, worst_details: {campo: {ticker: {filing, previous_value, si_lag}}}}`; `filing` é `{kind, dt_refer, versao, dt_receb}` (date ou ISO) e `previous_value` é arredondado a 2 casas dentro de `_adjusted`. O chamador passa `si_lag` falso para DY (DY não lê arquivamento, B-9). `compare.is_si_lag(field, raw, calc, calc_prev)`: DY em branco tem referência 0,00 e a regra de meio ponto de DY, como `summarize` (A-4).
- **Seen during design (evidence, not a conclusion, para T20)**: `b3_company_MBRF.json` tem `tradingName` `MARFRIG` e nenhum `stockDividends`; `b3_cash_MARFRIG_p1.json` tem 13 registros, o último com `lastDatePriorEx` 18/09/2025, fora da janela de 2026-11-05. MBRF3 tem DY 0,00 contra 18,41 só em 2026-11-05 no JSON v2 commitado. T20 testa a hipótese da busca primeiro.
- **Real data**: o cache do #59 está em `/home/caio/.t3/worktrees/TCC/t3-issue-59-d1f86a5f2296a99f/.cache/pit-rebuild` (362 MB, 585 entradas). Esta worktree ainda não tem `.cache/pit-rebuild`; T21 passo 1 copia com hard links. A CSV de 2026-11-06 está em `/home/caio/Documents/projetos/TCC/snapshots/data/selectors-2026-11-06/stocks.csv` (só leitura).
- **Pitfalls carried** (de #59 e lote 2, ainda valem): commit com `git commit -F <arquivo>` (o hook reescreve `-m` com a palavra "make"); `check_commit.py` avisa acima de 72 caracteres; o calendário de `tests/fixtures/pit_rebuild/cotahist.txt` não tem sessão entre 2026-08-21 e 2026-08-24, nem depois de 2026-11-06; `tests/test_pit_rebuild_cli.py` seta `sys.dont_write_bytecode`, e `py/` não pode conter o literal `baselines/`; `mkstemp` grava com modo 0600; no gate use `git diff --exit-code`, não `--stat`; a cópia de mutação de `py/` e `tests/` também precisa de `api/app/services/premium_policy.py`, `docs/validation/` e `data/raw/status_invest_fundamentals.csv`; `unittest` mostra erro de importação como `errors=1`, então confira o traceback antes de contar uma morte; `HTTPError` do urllib é um `_TemporaryFileWrapper`: feche as respostas de erro que descartar.
