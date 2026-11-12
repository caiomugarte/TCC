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
- **Phase / Task**: Execute. Lotes 1 a 4 de 5 concluídos (Phases 1 a 4, T1-T22). Próximo: lote 5 (Phase 5, T23-T24) e o Verifier (autor != verificador).
- **Completed** (commits locais, um por tarefa, Conventional Commits com `(#64)`):
  - T1-T13 (lotes 1 e 2): `1764831` ... `afb8111`.
  - T14-T18 (lote 3): `2339bfe`, `c76d95b`, `8b010fe`, `1240497`, `d0ccd8b`.
  - T19 `92257d3`: passe do filing anterior (B-9), `si_lag` e o bloco `adjusted_verdict`.
  - T20 `4b846c6`: causa da MBRF3 (MBR-02): nota em `report.INVESTIGATIONS`.
  - T21 `768d386`: relatório v3 regenerado offline (`report --offline`, 47 s) a partir do cache copiado com hard links.
  - T22 `d40cb37`: README da seção PIT com a revisão da #64; gate Build verde (raiz 1047 OK, 2 skipped).
  - Fora das tarefas: `e434054` (lote 3, fecha respostas de erro do cache); `550c89f` (correção S11 e S13 da SPEC: R é 2026-11-09; "I_d vazio").
- **In-progress (file:line)**: none.
- **Next step**: lote 5: T23 (`docs/validation/pit-rebuild-stocks-audit.md` e `AuditDocumentTests`), depois T24. Depois o Verifier. O README já aponta para `pit-rebuild-stocks-audit.md`: o arquivo tem de existir ao fim da feature.
- **Blockers**: none.
- **Uncommitted files**: none.
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline** (fim do lote 4): pit suite 460 OK (`python3 -m unittest discover -s tests -p "test_pit_rebuild*.py"`); suíte raiz 1047 OK, 2 skipped (`/home/caio/Documents/projetos/TCC/.venv/bin/python -m unittest discover -s tests -p "test_*.py"`); `compileall` ok; `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` vazio.
- **Lote 4: escolhas que o lote 5 precisa conhecer** (fora da SPEC; cada uma com teste ou com dado):
  - O relatório comitado é v3 (`docs/validation/`, em `768d386`). Os números: veredito oficial da classe `não reconstruível` (igual à v2); ajustado `com ressalvas`; critério parte 1 cumprida (menor: P/L em 2026-11-05, 72,77%); `si_lag` 32 e 350 (340 dos 350 de 2026-11-05 têm como filing mais recente o ITR de 2026-09-30, recebido de 3 a 5 de novembro; a conta é da sonda, não do JSON).
  - A T23 lê `adjusted_verdict.dates[d].fields[f].worst` (com `filing`, `previous_value`, `si_lag`) e `missing`. DY não tem filing; PRECO e LIQUIDEZ têm `filing` nulo e `previous_value` da recomputação.
  - Exemplos da issue confirmados: TIM (TIMS3, cd 24929) LL TTM 6,10 bi na v2 e 4,33 bi na v3, nas duas datas; Ambipar (AMBP3, cd 24961) `stale_filing` nas duas datas; Vivara (VIVA3, cd 24805) `share_count_suspect` em 2026-11-05, 236.197.769 → 82.118.477, ratio 0,3477.
  - Mudanças de fator em relação à v2 (todas por evento da B3 em (DT_REFER, d]): 11975 (AZEVEDO) 0,1 → 10 e resíduo 0,53 → 2,66 (a contagem da B3, 111.225.217, não bate com os eventos sob nenhuma potência de dez: caso para a auditoria); 14443 10 → 1 (a contagem cai à metade); 24961 e 25445 sem mudança de valor; 25160 cai a dois terços da v2.
  - Resíduos já fora de 1 na v2 (19879, 25682, 16993, 23574, 5380, 9393): não são efeito da #64.
  - MBRF3: não é a busca. O registro de 2025-09-19 (R$ 2,8105) dá DY 17,49 em 2026-08-07, igual à Status Invest; sai da janela em 2026-11-05. A Status Invest mantém 18,41 que nenhuma fonte em cache reproduz.
  - Oficial: só uma verdade por data muda (LIQ. CORRENTE, 2026-08-07, grupo sem os não classificados, 90,21% → 89,79%, por AMBP3 `stale_filing`).
- **Pitfalls carried** (de #59 e lotes anteriores, ainda valem): commit com `git commit -F <arquivo>` (o hook reescreve `-m` com a palavra "make"); `check_commit.py` avisa acima de 72 caracteres no título; o calendário de `tests/fixtures/pit_rebuild/cotahist.txt` não tem sessão entre 2026-08-21 e 2026-08-24, nem depois de 2026-11-06; `tests/test_pit_rebuild_cli.py` seta `sys.dont_write_bytecode`, e `py/` não pode conter o literal `baselines/`; `mkstemp` grava com modo 0600; no gate use `git diff --exit-code`, não `--stat`; `unittest` mostra erro de importação como `errors=1`, então confira o traceback antes de contar uma morte; `HTTPError` do urllib é um `_TemporaryFileWrapper`: feche as respostas de erro que descartar.
- **Pitfalls novos do lote 4**: o `Edit` falha quando a quebra de linha do README não é a da leitura: copie o texto de `grep -n` com `cat -A`; o cache grava por `os.replace` (troca de nome), então o hard link da cópia não altera o cache da #59; o histórico B3 só conta quando há registro com data-ex ≤ d − 365 (senão vai ao Yahoo, `b3_short_history`); `sleep` solto é bloqueado: para esperar um job use `run_in_background` ou um Monitor com `until`.
