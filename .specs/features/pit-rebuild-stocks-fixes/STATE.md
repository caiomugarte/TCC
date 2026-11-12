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
- **Phase / Task**: Execute concluída (T1-T24). Verifier `iw-64-verify-r1` (2026-11-12) rodou: **FAIL**, 2 blocking e 3 nonblocking, em `validation.md`. Gates: pit 471 OK; raiz 1058 OK, 2 skipped. Sensor: 25 mutantes, 23 mortos, 1 equivalente, 1 sobrevivente (V3). O relatório regenera do código commitado (só `tool_commit` e tempos mudam). `validate_state.py` falha pelo veredito FAIL, como esperado.
- **Completed** (commits locais, um por tarefa, Conventional Commits com `(#64)`):
  - T1-T22 (lotes 1 a 4): `1764831` ... `d40cb37`.
  - T23 `d0eb059`: auditoria de 2026-08-07 e `AuditDocumentTests` (8 testes; mutação verificada em cópia).
  - T24 `996490d`: 2026-11-05, resumo por data, critério de liberação (AUD-06) e testes (11 testes).
- **In-progress (file:line)**: none.
- **Next step**: (V1, blocking) decisão do usuário sobre o AC9/AUD-02: aceitar `não classificado` com motivo como resultado da auditoria (emendar AC9/AUD-02; a parte 2 de DEC-2 segue "não cumprida") ou classificar as 121 linhas com evidência. (V2, blocking) corrigir as linhas SIMH3 de 2026-08-07 (`pit-rebuild-stocks-audit.md:202` e `:229`): a contagem 583759291 já está no ITR 2026-06-30 v1 rec 2026-08-12; citar esse filing em vez do ITR 2026-09-30 rec 2026-11-04. Opcionais: V3 (teste para o corte de `reference_stale` nas marcas `si_lag`, `cli.py:932`), V4 (frase de referência zero no README, linha 725), V5 (emenda de AUD-02 para linhas dentro da tolerância). Depois, re-verificação (iteração 2 de 3).
- **Blockers**: nenhum técnico. O critério de liberação está **não cumprido** na parte 2 (121 linhas fora da tolerância sem explicação: 65 em 2026-08-07, 56 em 2026-11-05); a parte 1 está cumprida (menor marca 72,77 %, P/L, 2026-11-05). Decisão do usuário.
- **Uncommitted files**: none.
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline** (fim do lote 5): pit suite 471 OK; suíte raiz 1058 OK, 2 skipped; `compileall` ok; `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` vazio.
- **Lote 5: escolhas e achados que o Verifier e o usuário precisam conhecer** (cada um com evidência no documento):
  - Classes, 280 linhas (2026-08-07 + 2026-11-05): erro nosso 24 (fator A-18 com contagem mais de 2× a da B3 de R, mais HBRE3 por fonte Yahoo); erro do Status Invest 94; diferença de método 9; não classificado 153 (121 fora da tolerância, 32 dentro).
  - Desvio de AUD-02: AUD-02 pede uma das três classes. A lista de piores traz linhas com diferença 0 quando o campo tem menos de 10 linhas fora da tolerância; elas ficam como `não classificado` com o motivo `dentro da tolerância`. Cabe à spec decidir (excluir essas linhas da lista ou aceitar a classe).
  - Regras além de B-9/B-10, declaradas no cabeçalho: (a) dado de filing recebido depois de d (8 linhas de 2026-08-07: ORVR3 ×3, SIMH3 ×2, ONCO3, BRST3, EQPA3); (b) filing de dois passos (CAML3 e EMAE4, 2026-11-05).
  - **Ressalva de dados (importante)**: o CSV de 2026-08-07 (coleta declarada 2026-08-07) reproduz valores de filings recebidos até 2026-08-14 (ORVR3 ITR 2026-06-30 v1 em 08-13, v2 em 08-14; ONCO3 ITR 2026-06-30 em 08-14; BRST3 em 08-11; EQPA3 ITR 2026-06-30 em 08-12; SIMH3 com a contagem do ITR 2026-06-30 em 08-12. Corrigido pelo Verifier: a menção anterior a "EQPA3 DFP 2025 v2 em 10-02" não tem linha da auditoria que a sustente). Ou a coleta foi posterior ao declarado, ou o SI usa dado posterior. Confirmar a data real da coleta antes de usar 2026-08-07 como ponto PIT.
  - **SI repete valores entre as coletas**: 31 de 56 pares (ticker, campo) presentes nas duas listas têm o mesmo valor do SI nas duas datas, mesmo quando chegou filing novo entre elas (VTRU3, EMAE4, BHIA3, TIMS3). Nenhum filing dos últimos quatro reproduz esses valores.
  - **Achado no fator A-18**: `share_scale` marca `verified=True` com resíduo 2,5 (VVEO3) e 2,7 (AZEV). O rótulo de verificado não quer dizer que a contagem bate com a B3. Candidato a issue.
  - **TTM com comparativo zerado (B-5)**: TIMS3 (cd 24929) traz PENÚLTIMO = 0 em todos os flows no ITR de 2026-06-30; o fallback usa o ITR de 2025 e o LL vira 4,33 bi. O SI implica LL ~1,98 bi e PL ~12,4 bi. Não resolvido.
  - VTRU3 (cd 27057): o 3.11 do ITR de 2026-03-31 é 794,9 M no trimestre (16× o de 2025); o SI mostra prejuízo. Não resolvido.
  - DY: 13 linhas `erro do Status Invest` (o SI omite ou soma em dobro registros da B3: EQPA3, PFRM3, RIAA3, PASS3, MDNE3, VLID3; COCE5 em branco), 1 `erro nosso` (HBRE3, fonte Yahoo), 1 método (TGMA3, janela pelo último dia com direito, inferência de um caso), 5 não classificados (HBOR3, CLSC4, JSLG3, EQTL3, MBRF3).
- **Pitfalls novos do lote 5**: (1) `cli._evaluate_day` com `filings_day` posterior a `day` é barrado pelo guarda PIT-02 (`_assert_point_in_time`); testes de hipótese precisam desligá-lo só em scratch. (2) O arquivo de proventos usa o nome de pregão da empresa (`b3_company_<raiz>.json`: `EQTL PARA` -> `b3_cash_EQTL_PARA_p1.json`), não o ticker. (3) O store do CVM guarda só a versão mais recente de cada período, e só as contas de `STATEMENT_ACCOUNTS`. (4) Um comando de commit com linha sobrando travou por 120 s nesta tarefa: conferir cada linha antes de rodar.
- **Pitfalls carried** (de #59 e lotes anteriores): commit com `git commit -F <arquivo>` (o hook reescreve `-m`); `check_commit.py` avisa acima de 72 caracteres no título; `sleep` solto é bloqueado; no gate use `git diff --exit-code`, não `--stat`; `unittest` mostra erro de importação como `errors=1`.
