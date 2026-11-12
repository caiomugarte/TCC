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
- **Phase / Task**: Verifier iteração 3 (`iw-64-verify-r3`) concluída: **FAIL**, laço de 3 iterações esgotado, escalar ao usuário. V6, V7 e V8 fechados (regra escrita = código; 0 de 80 linhas `sem base` contábeis com reprodução, F0 incluído; HBOR3 DY sem subconjunto em 65 535; regeneração byte a byte igual, cache intacto; teste da tabela de origem mata A5). Novo V9 (blocking): o motivo das 47 linhas reproduzidas diz "o Status Invest reproduz o valor com <filing>" quando 38 são só dentro de 5 % (BHIA3 cita DFP 2019; 8 linhas preterem um reprodutor exato), e as 4 linhas de DY citam subconjuntos de 2007 a 2024 num conjunto que alcança ~98 % dos DY de duas casas; a exceção da regra 6 muda 3 classes (EGIE3 PSR ×2, OPCT3 EV/EBIT) por coincidência, com a contagem B3 mais próxima. Critério inalterado (cumprido, erro nosso 24). V10..V12 nonblocking. Relatório: `validation.md`. A rodada `iw-64-implement-r2-gaps` (commits `b8bdfd3`, `44768f9`, `81912e4`) é a verificada.
- **Gerador (V7)**: `scripts/pit_rebuild_audit.py` lê `.cache/pit-rebuild` (offline, conferido sem escrita) e o JSON (os dois CSV do SI por sha256) e grava só `--out`. Comando no cabeçalho do documento; roda em ~72 s; a regeneração é byte a byte igual. Testes de fixtures: `tests/test_pit_rebuild_audit.py`.
- **Busca (V6)**: toda versão de DFP e ITR do cache (nenhuma recebida depois de R = 2026-11-09), sem máscara STL-01; quatro visões contábeis (como arquivado, LL 3.11, PL 2.03, ambos) e a visão de contagem; tolerância CMP-01. DY: subconjunto de todas as linhas da B3 do cache, sem janela, a duas casas. Citação: o mais antigo (DT_RECEB, DT_REFER, VERSAO) entre as visões sem método; visão só de método por último. Contagem total da B3 (5b) cede a um demonstrativo que reproduz como arquivado (EGIE3 PSR).
- **Números (2026-08-07 / 2026-11-05)**: erro nosso 17 / 7 (24); erro do Status Invest 93 / 123 (216); diferença de método 6 / 2 (8); dentro da tolerância 24 / 8 (32); não classificado 0. Reproduzido 52 / 83 (135); sem base 41 / 40 (81). Critério cumprido, inalterado. Antes: erro do SI 213, método 11, reproduzido 101, sem base 112.
- **Mudanças de classe contra a rodada anterior (3, de 280)**: EGIE3 PSR 2026-08-07 e 2026-11-05 (método → erro do SI; DFP 2023 v1 reproduz 3,16 e 2,83); OPCT3 EV/EBIT 2026-11-05 (método → erro do SI; ITR 2025-06-30 v2 reproduz 14,12 pela mesma regra da contagem). Os outros 277 mantêm a classe.
- **Citações contábeis**: 47 reproduções; 9 exatas a duas casas, 38 só dentro da tolerância (CMP-01). O documento diz isso na seção Limites. Exemplo: BHIA3 LIQ. CORRENTE cita o DFP 2019 v1 (recalculado 0,79 contra SI 0,76, 4,1 %), porque é o mais antigo dentro da tolerância; o ITR 2024-06-30 do exemplo da #64 é exato.
- **DY reproduzido por subconjunto (4 linhas)**: CLSC4 (dois registros de 2012 e 2022), JSLG3 (15 registros), EQTL3 (5), MBRF3 (2, de 2010 e 2024). Vários registros não têm cotação para a data-ex (ex-date fora do COTAHIST). Só HBOR3 fica `sem base`.
- **Decisões do usuário pendentes (não tomadas por mim)**:
  1. Tolerância de reprodução: manter CMP-01 (definição do documento; 38 de 47 citações são aproximadas) ou exigir igualdade a duas casas (muda classes de linhas hoje `erro do SI` reproduzido).
  2. DY: registros sem cotação para a data-ex entram na busca (leitura literal de "todos os proventos do cache"). Sem eles, JSLG3, EQTL3, MBRF3 e CLSC4 voltam a `sem base`.
  3. Ordem da regra 5b: a contagem da B3 cede a reprodução como arquivado (EGIE3, OPCT3); LL e PL ficam na ordem original. Confirmar.
  4. V3 (rodada anterior): remover o corte `reference_stale` (código morto) ou dar a ele definição própria.
- **Verificações desta rodada**: suíte `pit` 480 OK; gate de build (venv) 1067 OK, 2 pulados, saída 0; `validate_spec.py` 0 erros e 0 avisos; `compileall` OK; `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` vazio; mutante A5 (segunda tabela, 52/41 → 53/40) morto por `test_the_origin_table_of_erro_do_status_invest_equals_its_rows_by_motive`.
- **Decisões já tomadas (rodadas 1 e 2, mantidas)**: `sem base` conta como `erro do Status Invest` (AC9, decisão do usuário); DY reproduzido exato a duas casas; versões só com índice não reproduzem e ficam marcadas; AUD-06 cumprido.
- **In-progress (file:line)**: none.
- **Next step**: o usuário decide os pontos 1 a 4 (o Verifier r3 recomenda: citar primeiro o reprodutor exato; exceção da regra 6 só com igualdade exata, o que devolve EGIE3 ×2 e OPCT3 a `diferença de método`; DY só com janelas contíguas de registros). Depois, um implementador aplica a correção mínima de V9 (motivos como medida: "dentro da tolerância de" / "igual a duas casas a" / "a coincidência não mostra a origem"; `audit.md:15` e `:49`), opcionalmente V10..V12, e regenera o documento.
- **Blockers**: V9 (blocking) e o laço de verificação esgotado; decisões 1 a 4 dependem do usuário.
- **Uncommitted files**: none (após o commit do relatório do Verifier r3).
- **Branch**: `t3/issue-64` (base `develop` em `ebaf1fd`); não enviada; sem PR.
- **Scratch (fora do repo, superado)**: `/tmp/claude-1000/-home-caio--t3-worktrees-TCC-t3-issue-64/3ec97855-9452-4251-bf93-af4f40242339/scratchpad/` (classify2.py, reproduce.py, views.py, dy_subsets.py, make_doc3.py). Substituídos por `scripts/pit_rebuild_audit.py`; `final_rows.json` lá é a referência das classes da rodada anterior.
- **Pitfalls novos desta rodada**:
  1. numpy só existe na venv do projeto; o gerador usa Python puro, porque a suíte `pit` roda com o `python3` do sistema.
  2. Teste que carrega arquivo por `spec_from_file_location` precisa de `sys.dont_write_bytecode = True` antes da chamada (AD-013; o gate `test_v0_zz_manifest_gates` varre isso).
  3. `COTAHIST` não cobre as datas-ex antigas: registros sem cotação têm `ex = None` (o rebuild os exclui da janela; a busca de DY os inclui).
  4. O maior `DT_RECEB` do cache é 2026-11-07 (ITR); nada depois de R.
- **Pitfalls carried** (#59 e rodadas anteriores): commit com `git commit -F <arquivo>` (o hook reescreve `-m`); `check_commit.py` avisa acima de 72 caracteres no título; no gate use `git diff --exit-code`, não `--stat`; `unittest` mostra erro de importação como `errors=1`.
