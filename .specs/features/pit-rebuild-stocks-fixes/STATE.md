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
- **Phase / Task**: Verifier iteração 2 (`iw-64-verify-r2`): **FAIL**, `validation.md` reescrito. V1..V5 fechados (V3 confirmado como mutante equivalente pelo código). Gap novo bloqueante V6; V7 e V8 não bloqueantes. Próximo: rodada de correção de V6 (+V8, V7 opcional) e Verifier iteração 3 (a última antes de escalar).
- **V6 (blocking)**: o documento de auditoria diz que todas as versões foram testadas por inteiro e pela contagem e que nenhuma reproduz o SI. O classificador (`views.py:205-206` no scratch `3ec97855-…`) zera toda versão com período acabado mais de 270 dias antes de d (STL-01), e o DY (`classify2.py:136`) só olha data-ex em (d − 730, d]. Sem essas restrições, 30 das 110 linhas "nenhum ... reproduz" têm versão que reproduz o SI (ex.: BHIA3 LIQ. CORRENTE 08-07, ITR 2024-06-30 v1, 12 733 / 16 771 = 0,76; AALR3 DIV. LIQ./PATRI. 08-07, ITR 2025-09-30 v1 = 0,34). JSLG3, EQTL3 e MBRF3 DY têm subconjunto exato em todos os registros. 5 linhas reproduzidas não citam o mais antigo (BRST3 MARGEM EBIT, ORVR3 P/VP, SIMH3 P/VP e PSR 08-07; CAML3 DIV. LIQ./PATRI. 11-05). Nenhuma classe muda e o critério continua "cumprido"; o texto, a tabela reproduzido × sem base e as citações do mais antigo estão falsos como escritos. Correção recomendada: declarar as duas restrições na regra (documento + spec B-10) e reescrever o motivo das linhas.
- **V7 (nonblocking)**: classificador fora do repositório; número "168 das 1148 versões de 49 empresas" não conferível pelo repo. **V8 (nonblocking)**: a segunda tabela do Resumo (`audit.md:47-50`) não tem teste (mutante A5 sobreviveu).
- **Conferido sem divergência**: as 464 linhas de "Demonstrativos conferidos" (26 empresas) = cache; os 108 `Conferidos: N` = cache; SIMH3 cita ITR 2026-06-30 v1 rec 2026-08-12 (583 759 291) = cache; EGIE3 método pela contagem B3 de R (1 416 381 520) reproduz 3,08 e 2,76.
- **Histórico da rodada `iw-64-implement-r1-gaps`** (antes do Verifier r2):
  - **Gaps tratados** (commits locais, um por gap, Conventional Commits com `(#64)`):
  - **V1 (blocking)**, decisão do usuário aplicada: linha fora da tolerância sem classe passa a `erro do Status Invest` quando nenhum demonstrativo do cache reproduz o valor do SI (`sem base`). A checagem: todas as versões de DFP e ITR recebidas até R (2026-11-09), cada uma como filing mais novo do ticker (demonstrativo inteiro, com os totais LL 3.11 e PL 2.03, e só a contagem de ações). Quando algum reproduz, a classe é a mesma e a evidência cita o mais antigo. Método só para a contagem da B3 de R (faixa 0,8–2 vezes, contagem não suspeita). Cada empresa tem a lista dos demonstrativos conferidos no documento (`Demonstrativos conferidos`); a linha cita `Conferidos: N`. Ficou com 0 `não classificado` fora da tolerância.
  - **V2 (blocking)**: SIMH3 (P/VP e PSR de 2026-08-07) cita `CVM ITR 2026-06-30 v1 rec 2026-08-12` (classe continua `erro do Status Invest`). Regra do gerador: o mais antigo que reproduz, por DT_RECEB, depois DT_REFER e VERSAO.
  - **V3 (nonblocking)**: M17 (`cli.py:932`, `left_out = set()`) é **mutante equivalente**. `cotahist.reference_stale` e `cli._unmapped_reason` leem o mesmo `quote_at`; todo ticker defasado tem motivo de não mapeado e não tem valor em nenhum campo, então nunca vira marca `si_lag`. Confirmado no JSON (23 e 17 defasados, nenhum no pior caso; GPIV33 é `no_cvm_code`). Não existe fixture em que a remoção do corte mude a saída. Acrescentado o teste `test_a_reference_stale_ticker_has_no_value_so_no_si_lag_mark_can_hold_it` (fixa o invariante; passa com e sem o corte). Em aberto para o usuário: remover o corte como código morto, ou dar a `reference_stale` uma definição distinta de `no_quote` (decisão de desenho).
  - **V4 (nonblocking)**: `docs/validation/README.md` (linha 725-727): a auditoria conta só `missing`; referência zero é limite declarado.
  - **V5 (nonblocking, spec)**: classe explícita `dentro da tolerância` (32 linhas, não é um dos três erros, fora da parte 2). AUD-02, B-10 e AUD-06 emendados em `spec.md` (r3).
  - **Decisões desta rodada** (registradas aqui; a seção Decisions é de Design e não foi tocada):
  - AC9/AUD-02: `sem base` conta como erro do Status Invest (decisão do usuário). No resumo, `erro do Status Invest` aparece dividido em reproduzido e sem base.
  - Método pela contagem da B3 de R na faixa 0,8–2 vezes (contagem não suspeita): sem essa faixa, EGIE3 PSR ficaria sem classe por 0,0065 do limiar 0,8 do lote 5. Os casos com contagem suspeita (SHR-05) continuam fora.
  - DY: reprodução exata a duas casas (a tolerância de 0,5 p.p. aceita subconjuntos sem sentido: EQTL3 teria três).
  - Versões com só o índice no cache (`168` de `1148` nas 49 empresas conferidas) não reproduzem nada e não entram na contagem; estão marcadas no documento.
  - **Números (fim da rodada)**: 2026-08-07: erro nosso 17, erro do SI 92, método 7, dentro da tolerância 24, não classificado 0 (140). 2026-11-05: 7, 121, 4, 8, 0 (140). Total 24 / 213 / 11 / 32 / 0 (280). Erro do SI: 101 reproduzidos (defasagem B-9, preço de d − 1, DY com subconjunto exato, filing ou contagem de outro demonstrativo), 112 sem base (108 de contas e participação, 4 de DY).
  - **Critério (AUD-06)**: `cumprido` (parte 1 cumprida, menor marca 72,77 %; parte 2: 0 linhas fora da tolerância sem classe). É consequência da regra de `sem base`; o usuário deve conferir antes de liberar.
- **In-progress (file:line)**: none.
- **Next step**: rodada de correção de V6 (declarar as restrições STL-01 e janela de DY, ou rodar sem elas e reclassificar as 30 linhas), V8 (teste da tabela reproduzido × sem base) e V7 opcional; depois Verifier iteração 3.
- **Blockers**: nenhum técnico. Decisão do usuário pendente: o corte `reference_stale` (V3), remover ou redefinir.
- **Uncommitted files**: none (após os commits desta rodada).
- **Branch**: `t3/issue-64` (base `develop` at `ebaf1fd`); not pushed, no PR.
- **Baseline** (Verifier r2, a29381a): pit suite 473 OK; suíte raiz (venv) 1060 OK, 2 skipped. Antes, fim da rodada: pit suite 473 OK; suíte raiz (venv) 1060 OK, 2 skipped; `compileall` ok; `git diff --exit-code ebaf1fd -- data snapshots baselines py/core api py/config.py py/pit_validation.py` vazio.
- **Scratch (fora do repo)**: `classify2.py` (classes finais), `reproduce.py` + `views.py` (demonstrativos como filing mais novo), `dy_subsets.py`, `make_doc3.py` (gerador do documento) em `/tmp/claude-1000/-home-caio--t3-worktrees-TCC-t3-issue-64/3ec97855-9452-4251-bf93-af4f40242339/scratchpad/`. O lote 5 (`classify.py`, `make_doc2.py`) continua como referência do cálculo de base.
- **Achados que continuam valendo** (do lote 5, com evidência no documento):
  - O CSV de 2026-08-07 reproduz filings recebidos até 2026-08-14 (ORVR3, ONCO3, BRST3, EQPA3, SIMH3). Confirmar a data real da coleta antes de usar 2026-08-07 como ponto PIT.
  - 31 de 56 pares (ticker, campo) presentes nas duas coletas têm o mesmo valor do SI nas duas datas, mesmo com filing novo entre elas (VTRU3, EMAE4, BHIA3, TIMS3).
  - `share_scale` marca `verified=True` com resíduo 2,5 (VVEO3) e 2,7 (AZEV). Candidato a issue.
  - TTM com comparativo zerado (B-5): TIMS3 (cd 24929) traz PENÚLTIMO = 0; o LL vira 4,33 bi, e o SI implica ~1,98 bi. Não resolvido.
  - VTRU3 (cd 27057): o 3.11 do ITR de 2026-03-31 é 16 vezes o de 2025; o SI mostra prejuízo. Não resolvido.
  - DY: 13 linhas `erro do Status Invest` de registros (SI omite ou soma em dobro), 1 `erro nosso` (HBRE3, fonte Yahoo), 1 método (TGMA3, janela por último dia com direito; inferência de um caso), 4 `sem base` (HBOR3, JSLG3, EQTL3, MBRF3; a investigação do MBRF3 já diz que nenhuma fonte em cache reproduz 18,41).
- **Pitfalls novos desta rodada**:
  1. A contagem sozinha (COUNT) lê os demais insumos em d; o demonstrativo inteiro de um filing posterior a d lê a CVM em R. Misturar os dois gerou reproduções perdidas (corrigido no scratch).
  2. `indicators._assert_point_in_time` bloqueia views com filing posterior a d: desligado só no scratch.
  3. O zip da CVM no cache traz as linhas de demonstrativo de alguns períodos só para a versão mais nova: as anteriores aparecem só no índice.
  4. O JSON de proventos da B3 repete cada linha (CELESC: duas linhas idênticas); a regra A-15 mantém uma na janela e na evidência; a multiplicidade só entra na busca exata.
- **Pitfalls carried** (de #59 e lotes anteriores): commit com `git commit -F <arquivo>` (o hook reescreve `-m`); `check_commit.py` avisa acima de 72 caracteres no título; `sleep` solto é bloqueado; no gate use `git diff --exit-code`, não `--stat`; `unittest` mostra erro de importação como `errors=1`.
