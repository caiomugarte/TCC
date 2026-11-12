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
- **Phase / Task**: Execute, rodada 4 (`iw-64-implement-r3-gaps`): gaps V9 a V12 do Verifier iteração 3 e decisões 1 a 6 do usuário **tratados**. Pronto para o Verifier iteração 4 (`iw-64-verify-r4`). O `validation.md` no repo continua o da iteração 3 (FAIL) até o novo relatório.
- **Commits da rodada 4** (de `5e84eca`, sem mudança em `py/`): `932ed5d` refatoração (`decide()` e `candidate_reads()`, saída byte a byte igual); `2477a21` citação exato primeiro e motivos como medida (V9, V10); `0f58706` regra 6 só com igualdade exata (decisão 3); `4773f4c` janelas contíguas do DY (decisão 4); `d9308eb` LL 3.11 e PL 2.03 como medida, com teste-guarda; `506c7db` recontagem das frases de contagem (V11); `0e33887` teste de citação exato antes do mais próximo (V11); `bed113e` sem bytecode (V12); `76c5e59` motivo de defasagem (B-9) como medida (71 linhas, decisão 1; achado na varredura final dos motivos); `4ccb098` cabeçalhos do resumo sem afirmar origem ("reproduzido por", "a origem da classe"); este commit, o handoff.
- **Decisões do usuário aplicadas** (2026-11-12):
  1. Texto como medida: "igual a duas casas a <filing> (<valor>)" e "dentro da tolerância CMP-01 de <filing> (<valor> contra <SI>); a coincidência não mostra a origem", em todo motivo de demonstrativo, de contagem da B3, de LL/PL, de pós-d, de defasagem (B-9) e de DY. Cabeçalhos do resumo também (`4ccb098`). Varredura dos 280 motivos: restam só frases de medida; "reproduz o valor do SI" aparece só como negação em `sem base`. `audit.md:15` e `:49` corrigidos.
  2. Citação: igual a duas casas primeiro (menor DT_RECEB; empate DT_REFER, VERSAO); sem nenhum, o mais próximo do SI (empate: o mais antigo). Visões como arquivado e contagem primeiro; método só se nenhuma reproduz (V10: parágrafo "Citação" no documento; B-10 r5).
  3. Regra 6: exceção só com igualdade exata a duas casas. EGIE3 PSR (duas datas) e OPCT3 EV/EBIT de 2026-11-05 voltam a `diferença de método`. A contagem da B3 de R bate exato nas três (3,08; 2,76; 14,12), e o script decidiu pela regra.
  4. DY: só janelas contíguas de registros da classe (ESPECI do COTAHIST em d), na ordem de data-ex; a mais curta, e entre as de igual tamanho a mais antiga. Quem deixa de reproduzir vira `sem base`, ainda `erro do Status Invest`.
  5. V11: fixtures de escopo, ordem das regras (regra 6 só exata), janela contígua e citação (exato antes do mais próximo); recontagem de "São N linhas" e "Das N citações …, M são iguais". Mutantes: 20 de 20 mortos em cópias fora da árvore; o controle passa.
  6. V12: `sys.dont_write_bytecode = True` antes dos imports; o teste roda o gerador numa cópia sem `PYTHONDONTWRITEBYTECODE` e exige zero `.pyc`.
- **Números** (2026-08-07 / 2026-11-05): erro nosso 17 / 7 (24); erro do SI 92 / 121 (213); diferença de método 7 / 4 (11); dentro da tolerância 24 / 8 (32); não classificado 0. Origem do erro do SI: reproduzido 48 / 75 (123); sem base 44 / 46 (90). Citações contábeis: 44, das quais 17 iguais a duas casas e 27 só dentro da tolerância. Critério: **cumprido**, erro nosso 24.
- **Antes desta rodada** (r3): erro do SI 93 / 123 (216), método 6 / 2 (8), reproduzido 52 / 83 (135), sem base 41 / 40 (81); 47 citações, 9 exatas.
- **Mudanças de classe (3, de 280)**: EGIE3 PSR 2026-08-07 e 2026-11-05 e OPCT3 EV/EBIT 2026-11-05, de erro do SI para diferença de método (decisão 3). Nenhuma outra linha muda de classe. Nove linhas de DY mudam de origem, reproduzido para sem base, sem mudar de classe: 2026-08-07 RIAA3, MDNE3 e COCE5; 2026-11-05 RIAA3, MDNE3, EQPA3, COCE5, JSLG3 e MBRF3.
- **DY manual que sobrou** (`DY_DECISIONS`, 4 linhas): HBRE3 `erro nosso` (histórico curto da B3); TGMA3 `diferença de método` (inferência de um caso: o corte do SI parece ser pelo lastDatePrior); COCE5 nas duas datas, em branco no CSV (nenhuma janela dá 0,00).
- **Leituras antigas corrigidas pelo cálculo**: MDNE3 e VLID3 ("quatro cópias do registro") não reproduzem o SI em nenhum dos dois fechamentos; RIAA3 e EQPA3 (2026-11-05) eram somas de registros não contíguos.
- **Pendências do usuário (não tratadas nesta rodada)**: item 4 da rodada anterior (V3: `reference_stale`, código morto ou definição própria); e registros da B3 sem cotação para a data-ex, que seguem na busca, ordenados por lastDatePriorEx.
- **Verificações desta rodada** (código final): suíte `pit` 495 OK (eram 480); gate Build (venv) `Ran 1082 tests`, `OK (skipped=2)`, saída 0 (os 2 pulados são os de antes, fora da feature); mutantes 20 de 20 mortos em cópias fora da árvore, mais o V12 sem a flag, e o controle passa; regeneração duas vezes, byte a byte igual, e o documento commitado não muda; `validate_spec.py` 0 erros e 0 avisos; `validate_tasks.py` 0 erros e 1 aviso (T22, anterior; `tasks.md` não mudou); diff protegido (`data`, `snapshots`, `baselines`, `py/core`, `api`, `py/config.py`, `py/pit_validation.py`) vazio.
- **README**: `docs/validation/README.md` aponta para o documento sem números; sem alteração.
- **Próximo passo**: Verifier iteração 4 (`iw-64-verify-r4`). Ler a spec r5 (B-10, AUD-06) e o documento regenerado; conferir DY, citação e regra 6 contra o cache.
- **Blockers**: nenhum.
- **Uncommitted files**: nenhum após este commit.
- **Branch**: `t3/issue-64` (base `develop`, `ebaf1fd`); não enviada; sem PR.
- **Pitfalls novos desta rodada**:
  1. Contagens por prefixo: `sem base` e citação são contados pelo início do motivo. O prefixo em `SEM_BASE_PREFIXES` (script) e `SEM_BASE_MOTIVES` (teste) e a frase do documento devem andar juntos; o teste de recontagem pega o desvio.
  2. `typeStock` é o ESPECI do COTAHIST (ON, PN, PNA, PNB, UNT). O sufixo do ticker não basta: COCE5 é PNA.
  3. A regeneração leva ~75 s e o timeout padrão do shell é 120 s: passar um timeout maior.
  4. "igual a duas casas a <objeto>": o objeto não pode começar com "a" ou "o" (saía "a a contagem" antes da correção).
- **Pitfalls carried** (#59 e rodadas anteriores): commit com `git commit -F <arquivo>` (o hook reescreve `-m`); `check_commit.py` avisa acima de 72 caracteres no título; no gate use `git diff --exit-code`, não `--stat`; `unittest` mostra erro de importação como `errors=1`.
