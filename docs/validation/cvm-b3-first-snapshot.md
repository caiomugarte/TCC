# Primeiro snapshot de ações em CVM e B3 (#80)

## O snapshot

O registry `snapshots/` não é versionado. As cópias byte a byte do manifesto, dos dois CSVs de seleção e das duas
proveniências estão em [`proofs/cvm-b3-2026-11-17/`](proofs/cvm-b3-2026-11-17/), e todos os números desta seção saem
delas:

| Arquivo | SHA-256 do arquivo |
|---|---|
| `cvm-b3-2026-11-17.json` (manifesto) | `57aec798759837b0dd073b5c6c80cfef58ae6c561de274ae9a1856329b952dd6` |
| `stocks.csv` | `7d12d6347660954758f0d853b9e3328e2ce4825985102542753faa8a3affcd0c` |
| `stocks.provenance.json` | `e090db0e731b165a56c13e5c24a5245b5989695a65be8254d0a12ca96cfc0cae` |
| `fiis.csv` | `99889381df013ca6b73c2ba08c63999a72531e0969b04f1f60a0be18a99eba43` |
| `fiis.provenance.json` | `c6ce8e368b7a46a8128de7677f472f708121c415e8ae3124a83e2a33c0a7e3de` |

- Manifesto `cvm-b3-2026-11-17` (registry `snapshots/`, estado `published`), `cutoff_date` 2026-11-16,
  `manifest_sha256` `aa1da3f43ad5a7d38cb2b6a157fd7512e7d6685ca7c1df58da6530327e57735e`. Esse é o hash canônico do
  conteúdo (`manifest_canonical_sha256` em `py/snapshot_manifest.py`, sem `state` nem `state_changed_at`), não o hash
  do arquivo; ele recalcula igual a partir da cópia commitada.
- Ações: `data/selectors-2026-11-17/stocks.csv`, provedor `cvm-b3`, `as_of` 2026-11-16, SHA-256
  `7d12d6347660954758f0d853b9e3328e2ce4825985102542753faa8a3affcd0c`. O manifesto guarda os hashes dos brutos por
  tipo (`stock_source.metadata.raw_files`): 11 arquivos DFP/ITR, 2 COTAHIST, 374 de proventos, 300 de eventos, 299 de
  classificação setorial e 18 de cadastro.
- FIIs: `data/selectors-2026-11-17/fiis.csv`, provedor `statusinvest`, baixado em 2026-11-17T14:53:41Z, SHA-256
  `99889381df013ca6b73c2ba08c63999a72531e0969b04f1f60a0be18a99eba43`. Dos 593 FIIs do Status Invest, 275 saíram pelo
  filtro de negociação recente da B3 em 2026-11-16 (`recent_trade_filter.removed.fiis` em `fiis.provenance.json`) e
  ficaram 318 (as linhas de `fiis.csv`).
- Universo de ações: 384 papéis (as linhas de `stocks.csv`), com 1 deixado de fora (`IBOV11`, sem código CVM, em
  `excluded` de `stocks.provenance.json`). Entre os 384, 38 são financeiros (coluna `SUBSETOR` "Intermediários
  Financeiros" ou "Previdência e Seguros"). Com a política do run abaixo (`cvm-b3-first-snapshot/policy.json`), 19
  deles passam pelos filtros de elegibilidade do pré-processamento e todos os 19 recebem SCORE, entre eles ITUB4,
  SANB11, BBDC4 e BBAS3.
- BPAC11 continua sem indicadores contábeis por causa do dado. O ITR de 2026-09-30 do BTG está no índice da CVM
  (versão 2, recebida em 2026-11-13), mas o arquivo aberto da CVM ainda não traz as demonstrações dele. Como o ITR
  mais novo utilizável não tem linhas, os campos contábeis ficam vazios.

## O run Premium

O run `5d82ad49-4482-417f-82cd-1db773536ba2` foi executado sobre esse manifesto, com a política `premium-policy-v4`,
o mesmo perfil do run anterior (`7e1d60a1…`) e todas as chamadas de socket bloqueadas no processo. Ele terminou em
`completed`, e `provenance.selector_sources` traz os hashes verificados na leitura, iguais aos do manifesto:
`stocks` `cvm-b3` `7d12d634…` e `fiis` `statusinvest` `99889381…`. O run rodou numa cópia do banco de
desenvolvimento, não no `api/prumo-dev.db`.

## Comparação com o último run sobre o Status Invest

O último run Premium sobre o Status Invest é o `4be2248e-6b30-4a63-9a1d-61c8730f2c04` (manifesto `dev-2026-11-06`,
`premium-policy-v4`). Ele baixou o CSV de ações ao vivo em 2026-11-09 (SHA-256 `a984a6c7…`), e esse arquivo não foi
guardado. As notas abaixo usam então o CSV de ações registrado no `dev-2026-11-06`
(`data/selectors-2026-11-06/stocks.csv`, Status Invest de 2026-11-06, SHA-256 `87c15cb5…`) como referência do
Status Invest. As duas carteiras são as dos runs. Rank e SCORE vêm do pré-processamento e da pontuação do seletor,
com a política de ações do run novo e sem o GA.

Para separar a fórmula do resto, uma terceira variante é o CSV da CVM/B3 com `ROIC` e `PEG RATIO` trocados pelos do
Status Invest ("híbrido"). Quando o rank no híbrido volta perto do rank no Status Invest, a troca vem da fórmula.
Evidência completa em `cvm-b3-first-snapshot/score-comparison.json`, gerado por
`scripts/compare_stock_selector_sources.py` com `cvm-b3-first-snapshot/policy.json`.

Dos 13 papéis, 4 ficaram (CURY3, FIQE3, ISAE4, RDOR3), 9 entraram e 9 saíram. A zona de proteção foi aplicada
com o run anterior como incumbente, mas não substituiu nenhum papel: a carteira final é a do consenso.

| Papel | Mudança | Rank SI → CVM/B3 (híbrido) | Motivo |
|---|---|---|---|
| ABCB4 | entrou | 57 → 4 (2) | Regra dos financeiros: o grupo de solidez passa a ser PL/ativos contra os bancos (AVG_LIQUIDEZ −0,258 → 1,134). |
| BAZA3 | entrou | sem SCORE → 1 (1) | Regra dos financeiros: no Status Invest saía por SCORE vazio; agora pontua com o conjunto dos bancos. |
| BEEF3 | entrou | 28 → 10 (22) | Fórmula do PEG e dado: o lucro de 12 meses da CVM é negativo (P/L −9,04 contra 5,54), então o PEG fica vazio. |
| BRAV3 | entrou | 162 → 2 (54) | Fórmula do PEG e dado: P/L −29,18 na CVM contra 134,44 no Status Invest; com P/L negativo o PEG fica vazio e o grupo de crescimento fica só com o CAGR de receitas (124,4 %). |
| DEXP3 | entrou | 31 → 15 (14) | Dado: o híbrido não volta; P/L 5,50 contra 6,04 e DY 7,04 contra 7,82, num universo de 384 papéis em vez de 617. |
| INTB3 | entrou | 29 → 13 (36) | Fórmula do PEG: 0,60 na CVM/B3 contra 0,23 no Status Invest. |
| MILS3 | entrou | 18 → 7 (8) | Dado: ROE 25,36 contra 23,81 e margem líquida 26,71 contra 23,53. |
| POMO3 | entrou | 49 → 14 (37) | Fórmula do PEG: 0,07 contra −7,36 no Status Invest. |
| SLCE3 | entrou | 54 → 6 (18) | Fórmula do PEG e dado: o CAGR de lucros em 5 anos é +4,07 na CVM e −4,49 no Status Invest. |
| BBSE3 | saiu | 1 → 19 (5) | Fórmula do PEG, depois a regra dos financeiros: PEG 0,41 contra 2,62; ROIC e EV/EBIT não se aplicam a seguradora. |
| BMOB3 | saiu | 10 → 40 (13) | Fórmula do ROIC e do PEG. |
| BRSR6 | saiu | 3 → 114 (104) | Regra dos financeiros: a solidez vira PL/ativos contra os bancos (AVG_LIQUIDEZ 3,937 → −1,019). |
| CXSE3 | saiu | 8 → 80 (16) | Fórmula do PEG (0,68 contra 3,20), depois a regra dos financeiros. |
| FESA4 | saiu | 5 → 47 (29) | Dado: DY 0 pela B3 contra 12,95 no Status Invest. |
| LAVV3 | saiu | 9 → 24 (19) | Dado: DY 22,30 contra 32,78 e ROE 22,61 contra 24,90. |
| LOGG3 | saiu | 2 → 111 (49) | Fórmula do PEG e dado: a classificação da B3 (setor "Financeiro", subsetor "Exploração de Imóveis") troca os pares do setor. |
| MOTV3 | saiu | 17 → 17 (7) | Nenhuma das três diretamente: o rank é o mesmo, e os papéis que subiram passaram à frente no consenso do GA. |
| SBSP3 | saiu | 44 → 65 (67) | Dado: P/L 13,15 contra 11,00 e ROE 15,55 contra 17,80. |

## Limites

- A referência do Status Invest é o CSV de 2026-11-06, não o de 2026-11-09 que o run anterior usou.
- A seed muda com o manifesto (`derive_premium_seed`), então parte da diferença no consenso do GA não vem dos dados.
- O rank sem o GA explica a direção de cada troca, não a escolha final, que também pesa HHI e frequência no consenso.
- Os ranks, os SCORE e as carteiras desta página foram calculados antes da correção de `INVERT_METRICS`
  (`py/config.py`), que passou a inverter `EV/EBIT` (grupo value) e `PEG RATIO` (grupo growth). Nos números acima,
  um `EV/EBIT` ou um `PEG RATIO` maior ainda contava a favor do papel. As provas em
  [`proofs/cvm-b3-2026-11-17/`](proofs/cvm-b3-2026-11-17/) e `cvm-b3-first-snapshot/score-comparison.json` não foram
  regeneradas: o run e a comparação dependem de rede e do banco de desenvolvimento.
