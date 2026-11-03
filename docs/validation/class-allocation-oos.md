# Validação point-in-time fora da amostra: alocação entre classes

- Protocolo: `pit-walk-forward-v1`
- `protocol_sha256`: `c6e0e695392552877c9a6e1be5dbee5fb79f5135cebaf6cf3002e9e5e83a7f7c`
- `audit_sha256`: `d09f3177159cc8cb3ac0cd9aaa70a1e7b276529ae05f446ecd9060efc8e017db`
- `run_period`: 2016-10-31 a 2026-07-21 (`unverified_common_rows` 69)

## Veredito por camada

| Camada | Veredito | Evidência | Sobreposição com a escolha do método |
| --- | --- | --- | --- |
| `class_allocation` | `VALIDATED` | `non_revision` | `full` |
| `stock_selection` | `UNVERIFIABLE` | `-` | `-` |
| `fii_selection` | `UNVERIFIABLE` | `-` | `-` |
| `basic_recommendation` | `VALIDATED` | `non_revision` | `full` |
| `premium_recommendation` | `UNVERIFIABLE` | `-` | `-` |

## O que falta

- `class_allocation`: nada
- `stock_selection`: Status Invest fundamentals as published on each decision date; historical listed universe including delisted tickers; as-of (not retroactively adjusted) total-return prices per ticker
- `fii_selection`: Status Invest FII fundamentals as published on each decision date; historical listed FII universe including delisted tickers; as-of (not retroactively adjusted) total-return prices per ticker; as-of FII eligibility data (liquidity, segment, listing)
- `basic_recommendation`: nada
- `premium_recommendation`: stock_selection: UNVERIFIABLE; fii_selection: UNVERIFIABLE

## Auditoria point-in-time por série

Regra (AD-018): par entre o grupo do arquivo lido e outra captação com `days_apart` de pelo menos 30 dias, pelo menos 250 datas compartilhadas, nenhuma data faltando no intervalo compartilhado e fatores diários iguais dentro de `1e-9`.

| Série | Tipo | Status | Motivos | `days_apart` | Datas compartilhadas | Datas não verificadas | Primeira data | Última data | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ibovespa` | `exchange_index_close` | `PIT_QUALIFIED` | - | 97 | 2413 | 70 | 2016-10-31 | 2026-07-21 | `d286df77b24ba20ae836ba8c6d78cef6463484bee3482085658273997cb02754` |
| `ifix` | `exchange_index_close` | `PIT_QUALIFIED` | - | 97 | 2413 | 70 | 2016-10-31 | 2026-07-21 | `02822b0beb0fd858ae12ecd83cce82ecca8ef9d57df036830e944fc4767bd1c2` |
| `sp500_total_return_usd` | `exchange_index_close` | `PIT_QUALIFIED` | - | 97 | 2442 | 71 | 2016-10-31 | 2026-07-21 | `2f0a1973bc4832a86b9c33534e895d54f2a18658d54cf83880dadbe562404733` |
| `di` | `official_rate` | `PIT_QUALIFIED` | - | 97 | 2439 | 70 | 2016-10-31 | 2026-07-21 | `9395ea9dd50b562082ad3ed3ce0c137a75f51dad0e3e777aa14427d2b871598f` |
| `btc_usd` | `spot_close` | `PIT_QUALIFIED` | - | 97 | 3553 | 100 | 2016-10-29 | 2026-07-21 | `148edd4b55638bf5aa9dc6fee99c8ac0c3d9d7642ec366df444eddfabc3c2212` |
| `ptax` | `official_fixing` | `PIT_QUALIFIED` | - | 97 | 2439 | 70 | 2016-10-31 | 2026-07-21 | `78dc94fa41fec683b7780a21fbbe8c150b164912190c5a8b4260871b1454110a` |

## Leitura obrigatória

- O método (ADR 0003), seus tetos (`ALLOCATION_PROFILE_ANCHORS`) e sua janela de 10 anos (`WINDOW_YEARS`) foram escolhidos em 2026-11-01 depois de ver o período 2019-2026.
- `selection_overlap`: `full`.
- Toda janela avaliada precede essa data: o período avaliado é o mesmo em que o método foi escolhido, então o veredito diz que os números eram alcançáveis com os dados da época, não que a escolha do método foi testada fora da amostra.
- Os 12 deslocamentos se sobrepõem no calendário: mostram a sensibilidade à data de início e não são amostras independentes.
- Nenhuma decisão vê a janela de 10 anos da produção: o lookback vai de 3 anos até o histórico verificado (`full_history` verdadeiro em todas as decisões). `full_history` verdadeiro significa que o histórico disponível na decisão era menor que a janela de 10 anos, então a estratégia usou todo ele; e a ADR 0003 rejeita janelas de 3 anos por não conterem nenhuma crise.
- Nível de evidência: não revisão observada entre duas captações (`days_apart` dias); nenhum vintage as-of. `days_apart` por série: `ibovespa` 97, `ifix` 97, `sp500_total_return_usd` 97, `di` 97, `btc_usd` 97, `ptax` 97.
- A série IBOV (Yahoo `^BVSP`) é rotulada "retorno total (metodologia B3)". As duas declarações conflitam: `data/allocation/metadata.json` diz "price index; dividends not included"; a ADR 0003 trata o Ibovespa como retorno total.
- Custos: `cost_bps` 10 por lado (one-way) sobre o valor negociado em todas as classes e benchmarks; impostos excluídos.

## Distribuição por perfil

### conservador

Retorno anualizado (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 13,82% | 14,70% | 15,39% | 16,09% | 18,31% | 12 |
| CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |

Volatilidade anualizada (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 6,36% | 6,52% | 6,86% | 7,47% | 8,12% | 12 |
| CDI | 0,28% | 0,30% | 0,31% | 0,31% | 0,31% | 12 |
| IBOV | 18,21% | 18,46% | 19,16% | 24,47% | 24,56% | 12 |
| 1/5 ingênuo | 18,04% | 19,68% | 20,58% | 21,06% | 22,22% | 12 |

Drawdown máximo (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | -16,50% | -14,30% | -11,16% | -6,96% | -6,29% | 12 |
| CDI | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% | 12 |
| IBOV | -46,82% | -45,84% | -26,50% | -26,50% | -26,50% | 12 |
| 1/5 ingênuo | -41,28% | -34,26% | -28,74% | -27,13% | -23,09% | 12 |

Excesso de retorno anualizado da estratégia sobre cada benchmark:

| Benchmark | min | p25 | mediana | p75 | max | caminhos | positivos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CDI | 3,39% | 4,98% | 5,69% | 6,23% | 8,35% | 12 | 100,00% |
| IBOV | 0,36% | 3,32% | 7,03% | 9,17% | 11,43% | 12 | 100,00% |
| 1/5 ingênuo | -13,21% | -11,86% | -10,05% | -9,02% | -6,34% | 12 | 0,00% |

Sensibilidade ao custo (retorno anualizado, mesmos caminhos e benchmarks):

| `cost_bps` | Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | estratégia | 13,85% | 14,73% | 15,43% | 16,12% | 18,35% | 12 |
| 0 | CDI | 9,27% | 9,66% | 9,92% | 10,17% | 10,45% | 12 |
| 0 | IBOV | 5,18% | 6,24% | 8,44% | 10,53% | 17,63% | 12 |
| 0 | 1/5 ingênuo | 20,20% | 24,15% | 25,71% | 27,38% | 30,63% | 12 |
| 10 | estratégia | 13,82% | 14,70% | 15,39% | 16,09% | 18,31% | 12 |
| 10 | CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| 10 | IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 10 | 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |
| 50 | estratégia | 13,70% | 14,56% | 15,25% | 15,94% | 18,16% | 12 |
| 50 | CDI | 9,17% | 9,57% | 9,82% | 10,07% | 10,36% | 12 |
| 50 | IBOV | 5,09% | 6,15% | 8,34% | 10,44% | 17,53% | 12 |
| 50 | 1/5 ingênuo | 19,99% | 23,89% | 25,45% | 27,09% | 30,35% | 12 |

### moderado

Retorno anualizado (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 15,65% | 17,50% | 18,45% | 19,36% | 22,00% | 12 |
| CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |

Volatilidade anualizada (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 9,65% | 10,09% | 10,39% | 11,40% | 12,31% | 12 |
| CDI | 0,28% | 0,30% | 0,31% | 0,31% | 0,31% | 12 |
| IBOV | 18,21% | 18,46% | 19,16% | 24,47% | 24,56% | 12 |
| 1/5 ingênuo | 18,04% | 19,68% | 20,58% | 21,06% | 22,22% | 12 |

Drawdown máximo (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | -23,26% | -20,27% | -16,60% | -11,31% | -9,48% | 12 |
| CDI | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% | 12 |
| IBOV | -46,82% | -45,84% | -26,50% | -26,50% | -26,50% | 12 |
| 1/5 ingênuo | -41,28% | -34,26% | -28,74% | -27,13% | -23,09% | 12 |

Excesso de retorno anualizado da estratégia sobre cada benchmark:

| Benchmark | min | p25 | mediana | p75 | max | caminhos | positivos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CDI | 5,21% | 7,68% | 8,67% | 9,69% | 12,03% | 12 | 100,00% |
| IBOV | 3,06% | 5,24% | 10,08% | 12,06% | 15,12% | 12 | 100,00% |
| 1/5 ingênuo | -9,52% | -8,41% | -7,00% | -6,34% | -4,52% | 12 | 0,00% |

Sensibilidade ao custo (retorno anualizado, mesmos caminhos e benchmarks):

| `cost_bps` | Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | estratégia | 15,68% | 17,54% | 18,49% | 19,40% | 22,04% | 12 |
| 0 | CDI | 9,27% | 9,66% | 9,92% | 10,17% | 10,45% | 12 |
| 0 | IBOV | 5,18% | 6,24% | 8,44% | 10,53% | 17,63% | 12 |
| 0 | 1/5 ingênuo | 20,20% | 24,15% | 25,71% | 27,38% | 30,63% | 12 |
| 10 | estratégia | 15,65% | 17,50% | 18,45% | 19,36% | 22,00% | 12 |
| 10 | CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| 10 | IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 10 | 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |
| 50 | estratégia | 15,51% | 17,34% | 18,29% | 19,18% | 21,82% | 12 |
| 50 | CDI | 9,17% | 9,57% | 9,82% | 10,07% | 10,36% | 12 |
| 50 | IBOV | 5,09% | 6,15% | 8,34% | 10,44% | 17,53% | 12 |
| 50 | 1/5 ingênuo | 19,99% | 23,89% | 25,45% | 27,09% | 30,35% | 12 |

### arrojado

Retorno anualizado (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 17,36% | 20,43% | 21,51% | 22,78% | 25,28% | 12 |
| CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |

Volatilidade anualizada (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | 12,97% | 13,87% | 14,20% | 15,07% | 16,30% | 12 |
| CDI | 0,28% | 0,30% | 0,31% | 0,31% | 0,31% | 12 |
| IBOV | 18,21% | 18,46% | 19,16% | 24,47% | 24,56% | 12 |
| 1/5 ingênuo | 18,04% | 19,68% | 20,58% | 21,06% | 22,22% | 12 |

Drawdown máximo (custo `cost_bps` 10):

| Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- |
| estratégia | -29,48% | -26,27% | -21,56% | -16,83% | -12,90% | 12 |
| CDI | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% | 12 |
| IBOV | -46,82% | -45,84% | -26,50% | -26,50% | -26,50% | 12 |
| 1/5 ingênuo | -41,28% | -34,26% | -28,74% | -27,13% | -23,09% | 12 |

Excesso de retorno anualizado da estratégia sobre cada benchmark:

| Benchmark | min | p25 | mediana | p75 | max | caminhos | positivos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CDI | 6,92% | 10,64% | 11,68% | 13,08% | 15,32% | 12 | 100,00% |
| IBOV | 5,79% | 7,57% | 13,24% | 14,94% | 19,22% | 12 | 100,00% |
| 1/5 ingênuo | -5,52% | -5,04% | -3,86% | -3,46% | -2,80% | 12 | 0,00% |

Sensibilidade ao custo (retorno anualizado, mesmos caminhos e benchmarks):

| `cost_bps` | Série | min | p25 | mediana | p75 | max | caminhos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | estratégia | 17,39% | 20,48% | 21,56% | 22,83% | 25,33% | 12 |
| 0 | CDI | 9,27% | 9,66% | 9,92% | 10,17% | 10,45% | 12 |
| 0 | IBOV | 5,18% | 6,24% | 8,44% | 10,53% | 17,63% | 12 |
| 0 | 1/5 ingênuo | 20,20% | 24,15% | 25,71% | 27,38% | 30,63% | 12 |
| 10 | estratégia | 17,36% | 20,43% | 21,51% | 22,78% | 25,28% | 12 |
| 10 | CDI | 9,25% | 9,64% | 9,90% | 10,15% | 10,43% | 12 |
| 10 | IBOV | 5,16% | 6,23% | 8,42% | 10,51% | 17,61% | 12 |
| 10 | 1/5 ingênuo | 20,16% | 24,09% | 25,66% | 27,33% | 30,57% | 12 |
| 50 | estratégia | 17,20% | 20,25% | 21,33% | 22,59% | 25,08% | 12 |
| 50 | CDI | 9,17% | 9,57% | 9,82% | 10,07% | 10,36% | 12 |
| 50 | IBOV | 5,09% | 6,15% | 8,34% | 10,44% | 17,53% | 12 |
| 50 | 1/5 ingênuo | 19,99% | 23,89% | 25,45% | 27,09% | 30,35% | 12 |


## Como regenerar

```sh
python3 py/pit_validation.py audit --protocol docs/validation/protocol.json --out docs/validation/pit-audit.json
python3 py/pit_validation.py report --protocol docs/validation/protocol.json --audit docs/validation/pit-audit.json --out-dir docs/validation
```
