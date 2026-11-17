# Reconstrução PIT dos indicadores de FIIs

- Esquema: `pit-rebuild-fiis/v1` (issue #60, spec #75)
- Commit da ferramenta: `b290f5ef6d2bcb2c42f774752579018aebdaa2b8`
- Datas de decisão: 2026-08-10, 2026-11-05
- Status Invest de 2026-08-10 (coletado em 2026-08-10): `status_invest_fii.csv`, 588 linhas, sha256 `f69bc7d05772d7b2d266ea7b5d320df2da67333942ab2440d005ed574554b105`
- Status Invest de 2026-11-05 (coletado em 2026-11-06): `fiis.csv`, 592 linhas, sha256 `11da8a99e03531d39581edb0abc6949288e1bc3b9a9dbd5d19faae6bf739ba52`

## Veredito

**Veredito da classe:** `não reconstruível`.

Critério de liberação, parte 1 (≥ 70% nas duas datas): não cumprida (`PERCENTUAL EM CAIXA`, `CAGR DIVIDENDOS 3 ANOS`, `CAGR VALOR COTA 3 ANOS`, `SETOR`; sem referência, fora da conta: `GESTAO`).
Critério de liberação, parte 2 (piores casos classificados): não cumprida (94 de 538 casos sem classificação).
Critério inteiro: não cumprido.

| indicador | 2026-08-10: % na tolerância (n) | 2026-11-05: % na tolerância (n) | veredito |
| --- | --- | --- | --- |
| `LIQUIDEZ MEDIA DIARIA` | 72,22% (162) | 88,55% (166) | com ressalvas |
| `N COTISTAS` | 89,51% (162) | 73,49% (166) | com ressalvas |
| `PATRIMONIO` | 88,89% (162) | 86,14% (166) | com ressalvas |
| `N COTAS` | 88,89% (162) | 87,35% (166) | com ressalvas |
| `PERCENTUAL EM CAIXA` | 87,04% (162) | 35,54% (166) | não reconstruível |
| `P/VP` | 87,65% (162) | 86,75% (166) | com ressalvas |
| `CAGR DIVIDENDOS 3 ANOS` | 8,40% (131) | 8,45% (142) | não reconstruível |
| `CAGR VALOR COTA 3 ANOS` | 0,75% (134) | 2,08% (144) | não reconstruível |
| `DY` | 79,01% (162) | 82,53% (166) | com ressalvas |
| `ULTIMO DIVIDENDO` | 89,38% (160) | 90,91% (165) | com ressalvas |
| `PRECO` | 94,44% (162) | 93,98% (166) | reconstruível |
| `GESTAO` | - (0) | - (0) | sem referência |
| `SETOR` | 27,16% (162) | 28,31% (166) | não reconstruível |

## Regras

- Tolerância: diferença relativa de até 5%; nos indicadores em percentual (`DY`, `PERCENTUAL EM CAIXA`, `CAGR DIVIDENDOS 3 ANOS`, `CAGR VALOR COTA 3 ANOS`), até 0,5 ponto percentual. O valor reconstruído é arredondado a duas casas antes da comparação. São as regras do relatório de ações.
- Denominador: os FIIs elegíveis com valor no Status Invest. Um FII do denominador sem valor reconstruído conta como fora da tolerância.
- Veredito por indicador e data: `reconstruível` com 90% ou mais, `com ressalvas` com 70% ou mais, `não reconstruível` abaixo de 70%, `sem referência` com denominador zero. Entre as datas vale o pior.
- Veredito da classe: o pior dos 13 indicadores.
- Critério de liberação (#64): parte 1, todo indicador com pelo menos 70% na tolerância nas duas datas; parte 2, os 10 piores de cada indicador e data classificados (erro nosso, erro do Status Invest ou diferença de método).
- Ponto no tempo: só entram informes cuja versão mais recente foi entregue até `d` (`Data_Entrega`) e pregões até `d`. Um informe de mês encerrado há mais de 270 dias não dá valor (`stale_informe`).
- Proventos: a página de fundos da B3 primeiro; o Yahoo só quando a B3 não alcança o início da janela, e a fonte e o motivo ficam marcados. O informe mensal da CVM não traz provento por cota.

| indicador | fórmula |
| --- | --- |
| `PRECO` | fechamento do COTAHIST (CODBDI 12) no último pregão até `d`, no máximo 30 dias antes |
| `LIQUIDEZ MEDIA DIARIA` | média do `VOLTOT` dos últimos 21 pregões até `d` |
| `N COTISTAS` | `Total_Numero_Cotistas` do informe em `d` |
| `PATRIMONIO` | `Patrimonio_Liquido` do informe em `d` |
| `N COTAS` | `Cotas_Emitidas` do informe em `d` |
| `PERCENTUAL EM CAIXA` | `Total_Necessidades_Liquidez` / `Patrimonio_Liquido` × 100 |
| `P/VP` | `PRECO` / `Valor_Patrimonial_Cotas` do informe em `d` |
| `CAGR VALOR COTA 3 ANOS` | `Valor_Patrimonial_Cotas` sobre o do informe 36 meses antes (conhecido em `d`), ^(1/3) − 1 |
| `DY` | proventos com data-ex em (d − 365 dias, d] / `PRECO` × 100 |
| `ULTIMO DIVIDENDO` | o último provento com data-ex até `d` |
| `CAGR DIVIDENDOS 3 ANOS` | proventos dos 12 meses até `d` sobre os dos 12 meses até `d` − 3 anos, ^(1/3) − 1 |
| `SETOR` | `Segmento_Atuacao` do informe em `d`, pela tabela de correspondência |
| `GESTAO` | `Tipo_Gestao` do informe em `d`, pela tabela de correspondência |

Universo elegível: `py/fii_selection.py: CORE_POSITIVE_COLUMNS, apply_fii_eligibility`; `api/app/services/premium_policy.py: FII_RULES['arrojado']['filters']`. Positivos: PRECO. P/VP. LIQUIDEZ MEDIA DIARIA. PATRIMONIO. N COTISTAS. N COTAS; DY ≥ 0; PATRIMONIO ≥ 200.000.000; LIQUIDEZ MEDIA DIARIA ≥ 50.000.

## Universo e ligação com os informes

| data | elegíveis | com informe em d | ISIN só num informe posterior | sem informe (motivo) | com informe entregue depois de d | sessão que bate com os preços |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-10 | 162 | 145 | 7 | no_cvm_match: 11, no_isin: 2, stale_informe: 4 | 144 | 2026-08-10 (72) |
| 2026-11-05 | 166 | 150 | 8 | no_cvm_match: 9, no_isin: 2, stale_informe: 5 | 148 | 2026-11-05 (148) |

Aviso: em 2026-08-10, só 72 dos 162 preços elegíveis são o fechamento de 2026-08-10 no COTAHIST; nenhuma sessão entre d − 7 e d + 3 bate com a maioria deles.

## Fontes de proventos

Contagem de FIIs elegíveis por fonte (`b3`, `yahoo` marcado como reserva, `none` sem fonte) e motivo do uso do Yahoo.

| data | indicador | fontes | motivos |
| --- | --- | --- | --- |
| 2026-08-10 | `DY` | none: 4, yahoo: 158 | b3_empty: 9, b3_empty_yahoo_failed: 1, b3_failed: 1, b3_failed_yahoo_failed: 3, b3_short_history: 148 |
| 2026-08-10 | `ULTIMO DIVIDENDO` | b3: 147, none: 4, yahoo: 11 | b3_empty: 10, b3_empty_yahoo_failed: 1, b3_failed: 1, b3_failed_yahoo_failed: 3 |
| 2026-08-10 | `CAGR DIVIDENDOS 3 ANOS` | none: 4, yahoo: 158 | b3_empty: 9, b3_empty_yahoo_failed: 1, b3_failed: 1, b3_failed_yahoo_failed: 3, b3_short_history: 148 |
| 2026-11-05 | `DY` | none: 3, yahoo: 163 | b3_empty: 10, b3_empty_yahoo_failed: 1, b3_failed_yahoo_failed: 2, b3_short_history: 153 |
| 2026-11-05 | `ULTIMO DIVIDENDO` | b3: 153, none: 3, yahoo: 10 | b3_empty: 10, b3_empty_yahoo_failed: 1, b3_failed_yahoo_failed: 2 |
| 2026-11-05 | `CAGR DIVIDENDOS 3 ANOS` | none: 3, yahoo: 163 | b3_empty: 10, b3_empty_yahoo_failed: 1, b3_failed_yahoo_failed: 2, b3_short_history: 153 |

Na prática, a fonte do `DY` e do `CAGR DIVIDENDOS 3 ANOS` dos FIIs é o Yahoo, não a B3: em 2026-08-10, `DY`: 158 de 162 do Yahoo e 0 da B3 (148 por `b3_short_history`); em 2026-11-05, `DY`: 163 de 166 do Yahoo e 0 da B3 (153 por `b3_short_history`).
`b3_short_history` quer dizer que a página de fundos da B3 não alcança o início da janela de 12 meses do `DY` (2025-08-10 para 2026-08-10, 2025-11-05 para 2026-11-05): ela só lista os proventos recentes (primeiro registro com mediana 2025-11-28 entre 157 fundos, ver Cobertura histórica). A regra das ações da #77 (um histórico da B3 que cobre a janela vence o Yahoo) já é a dos FIIs: o Yahoo só entra, marcado, quando a B3 não cobre a janela. O `ULTIMO DIVIDENDO` não depende da janela e vem da B3 em 147 de 162 em 2026-08-10; 153 de 166 em 2026-11-05.

`DY só com a B3` (sensibilidade, não entra no veredito): os proventos da B3 com data-ex até `d`, na janela truncada no primeiro registro dela.

| data | % na tolerância (n) |
| --- | --- |
| 2026-08-10 | 0,00% (162) |
| 2026-11-05 | 62,65% (166) |

## Setor e gestão

`GESTAO`: nenhum FII elegível tem valor no Status Invest (162 em 2026-08-10, 166 em 2026-11-05), e o indicador fica `sem referência`.

Tabela de correspondência, escrita antes da medição (CVM → Status Invest):

| campo | CVM | Status Invest |
| --- | --- | --- |
| `SETOR` | Shoppings | Shoppings |
| `SETOR` | Logística | Imóveis Industriais e Logísticos, Logística |
| `SETOR` | Escritórios | Lajes Corporativas |
| `SETOR` | Lajes Corporativas | Lajes Corporativas |
| `SETOR` | Residencial | Imóveis Residenciais |
| `SETOR` | Hospital | Hospitalar |
| `SETOR` | Hotel | Hotéis |
| `SETOR` | Varejo | Varejo |
| `SETOR` | Educacional | Educacional |
| `SETOR` | Híbrido | Misto |
| `SETOR` | Multicategoria | Misto |
| `SETOR` | Outros | sem correspondente |
| `GESTAO` | Ativa | Ativa |
| `GESTAO` | Passiva | Passiva |
| `GESTAO` | Definida | Passiva |

<details>
<summary>GESTAO, 2026-08-10: - na mesma categoria depois do mapeamento (0 de 0; 162 sem valor no Status Invest)</summary>

| CVM | Status Invest | FIIs | mesma categoria |
| --- | --- | --- | --- |

Sem valor da CVM (motivo: classe): nenhum.

</details>

<details>
<summary>GESTAO, 2026-11-05: - na mesma categoria depois do mapeamento (0 de 0; 166 sem valor no Status Invest)</summary>

| CVM | Status Invest | FIIs | mesma categoria |
| --- | --- | --- | --- |

Sem valor da CVM (motivo: classe): nenhum.

</details>

<details>
<summary>SETOR, 2026-08-10: 27,16% na mesma categoria depois do mapeamento (44 de 162; 0 sem valor no Status Invest)</summary>

| CVM | Status Invest | FIIs | mesma categoria |
| --- | --- | --- | --- |
| Multicategoria | Papéis | 22 | não |
| Logística | Imóveis Industriais e Logísticos | 12 | sim |
| Shoppings | Shoppings | 12 | sim |
| Outros | Papéis | 11 | não |
| Multicategoria | Indefinido | 10 | não |
| Escritórios | Lajes Corporativas | 9 | sim |
| Multicategoria | Imóveis Industriais e Logísticos | 7 | não |
| Multicategoria | Lajes Corporativas | 7 | não |
| Multicategoria | Misto | 7 | sim |
| Multicategoria | Fundo de Fundos | 6 | não |
| Outros | Misto | 6 | não |
| Outros | Fundo de Fundos | 4 | não |
| Outros | Indefinido | 4 | não |
| Logística | Misto | 3 | não |
| Multicategoria | Fundo de Desenvolvimento | 2 | não |
| Multicategoria | Shoppings | 2 | não |
| Outros | Fundo de Desenvolvimento | 2 | não |
| Outros | Lajes Corporativas | 2 | não |
| Escritórios | Fundo de Desenvolvimento | 1 | não |
| Hospital | Hospitalar | 1 | sim |
| Hospital | Indefinido | 1 | não |
| Hotel | Hotéis | 1 | sim |
| Logística | Imóveis Residenciais | 1 | não |
| Logística | Logística | 1 | sim |
| Logística | Papéis | 1 | não |
| Logística | Serviços Financeiros Diversos | 1 | não |
| Multicategoria | Educacional | 1 | não |
| Multicategoria | Imóveis Comerciais - Outros | 1 | não |
| Outros | Imóveis Comerciais - Outros | 1 | não |
| Outros | Shoppings | 1 | não |
| Residencial | Imóveis Comerciais - Outros | 1 | não |
| Shoppings | Indefinido | 1 | não |
| Shoppings | Serviços Financeiros Diversos | 1 | não |
| Varejo | Agências de Bancos | 1 | não |
| Varejo | Varejo | 1 | sim |

Sem valor da CVM (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>SETOR, 2026-11-05: 28,31% na mesma categoria depois do mapeamento (47 de 166; 0 sem valor no Status Invest)</summary>

| CVM | Status Invest | FIIs | mesma categoria |
| --- | --- | --- | --- |
| Multicategoria | Papéis | 22 | não |
| Shoppings | Shoppings | 13 | sim |
| Logística | Imóveis Industriais e Logísticos | 12 | sim |
| Escritórios | Lajes Corporativas | 11 | sim |
| Outros | Papéis | 11 | não |
| Multicategoria | Indefinido | 9 | não |
| Multicategoria | Imóveis Industriais e Logísticos | 7 | não |
| Multicategoria | Lajes Corporativas | 7 | não |
| Multicategoria | Misto | 7 | sim |
| Outros | Misto | 7 | não |
| Multicategoria | Fundo de Fundos | 6 | não |
| Outros | Indefinido | 5 | não |
| Outros | Fundo de Fundos | 4 | não |
| Logística | Misto | 3 | não |
| Multicategoria | Fundo de Desenvolvimento | 3 | não |
| Outros | Fundo de Desenvolvimento | 2 | não |
| Outros | Lajes Corporativas | 2 | não |
| Escritórios | Fundo de Desenvolvimento | 1 | não |
| Hospital | Hospitalar | 1 | sim |
| Hospital | Indefinido | 1 | não |
| Hotel | Hotéis | 1 | sim |
| Logística | Imóveis Residenciais | 1 | não |
| Logística | Logística | 1 | sim |
| Logística | Papéis | 1 | não |
| Logística | Serviços Financeiros Diversos | 1 | não |
| Multicategoria | Educacional | 1 | não |
| Multicategoria | Imóveis Comerciais - Outros | 1 | não |
| Multicategoria | Shoppings | 1 | não |
| Outros | Imóveis Comerciais - Outros | 1 | não |
| Outros | Shoppings | 1 | não |
| Residencial | Imóveis Comerciais - Outros | 1 | não |
| Shoppings | Indefinido | 1 | não |
| Shoppings | Serviços Financeiros Diversos | 1 | não |
| Varejo | Agências de Bancos | 1 | não |
| Varejo | Imóveis Comerciais - Outros | 1 | não |
| Varejo | Varejo | 1 | sim |

Sem valor da CVM (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

## Cobertura histórica

Anos dos informes mensais na listagem da CVM: 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026. Para cada indicador, o primeiro ano em que algum informe tem a coluna preenchida e, por ano, o % dos 155 fundos ligados na última data que têm a coluna preenchida.

| indicador | coluna | ano mais antigo | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N COTISTAS` | `Total_Numero_Cotistas` | 2016 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `PATRIMONIO` | `Patrimonio_Liquido` | 2016 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `N COTAS` | `Cotas_Emitidas` | 2016 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `PERCENTUAL EM CAIXA` | `Total_Necessidades_Liquidez` | 2016 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `P/VP` | `Valor_Patrimonial_Cotas` | 2016 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `CAGR VALOR COTA 3 ANOS` | `Valor_Patrimonial_Cotas` | 2019 | 10,97% | 30,97% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `SETOR` | `Segmento_Atuacao` | 2016 | 10,32% | 29,68% | 40,00% | 55,48% | 69,03% | 79,35% | 85,81% | 93,55% | 99,35% | 100,00% | 96,77% |
| `GESTAO` | `Tipo_Gestao` | 2016 | 10,32% | 29,68% | 40,00% | 55,48% | 69,03% | 79,35% | 83,87% | 91,61% | 97,42% | 100,00% | 96,77% |

- `CAGR VALOR COTA 3 ANOS` precisa de 36 meses: o ano mais antigo é o primeiro ano da coluna mais 3.
- Proventos da B3 (`DY`, `ULTIMO DIVIDENDO`): 157 fundos com registro; primeiro registro (data com) mais antigo 2016-08-31, mediana 2025-11-28.
- Proventos do Yahoo (reserva): 164 tickers com registro; mais antigo 2017-09-01, mediana 2022-05-02.
- `PRECO` e `LIQUIDEZ MEDIA DIARIA`: COTAHIST anual; esta medição carregou 2025, 2026 e não sondou anos anteriores.

## Os 10 piores de cada indicador

Classificação pela evidência das fontes (ver Limitações). `usado` é o informe, o pregão ou a fonte do valor reconstruído; `evidência` é o que sustenta a classificação.

Contagem por classe dos 10 piores e de todos os FIIs sem valor reconstruído (estes pelo motivo).

| data | indicador | erro nosso | erro do Status Invest | diferença de método | sem classificação |
| --- | --- | --- | --- | --- | --- |
| 2026-08-10 | `LIQUIDEZ MEDIA DIARIA` | 0 | 0 | 2 | 10 |
| 2026-08-10 | `N COTISTAS` | 13 | 0 | 4 | 0 |
| 2026-08-10 | `PATRIMONIO` | 13 | 1 | 4 | 0 |
| 2026-08-10 | `N COTAS` | 13 | 1 | 4 | 0 |
| 2026-08-10 | `PERCENTUAL EM CAIXA` | 13 | 2 | 5 | 1 |
| 2026-08-10 | `P/VP` | 13 | 1 | 6 | 0 |
| 2026-08-10 | `CAGR DIVIDENDOS 3 ANOS` | 46 | 0 | 0 | 10 |
| 2026-08-10 | `CAGR VALOR COTA 3 ANOS` | 8 | 0 | 4 | 10 |
| 2026-08-10 | `DY` | 1 | 0 | 9 | 10 |
| 2026-08-10 | `ULTIMO DIVIDENDO` | 5 | 0 | 4 | 6 |
| 2026-08-10 | `PRECO` | 0 | 0 | 9 | 0 |
| 2026-08-10 | `GESTAO` | 0 | 0 | 0 | 0 |
| 2026-08-10 | `SETOR` | 13 | 0 | 14 | 0 |
| 2026-11-05 | `LIQUIDEZ MEDIA DIARIA` | 0 | 0 | 2 | 10 |
| 2026-11-05 | `N COTISTAS` | 11 | 7 | 5 | 3 |
| 2026-11-05 | `PATRIMONIO` | 11 | 7 | 5 | 0 |
| 2026-11-05 | `N COTAS` | 11 | 4 | 5 | 1 |
| 2026-11-05 | `PERCENTUAL EM CAIXA` | 11 | 9 | 5 | 1 |
| 2026-11-05 | `P/VP` | 11 | 5 | 6 | 0 |
| 2026-11-05 | `CAGR DIVIDENDOS 3 ANOS` | 50 | 0 | 0 | 10 |
| 2026-11-05 | `CAGR VALOR COTA 3 ANOS` | 9 | 0 | 5 | 10 |
| 2026-11-05 | `DY` | 0 | 0 | 14 | 5 |
| 2026-11-05 | `ULTIMO DIVIDENDO` | 4 | 0 | 4 | 6 |
| 2026-11-05 | `PRECO` | 0 | 0 | 9 | 1 |
| 2026-11-05 | `GESTAO` | 0 | 0 | 0 | 0 |
| 2026-11-05 | `SETOR` | 11 | 0 | 15 | 0 |

<details>
<summary>LIQUIDEZ MEDIA DIARIA, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `HDOF11` | 251.437,50 | 1.593.099,55 | 533,60 | sem classificação | pregão de 2026-07-15 | pregão de 2026-07-15 |
| `CPOF11` | 3.531.252,67 | 10.687.922,67 | 202,67 | sem classificação | pregão de 2026-08-10 | pregão de 2026-08-10 |
| `RZZV11` | 105.000,00 | 270.585,71 | 157,70 | sem classificação | sem pregão | sem pregão |
| `SARE11` | 348.855,00 | 818.356,64 | 134,58 | sem classificação | sem pregão | sem pregão |
| `JCIN11` | 9.973.769,00 | 23.069.962,96 | 131,31 | sem classificação | sem pregão | sem pregão |
| `IRDM11` | 5.300.561,00 | 9.434.260,83 | 77,99 | sem classificação | sem pregão | sem pregão |
| `HGFF11` | 1.552.502,00 | 542.900,74 | 65,03 | sem classificação | sem pregão | sem pregão |
| `HBCR11` | 44.514.463,00 | 24.093.062,94 | 45,88 | sem classificação | pregão de 2026-07-27 | pregão de 2026-07-27 |
| `BPFF11` | 512.062,00 | 301.357,24 | 41,15 | sem classificação | sem pregão | sem pregão |
| `KORE11` | 1.768.940,33 | 2.374.727,84 | 34,25 | sem classificação | pregão de 2026-08-10 | pregão de 2026-08-10 |

Sem valor reconstruído (motivo: classe): `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método).

</details>

<details>
<summary>LIQUIDEZ MEDIA DIARIA, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `HBCR11` | 5.597.392,56 | 25.553.517,33 | 356,53 | sem classificação | pregão de 2026-10-28 | pregão de 2026-10-28 |
| `HDOF11` | 272.727,27 | 1.233.044,02 | 352,12 | sem classificação | pregão de 2026-10-13 | pregão de 2026-10-13 |
| `RZZV11` | 105.000,00 | 270.585,71 | 157,70 | sem classificação | sem pregão | sem pregão |
| `SARE11` | 348.855,00 | 818.356,64 | 134,58 | sem classificação | sem pregão | sem pregão |
| `VXXV11` | 10.102.446,90 | 23.667.205,00 | 134,27 | sem classificação | pregão de 2026-11-05 | pregão de 2026-11-05 |
| `JCIN11` | 9.973.769,00 | 23.069.962,96 | 131,31 | sem classificação | sem pregão | sem pregão |
| `CPOF11` | 8.560.776,21 | 17.637.077,37 | 106,02 | sem classificação | pregão de 2026-11-04 | pregão de 2026-11-04 |
| `IRDM11` | 5.300.561,00 | 9.434.260,83 | 77,99 | sem classificação | sem pregão | sem pregão |
| `VIML11` | 80.396,48 | 24.339,45 | 69,73 | sem classificação | pregão de 2026-11-05 | pregão de 2026-11-05 |
| `JASC11` | 1.339.273,33 | 433.385,87 | 67,64 | sem classificação | pregão de 2026-10-09 | pregão de 2026-10-09 |

Sem valor reconstruído (motivo: classe): `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método).

</details>

<details>
<summary>N COTISTAS, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |

Sem valor reconstruído (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>N COTISTAS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `IBBP11` | 2.037,00 | 871,00 | 57,24 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `KNHY11` | 54.767,00 | 27.439,00 | 49,90 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `KCRE11` | 29.608,00 | 16.320,00 | 44,88 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `KNCR11` | 904.998,00 | 593.709,00 | 34,40 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `VGHF11` | 368.182,00 | 280.719,00 | 23,76 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `ADSH11` | 170,00 | 210,00 | 23,53 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `SNEL11` | 119.274,00 | 141.612,00 | 18,73 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-07 v1, entregue em 2026-08-13 |
| `VCJR11` | 32.507,00 | 26.875,00 | 17,33 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-06 v1, entregue em 2026-07-15 |
| `VXXV11` | 14,00 | 16,00 | 14,29 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `LASC11` | 618,00 | 557,00 | 9,87 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-15 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>PATRIMONIO, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `KISU11` | 356.882.460,97 | 455.684.754,26 | 27,68 | erro do Status Invest | informe 2026-06 v1, entregue em 2026-07-23 | informe 2026-04 v1, entregue em 2026-05-13 |

Sem valor reconstruído (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>PATRIMONIO, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `SNEL11` | 929.530.771,72 | 1.814.685.273,70 | 95,23 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-07 v1, entregue em 2026-08-13 |
| `VXXV11` | 203.357.360,42 | 264.838.264,30 | 30,23 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `KNSC11` | 1.765.949.116,43 | 2.086.450.423,72 | 18,15 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `KNUQ11` | 2.174.949.969,53 | 2.531.005.740,01 | 16,37 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `CJCT11` | 231.577.871,31 | 265.844.582,72 | 14,80 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `TRXB11` | 449.736.535,65 | 396.971.297,83 | 11,73 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `BBFO11` | 287.294.548,98 | 268.053.648,19 | 6,70 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-10 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>N COTAS, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `KISU11` | 44.196.050,00 | 59.544.167,00 | 34,73 | erro do Status Invest | informe 2026-06 v1, entregue em 2026-07-23 | informe 2026-04 v1, entregue em 2026-05-13 |

Sem valor reconstruído (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>N COTAS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `SNEL11` | 115.047.448,00 | 227.034.697,00 | 97,34 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-07 v1, entregue em 2026-08-13 |
| `IBBP11` | 102.092.483,00 | 75.370.654,52 | 26,17 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `KNUQ11` | 21.484.730,00 | 25.187.456,00 | 17,23 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-08 v2, entregue em 2026-09-25 |
| `KNSC11` | 202.202.385,00 | 235.725.514,00 | 16,58 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `VXXV11` | 477.875,00 | 513.895,00 | 7,54 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-08 v1, entregue em 2026-09-15 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>PERCENTUAL EM CAIXA, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `BBFO11` | 1,04 | 4,82 | 3,78 | erro do Status Invest | informe 2026-07 v1, entregue em 2026-08-07 | informe 2026-06 v1, entregue em 2026-07-10 |
| `HDOF11` | 4,66 | 1,97 | 2,69 | diferença de método | informe 2026-05 v1, entregue em 2026-06-12 | informe 2026-06 v2, entregue em 2026-09-14 |
| `XPML11` | 3,50 | 2,28 | 1,22 | erro do Status Invest | informe 2026-05 v2, entregue em 2026-06-22 | informe 2026-04 v2, entregue em 2026-05-28 |
| `RZTR11` | 2,61 | 1,41 | 1,20 | sem classificação | informe 2026-05 v2, entregue em 2026-06-18 | informe 2026-05 v2, entregue em 2026-06-18 |

Sem valor reconstruído (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>PERCENTUAL EM CAIXA, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `KISU11` | 100,05 | 14,30 | 85,75 | sem classificação | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-09 v1, entregue em 2026-10-14 |
| `MCRE11` | 3,73 | 32,07 | 28,34 | erro do Status Invest | informe 2026-09 v2, entregue em 2026-10-20 | informe 2026-08 v1, entregue em 2026-09-14 |
| `VGIR11` | 17,39 | 1,54 | 15,85 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-06 v1, entregue em 2026-07-13 |
| `BBFO11` | 1,04 | 16,67 | 15,63 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-10 |
| `MXRF11` | 17,65 | 5,52 | 12,13 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `KNUQ11` | 9,57 | 20,78 | 11,21 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `JSCR11` | 11,53 | 1,44 | 10,09 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `MCCI11` | 0,58 | 10,30 | 9,72 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-14 | informe 2026-06 v1, entregue em 2026-07-13 |
| `KNIP11` | 11,35 | 2,28 | 9,07 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `RBRL11` | 9,29 | 0,74 | 8,55 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>P/VP, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `KISU11` | 0,77 | 0,82 | 6,49 | erro do Status Invest | informe 2026-06 v1, entregue em 2026-07-23 | informe 2026-04 v1, entregue em 2026-05-13 |

Sem valor reconstruído (motivo: classe): `BGRB11` (no_cvm_match: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (no_quote: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>P/VP, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `VXXV11` | 1,94 | 1,60 | 17,53 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-14 |
| `TRXB11` | 1,72 | 1,95 | 13,37 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `CJCT11` | 1,15 | 1,05 | 8,70 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |
| `BBFO11` | 0,85 | 0,91 | 7,06 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-06 v1, entregue em 2026-07-10 |
| `IBBP11` | 0,78 | 0,82 | 5,13 | erro do Status Invest | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-07 v1, entregue em 2026-08-14 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `BTCR11` (no_isin: erro nosso), `FTCE11B` (no_isin: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `JCIN11` (no_cvm_match: erro nosso), `MCLO11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>CAGR DIVIDENDOS 3 ANOS, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `BPML11` | 1,01 | 90,84 | 89,83 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `SARE11` | -28,39 | -100,00 | 71,61 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `VPPR11` | -42,64 | -100,00 | 57,36 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `HGFF11` | 8,82 | -15,82 | 24,64 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `HTMX11` | -15,56 | 9,07 | 24,63 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `HCTR11` | -7,25 | -29,56 | 22,31 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `NSLU11` | 20,42 | 41,50 | 21,08 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `RECR11` | 2,87 | 19,40 | 16,53 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `PSEC11` | -5,13 | -21,22 | 16,09 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `TEPP11` | 6,20 | 22,13 | 15,93 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |

Sem valor reconstruído (motivo: classe): `AFHI11` (short_or_zero_history: erro nosso), `BBFO11` (short_or_zero_history: erro nosso), `BLMG11` (short_or_zero_history: erro nosso), `BPFF11` (b3_empty_yahoo_failed: erro nosso), `BROF11` (short_or_zero_history: erro nosso), `BTCR11` (b3_failed_yahoo_failed: erro nosso), `CACR11` (short_or_zero_history: erro nosso), `CCME11` (short_or_zero_history: erro nosso), `CPSH11` (short_or_zero_history: erro nosso), `CPUR11` (short_or_zero_history: erro nosso), `CXCO11` (short_or_zero_history: erro nosso), `CYCR11` (short_or_zero_history: erro nosso), `DEVA11` (short_or_zero_history: erro nosso), `GAME11` (short_or_zero_history: erro nosso), `GARE11` (short_or_zero_history: erro nosso), `HBCR11` (short_or_zero_history: erro nosso), `HOFC11` (short_or_zero_history: erro nosso), `HREC11` (short_or_zero_history: erro nosso), `HSAF11` (short_or_zero_history: erro nosso), `HSLG11` (short_or_zero_history: erro nosso), `HSRE11` (b3_failed_yahoo_failed: erro nosso), `IRDM11` (short_or_zero_history: erro nosso), `IRIM11` (short_or_zero_history: erro nosso), `KISU11` (short_or_zero_history: erro nosso), `KNHF11` (short_or_zero_history: erro nosso), `KNUQ11` (short_or_zero_history: erro nosso), `LASC11` (short_or_zero_history: erro nosso), `LIFE11` (short_or_zero_history: erro nosso), `MANA11` (short_or_zero_history: erro nosso), `MCRE11` (short_or_zero_history: erro nosso), `PVBI11` (short_or_zero_history: erro nosso), `RBRX11` (short_or_zero_history: erro nosso), `RINV11` (short_or_zero_history: erro nosso), `RPRI11` (short_or_zero_history: erro nosso), `RZAK11` (short_or_zero_history: erro nosso), `RZAT11` (short_or_zero_history: erro nosso), `RZTR11` (short_or_zero_history: erro nosso), `RZZR11` (short_or_zero_history: erro nosso), `SNFF11` (short_or_zero_history: erro nosso), `TJKB11` (short_or_zero_history: erro nosso), `TRBL11` (short_or_zero_history: erro nosso), `TVRI11` (short_or_zero_history: erro nosso), `URPR11` (short_or_zero_history: erro nosso), `VGHF11` (short_or_zero_history: erro nosso), `VSLH11` (short_or_zero_history: erro nosso), `VVMR11` (short_or_zero_history: erro nosso).

</details>

<details>
<summary>CAGR DIVIDENDOS 3 ANOS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `BPML11` | -4,08 | 113,09 | 117,17 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `HGFF11` | 8,82 | -100,00 | 108,82 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `SARE11` | -28,39 | -100,00 | 71,61 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `VPPR11` | -42,64 | -100,00 | 57,36 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `RBRF11` | 6,82 | -21,26 | 28,08 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `CNES11` | 8,94 | -12,63 | 21,57 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `RBRP11` | -5,27 | 13,15 | 18,42 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `NSLU11` | 21,11 | 39,38 | 18,27 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `PSEC11` | -5,97 | -23,05 | 17,08 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `HCTR11` | -9,39 | -25,98 | 16,59 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |

Sem valor reconstruído (motivo: classe): `AFHI11` (short_or_zero_history: erro nosso), `BBFO11` (short_or_zero_history: erro nosso), `BLMG11` (short_or_zero_history: erro nosso), `BPFF11` (b3_empty_yahoo_failed: erro nosso), `BROF11` (short_or_zero_history: erro nosso), `BTCR11` (b3_failed_yahoo_failed: erro nosso), `BTRU11` (short_or_zero_history: erro nosso), `CACR11` (short_or_zero_history: erro nosso), `CCME11` (short_or_zero_history: erro nosso), `CJCT11` (short_or_zero_history: erro nosso), `CLIN11` (short_or_zero_history: erro nosso), `CPSH11` (short_or_zero_history: erro nosso), `CPUR11` (short_or_zero_history: erro nosso), `CXCO11` (short_or_zero_history: erro nosso), `CYCR11` (short_or_zero_history: erro nosso), `DEVA11` (short_or_zero_history: erro nosso), `GAME11` (short_or_zero_history: erro nosso), `GARE11` (short_or_zero_history: erro nosso), `HBCR11` (short_or_zero_history: erro nosso), `HOFC11` (short_or_zero_history: erro nosso), `HREC11` (short_or_zero_history: erro nosso), `HSAF11` (short_or_zero_history: erro nosso), `HSLG11` (short_or_zero_history: erro nosso), `IRDM11` (short_or_zero_history: erro nosso), `IRIM11` (short_or_zero_history: erro nosso), `JCCJ11` (short_or_zero_history: erro nosso), `KISU11` (short_or_zero_history: erro nosso), `KNHF11` (short_or_zero_history: erro nosso), `KNUQ11` (short_or_zero_history: erro nosso), `LASC11` (short_or_zero_history: erro nosso), `MCRE11` (short_or_zero_history: erro nosso), `PVBI11` (short_or_zero_history: erro nosso), `RINV11` (short_or_zero_history: erro nosso), `RZAK11` (short_or_zero_history: erro nosso), `RZAT11` (short_or_zero_history: erro nosso), `RZTR11` (short_or_zero_history: erro nosso), `RZZR11` (short_or_zero_history: erro nosso), `SAPI11` (short_or_zero_history: erro nosso), `SNEL11` (short_or_zero_history: erro nosso), `SNFF11` (short_or_zero_history: erro nosso), `TJKB11` (short_or_zero_history: erro nosso), `TRBL11` (short_or_zero_history: erro nosso), `TRXB11` (short_or_zero_history: erro nosso), `TVRI11` (short_or_zero_history: erro nosso), `URPR11` (short_or_zero_history: erro nosso), `VGHF11` (short_or_zero_history: erro nosso), `VIML11` (short_or_zero_history: erro nosso), `VRTM11` (short_or_zero_history: erro nosso), `VSLH11` (short_or_zero_history: erro nosso), `VVMR11` (short_or_zero_history: erro nosso).

</details>

<details>
<summary>CAGR VALOR COTA 3 ANOS, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `CPUR11` | 15,87 | -53,87 | 69,74 | sem classificação | informe 2026-06 v1, entregue em 2026-07-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `PQDP11` | 61,52 | -6,03 | 67,55 | sem classificação | informe 2026-06 v1, entregue em 2026-07-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `KNSC11` | 12,09 | -54,35 | 66,44 | sem classificação | informe 2026-06 v1, entregue em 2026-07-14 | informe 2026-06 v1, entregue em 2026-07-14 |
| `HREC11` | 10,28 | -54,51 | 64,79 | sem classificação | informe 2026-06 v1, entregue em 2026-07-14 | informe 2026-06 v1, entregue em 2026-07-14 |
| `PORD11` | 10,33 | -54,04 | 64,37 | sem classificação | informe 2026-06 v1, entregue em 2026-07-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `KCRE11` | 7,14 | -55,44 | 62,58 | sem classificação | informe 2026-06 v1, entregue em 2026-07-14 | informe 2026-06 v1, entregue em 2026-07-14 |
| `SARE11` | 0,00 | -62,38 | 62,38 | sem classificação | informe 2025-11 v1, entregue em 2025-12-15 | informe 2025-11 v1, entregue em 2025-12-15 |
| `CPTS11` | 6,71 | -54,35 | 61,06 | sem classificação | informe 2026-06 v1, entregue em 2026-07-15 | informe 2026-06 v1, entregue em 2026-07-15 |
| `GGRC11` | 5,49 | -54,08 | 59,57 | sem classificação | informe 2026-06 v2, entregue em 2026-07-30 | informe 2026-06 v2, entregue em 2026-07-30 |
| `HLOG11` | 6,33 | -53,17 | 59,50 | sem classificação | informe 2026-06 v1, entregue em 2026-07-14 | informe 2026-06 v1, entregue em 2026-07-14 |

Sem valor reconstruído (motivo: classe): `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `HGFF11` (stale_informe: diferença de método), `HSRE11` (no_cvm_match: erro nosso), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>CAGR VALOR COTA 3 ANOS, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `CPUR11` | 12,88 | -53,44 | 66,32 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `HREC11` | 11,19 | -53,77 | 64,96 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `PQDP11` | 58,47 | -5,85 | 64,32 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `KNSC11` | 10,27 | -53,91 | 64,18 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `PORD11` | 10,02 | -53,84 | 63,86 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `KCRE11` | 7,01 | -54,41 | 61,42 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `HLOG11` | 7,75 | -53,09 | 60,84 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `CPSH11` | 5,96 | -53,01 | 58,97 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `HFOF11` | 3,45 | -55,07 | 58,52 | sem classificação | informe 2026-09 v1, entregue em 2026-10-15 | informe 2026-09 v1, entregue em 2026-10-15 |
| `GGRC11` | 2,96 | -54,04 | 57,00 | sem classificação | informe 2026-09 v2, entregue em 2026-10-30 | informe 2026-09 v2, entregue em 2026-10-30 |

Sem valor reconstruído (motivo: classe): `AJFI11` (no_informe_36_months_before: erro nosso), `BPFF11` (stale_informe: diferença de método), `BTCI11` (no_cvm_match: erro nosso), `HGFF11` (stale_informe: diferença de método), `IRDM11` (stale_informe: diferença de método), `JCCJ11` (no_cvm_match: erro nosso), `PCIP11` (no_cvm_match: erro nosso), `PMLL11` (no_cvm_match: erro nosso), `PSEC11` (no_cvm_match: erro nosso), `RBRF11` (stale_informe: diferença de método), `SAPI11` (no_informe_36_months_before: erro nosso), `SARE11` (stale_informe: diferença de método), `TVRI11` (no_cvm_match: erro nosso), `VVMR11` (no_cvm_match: erro nosso).

</details>

<details>
<summary>DY, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `CPLG11` | 14,31 | 0,00 | 14,31 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `VVMR11` | 1,18 | 12,02 | 10,84 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `NSLU11` | 18,86 | 28,48 | 9,62 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `APXM11` | 10,40 | 0,82 | 9,58 | sem classificação | proventos: yahoo (b3_failed) | proventos da fonte yahoo |
| `CACR11` | 74,26 | 64,99 | 9,27 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `PSEC11` | 14,48 | 8,40 | 6,08 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `RZAK11` | 16,84 | 15,45 | 1,39 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `HSAF11` | 15,08 | 13,82 | 1,26 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `MANA11` | 14,94 | 13,73 | 1,21 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `CLIN11` | 12,36 | 13,46 | 1,10 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |

Sem valor reconstruído (motivo: classe): `BPFF11` (no_quote: diferença de método), `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método), `HGFF11` (no_quote: diferença de método), `HSRE11` (b3_failed_yahoo_failed: erro nosso), `IRDM11` (no_quote: diferença de método), `JCIN11` (no_quote: diferença de método), `RBRF11` (no_quote: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (no_quote: diferença de método).

</details>

<details>
<summary>DY, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (p.p.) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `CPLG11` | 15,78 | 0,00 | 15,78 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `VVMR11` | 0,51 | 15,47 | 14,96 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `CACR11` | 65,31 | 54,20 | 11,11 | diferença de método | proventos: yahoo (b3_short_history) | proventos da fonte b3 |
| `NSLU11` | 19,12 | 28,72 | 9,60 | diferença de método | proventos: yahoo (b3_short_history) | proventos da fonte b3 |
| `BTRU11` | 10,35 | 1,01 | 9,34 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `PSEC11` | 16,67 | 9,24 | 7,43 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `HBCR11` | 4,74 | 10,38 | 5,64 | diferença de método | proventos: yahoo (b3_short_history) | proventos da fonte b3 |
| `GSFI11` | 3,78 | 1,93 | 1,85 | sem classificação | proventos: yahoo (b3_short_history) | proventos da fonte yahoo |
| `TRXF11` | 18,18 | 16,75 | 1,43 | diferença de método | proventos: yahoo (b3_short_history) | proventos da fonte b3 |
| `MANA11` | 16,57 | 15,24 | 1,33 | diferença de método | proventos: yahoo (b3_short_history) | proventos da fonte b3 |

Sem valor reconstruído (motivo: classe): `BPFF11` (no_quote: diferença de método), `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método), `HGFF11` (no_quote: diferença de método), `IRDM11` (no_quote: diferença de método), `JCIN11` (no_quote: diferença de método), `RBRF11` (no_quote: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (no_quote: diferença de método).

</details>

<details>
<summary>ULTIMO DIVIDENDO, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `VXXV11` | 93,02 | 366,37 | 293,86 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `RBRF11` | 0,12 | 0,33 | 175,00 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `JCIN11` | 7,08 | 8,60 | 21,42 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `SARE11` | 0,03 | 0,03 | 15,38 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `AJFI11` | 0,06 | 0,06 | 9,09 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `SHPP11` | 0,06 | 0,06 | 9,09 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `VSLH11` | 0,02 | 0,02 | 9,09 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `PSEC11` | 0,60 | 0,55 | 8,33 | sem classificação | proventos: b3 | proventos da fonte b3 |
| `PCIP11` | 1,00 | 1,07 | 7,00 | sem classificação | proventos: b3 | proventos da fonte b3 |
| `AZPL11` | 0,07 | 0,07 | 6,67 | diferença de método | proventos: b3 | proventos da fonte yahoo |

Sem valor reconstruído (motivo: classe): `BPFF11` (b3_empty_yahoo_failed: erro nosso), `BTCR11` (b3_failed_yahoo_failed: erro nosso), `FTCE11B` (b3_failed_yahoo_failed: erro nosso), `HSRE11` (b3_failed_yahoo_failed: erro nosso), `IRDM11` (no_distribution_by_d: erro nosso).

</details>

<details>
<summary>ULTIMO DIVIDENDO, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `VXXV11` | 93,02 | 366,37 | 293,86 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `RBRF11` | 0,12 | 0,33 | 175,00 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `CNES11` | 0,01 | 0,01 | 34,43 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `JCIN11` | 7,08 | 8,60 | 21,42 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `SARE11` | 0,03 | 0,03 | 15,38 | sem classificação | proventos: yahoo (b3_empty) | proventos da fonte yahoo |
| `VINO11` | 0,05 | 0,04 | 13,04 | sem classificação | proventos: b3 | proventos da fonte b3 |
| `GSFI11` | 0,10 | 0,11 | 10,00 | sem classificação | proventos: b3 | proventos da fonte b3 |
| `AZPL11` | 0,07 | 0,07 | 6,67 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `VGRI11` | 0,07 | 0,07 | 6,67 | diferença de método | proventos: b3 | proventos da fonte yahoo |
| `BTCI11` | 0,10 | 0,10 | 5,26 | diferença de método | proventos: b3 | proventos da fonte yahoo |

Sem valor reconstruído (motivo: classe): `BPFF11` (b3_empty_yahoo_failed: erro nosso), `BTCR11` (b3_failed_yahoo_failed: erro nosso), `FTCE11B` (b3_failed_yahoo_failed: erro nosso), `IRDM11` (no_distribution_by_d: erro nosso).

</details>

<details>
<summary>PRECO, 2026-08-10</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |

Sem valor reconstruído (motivo: classe): `BPFF11` (no_quote: diferença de método), `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método), `HGFF11` (no_quote: diferença de método), `IRDM11` (no_quote: diferença de método), `JCIN11` (no_quote: diferença de método), `RBRF11` (no_quote: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (no_quote: diferença de método).

</details>

<details>
<summary>PRECO, 2026-11-05</summary>

| ticker | referência | reconstruído | diferença (%) | classe | usado | evidência |
| --- | --- | --- | --- | --- | --- | --- |
| `VVMR11` | 79,00 | 84,80 | 7,34 | sem classificação | pregão de 2026-11-05 | pregão de 2026-11-05 |

Sem valor reconstruído (motivo: classe): `BPFF11` (no_quote: diferença de método), `BTCR11` (no_quote: diferença de método), `FTCE11B` (no_quote: diferença de método), `HGFF11` (no_quote: diferença de método), `IRDM11` (no_quote: diferença de método), `JCIN11` (no_quote: diferença de método), `RBRF11` (no_quote: diferença de método), `RZZV11` (no_quote: diferença de método), `SARE11` (no_quote: diferença de método).

</details>

<details>
<summary>SETOR, 2026-08-10: os 10 primeiros fora da categoria (por ticker)</summary>

| ticker | Status Invest | CVM | classe | evidência |
| --- | --- | --- | --- | --- |
| `AFHI11` | Papéis | Multicategoria | diferença de método | informe 2026-06 v1, entregue em 2026-07-13 |
| `AZPL11` | Serviços Financeiros Diversos | Logística | diferença de método | informe 2026-06 v1, entregue em 2026-07-13 |
| `BBFO11` | Fundo de Fundos | Multicategoria | diferença de método | informe 2026-07 v1, entregue em 2026-08-07 |
| `BBIG11` | Serviços Financeiros Diversos | Shoppings | diferença de método | informe 2026-07 v1, entregue em 2026-08-07 |
| `BCIA11` | Fundo de Fundos | Multicategoria | diferença de método | informe 2026-06 v1, entregue em 2026-07-14 |
| `BCRI11` | Papéis | Outros | diferença de método | informe 2026-06 v1, entregue em 2026-07-14 |
| `BLMG11` | Imóveis Industriais e Logísticos | Multicategoria | diferença de método | informe 2026-06 v1, entregue em 2026-07-15 |
| `BPML11` | Shoppings | Multicategoria | diferença de método | informe 2026-06 v1, entregue em 2026-07-15 |
| `BRCR11` | Lajes Corporativas | Multicategoria | diferença de método | informe 2026-06 v2, entregue em 2026-07-31 |
| `BROF11` | Lajes Corporativas | Multicategoria | diferença de método | informe 2026-06 v1, entregue em 2026-07-15 |

</details>

<details>
<summary>SETOR, 2026-11-05: os 10 primeiros fora da categoria (por ticker)</summary>

| ticker | Status Invest | CVM | classe | evidência |
| --- | --- | --- | --- | --- |
| `AFHI11` | Papéis | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-14 |
| `AZPL11` | Serviços Financeiros Diversos | Logística | diferença de método | informe 2026-09 v1, entregue em 2026-10-13 |
| `BBFO11` | Fundo de Fundos | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |
| `BBIG11` | Serviços Financeiros Diversos | Shoppings | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |
| `BCIA11` | Fundo de Fundos | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-13 |
| `BCRI11` | Papéis | Outros | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |
| `BLMG11` | Imóveis Industriais e Logísticos | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |
| `BPML11` | Shoppings | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |
| `BRCR11` | Lajes Corporativas | Multicategoria | diferença de método | informe 2026-09 v2, entregue em 2026-10-22 |
| `BROF11` | Lajes Corporativas | Multicategoria | diferença de método | informe 2026-09 v1, entregue em 2026-10-15 |

</details>

## Limitações

- A página de fundos da B3 traz só os proventos dos últimos meses a partir do dia da coleta. Para as duas datas, a janela de 12 meses começa antes do primeiro registro dela na maioria dos FIIs, e o DY cai no Yahoo, marcado. A linha `DY só com a B3` mostra o que a B3 sozinha dá na janela truncada.
- O `CAGR DIVIDENDOS 3 ANOS` precisa de quatro anos de proventos, que a B3 não publica: ele vem do Yahoo quando há, marcado.
- O campo `Percentual_Dividend_Yield_Mes` do informe mensal não é usado: no informe do XP Malls (CNPJ 28.757.546/0001-00), versão 1 de 2026-01, ele vale −0,058778, o que não é um rendimento distribuído.
- Os arquivos anuais da CVM trazem só a versão mais recente de cada informe. Um informe reapresentado depois de `d` some em `d`, e o fundo usa o informe anterior.
- A ligação ticker → fundo é o ISIN do último pregão até `d` (COTAHIST) igual ao `Codigo_ISIN` do informe. Alguns fundos informam outro ISIN (por exemplo `BR0EI9CTF007`) e só passam ao ISIN negociado num informe posterior a `d`: esse informe só identifica o fundo, e os valores continuam vindo dos informes entregues até `d` (`isin_from_later_informe`). Quando mais de um fundo informa o mesmo ISIN, vale o de número de cotas mais perto do da página da B3, obtida na coleta e não em `d`.
- A tabela de correspondência de setor e gestão foi escrita antes da medição. A CVM classifica muitos fundos como `Multicategoria` ou `Outros`, categorias que o Status Invest divide em várias.
- A classificação dos 10 piores é automática: ela procura uma fonte que dê o valor do Status Invest. O caso sem fonte fica `sem classificação` e precisa de leitura à mão.
