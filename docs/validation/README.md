# Validação point-in-time fora da amostra

Protocolo `pit-walk-forward-v1` para validar a alocação entre classes
(`equal-weight-risky-v1`, ADR 0003) sem olhar para o futuro. As decisões que o
fixam estão em `.specs/STATE.md` (AD-018, AD-019 e AD-020); os valores
declarados estão em [`protocol.json`](protocol.json).

| Arquivo | O que é |
| --- | --- |
| `protocol.json` | o protocolo declarado: séries, tipos, regras de decisão, perfis, benchmarks e custo |
| `pit-audit.json` | a auditoria point-in-time de cada série, gerada pelo comando `audit` |
| `class-allocation-oos.json` | o veredito por camada e, quando `VALIDATED`, a distribuição por perfil |
| `class-allocation-oos.md` | o mesmo relatório em texto |

Nenhum desses números é escrito à mão: os três arquivos gerados vêm dos
comandos abaixo e um teste confere que a regeneração reproduz os mesmos bytes.

## Regras do walk-forward

- **Regra de dados:** uma decisão na data `t` usa só linhas datadas `<= t`
  (a "data de decisão"). A data de corte do snapshot continua sendo a última
  data comum dele, e o período da execução (`run_period.start` a
  `run_period.end`) é o intervalo verificado pela auditoria.
- **Regra de execução:** os pesos decididos em `t` entram no fechamento da
  próxima data comum (a data de execução); o retorno da linha datada nesse dia
  ainda é dos pesos anteriores. Na primeira execução a carteira começa em 1,0
  e essa linha não rende nada. O S&P 500 TR de `t` (fechamento da NYSE), a
  barra do BTC-USD de `t` (fecha às 24:00 UTC) e o DI de `t` (publicado depois
  do pregão) não são conhecidos no fechamento da B3 em `t`.
- **Rebalanceamento:** volta ao alvo só em cada data de execução, com deriva
  entre uma execução e a próxima; uma nova decisão a cada 12 meses.
- **12 deslocamentos mensais:** a âncora da decisão `j` no deslocamento `k`
  (`k` de 0 a 11) é `run_period.start` + (12 x 3 + `k` + 12`j`) meses, calculada
  de uma vez e com o dia limitado ao fim do mês; a decisão é a última data
  comum `<=` âncora, e a execução é a data comum seguinte.
- **Lookback:** a estratégia vê todo o histórico verificado até `t`, a partir
  de 3 anos (`min_lookback_years`); o teto de 10 anos da produção
  (`WINDOW_YEARS`) nunca é alcançado, porque o histórico verificado é menor.
- **Período de cada caminho:** da primeira execução até a última execução em
  ou antes de `run_period.end`. A janela de teste `j` são as linhas em
  `(exec_j, exec_{j+1}]`; as janelas de um caminho são contíguas, não se
  sobrepõem e cobrem todo o período declarado.
- **Benchmarks:** `cdi` (100% renda fixa), `ibov` (100% ações brasileiras, a
  mesma série `ibovespa.csv` da estratégia) e `naive_equal_weight` (20% em
  cada uma das cinco classes), com o mesmo calendário, a mesma defasagem, o
  mesmo rebalanceamento e o mesmo custo da estratégia.
- **Custo:** `cost_bps` 10 por lado (one-way) sobre o valor negociado, em todas
  as classes e benchmarks: em cada execução o valor é multiplicado por
  `1 - cost_bps / 10000 x giro`, com giro = soma de `|alvo - pesos derivados|`.
  Impostos e taxas de fundos estão excluídos. A sensibilidade ao custo
  (`sensitivity_bps`: 0, 10 e 50) refaz os mesmos caminhos ao lado do resultado
  principal.

## Séries e tipos (`series_kind`)

| Série | `series_kind` | Fonte |
| --- | --- | --- |
| `ibovespa` | `exchange_index_close` | Yahoo `^BVSP`, espelho do índice oficial da B3; rotulado retorno total (metodologia B3) |
| `ifix` | `exchange_index_close` | B3 Index Statistics API, índice IFIX |
| `sp500_total_return_usd` | `exchange_index_close` | Yahoo `^SP500TR`, espelho do índice oficial S&P 500 Total Return |
| `di` | `official_rate` | Banco Central do Brasil, SGS série 12 (CDI diário) |
| `ptax` | `official_fixing` | Banco Central do Brasil, PTAX de fechamento (publicada por volta das 13:00) |
| `btc_usd` | `spot_close` | Yahoo BTC-USD, cotação composta de corretoras com carimbo 00:00 UTC, sem ajuste que reescreva a história |

Tipos que qualificam: `exchange_index_close`, `official_fixing`,
`official_rate`, `spot_close`. Tipos revisáveis, que nunca qualificam:
`vendor_adjusted_close`, `vendor_composite_close`. Qualquer outro valor faz o
`report` sair com código 2.

O rótulo do IBOV segue a metodologia da B3 citada na ADR 0003 (retorno total);
`data/allocation/metadata.json` ainda diz "price index; dividends not
included", uma declaração anterior que fica sem alteração.

## Regra point-in-time (AD-018)

1. Captações que guardam um arquivo da série com o mesmo sha256 formam um
   grupo, datado pela captação mais recente (`same_bytes_as`).
2. Pares só se formam entre o grupo do arquivo que a execução lê
   (`snapshot_dir`) e cada outro grupo com o mesmo provedor e identificador.
3. Um par qualifica quando os grupos estão a pelo menos 30 dias um do outro
   (`days_apart`), compartilham pelo menos 250 datas, nenhuma data do snapshot
   falta entre a primeira e a última data compartilhada, e os fatores diários
   coincidem dentro de `1e-9` (relativo).
4. Uma série é `PIT_QUALIFIED` quando o tipo qualifica e um par qualifica;
   senão é `PIT_MISSING` com motivos entre `series_absent`, `kind_revisable`,
   `provider_mismatch`, `no_retrieval_pair_30d`, `overlap_below_250`,
   `overlap_incomplete` e `values_revised`.
5. **Corte para as datas verificadas:** as datas do snapshot fora do intervalo
   compartilhado são `unverified_dates` e a execução nunca as usa. Ela roda só
   na interseção dos seis intervalos verificados, e a primeira linha desse
   intervalo tem retorno zero.

Nível de evidência: não revisão observada entre duas captações (`days_apart` dias); nenhum vintage as-of.
Concordar por alguns meses não prova que um valor de 2019 nunca foi revisto
antes de julho de 2026.

## Regra do veredito

`class_allocation` é `VALIDATED` só quando as seis séries são `PIT_QUALIFIED`,
todas as verificações de integridade passaram e cada perfil tem pelo menos 2
caminhos; senão é `UNVERIFIABLE`, com a lista do que falta.
`basic_recommendation` segue `class_allocation`. `stock_selection`,
`fii_selection` e `premium_recommendation` são `UNVERIFIABLE`: não existem
fundamentos datados nem um universo histórico de tickers. `VALIDATED` é
evidencial, não um teste de desempenho: diz que o protocolo rodou com dados
point-in-time; bater um benchmark é relatado, nunca exigido. O relatório
também dá `selection_overlap` (`full`, `partial` ou `none`), a sobreposição do
período avaliado com o período em que o método foi escolhido (2026-11-01).
Só uma camada `VALIDATED` publica números.

## Como regenerar

A partir da raiz do repositório:

```sh
python3 py/pit_validation.py audit --protocol docs/validation/protocol.json --out docs/validation/pit-audit.json
python3 py/pit_validation.py report --protocol docs/validation/protocol.json --audit docs/validation/pit-audit.json --out-dir docs/validation
```

Códigos de saída: 0 escrito (qualquer veredito), 2 protocolo ou argumentos
inválidos, 3 caminho inexistente, 4 violação de integridade (vazamento de
data, sobreposição, buraco ou janela perdida, corte do snapshot divergente ou
auditoria de outros arquivos). Uma falha não escreve nada.
