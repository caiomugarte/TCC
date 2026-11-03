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
auditoria que difere de uma nova auditoria das captações, que o `report`
refaz antes de rodar). Uma falha não escreve nada.

# Convergência do multi-run (issue #9, fase 1)

Experimento `multirun-convergence-v1`: o Premium escolhe ações e FIIs como o
consenso por frequência de até 150 execuções do GA por seletor. O experimento
mede, no snapshot `dev-2026-10-29-ibov`, quanto esse consenso muda com 30, 60,
100 e 150 execuções e entre seeds, e recomenda `run_count`, os alvos do critério
adaptativo e a zona de proteção (K, δ) do #46 por regras registradas **antes**
da execução. As decisões estão em `.specs/STATE.md` (AD-022, AD-023 e AD-024).

| Arquivo | O que é |
| --- | --- |
| `multirun-convergence-protocol.json` | o protocolo pré-registrado: perfis, seeds, prefixos, regra de parada da produção e as três regras de decisão; commitado antes de qualquer lote medido |
| `multirun-convergence/lots/<perfil>/seed-<seed>.json` | um lote: as 150 execuções de cada seletor (tickers, fitness, HHI, gerações) e o SCORE e o SETOR de cada ticker escolhido, sem tempos nem caminhos |
| `multirun-convergence/lots/<perfil>/seed-<seed>.cost.json` | o custo medido do lote (wall, CPU, RSS, máquina) |
| `multirun-convergence/lots/run-summary.json` | o tempo de parede do experimento e `jobs` |
| `multirun-convergence.json` | o relatório gerado: curvas, frequências, giro, folga de ranking, zona e as recomendações |
| `multirun-convergence-cost.json` | o custo por lote, copiado dos arquivos de custo |
| `multirun-convergence.md` | o mesmo relatório em texto |

**Giro entre seeds** é a diferença entre os consensos de seeds diferentes no
mesmo snapshot: `trocados = n - |C_i ∩ C_j|` para cada par de seeds. Não é o
giro mês a mês (entre snapshots consecutivos), que a fase 2 mede no #46: com
os dados fixos, todo giro entre seeds é ruído do GA, o piso da instabilidade.
O giro dentro da seed (o consenso de uma seed com k execuções contra o mesmo
com 150) é descritivo e não entra em nenhuma regra.

## Regras de decisão (pré-registradas)

- **`run_count`:** para cada célula (perfil x seletor), `S(k)` e `M(k)` são a
  média e a mediana de `trocados` sobre os 10 pares de seeds no prefixo k. A
  célula achatou quando `S(100) - S(150) <= 0,5` (os dois últimos candidatos
  declarados); seu `k_min` é o menor candidato (30, 60, 100, 150) com
  `S(k) <= S(150) + 0,5` **e** `M(k) <= M(150) + 0,5`, ε = 0,5 troca na média e
  na mediana. O valor é o maior `k_min` entre as células, e só existe se todas
  achataram; senão é `data_insufficient`. A regra roda de novo deixando cada
  seed de fora (os 6 pares das outras quatro) e a recomendação é o **máximo**
  das seis (cheia e cinco leave-one-seed-out); `data_insufficient` em qualquer
  uma vale para a recomendação inteira.
- **Alvos do critério adaptativo:** julgados juntos a 150 execuções, como a
  produção para (só para quando os dois valem). `target_jaccard` 0,75 é
  `reachable` numa célula quando todas as seeds têm `jaccard_mean >= 0,75`,
  `unreachable` quando nenhuma tem e `mixed` no resto; `target_cv` 0,02 da
  mesma forma com `fitness_cv <= 0,02`. A ação é `keep` só quando os dois
  vereditos são `reachable` em todas as células; senão `use_fixed_run_count`
  com o `run_count` recomendado (150 quando `data_insufficient`).
- **Zona de proteção (AD-022):** no prefixo recomendado, `K*` é o quantil 0,95
  por CDF invertida dos rankings, no lote da outra seed, de cada titular, nos 20
  pares ordenados de seeds (um titular sem ranking conta como infinito; abaixo
  de n sobe para n; acima de 30 é nulo); a recomendação é `k_offset = K* - n`.
  `δ*` é o quantil 0,95 por CDF invertida de `|f_i(t) - f_j(t)|` nos 10 pares,
  calculado exatamente a partir das contagens (frações, sem float) e
  arredondado para cima em 0,01. Para cada seletor vale o maior `k_offset` e o
  maior δ entre os perfis.

Os valores (ε, quantil, arredondamento, prefixos, seeds, grade de δ) estão no
protocolo; o relatório não tem número escrito à mão.

## Como rodar e regenerar

A partir da raiz do repositório, com o `.venv` do projeto:

```sh
.venv/bin/python scripts/multirun_convergence.py run --protocol docs/validation/multirun-convergence-protocol.json --registry /home/caio/Documents/projetos/TCC/snapshots --lots-dir docs/validation/multirun-convergence/lots --jobs 5
.venv/bin/python scripts/multirun_convergence.py report --protocol docs/validation/multirun-convergence-protocol.json --lots-dir docs/validation/multirun-convergence/lots --out-dir docs/validation
```

`run` recusa (saída 6) um protocolo não commitado ou com mudanças, ou código
modificado em `api/app`, `py` e nos quatro scripts do experimento e do
benchmark; pula lotes já gravados e válidos; nunca sobrescreve um lote
inválido (saída 4). `report` reconstrói os três arquivos byte a byte a partir
do protocolo e dos lotes, sem o registro. Códigos de saída: 0 escrito, 1 um
processo filho falhou, 2 protocolo ou argumentos inválidos, 3 o registro não
tem o snapshot declarado, 4 integridade dos lotes, 6 pré-registro violado.
