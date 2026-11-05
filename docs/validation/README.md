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
| `multirun-convergence.md` | o mesmo relatório em texto (a frase da linha 57 sobre a execução de produção está errada: ver a Errata no fim desta seção) |

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

## Limitação: o giro com a zona é medido dentro da amostra

- **O giro no ponto recomendado é otimista.** `recommend_zone` ajusta `K*` e
  `δ*` nos 20 pares ordenados de seeds (`scripts/convergence_analysis.py:1134`)
  e simula o ponto recomendado nesses mesmos pares (`:1148`). A coluna "giro no
  ponto recomendado" do relatório (média 0,40 / 0,10 / 0,70 / 0,00) é, portanto,
  dentro da amostra e otimista.
- **Sensibilidade leave-one-seed-out (exploratória, não pré-registrada).** Uma
  sonda da revisão do PR #51 (não commitada) ajusta K e δ com a mesma regra
  em 4 seeds (o maior entre os perfis) e simula só os 8 pares ordenados com a
  seed deixada de fora. O giro médio com a zona sobe para 0,300 em
  `premium-2026-11-02/stocks` (0,10 dentro da amostra), 0,725 em
  `arrojado-anchor/stocks` (0,40) e 0,775 em `arrojado-anchor/fiis` (0,70);
  `premium-2026-11-02/fiis` fica em 0,000. Sem a zona (K = 0) o giro é o mesmo
  dos dois jeitos: 1,10, 1,40 e 1,30.
- **K depende do conjunto de seeds.** Nos cinco ajustes sem uma seed, o
  `k_offset` de ações vai de 2 a 6 (6 com as cinco seeds), o de FIIs de 1 a 4
  (3) e δ de 0,10 a 0,12 (0,11).
- **A recomendação pré-registrada não muda.** `k_offset` 6 / δ 0,11 para ações
  e `k_offset` 3 / δ 0,11 para FIIs seguem sendo a recomendação; a sonda é uma
  sensibilidade e não a substitui. O #46 precisa considerar que o giro com a
  zona fora da amostra é maior e que K é sensível ao conjunto de seeds ao
  dimensionar K e δ.

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

## Custo do experimento medido

Os números por lote estão em `multirun-convergence-cost.json` e na tabela de
custo do `.md`, copiados dos arquivos de custo: 24,22 CPU-horas medidas nos 10
lotes (11,64 nos 5 de `arrojado-anchor`, 12,58 nos 5 de `premium-2026-11-02`),
com `jobs` 5 numa AMD Ryzen 5 7600X (6 núcleos / 12 threads). O que os arquivos
de custo não mostram:

- **A execução teve duas invocações.** A primeira começou em 2026-11-03 às
  15:02 (-03) e gravou os 5 lotes de `arrojado-anchor` entre 17:27 e 17:32
  (8.984,8 s do primeiro início ao último fim). Os 5 processos de
  `premium-2026-11-02` dela rodaram cerca de 1h20 e foram interrompidos de
  propósito para reiniciar a máquina, sem gravar nada. O mesmo comando,
  retomado em 2026-11-04 às 00:53 (-03), pulou os 5 lotes válidos, gravou os 5
  de `premium-2026-11-02` e saiu com 0 por volta de 03:30 (9.416,4 s).
- **O tempo de parede do relatório cobre só a segunda invocação.**
  `run-summary.json` é escrito pela invocação que iniciou filhos, então os
  9.416,4 s do relatório são só os lotes de `premium-2026-11-02`. Somando as
  duas janelas que gravaram lotes, o experimento levou 18.401,2 s (cerca de
  5,1 h) de parede, fora a parte descartada.
- **Computação descartada, fora das 24,22 CPU-horas:** cerca de 1h20 x 5
  processos (por volta de 6,7 horas-processo) da primeira tentativa de
  `premium-2026-11-02`. Não foi medida: o log da primeira invocação ficava em
  `/tmp` e não sobreviveu à reinicialização, então nem o 1h20 (estimativa do
  operador) nem o total de cerca de 31 citado antes têm registro. As duas
  grandezas ficam separadas: 24,22 CPU-horas medidas nos lotes gravados e
  cerca de 6,7 horas-processo estimadas na parte descartada. Horas-processo
  não são CPU-horas medidas, então não se somam.
- **O kernel mudou na reinicialização** (7.0.0-38-generic nos lotes de
  `arrojado-anchor`, 7.0.0-42-generic nos de `premium-2026-11-02`), por isso o
  relatório lista dois fingerprints de máquina para o mesmo processador. Python
  (3.14.4), numpy (2.5.4) e pandas (3.0.6) são os mesmos nos 10 arquivos de
  custo (`machine.libraries`), então os lotes vêm do mesmo build.

## Errata: a execução de produção de 2026-11-02 não foi refeita

O relatório gerado diz, em `docs/validation/multirun-convergence.md:57`
(escrito por `scripts/convergence_analysis.py:1435-1436`): "O lote
`premium-2026-11-02` com a seed 46641103 refaz a execução
`9e22cc16-cfda-495e-a8c9-c0d202ef32b6` sem o critério adaptativo". Isso está
errado. O `.md` não é corrigido à mão porque `report` o regenera byte a byte a
partir de código congelado para a auditoria (nada em `scripts/` mudou depois de
`9b3b375`, o commit de código dos lotes).

O lote reusa o código, o perfil, a política, a seed e o snapshot de alocação
da execução de produção, e o critério adaptativo não disparou nela (a
estabilidade final, Jaccard 0,137 / CV 0,054 nas ações e 0,386 / 0,042 nos
FIIs, nunca atingiu 0,75 / 0,02, então ela rodou as 150 execuções). O que
difere são as entradas dos seletores. A produção baixa de novo o Status Invest
para o workspace a cada execução (`api/app/services/premium_executor.py:104`,
que chama `status_invest_inputs.refresh_inputs`) e entrega esses arquivos aos
dois seletores. No banco de desenvolvimento, a execução `9e22cc16` registra em
`result_json.provenance.selector_sources` o sha256 `4323377c…` para ações
(baixado em 2026-11-02T20:47:25Z) e `7c5fb662…` para FIIs. Esse banco é a
cópia local, não commitada, de `api/prumo-dev.db` no checkout principal: o
`api/prumo-dev.db` commitado não tem a execução `9e22cc16`. Os valores
completos lá registrados são, para ações, sha256
`4323377c9204d5bdba8947e4b310c31e971bacfef9c3628c6c654c75679ce648`
(`retrieved_at` 2026-11-02T20:47:25.048114Z) e, para FIIs, sha256
`7c5fb66236d02ed83901fa45573038ae9ff0d9fa900aa01ab498b477a9586419`
(`retrieved_at` 2026-11-02T20:47:29.649295Z). O experimento usa
os CSVs congelados do snapshot: `690e9ae9…` (ações) e `549af15b…` (FIIs).
Então o `production_run_match` falso nos dois seletores mede uma diferença de
entrada, não o GA nem o critério de parada.

**Limitação.** Como cada execução Premium baixa as entradas de novo, o giro que
o usuário vê na produção inclui também a deriva dos dados baixados dentro do
mesmo snapshot, além do ruído do GA. Este experimento, com entradas
congeladas, mede só o ruído do GA; a deriva de entrada não está medida. O #46
precisa considerá-la ao dimensionar K e δ.

**Achado fora do #9.** A proveniência da política da execução
(`policy_json.provenance.source_snapshot_hashes.stock` e `.fii`) registra os hashes
do snapshot (`690e9ae9…` e `549af15b…`), embora os seletores tenham rodado
sobre o download novo (`4323377c…` e `7c5fb662…`). Quem lê a proveniência
acredita que a execução usou o snapshot. Isso fica para uma issue separada; o
#9 não muda código de produção.

# Giro mês a mês com a zona de proteção (issue #46, fase 2 da #9)

**Estado: pendente: snapshot de novembro.** Não há manifesto publicado com corte
em novembro (só `dev-2026-10-29` e `dev-2026-10-29-ibov`), então a medição não
rodou e não há número aqui. Os dois arquivos abaixo não existem até ela rodar.

| Arquivo | O que é |
| --- | --- |
| `protection-zone-turnover.json` | o giro de outubro para novembro por classe, com e sem a zona, e o que ficou, trocou e saiu |
| `protection-zone-turnover.md` | o mesmo relatório em texto |

`scripts/protection_zone_turnover.py measure` roda o pipeline Premium de
produção uma vez num manifesto publicado de novembro, com o registro de perfil
`premium-2026-11-02`, a política `premium-policy-v3`, a seed 46641103 e, como
titulares, os tickers que a execução de produção
`9e22cc16-cfda-495e-a8c9-c0d202ef32b6` entregou em outubro
(`multirun-convergence-protocol.json`, `production_run`). A zona roda depois do
consenso, então as mesmas execuções do GA dão os dois números: `turnover` (com
a zona) e `consensus_turnover` (consenso puro), ambos contados como entradas
contra a carteira de outubro.

## O que a medição precisa

1. Um snapshot de novembro construído, validado e publicado com
   `py/snapshot_registry.py` (`build`, `validate`, `publish`), com corte em ou
   depois de 2026-11-01.
2. Um `HEAD` commitado e limpo: o relatório grava o commit e se a árvore tinha
   mudanças.
3. Cerca de 2,5 h de parede (uma execução Premium 13 + 13 sem paralelismo).

```sh
.venv/bin/python scripts/protection_zone_turnover.py measure --registry /home/caio/Documents/projetos/TCC/snapshots --manifest-id <manifesto-de-novembro> --out-dir docs/validation
```

Códigos de saída: 0 escrito, 1 o pipeline falhou ou um seletor não aplicou a
zona, 2 argumentos ou protocolo inválidos, 3 manifesto ausente, não publicado,
inválido, o de outubro (`dev-2026-10-29-ibov`) ou com corte antes de
2026-11-01. Uma falha não escreve nada. Rodar de novo com o mesmo manifesto, o
mesmo código e a mesma seed reescreve os dois arquivos byte a byte.

## Limitações conhecidas antes de medir

- **Pontuação de outubro anterior ao #52.** Os titulares vêm da execução
  `9e22cc16`, pontuada antes do #52 (ações com SCORE NaN ainda podiam ser
  escolhidas). Novembro descarta essas ações antes do GA, então parte das
  saídas de ações pode vir da mudança de pontuação e não do snapshot novo.
- **Seed fixa.** A medição usa a seed de outubro. A produção deriva a seed da
  versão da política (`derive_premium_seed`), então uma execução real em
  `premium-policy-v3` na mesma conta, perfil e manifesto sorteia execuções do
  GA diferentes.
- **Entradas congeladas.** A medição lê os CSVs do manifesto; a execução de
  outubro rodou sobre o Status Invest baixado na hora (ver a Errata acima).
  A deriva dos dados baixados não entra no número.
- **Um perfil e um par de meses.** K e δ vêm da calibração da #9, feita em
  lotes que ainda tinham ações com SCORE NaN.

A primeira, a segunda e a última também ficam gravadas em `limitations` no
relatório gerado.
