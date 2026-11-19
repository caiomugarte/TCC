# Baseline de performance Premium com dados reais (P0-03a, issue #8)

Regenerada em 2026-11-19 depois da issue #95 (fitness do GA avaliado em lote).
Os números da baseline original estão em "Antes e depois da #95".

> **Estado: medido.** Os valores das seções "Comando e commit", "Máquina" e
> "Resultados por perfil e etapa" são **medidos** e vêm do `summary.json` da
> execução registrada abaixo, salvo em
> [`premium-performance-baseline.json`](premium-performance-baseline.json).
> A seção "Modelo inicial de capacidade" é **derivada/estimada**: aplica
> premissas explícitas sobre os números medidos e não é medição.

## Objetivo e escopo

A estimativa anterior de 45 min a 2,7 h por execução Premium vinha de um
benchmark sintético e é um piso, não uma medição. Este relatório mede poucas
execuções reais:

- 3 perfis (conservador, moderado, arrojado), 1 seed cada;
- pipeline completo de dois estágios (`run_premium_optimization`) sobre o
  snapshot declarado;
- por etapa: alocação, ações, FIIs e composição.

Fora de escopo: convergência, qualidade ou estabilidade das carteiras;
alteração de `run_count`; validação fora da amostra.

**Este relatório não conclui nada sobre reduzir ou manter os 150 runs.** Os
números servem só para dimensionar o custo da configuração atual.

## Método

- **Entrada do pipeline.** O script chama
  `run_premium_optimization` (`api/app/adapters/premium_optimization.py`) e
  injeta wrappers cronometrados no lugar de `snapshot_loader`,
  `allocation_engine`, `stock_engine` e `fii_engine`. Cada wrapper chama o
  engine padrão do próprio adapter (`_default_*`); o código de produção não
  muda.
- **Política.** Cada perfil é um Profile v1 sintético no score âncora do
  produto (`ANCHOR_SCORES`: 0,0 / 0,5 / 1,0; todas as dimensões iguais ao
  score; restrição `nenhuma`). A política sai de
  `resolve_recommendation_policy(..., plan="premium")` com os mesmos
  argumentos que `PremiumRecommendationService` usa (ids e hashes do
  manifest, `cutoff_date`, seed). Os parâmetros de GA são os de
  `SYSTEM_GA_CONFIG` (`run_count` 150, `adaptive_mode`, `min_runs` 40,
  população 300, 400 gerações).
- **Seed.** Fixa (`--seed`, padrão 42) e igual para os três perfis.
- **Snapshot.** Declarado em
  [`premium-performance-snapshot.json`](premium-performance-snapshot.json):
  `manifest_id`, `manifest_sha256` e o SHA-256 (`sha256_path` de
  `py/snapshot_manifest.py`) do diretório de alocação e dos CSVs congelados
  do Status Invest (`<registry>/stocks.csv`, `<registry>/fiis.csv`, layout de
  [`docs/snapshot_registry.md`](../snapshot_registry.md)). O script recalcula
  os hashes e roda `validate_manifest` antes de medir qualquer coisa; se algo
  divergir, sai com código 3. Os seletores leem esses CSVs congelados, não o
  `refresh_inputs` ao vivo, então o download fica fora da medição.
- **Etapas.** `snapshot_load` e `allocation` (juntas formam a etapa de
  alocação), `stocks`, `fiis` e `total`, que é a chamada inteira de
  `run_premium_optimization`. A `composition` é derivada:
  `total − (snapshot_load + allocation + stocks + fiis)`, para wall e CPU.
- **Tempo.** Wall time com `time.perf_counter`. CPU com
  `resource.getrusage(RUSAGE_SELF)` (user + sys = "CPU-segundos") e
  `time.process_time`. `RUSAGE_CHILDREN` é registrado à parte. Os dois
  seletores rodam com `parallel=False`, então não há processos filhos e
  "CPU filhos" deve ser 0; se não for, o valor aparece na coluna própria e
  entra em `cpu_seconds_total`.
- **Memória.** Uma thread lê `VmRSS` de `/proc/self/status` a cada 1 s, com
  amostras extras no início e no fim de cada etapa. O pico amostrado por
  etapa é o maior valor dentro da janela da etapa. Ao fim, o pico amostrado
  do processo é conferido com o pico do kernel, `max(ru_maxrss, VmHWM)`:
  `invalid` se o amostrado passar do pico do kernel em mais de 2% ou se
  `ru_maxrss` e `VmHWM` divergirem em mais de 2% (sai com código 4);
  `undersampled` se o amostrado ficar abaixo de 90% do pico. Cada perfil roda
  num processo Python novo (`--isolation process`), então `ru_maxrss` é o
  pico daquele perfil.
- **Gerações e runs.** Só para ações e FIIs. Desde a ADR 0003 a alocação usa
  pesos iguais e não roda GA. Ações: `n_runs` e `all_runs[*].generations_run`
  do resultado de `run_multi_execution_profile`. FIIs: `n_runs` do resultado
  de `run_fii_selection`; o `generations_run` de cada execução é lido por um
  proxy que só observa `fii_selection._run_fii_execution` durante a chamada e
  é restaurado em seguida.
- **Plataforma.** Só Linux (`/proc`, `resource`).

## Comando e commit

Comando para reproduzir (a partir da raiz do repositório, com o venv do
projeto; o diretório de saída é neutro e fica fora do repo):

```bash
.venv/bin/python scripts/benchmark_premium.py \
  --registry <repo>/snapshots \
  --output-dir /tmp/prumo-bench/premium-baseline
```

A execução registrada rodou no commit `086129e` (fitness do GA avaliado em
lote, issue #95) com a árvore limpa, num worktree do repositório, com
`--registry /home/caio/Documents/projetos/TCC/snapshots` e um diretório de
saída temporário (o campo `command` do JSON guarda esse caminho original). O
diretório de saída não entra nas condições comparadas pelo gate.

| Campo | Valor (medido) |
| --- | --- |
| Comando exato | `python scripts/benchmark_premium.py --registry /home/caio/Documents/projetos/TCC/snapshots --output-dir <scratch>/bench/run` (padrões: `--profiles conservador,moderado,arrojado`, `--seed 42`, `--capital 100000`, `--isolation process`) |
| Commit (`git rev-parse HEAD`) | `086129e58b88a6b961d1e246cdf286c933a57c6a` |
| Árvore suja (`dirty` / `code_dirty`) | `false` / `false` |
| `code_fingerprint` | `52ac99b7c13a04dbaa98e4348a94964e1e05dc280397b284fdd142e404abc569` (trees: `api/app` `18a31ad`, `py` `abd65be`, `scripts/bench_measure.py` `0aa2ebc`, `scripts/benchmark_premium.py` `806df9f`) |
| Snapshot (`manifest_id` / `manifest_sha256`) | `dev-2026-10-29-ibov` / `b66a5764bdcb8d8a20cfc4e10f9ad4d7b14448e3c997fec8e597fdcaccd2d014` (`cutoff_date` 2026-10-28) |
| Seed | 42 (mesma nos três perfis) |
| Capital | R$ 100.000,00 |
| Modo / isolamento | `measured` / `process` (um processo Python novo por perfil, em sequência) |
| Data da execução (UTC) | início em 2026-11-19T02:04:24Z (`created_at`, gravado antes do primeiro perfil); duração somada dos três perfis ≈ 488 s ≈ 8,1 min |
| Carga da máquina durante a execução | Não dedicada. Load average ≈ 0,4–1,1 antes de começar; durante a medição rodaram também as suítes de teste (`nice -n 19`) e uma checagem curta de equivalência. Registrado como condição da medição, não controlado. |

O `summary.json` da execução está em
[`docs/validation/premium-performance-baseline.json`](premium-performance-baseline.json)
(cópia byte a byte) e é a referência do gate de regressão.

## Máquina

Máquina local. A conversão para o servidor é uma premissa do modelo de
capacidade, não uma medição.

| Campo | Valor (medido) |
| --- | --- |
| CPU | AMD Ryzen 5 7600X 6-Core Processor (x86_64) |
| Núcleos físicos / threads | 6 / 12 |
| RAM | 31.358.116 kB (29,91 GiB) |
| SO / kernel | Ubuntu 26.04.1 LTS / 7.0.0-43-generic |
| Python | CPython 3.14.4 |
| Bibliotecas (`machine.libraries`) | numpy 2.5.4, pandas 3.0.6 |
| `machine.fingerprint` | `993d05af1e0b9377bd93f221b522a100883613fe528b923d127a8b032049c244` (inclui as versões de numpy e pandas, o kernel e a RAM total) |

## Resultados por perfil e etapa (medido)

Todos os valores desta seção são **medidos** (`mode: measured`). Fonte:
`summary.json` (seção `profiles.<perfil>.stages` e `.engines`).

| Perfil | Etapa | Wall (s) | CPU-s (user+sys) | CPU filhos (s) | CPU/wall | Pico RSS amostrado (MiB) | n_runs efetivo | Gerações (total / média / mín–máx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conservador | alocação (`snapshot_load` + `allocation`) | 0,47 | 0,47 | 0,00 | 1,00 | 147,2 | n/a | n/a (sem GA, ADR 0003) |
| conservador | ações | 85,08 | 85,09 | 0,00 | 1,00 | 192,9 | 150 (de 150) | 12.680 / 84,5 / 51–176 |
| conservador | FIIs | 9,66 | 9,66 | 0,00 | 1,00 | 196,6 | **40** (de 150) | 2.058 / 51,5 / 51–55 |
| conservador | composição (derivada) | 0,001 | 0,001 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| conservador | **total** | **95,20** (1,6 min) | 95,21 | 0,00 | 1,00 | 196,6 | n/a | n/a |
| moderado | alocação (`snapshot_load` + `allocation`) | 0,46 | 0,46 | 0,00 | 1,00 | 146,3 | n/a | n/a (sem GA, ADR 0003) |
| moderado | ações | 108,61 | 108,62 | 0,00 | 1,00 | 192,9 | 150 (de 150) | 13.017 / 86,8 / 51–206 |
| moderado | FIIs | 68,84 | 68,84 | 0,00 | 1,00 | 205,0 | 150 (de 150) | 13.041 / 86,9 / 51–202 |
| moderado | composição (derivada) | 0,001 | 0,001 | 0,00 | 1,01 | n/a (sem amostra) | n/a | n/a |
| moderado | **total** | **177,90** (3,0 min) | 177,92 | 0,00 | 1,00 | 205,0 | n/a | n/a |
| arrojado | alocação (`snapshot_load` + `allocation`) | 0,48 | 0,48 | 0,00 | 1,00 | 146,1 | n/a | n/a (sem GA, ADR 0003) |
| arrojado | ações | 117,87 | 117,89 | 0,00 | 1,00 | 192,9 | 150 (de 150) | 11.232 / 74,9 / 51–169 |
| arrojado | FIIs | 96,64 | 96,65 | 0,00 | 1,00 | 202,4 | 150 (de 150) | 13.248 / 88,3 / 51–210 |
| arrojado | composição (derivada) | 0,002 | 0,002 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| arrojado | **total** | **215,00** (3,6 min) | 215,02 | 0,00 | 1,00 | 202,4 | n/a | n/a |

Notas da tabela (medido): a alocação se divide em `snapshot_load` ≈ 0,03 s
e `allocation` ≈ 0,44 s nos três perfis. A composição dura poucos
milissegundos e não recebeu amostra periódica de RSS (`rss_samples: 0`). Os
CPU/wall de etapas curtas são ruidosos (podem passar de 1,00 por
arredondamento). "CPU filhos" é 0 em todas as etapas, como esperado com
`parallel=False`. Gerações configuradas: 400 por run, população 300.

Parada adaptativa dos seletores (medido, `profiles.<perfil>.engines.<seletor>`;
alvos registrados em todos: `target_cv` 0,02, `target_jaccard` 0,75,
`min_runs` 40):

| Perfil | Seletor | `stop_reason` | `stability_target_met` | `fitness_cv` | `jaccard_mean` | `runs_observed` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| conservador | ações | `run_count` | `false` | 0,0528 | 0,2364 | 150 |
| conservador | FIIs | `stability_target` | `true` | 0,0000 | 1,0000 | 40 |
| moderado | ações | `run_count` | `false` | 0,0528 | 0,1448 | 150 |
| moderado | FIIs | `run_count` | `false` | 0,0261 | 0,5908 | 150 |
| arrojado | ações | `run_count` | `false` | 0,0634 | 0,1176 | 150 |
| arrojado | FIIs | `run_count` | `false` | 0,0494 | 0,1921 | 150 |

Validação do RSS (medido):

| Perfil | Pico amostrado (MiB) | `ru_maxrss` (MiB) | `VmHWM` (MiB) | Razão amostrado/pico | Amostras / maior intervalo (s) | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| conservador | 196,7 | 196,5 | 196,7 | 1,00 | 107 / 1,04 | `ok` |
| moderado | 205,1 | 204,5 | 205,1 | 1,00 | 189 / 1,02 | `ok` |
| arrojado | 202,5 | 201,9 | 202,5 | 1,00 | 226 / 1,03 | `ok` |

`ru_maxrss` e `VmHWM` divergem menos de 0,5% (limite: 2%).
`ru_maxrss_children` é 0.

### Observações factuais

Registradas sem conclusão sobre qualidade, convergência ou sobre o número de
runs.

- **O run Premium inteiro cabe em 1,6 a 3,6 min num núcleo.** Os seletores
  continuam com mais de 99% do wall; alocação (< 0,5 s) e composição
  (< 0,01 s) são desprezíveis.
- **O arrojado passou a ser o perfil mais lento** (215 s), com os FIIs em
  96,6 s. O conservador continua o mais rápido porque os FIIs param em 40
  runs pelo critério de estabilidade (`fitness_cv` 0,0 e `jaccard_mean` 1,0).
  Nos outros cinco seletores o JSON registra `stop_reason: run_count`.
- **O custo por geração ficou entre 4,7 e 10,5 ms** e cresce do conservador
  para o arrojado: ações 6,71 / 8,34 / 10,49 ms e FIIs 4,69 / 5,28 /
  7,30 ms por geração (conservador / moderado / arrojado). Por run, ações
  0,57 / 0,72 / 0,79 s. O fitness deixou de dominar; o que sobra por geração
  é seleção, crossover e o `mutate` (um `random.random()` por gene em
  Python), que a #95 manteve para não mudar o consumo do RNG, mais o custo
  fixo de preparação de cada run. Esta medição não separa as parcelas.
- **`converged_early_runs` mede outra coisa.** É o número de runs em que o GA
  parou antes das 400 gerações configuradas; foi igual ao número de runs em
  todos os seletores. Não é o mesmo critério do `n_runs` adaptativo.

### Antes e depois da #95

A baseline anterior (issue #8, rodada 2, commit `a7c7d38`, registrada em
`6f34f8b`) media 81,1 / 156,6 / 116,8 min (conservador / moderado /
arrojado). O total caiu 51× / 53× / 33×. A comparação direta dos totais não
é limpa: entre `a7c7d38` e `086129e` o código dos seletores mudou (por
exemplo, a exclusão de ações sem score da #52, a zona de proteção da #46, o
teto por segmento dos FIIs e a pontuação própria dos financeiros), e as
gerações mudaram junto (ações do conservador: 12.295 → 12.680). Por isso o
número mais honesto é o custo por geração:

| Perfil | Ações, s/geração (antes → depois) | Ganho | FIIs, s/geração (antes → depois) | Ganho |
| --- | --- | ---: | --- | ---: |
| conservador | 0,327 → 0,0067 | 49× | 0,403 → 0,0047 | 86× |
| moderado | 0,295 → 0,0083 | 35× | 0,374 → 0,0053 | 71× |
| arrojado | 0,264 → 0,0105 | 25× | 0,330 → 0,0073 | 45× |

- **A carteira não muda com a #95.** A troca é só na avaliação do fitness, e
  ela é igual bit a bit à de antes: `tests/test_ga_fitness_batch.py` compara
  o fitness em lote com o `GeneticAlgorithm.fitness` cromossomo a cromossomo
  (com e sem teto por segmento, com `hhi_max`, com cromossomos inviáveis e de
  tamanho errado) e fixa a carteira, o fitness e as gerações do `optimize`
  capturados antes da mudança. Numa checagem local, não commitada, nos
  `df_ranked` reais deste snapshot (ações e FIIs do moderado, com e sem os
  tetos), 120 mil avaliações deram arrays idênticos; a prova commitada é o
  teste, com frames sintéticos.
- As gerações desta execução são as mesmas do protótipo da pesquisa #90
  (`docs/research/premium-run-perf.md`, branch `research/premium-run-perf`),
  que rodou o mesmo snapshot e a mesma seed com o fitness vetorizado por
  monkeypatch.
- O gate se recusa a comparar com a baseline anterior (código 6): o
  `code_fingerprint` mudou e o `machine.fingerprint` também, porque o kernel
  passou de 7.0.0-38 para 7.0.0-43 e a RAM total reportada mudou 12 kB. Os ganhos acima foram calculados à parte,
  direto dos dois JSONs. O JSON anterior continua no histórico do git
  (`git show 6f34f8b:docs/validation/premium-performance-baseline.json`).

## Modelo inicial de capacidade

> **Derivado/estimado, não medido.** Esta seção aplica as premissas abaixo
> aos números medidos da seção anterior. Cada premissa é uma hipótese a
> confirmar; os valores aqui só valem enquanto ela valer. Não há aqui nenhuma
> recomendação sobre o `run_count`.

### Premissas

1. **P1, um worker.** O piloto usa `PremiumExecutor` com `max_workers=1` e
   `max_pending=2` (`api/app/services/premium_executor.py`): uma execução por
   vez, numa thread do processo da API, e no máximo duas aceitas (em execução
   + na fila). Mais de um worker exige mudança de código e, por causa do GIL,
   workers em processos separados, não threads.
2. **P2, CPU-bound de um núcleo.** Medido: CPU/wall ≈ 1,00 no total e "CPU
   filhos" 0. O modelo trata cada execução como ≈ 1 núcleo ocupado o tempo
   todo.
3. **P3, custo independente do capital e da conta.** O capital investível só
   escala os valores em BRL. A seed muda os resultados e, com o modo
   adaptativo, pode mudar o `n_runs` efetivo e as gerações (ver os FIIs do
   conservador); uma seed por perfil não mede essa variância.
4. **P4, conversão para o servidor (não verificada).** `T_servidor = k ×
   T_local`, com `k` = razão de desempenho single-thread entre o servidor e
   esta máquina. `k` **não foi medido**; o jeito mais barato de obtê-lo é
   rodar `--profiles moderado` uma vez no servidor (agora leva ≈ 3 min).
5. **P5, memória por worker ≈ 210 MiB.** Pico medido do pior perfil: 205,1
   MiB (moderado) num processo Python dedicado; o processo já está em ≈ 146
   MiB depois dos imports e da carga do snapshot.
6. **P6, mix de perfis.** Sem dados de uso, a média simples dos três perfis
   representa uma execução típica.
7. **P7, sem cache entre execuções.** Cada execução recalcula tudo; não há
   reaproveitamento entre contas.
8. **P8, núcleos físicos.** Um worker por núcleo físico; as threads SMT não
   contam como capacidade extra para carga CPU-bound.

### Valores derivados

Com `T_p` = wall total medido do perfil `p`: `T_cons` = 95,20 s,
`T_mod` = 177,90 s, `T_arr` = 215,00 s; `T̄ = média(T_p)` = 162,70 s
(2,7 min); `T_max` = 215,00 s (arrojado); `M_max` = 205,1 MiB.

Execuções por worker (1 núcleo, 24 h de uso contínuo, P1, P2, P7):

| Perfil | Wall total (min) | Execuções/h por worker | Execuções/dia por worker (local) | Execuções/dia por worker (servidor) |
| --- | ---: | ---: | ---: | --- |
| conservador | 1,6 | 37,8 | 908 | 908 / k |
| moderado | 3,0 | 20,2 | 486 | 486 / k |
| arrojado | 3,6 | 16,7 | 402 | 402 / k |
| média (P6) | 2,7 | 22,1 | 531 | 531 / k |

| Grandeza | Fórmula | Valor local (derivado) | Valor servidor (derivado, P4) |
| --- | --- | --- | --- |
| Espera no pior caso para a 2ª execução aceita (P1) | `2 × T_max` | 430 s ≈ 7,2 min | 7,2 × k min |
| Workers para N núcleos físicos (P2, P5, P8) | `min(N, floor(RAM_livre / M_max))` | N (a CPU limita antes da RAM) | N do servidor, conferindo a RAM livre |
| Execuções por dia com N workers | `N × 86400 / T̄` | 531 × N (6 núcleos: ≈ 3.186) | 531 × N / k |
| Timeout sugerido por execução | `1,5 × T_max`, arredondado | 322,5 s → **6 min** | 6 × k min |

O timeout usa uma margem de 50% sobre o pior perfil medido para cobrir
variância de seed e carga (P3), que não foi medida; a margem é uma escolha,
não um dado, e esta issue não muda nenhum timeout no código.

## Gate de regressão

O próprio script é o gate; não há CI.

```bash
.venv/bin/python scripts/benchmark_premium.py \
  --registry <repo>/snapshots \
  --output-dir /tmp/prumo-bench/premium-rerun \
  --baseline docs/validation/premium-performance-baseline.json
```

- Sai com código **5** se o wall total de qualquer perfil ficar mais de 20%
  acima do baseline (`--threshold`, padrão 0,20). Com totais de 1,6 a 3,6
  min, 20% são 19 a 43 s, perto do ruído de carga de uma máquina não
  dedicada.
- Sai com código **6** e não compara se mudar qualquer condição: código
  (`code_fingerprint`, os tree hashes de `api/app`, `py`,
  `scripts/benchmark_premium.py` e `scripts/bench_measure.py`, que não mudam
  com commits só de docs), árvore de código suja, snapshot, seed, máquina
  (`machine.fingerprint`, que inclui as versões de numpy e pandas e o
  kernel), modo (`measured` ou `smoke`), política do perfil ou perfil ausente
  do baseline.
- Para comparar dois `summary.json` já existentes sem rodar nada:
  `--compare-only <summary.json> --baseline <baseline.json>`. Conferência do
  JSON salvo contra ele mesmo (deve sair com 0):

  ```bash
  .venv/bin/python scripts/benchmark_premium.py \
    --compare-only docs/validation/premium-performance-baseline.json \
    --baseline docs/validation/premium-performance-baseline.json
  ```

- O baseline vale para esta máquina: uma execução no servidor sai com 6 por
  causa do `machine.fingerprint` e serve como nova medição (para obter `k`),
  não como comparação.

## Limitações

- Uma seed por perfil e uma execução por perfil: não mede a variância entre
  seeds nem entre contas, nem o ruído entre execuções. O limiar de 20% do
  gate não foi calibrado contra esse ruído.
- Máquina não dedicada: as suítes de teste rodaram com `nice -n 19` durante
  parte da medição. Os CPU/wall ≈ 1,00 sugerem pouca contenção, mas o efeito
  não foi isolado.
- `fitness_cv` e `jaccard_mean` são os valores do fim de cada seletor; o
  JSON não guarda a série por checagem.
- Máquina local; o servidor entra só pela premissa P4.
- O amostrador de 1 s pode perder picos de memória com menos de 1 s; o
  status `undersampled` sinaliza isso pela comparação com `VmHWM`.
- `composition` é derivada por subtração e absorve o overhead dos wrappers e
  do amostrador (desprezível, mas não zero).
- `--smoke` (1 run, 5 gerações) serve só para conferir a ligação com dados
  reais; não é medição e o gate se recusa a comparar com ele.
