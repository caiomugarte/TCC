# Baseline de performance Premium com dados reais (P0-03a, issue #8)

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

A execução registrada rodou no commit `a7c7d38` com a árvore limpa, com
`--registry /home/caio/Documents/projetos/TCC/snapshots` e um diretório de
saída temporário (o campo `command` do JSON guarda esse caminho original). O
diretório de saída não entra nas condições comparadas pelo gate.

| Campo | Valor (medido) |
| --- | --- |
| Comando exato | `python scripts/benchmark_premium.py --registry /home/caio/Documents/projetos/TCC/snapshots --output-dir <scratch>/bench-baseline-r2` (padrões: `--profiles conservador,moderado,arrojado`, `--seed 42`, `--capital 100000`, `--isolation process`) |
| Commit (`git rev-parse HEAD`) | `a7c7d38544102e08bbd120f0be5c4cc013b0e02e` |
| Árvore suja (`dirty` / `code_dirty`) | `false` / `false` |
| `code_fingerprint` | `1d52d44d1db268280112a0495a827a1589e82121c1655ef1bcc23cfb3c35ff40` (trees: `api/app` `9cc956f`, `py` `b86aec5`, `scripts/bench_measure.py` `0aa2ebc`, `scripts/benchmark_premium.py` `e057bf3`) |
| Snapshot (`manifest_id` / `manifest_sha256`) | `dev-2026-10-29-ibov` / `b66a5764bdcb8d8a20cfc4e10f9ad4d7b14448e3c997fec8e597fdcaccd2d014` (`cutoff_date` 2026-10-28) |
| Seed | 42 (mesma nos três perfis) |
| Capital | R$ 100.000,00 |
| Modo / isolamento | `measured` / `process` (um processo Python novo por perfil, em sequência) |
| Data da execução (UTC) | início em 2026-11-03T07:39:17Z (`created_at`, gravado antes do primeiro perfil); duração somada dos três perfis ≈ 21.266 s ≈ 5,9 h |
| Carga da máquina durante a execução | Não dedicada: havia outro trabalho leve. Load average ≈ 1,3–2,6 (12 threads lógicas). Registrado como condição da medição, não controlado. |

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
| RAM | 31.358.128 kB (29,91 GiB) |
| SO / kernel | Ubuntu 26.04.1 LTS / 7.0.0-38-generic |
| Python | CPython 3.14.4 |
| Bibliotecas (`machine.libraries`) | numpy 2.5.4, pandas 3.0.6 |
| `machine.fingerprint` | `7e180109c5fd263c7687373422045f6a29636375972351b195d017da41706955` (inclui as versões de numpy e pandas) |

## Resultados por perfil e etapa (medido)

Todos os valores desta seção são **medidos** (`mode: measured`). Fonte:
`summary.json` (seção `profiles.<perfil>.stages` e `.engines`).

| Perfil | Etapa | Wall (s) | CPU-s (user+sys) | CPU filhos (s) | CPU/wall | Pico RSS amostrado (MiB) | n_runs efetivo | Gerações (total / média / mín–máx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conservador | alocação (`snapshot_load` + `allocation`) | 0,48 | 0,48 | 0,00 | 1,00 | 145,8 | n/a | n/a (sem GA, ADR 0003) |
| conservador | ações | 4.020,37 | 3.984,34 | 0,00 | 0,991 | 191,8 | 150 (de 150) | 12.295 / 82,0 / 51–209 |
| conservador | FIIs | 842,70 | 834,94 | 0,00 | 0,991 | 194,1 | **40** (de 150) | 2.090 / 52,2 / 51–58 |
| conservador | composição (derivada) | 0,002 | 0,002 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| conservador | **total** | **4.863,55** (81,1 min) | 4.819,75 | 0,00 | 0,991 | 194,1 | n/a | n/a |
| moderado | alocação (`snapshot_load` + `allocation`) | 0,48 | 0,48 | 0,00 | 1,00 | 145,6 | n/a | n/a (sem GA, ADR 0003) |
| moderado | ações | 3.972,66 | 3.940,86 | 0,00 | 0,992 | 191,5 | 150 (de 150) | 13.475 / 89,8 / 51–176 |
| moderado | FIIs | 5.422,18 | 5.381,24 | 0,00 | 0,992 | 203,8 | 150 (de 150) | 14.490 / 96,6 / 51–232 |
| moderado | composição (derivada) | 0,003 | 0,003 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| moderado | **total** | **9.395,33** (156,6 min) | 9.322,58 | 0,00 | 0,992 | 203,8 | n/a | n/a |
| arrojado | alocação (`snapshot_load` + `allocation`) | 0,47 | 0,47 | 0,00 | 1,00 | 145,9 | n/a | n/a (sem GA, ADR 0003) |
| arrojado | ações | 3.068,71 | 3.034,70 | 0,00 | 0,989 | 191,5 | 150 (de 150) | 11.640 / 77,6 / 51–205 |
| arrojado | FIIs | 3.937,64 | 3.924,14 | 0,00 | 0,997 | 202,7 | 150 (de 150) | 11.918 / 79,5 / 51–202 |
| arrojado | composição (derivada) | 0,002 | 0,002 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| arrojado | **total** | **7.006,83** (116,8 min) | 6.959,32 | 0,00 | 0,993 | 202,7 | n/a | n/a |

Notas da tabela (medido): a alocação se divide em `snapshot_load` ≈ 0,03 s
e `allocation` ≈ 0,45 s nos três perfis. A composição dura poucos
milissegundos e não recebeu amostra periódica de RSS (`rss_samples: 0`). Os
CPU/wall de etapas abaixo de 1 s são ruidosos (podem passar de 1,00 por
arredondamento). "CPU filhos" é 0 em todas as etapas, como esperado com
`parallel=False`. Gerações configuradas: 400 por run, população 300.

Parada adaptativa dos seletores (medido, `profiles.<perfil>.engines.<seletor>`;
alvos registrados em todos: `target_cv` 0,02, `target_jaccard` 0,75,
`min_runs` 40):

| Perfil | Seletor | `stop_reason` | `stability_target_met` | `fitness_cv` | `jaccard_mean` | `runs_observed` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| conservador | ações | `run_count` | `false` | 0,0502 | 0,2659 | 150 |
| conservador | FIIs | `stability_target` | `true` | 0,0000 | 1,0000 | 40 |
| moderado | ações | `run_count` | `false` | 0,0587 | 0,1588 | 150 |
| moderado | FIIs | `run_count` | `false` | 0,0435 | 0,4735 | 150 |
| arrojado | ações | `run_count` | `false` | 0,0570 | 0,1158 | 150 |
| arrojado | FIIs | `run_count` | `false` | 0,0556 | 0,1885 | 150 |

Validação do RSS (medido):

| Perfil | Pico amostrado (MiB) | `ru_maxrss` (MiB) | `VmHWM` (MiB) | Razão amostrado/pico | Amostras / maior intervalo (s) | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| conservador | 194,2 | 193,4 | 194,2 | 1,00 | 4.869 / 1,04 | `ok` |
| moderado | 203,9 | 203,1 | 203,9 | 1,00 | 9.397 / 1,05 | `ok` |
| arrojado | 202,8 | 202,2 | 202,8 | 1,00 | 7.012 / 1,06 | `ok` |

O pico amostrado do processo inclui as amostras de fronteira e fica alguns
centésimos de MiB acima do pico da etapa `total`. `ru_maxrss` e `VmHWM`
divergem menos de 0,5% (limite: 2%). `ru_maxrss_children` é 0.

### Observações factuais

Registradas sem conclusão sobre qualidade, convergência ou sobre o número de
runs.

- **FIIs do conservador pararam em 40 runs, pelo critério de estabilidade.**
  O JSON registra `stop_reason: stability_target` e
  `stability_target_met: true`, com `fitness_cv` 0,0 (alvo ≤ 0,02) e
  `jaccard_mean` 1,0 (alvo ≥ 0,75) em 40 runs, que é o `min_runs`. Nos
  outros cinco seletores o JSON registra `stop_reason: run_count` e
  `stability_target_met: false`: rodaram os 150 runs configurados com
  `fitness_cv` entre 0,044 e 0,059 e `jaccard_mean` entre 0,116 e 0,474
  (tabela acima).
- **O critério de parada é checado em ritmos diferentes.** O seletor de FIIs
  (`run_fii_selection`, ramo `parallel=False`) checa a estabilidade depois de
  cada run e só pode parar a partir do `min_runs`; o de ações
  (`run_multi_execution_profile`) checa a cada 10 runs e imprime CV e
  Jaccard no `run.log`. Os valores de `fitness_cv` e `jaccard_mean` da tabela
  são os do fim de cada seletor.
- **`converged_early_runs` mede outra coisa.** É o número de runs em que o GA
  parou antes das 400 gerações configuradas (parada antecipada por geração);
  foi igual ao número de runs em todos os seletores (40/40 nos FIIs do
  conservador e 150/150 nos demais). Não é o mesmo critério do `n_runs`
  adaptativo.
- **O tempo por geração não é constante.** Wall por geração: ações 0,327 /
  0,295 / 0,264 s e FIIs 0,403 / 0,374 / 0,330 s (conservador / moderado /
  arrojado). Por run: ações 26,8 / 26,5 / 20,5 s e FIIs 21,1 / 36,1 / 26,3 s.
- **Os seletores dominam o tempo.** Ações + FIIs respondem por mais de
  99,99% do wall total nos três perfis; alocação (< 0,5 s) e composição
  (< 0,01 s) são desprezíveis.
- **O moderado foi o perfil mais lento** (9.395 s), puxado pelos FIIs
  (5.422 s, 96,6 gerações em média). O conservador foi o mais rápido
  (4.864 s), em boa parte porque os FIIs rodaram 40 runs.
- **Faixa medida contra a estimativa sintética anterior.** O total medido
  ficou entre 81 e 157 min por execução, dentro da faixa sintética de 45 min
  a 2,7 h citada no objetivo (e perto do teto no moderado).

### Repetição (rodada 1 contra rodada 2)

Fato registrado, não calibração. A rodada 1 (commit `7bf7284`, início
2026-11-03T01:07:14Z, load ≈ 2–4,7) e a rodada 2 (commit `a7c7d38`, a deste
relatório, load ≈ 1,3–2,6) usaram o mesmo código de engine (tree hashes
iguais nos dois JSONs: `api/app` `9cc956f` e `py` `b86aec5`), o mesmo
snapshot (`manifest_sha256` e `sources` iguais), a mesma seed (42), a mesma
política por perfil (`policy_sha256` igual) e o mesmo hardware (todos os
campos de `machine` iguais, exceto `libraries`, que a rodada 1 não
registrava). Mudaram só os arquivos do benchmark
(`scripts/benchmark_premium.py` e o novo `scripts/bench_measure.py`).

- Gerações por etapa idênticas: `generations_total`, mín, máx, `n_runs`
  efetivo e `runs_observed` de ações e FIIs coincidem nos três perfis, assim
  como os ativos escolhidos e os pesos por classe.
- Wall total da rodada 2 contra a rodada 1: conservador 4.945,83 →
  4.863,55 s (−1,66%), moderado 9.454,15 → 9.395,33 s (−0,62%), arrojado
  6.982,90 → 7.006,83 s (+0,34%). Por seletor: ações −1,72% / −0,42% /
  +0,95% e FIIs −1,37% / −0,77% / −0,13% (conservador / moderado /
  arrojado).
- O próprio gate se recusa a comparar as duas rodadas: `--compare-only`
  da rodada 2 contra a rodada 1 sai com código 6, porque o
  `code_fingerprint` (inclui os arquivos do benchmark) e o
  `machine.fingerprint` (agora inclui numpy e pandas) diferem. Os deltas
  acima foram calculados à parte, direto dos dois JSONs.
- É uma única repetição (n = 2) com cargas de máquina diferentes. Não mede a
  distribuição do ruído e não calibra o limiar de 20% do gate.

## Modelo inicial de capacidade

> **Derivado/estimado, não medido.** Esta seção aplica as premissas abaixo
> aos números medidos da seção anterior. Cada premissa é uma hipótese a
> confirmar; os valores aqui só valem enquanto ela valer. Não há aqui nenhuma
> recomendação sobre o `run_count`.

### Premissas

1. **P1, um worker.** O piloto usa `PremiumExecutor` com `max_workers=1` e
   `max_pending=2` (`api/app/services/premium_executor.py`): uma execução por
   vez, numa thread do processo da API, e no máximo duas aceitas (em execução
   + na fila). Hoje o executor rejeita `max_workers != 1`; mais de um worker
   exige mudança de código e, por causa do GIL, workers em processos
   separados, não threads.
2. **P2, CPU-bound de um núcleo.** Medido: CPU/wall de 0,991 a 0,993 no total
   e "CPU filhos" 0. O modelo trata cada execução como ≈ 1 núcleo ocupado o
   tempo todo (≈ 0,99 CPU/wall). A execução de referência rodou com a
   máquina sob carga leve (load ≈ 1,3–2,6), sem contenção de núcleo
   aparente.
3. **P3, custo independente do capital e da conta.** O capital investível só
   escala os valores em BRL. A seed muda os resultados e, com o modo
   adaptativo, pode mudar o `n_runs` efetivo e as gerações (ver os FIIs do
   conservador); uma seed por perfil não mede essa variância.
4. **P4, conversão para o servidor (não verificada).** `T_servidor = k ×
   T_local`, com `k` = razão de desempenho single-thread entre o servidor e
   esta máquina. `k` **não foi medido**; o jeito mais barato de obtê-lo é
   rodar `--profiles moderado` uma vez no servidor. Até lá, os valores de
   servidor abaixo são fórmulas em `k`, não números.
5. **P5, memória por worker ≈ 205 MiB.** Pico medido do pior perfil: 203,9
   MiB (moderado) num processo Python dedicado; o processo já está em ≈ 146
   MiB depois dos imports e da carga do snapshot, então a execução em si
   acrescenta ≈ 58 MiB. Num worker dentro do processo da API o número muda;
   o modelo usa 205 MiB por worker mais uma folga a definir para a API e o
   SO.
6. **P6, mix de perfis.** Sem dados de uso, a média simples dos três perfis
   representa uma execução típica.
7. **P7, sem cache entre execuções.** Cada execução recalcula tudo; não há
   reaproveitamento entre contas.
8. **P8, núcleos físicos.** Um worker por núcleo físico; as threads SMT não
   contam como capacidade extra para carga CPU-bound.

### Valores derivados

Com `T_p` = wall total medido do perfil `p`: `T_cons` = 4.863,55 s,
`T_mod` = 9.395,33 s, `T_arr` = 7.006,83 s; `T̄ = média(T_p)` = 7.088,57 s
(118,1 min ≈ 1,97 h); `T_max` = 9.395,33 s (moderado, ≈ 2,61 h);
`M_max` = 203,9 MiB.

Execuções por worker (1 núcleo, 24 h de uso contínuo, P1, P2, P7):

| Perfil | Wall total (h) | Execuções/h por worker | Execuções/dia por worker (local) | Execuções/dia por worker (servidor) |
| --- | ---: | ---: | ---: | --- |
| conservador | 1,35 | 0,74 | 17,8 | 17,8 / k |
| moderado | 2,61 | 0,38 | 9,2 | 9,2 / k |
| arrojado | 1,95 | 0,51 | 12,3 | 12,3 / k |
| média (P6) | 1,97 | 0,51 | 12,2 | 12,2 / k |

| Grandeza | Fórmula | Valor local (derivado) | Valor servidor (derivado, P4) |
| --- | --- | --- | --- |
| Execuções por hora, 1 worker (P1, P6) | `3600 / T̄` | 0,51 | 0,51 / k |
| Execuções por dia, 1 worker | `86400 / T̄` | 12,2 | 12,2 / k |
| Espera no pior caso para a 2ª execução aceita (P1) | `2 × T_max` (a que está rodando + a própria) | 18.791 s ≈ 5,22 h | 5,22 × k h |
| Workers para N núcleos físicos (P2, P5, P8) | `min(N, floor(RAM_livre / M_max))` | N (a CPU limita antes da RAM: cada worker usa ≈ 0,2 GiB) | N do servidor, conferindo a RAM livre |
| Execuções por dia com N workers | `N × 86400 / T̄` | 12,2 × N (6 núcleos: ≈ 73,1) | 12,2 × N / k |
| Memória para N workers (P5) | `N × 205 MiB` + folga | 1 worker ≈ 0,2 GiB; 4 ≈ 0,8 GiB; 6 ≈ 1,2 GiB | idem |
| Timeout sugerido por execução | `1,5 × T_max`, arredondado | 14.093 s (3,91 h) → **4 h** | 4 × k h (recalibrar após medir `k`) |
| Custo por geração, ações | `wall_ações / gerações_total_ações` | 0,327 / 0,295 / 0,264 s (cons / mod / arr) | × k |
| Custo por geração, FIIs | `wall_FIIs / gerações_total_FIIs` | 0,403 / 0,374 / 0,330 s (cons / mod / arr) | × k |
| Participação dos seletores no total | `(wall_ações + wall_FIIs) / T_p` | > 99,99% nos três perfis | n/a |

O timeout usa uma margem de 50% sobre o pior perfil medido para cobrir
variância de seed e carga (P3), que não foi medida; a margem é uma escolha,
não um dado. A espera de 5,22 h é o caso em que duas execuções moderadas
ficam enfileiradas no único worker.

O custo por geração permite reestimar `T_p` para outro número efetivo de
runs ou de gerações sem nova medição, mas a medição mostra que ele varia
entre perfis e seletores (0,26 a 0,40 s), então a extrapolação vale só
dentro do mesmo perfil e seletor. Isso é uma extrapolação, não uma
recomendação de mudar o `run_count`.

## Gate de regressão

O próprio script é o gate; não há CI.

```bash
.venv/bin/python scripts/benchmark_premium.py \
  --registry <repo>/snapshots \
  --output-dir /tmp/prumo-bench/premium-rerun \
  --baseline docs/validation/premium-performance-baseline.json
```

- Sai com código **5** se o wall total de qualquer perfil ficar mais de 20%
  acima do baseline (`--threshold`, padrão 0,20).
- Sai com código **6** e não compara se mudar qualquer condição: código
  (`code_fingerprint`, os tree hashes de `api/app`, `py`,
  `scripts/benchmark_premium.py` e `scripts/bench_measure.py`, que não mudam
  com commits só de docs), árvore de código suja, snapshot, seed, máquina
  (`machine.fingerprint`, que inclui as versões de numpy e pandas),
  modo (`measured` ou `smoke`), política do perfil ou perfil ausente do
  baseline.
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

- Uma seed por perfil e duas execuções por perfil (rodadas 1 e 2, n = 2):
  não mede a variância entre seeds nem entre contas, e uma repetição não
  mede a distribuição do ruído entre execuções. O limiar de 20% do gate não
  foi calibrado contra esse ruído.
- Máquina não dedicada: havia outro trabalho leve durante a medição (load
  average ≈ 1,3–2,6 em 12 threads). Os CPU/wall ≈ 0,99 sugerem pouca
  contenção, mas o efeito não foi isolado.
- `fitness_cv` e `jaccard_mean` são os valores do fim de cada seletor; o
  JSON não guarda a série por checagem.
- Máquina local; o servidor entra só pela premissa P4.
- O amostrador de 1 s pode perder picos de memória com menos de 1 s; o
  status `undersampled` sinaliza isso pela comparação com `VmHWM`.
- `composition` é derivada por subtração e absorve o overhead dos wrappers e
  do amostrador (desprezível, mas não zero).
- `--smoke` (1 run, 5 gerações) serve só para conferir a ligação com dados
  reais; não é medição e o gate se recusa a comparar com ele.
