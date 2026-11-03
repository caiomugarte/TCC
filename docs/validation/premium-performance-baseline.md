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

A execução registrada rodou no commit `7bf7284` com a árvore limpa, com
`--registry /home/caio/Documents/projetos/TCC/snapshots` e um diretório de
saída temporário (o campo `command` do JSON guarda esse caminho original). O
diretório de saída não entra nas condições comparadas pelo gate.

| Campo | Valor (medido) |
| --- | --- |
| Comando exato | `python scripts/benchmark_premium.py --registry /home/caio/Documents/projetos/TCC/snapshots --output-dir <scratch>/bench-baseline` (padrões: `--profiles conservador,moderado,arrojado`, `--seed 42`, `--capital 100000`, `--isolation process`) |
| Commit (`git rev-parse HEAD`) | `7bf7284c30ea2bbe903177fcae689d5f097a21ce` |
| Árvore suja (`dirty` / `code_dirty`) | `false` / `false` |
| `code_fingerprint` | `70d61e419faf68d87aef187440558a6f3642474277cc82cbc383924560731562` (trees: `api/app` `9cc956f`, `py` `b86aec5`, `scripts/benchmark_premium.py` `f6efe74`) |
| Snapshot (`manifest_id` / `manifest_sha256`) | `dev-2026-10-29-ibov` / `b66a5764bdcb8d8a20cfc4e10f9ad4d7b14448e3c997fec8e597fdcaccd2d014` (`cutoff_date` 2026-10-28) |
| Seed | 42 (mesma nos três perfis) |
| Capital | R$ 100.000,00 |
| Modo / isolamento | `measured` / `process` (um processo Python novo por perfil, em sequência) |
| Data da execução (UTC) | início em 2026-11-03T01:07:14Z (`created_at`, gravado antes do primeiro perfil); duração somada dos três perfis ≈ 21.383 s ≈ 5,9 h |
| Carga da máquina durante a execução | Não dedicada: havia outro trabalho leve. Load average ≈ 2–4,7 no início e ≈ 2,1–2,2 no fim (12 threads lógicas). Registrado como condição da medição, não controlado. |

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
| `machine.fingerprint` | `48000f0b2a53398f5ac05763b5204e8633c0ca8bff1694870b86f850a992002b` |

## Resultados por perfil e etapa (medido)

Todos os valores desta seção são **medidos** (`mode: measured`). Fonte:
`summary.json` (seção `profiles.<perfil>.stages` e `.engines`).

| Perfil | Etapa | Wall (s) | CPU-s (user+sys) | CPU filhos (s) | CPU/wall | Pico RSS amostrado (MiB) | n_runs efetivo | Gerações (total / média / mín–máx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conservador | alocação (`snapshot_load` + `allocation`) | 0,47 | 0,47 | 0,00 | 1,00 | 146,3 | n/a | n/a (sem GA, ADR 0003) |
| conservador | ações | 4.090,93 | 4.029,05 | 0,00 | 0,985 | 192,2 | 150 (de 150) | 12.295 / 82,0 / 51–209 |
| conservador | FIIs | 854,42 | 843,69 | 0,00 | 0,987 | 194,5 | **40** (de 150) | 2.090 / 52,2 / 51–58 |
| conservador | composição (derivada) | 0,002 | 0,002 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| conservador | **total** | **4.945,83** (82,4 min) | 4.873,21 | 0,00 | 0,985 | 194,5 | n/a | n/a |
| moderado | alocação (`snapshot_load` + `allocation`) | 0,47 | 0,47 | 0,00 | 1,00 | 146,1 | n/a | n/a (sem GA, ADR 0003) |
| moderado | ações | 3.989,41 | 3.949,02 | 0,00 | 0,990 | 192,0 | 150 (de 150) | 13.475 / 89,8 / 51–176 |
| moderado | FIIs | 5.464,26 | 5.419,88 | 0,00 | 0,992 | 204,1 | 150 (de 150) | 14.490 / 96,6 / 51–232 |
| moderado | composição (derivada) | 0,003 | 0,003 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| moderado | **total** | **9.454,15** (157,6 min) | 9.369,37 | 0,00 | 0,991 | 204,1 | n/a | n/a |
| arrojado | alocação (`snapshot_load` + `allocation`) | 0,47 | 0,47 | 0,00 | 1,00 | 146,3 | n/a | n/a (sem GA, ADR 0003) |
| arrojado | ações | 3.039,69 | 3.013,73 | 0,00 | 0,991 | 192,1 | 150 (de 150) | 11.640 / 77,6 / 51–205 |
| arrojado | FIIs | 3.942,73 | 3.907,22 | 0,00 | 0,991 | 203,2 | 150 (de 150) | 11.918 / 79,5 / 51–202 |
| arrojado | composição (derivada) | 0,003 | 0,003 | 0,00 | 1,00 | n/a (sem amostra) | n/a | n/a |
| arrojado | **total** | **6.982,90** (116,4 min) | 6.921,43 | 0,00 | 0,991 | 203,2 | n/a | n/a |

Notas da tabela (medido): a alocação se divide em `snapshot_load` ≈ 0,03 s
e `allocation` ≈ 0,45 s nos três perfis. A composição dura poucos
milissegundos e não recebeu amostra periódica de RSS (`rss_samples: 0`). Os
CPU/wall de etapas abaixo de 1 s são ruidosos (podem passar de 1,00 por
arredondamento). "CPU filhos" é 0 em todas as etapas, como esperado com
`parallel=False`. Gerações configuradas: 400 por run, população 300.

Validação do RSS (medido):

| Perfil | Pico amostrado (MiB) | `ru_maxrss` (MiB) | `VmHWM` (MiB) | Razão amostrado/pico | Amostras / maior intervalo (s) | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| conservador | 194,6 | 193,3 | 194,6 | 1,00 | 4.951 / 1,11 | `ok` |
| moderado | 204,2 | 203,4 | 204,2 | 1,00 | 9.455 / 1,08 | `ok` |
| arrojado | 203,3 | 202,2 | 203,3 | 1,00 | 6.988 / 1,04 | `ok` |

O pico amostrado do processo inclui as amostras de fronteira e fica alguns
centésimos de MiB acima do pico da etapa `total`. `ru_maxrss` e `VmHWM`
divergem menos de 1% (limite: 2%). `ru_maxrss_children` é 0.

### Observações factuais

Registradas sem conclusão sobre qualidade, convergência ou sobre o número de
runs.

- **FIIs do conservador pararam em 40 runs.** Nos outros cinco seletores o
  `n_runs` efetivo foi 150. A causa está no código: com `adaptive_mode`, o
  laço sequencial de `run_fii_selection` (`py/fii_selection.py`, ramo
  `parallel=False`) recalcula `_fii_stability` depois de **cada** run e
  interrompe quando `len(resultados) >= min_runs` **e** `fitness_cv <=
  target_cv` **e** `jaccard_mean >= target_jaccard`. A política Premium passa
  `min_runs` 40, `target_cv` 0,02 e `target_jaccard` 0,75
  (`SYSTEM_GA_CONFIG` em `api/app/services/premium_policy.py`, repassados por
  `_default_fii_engine` em `api/app/adapters/premium_optimization.py`). O
  `n_runs` 40 indica que o critério foi satisfeito já na primeira checagem
  elegível (run 40). Os valores de CV e Jaccard dos FIIs não ficam no
  `summary.json` nem no log (o seletor de FIIs não imprime a verificação),
  então não estão registrados aqui.
- **O critério de parada das ações é checado em outro ritmo.**
  `run_multi_execution_profile` (`py/pipelines/multi_run.py`) checa a
  estabilidade a cada 10 runs e imprime CV e Jaccard no `run.log`; nas três
  execuções de ações o critério não foi atingido e os 150 runs rodaram. As
  linhas de verificação de convergência por run do log ficam fora do escopo
  deste relatório.
- **`converged_early_runs` mede outra coisa.** É o número de runs em que o GA
  parou antes das 400 gerações configuradas (parada antecipada por geração);
  foi igual ao número de runs em todos os seletores (por exemplo, 40/40 nos
  FIIs do conservador e 150/150 nos demais). Não é o mesmo critério do
  `n_runs` adaptativo.
- **O tempo por geração não é constante.** Wall por geração: ações 0,333 /
  0,296 / 0,261 s e FIIs 0,409 / 0,377 / 0,331 s (conservador / moderado /
  arrojado). Por run: ações 27,3 / 26,6 / 20,3 s e FIIs 21,4 / 36,4 / 26,3 s.
- **Os seletores dominam o tempo.** Ações + FIIs respondem por mais de
  99,99% do wall total nos três perfis; alocação (< 0,5 s) e composição
  (< 0,01 s) são desprezíveis.
- **O moderado foi o perfil mais lento** (9.454 s), puxado pelos FIIs
  (5.464 s, 96,6 gerações em média). O conservador foi o mais rápido
  (4.946 s), em boa parte porque os FIIs rodaram 40 runs.
- **Faixa medida contra a estimativa sintética anterior.** O total medido
  ficou entre 82 e 158 min por execução, dentro da faixa sintética de 45 min
  a 2,7 h citada no objetivo (e perto do teto no moderado).

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
2. **P2, CPU-bound de um núcleo.** Medido: CPU/wall de 0,985 a 0,991 no total
   e "CPU filhos" 0. O modelo trata cada execução como ≈ 1 núcleo ocupado o
   tempo todo (≈ 0,99 CPU/wall). A execução de referência rodou com a
   máquina sob carga leve (load ≈ 2), sem contenção de núcleo aparente.
3. **P3, custo independente do capital e da conta.** O capital investível só
   escala os valores em BRL. A seed muda os resultados e, com o modo
   adaptativo, pode mudar o `n_runs` efetivo e as gerações (ver os FIIs do
   conservador); uma seed por perfil não mede essa variância.
4. **P4, conversão para o servidor (não verificada).** `T_servidor = k ×
   T_local`, com `k` = razão de desempenho single-thread entre o servidor e
   esta máquina. `k` **não foi medido**; o jeito mais barato de obtê-lo é
   rodar `--profiles moderado` uma vez no servidor. Até lá, os valores de
   servidor abaixo são fórmulas em `k`, não números.
5. **P5, memória por worker ≈ 205 MiB.** Pico medido do pior perfil: 204,2
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

Com `T_p` = wall total medido do perfil `p`: `T_cons` = 4.945,83 s,
`T_mod` = 9.454,15 s, `T_arr` = 6.982,90 s; `T̄ = média(T_p)` = 7.127,63 s
(118,8 min ≈ 1,98 h); `T_max` = 9.454,15 s (moderado, ≈ 2,63 h);
`M_max` = 204,2 MiB.

Execuções por worker (1 núcleo, 24 h de uso contínuo, P1, P2, P7):

| Perfil | Wall total (h) | Execuções/h por worker | Execuções/dia por worker (local) | Execuções/dia por worker (servidor) |
| --- | ---: | ---: | ---: | --- |
| conservador | 1,37 | 0,73 | 17,5 | 17,5 / k |
| moderado | 2,63 | 0,38 | 9,1 | 9,1 / k |
| arrojado | 1,94 | 0,52 | 12,4 | 12,4 / k |
| média (P6) | 1,98 | 0,51 | 12,1 | 12,1 / k |

| Grandeza | Fórmula | Valor local (derivado) | Valor servidor (derivado, P4) |
| --- | --- | --- | --- |
| Execuções por hora, 1 worker (P1, P6) | `3600 / T̄` | 0,51 | 0,51 / k |
| Execuções por dia, 1 worker | `86400 / T̄` | 12,1 | 12,1 / k |
| Espera no pior caso para a 2ª execução aceita (P1) | `2 × T_max` (a que está rodando + a própria) | 18.908 s ≈ 5,25 h | 5,25 × k h |
| Workers para N núcleos físicos (P2, P5, P8) | `min(N, floor(RAM_livre / M_max))` | N (a CPU limita antes da RAM: cada worker usa ≈ 0,2 GiB) | N do servidor, conferindo a RAM livre |
| Execuções por dia com N workers | `N × 86400 / T̄` | 12,1 × N (6 núcleos: ≈ 72,7) | 12,1 × N / k |
| Memória para N workers (P5) | `N × 205 MiB` + folga | 1 worker ≈ 0,2 GiB; 4 ≈ 0,8 GiB; 6 ≈ 1,2 GiB | idem |
| Timeout sugerido por execução | `1,5 × T_max`, arredondado | 14.181 s → **4 h** | 4 × k h (recalibrar após medir `k`) |
| Custo por geração, ações | `wall_ações / gerações_total_ações` | 0,333 / 0,296 / 0,261 s (cons / mod / arr) | × k |
| Custo por geração, FIIs | `wall_FIIs / gerações_total_FIIs` | 0,409 / 0,377 / 0,331 s (cons / mod / arr) | × k |
| Participação dos seletores no total | `(wall_ações + wall_FIIs) / T_p` | > 99,99% nos três perfis | n/a |

O timeout usa uma margem de 50% sobre o pior perfil medido para cobrir
variância de seed e carga (P3), que não foi medida; a margem é uma escolha,
não um dado. A espera de 5,25 h é o caso em que duas execuções moderadas
ficam enfileiradas no único worker.

O custo por geração permite reestimar `T_p` para outro número efetivo de
runs ou de gerações sem nova medição, mas a medição mostra que ele varia
entre perfis e seletores (0,26 a 0,41 s), então a extrapolação vale só
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
  (`code_fingerprint`, os tree hashes de `api/app`, `py` e
  `scripts/benchmark_premium.py`, que não mudam com commits só de docs),
  árvore de código suja, snapshot, seed, máquina (`machine.fingerprint`),
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

- Uma seed por perfil e uma execução por perfil: não mede a variância entre
  seeds, entre contas nem o ruído entre repetições da mesma execução. O
  limiar de 20% do gate não foi calibrado contra esse ruído.
- Máquina não dedicada: havia outro trabalho leve durante a medição (load
  average ≈ 2–4,7 no início, ≈ 2,1–2,2 no fim, em 12 threads). Os CPU/wall
  ≈ 0,99 sugerem pouca contenção, mas o efeito não foi isolado.
- O `summary.json` não registra o CV de fitness nem o Jaccard da parada
  adaptativa dos FIIs; o motivo dos 40 runs no conservador vem do código, não
  de um valor registrado.
- Máquina local; o servidor entra só pela premissa P4.
- O amostrador de 1 s pode perder picos de memória com menos de 1 s; o
  status `undersampled` sinaliza isso pela comparação com `VmHWM`.
- `composition` é derivada por subtração e absorve o overhead dos wrappers e
  do amostrador (desprezível, mas não zero).
- `--smoke` (1 run, 5 gerações) serve só para conferir a ligação com dados
  reais; não é medição e o gate se recusa a comparar com ele.
