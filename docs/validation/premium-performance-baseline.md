# Baseline de performance Premium com dados reais (P0-03a, issue #8)

> **Estado: PENDENTE.** A estrutura, o método e o script estão prontos; as
> células marcadas `PENDENTE` serão preenchidas com a saída de
> `scripts/benchmark_premium.py`. Nenhum número deste documento é estimado:
> cada valor numérico vem do `summary.json` da execução registrada abaixo e
> está marcado como **medido**.

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

Execução medida (registrar aqui o comando exato e o commit do
`summary.json`):

```bash
python scripts/benchmark_premium.py \
  --registry <repo>/snapshots \
  --output-dir <dir-fora-do-repo>/premium-bench
```

| Campo | Valor |
| --- | --- |
| Comando exato | PENDENTE |
| Commit (`git rev-parse HEAD`) | PENDENTE |
| Árvore suja (`dirty` / `code_dirty`) | PENDENTE |
| `code_fingerprint` | PENDENTE |
| Snapshot (`manifest_id` / `manifest_sha256`) | `dev-2026-10-29-ibov` / `b66a5764bdcb8d8a20cfc4e10f9ad4d7b14448e3c997fec8e597fdcaccd2d014` (declarado) |
| Seed | PENDENTE |
| Data da execução (UTC) | PENDENTE |

O `summary.json` da execução fica em
`docs/validation/premium-performance-baseline.json` e é a referência do gate
de regressão.

## Máquina

Máquina local. A conversão para o servidor é uma premissa do modelo de
capacidade, não uma medição.

| Campo | Valor |
| --- | --- |
| CPU | PENDENTE |
| Núcleos físicos / threads | PENDENTE |
| RAM | PENDENTE |
| SO / kernel | PENDENTE |
| Python | PENDENTE |
| `machine.fingerprint` | PENDENTE |

## Resultados por perfil e etapa (medido)

Todos os valores desta seção são **medidos** (`mode: measured`). Fonte:
`summary.json` (seção `profiles.<perfil>.stages` e `.engines`).

| Perfil | Etapa | Wall (s) | CPU-s (user+sys) | CPU filhos (s) | CPU/wall | Pico RSS amostrado (MiB) | n_runs efetivo | Gerações (total / média / mín–máx) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conservador | alocação (`snapshot_load` + `allocation`) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a (sem GA, ADR 0003) |
| conservador | ações | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| conservador | FIIs | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| conservador | composição (derivada) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a | n/a |
| conservador | **total** | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a |
| moderado | alocação (`snapshot_load` + `allocation`) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a (sem GA, ADR 0003) |
| moderado | ações | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| moderado | FIIs | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| moderado | composição (derivada) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a | n/a |
| moderado | **total** | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a |
| arrojado | alocação (`snapshot_load` + `allocation`) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a (sem GA, ADR 0003) |
| arrojado | ações | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| arrojado | FIIs | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| arrojado | composição (derivada) | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a | n/a |
| arrojado | **total** | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | n/a | n/a |

Validação do RSS (medido):

| Perfil | Pico amostrado (MiB) | `ru_maxrss` (MiB) | `VmHWM` (MiB) | Razão amostrado/pico | Amostras / maior intervalo (s) | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| conservador | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| moderado | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |
| arrojado | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |

## Modelo inicial de capacidade

Derivado só dos números medidos acima e das premissas listadas. Cada
premissa é uma hipótese a confirmar, não um fato medido.

### Premissas

1. **P1, um worker.** O piloto usa `PremiumExecutor` com `max_workers=1` e
   `max_pending=2` (`api/app/services/premium_executor.py`): uma execução por
   vez e no máximo duas aceitas (em execução + na fila).
2. **P2, CPU de um núcleo.** Os seletores rodam sequenciais
   (`parallel=False`); espera-se CPU/wall ≈ 1. Se a medição mostrar outro
   valor, o modelo usa o valor medido.
3. **P3, custo independente do capital e da conta.** O capital investível só
   escala os valores em BRL; a seed muda, mas o custo por geração não deve
   mudar. Uma seed por perfil não mede essa variância.
4. **P4, conversão para o servidor.** `T_servidor = k × T_local`, com `k` = a
   razão de desempenho single-thread entre o servidor e a máquina local.
   `k` não foi medido; o jeito mais barato de obtê-lo é rodar
   `--profiles moderado` uma vez no servidor.
5. **P5, memória por worker.** Cada worker precisa do pico de RSS medido do
   pior perfil, mais uma folga a definir para a API e o SO.
6. **P6, mix de perfis.** Sem dados de uso, a média simples dos três perfis
   representa uma execução típica.
7. **P7, sem cache entre execuções.** Cada execução recalcula tudo; não há
   reaproveitamento entre contas.

### Fórmulas

Com `T_p` = wall total medido do perfil `p`, `T̄ = média(T_p)`,
`T_max = máx(T_p)` e `M_max` = pico de RSS medido do pior perfil:

| Grandeza | Fórmula | Valor local (medido + premissas) | Valor servidor |
| --- | --- | --- | --- |
| Execuções por hora, 1 worker (P1, P6) | `3600 / T̄` | PENDENTE | PENDENTE (× 1/k, P4) |
| Execuções por dia, 1 worker | `86400 / T̄` | PENDENTE | PENDENTE |
| Espera no pior caso para a 2ª execução aceita (P1) | `2 × T_max` (a que está rodando + a própria) | PENDENTE | PENDENTE |
| Workers que cabem na máquina (P2, P5) | `min(núcleos, floor(RAM_livre / M_max))` | PENDENTE | PENDENTE |
| Custo por geração, ações | `wall_ações / gerações_total_ações` | PENDENTE | PENDENTE |
| Custo por geração, FIIs | `wall_FIIs / gerações_total_FIIs` | PENDENTE | PENDENTE |
| Participação dos seletores no total | `(wall_ações + wall_FIIs) / T_p` | PENDENTE | n/a |

O custo por geração permite reestimar `T_p` para outro número efetivo de
runs ou de gerações sem nova medição, desde que o custo por geração continue
constante. Isso é uma extrapolação, não uma recomendação de mudar o
`run_count`.

## Gate de regressão

O próprio script é o gate; não há CI.

```bash
python scripts/benchmark_premium.py \
  --registry <repo>/snapshots \
  --output-dir <dir-fora-do-repo>/premium-bench-rerun \
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
  `--compare-only <summary.json> --baseline <baseline.json>`.

## Limitações

- Uma seed por perfil: não mede a variância entre seeds nem entre contas.
- Máquina local; o servidor entra só pela premissa P4.
- O amostrador de 1 s pode perder picos de memória com menos de 1 s; o
  status `undersampled` sinaliza isso pela comparação com `VmHWM`.
- `composition` é derivada por subtração e absorve o overhead dos wrappers e
  do amostrador (desprezível, mas não zero).
- `--smoke` (1 run, 5 gerações) serve só para conferir a ligação com dados
  reais; não é medição e o gate se recusa a comparar com ele.
