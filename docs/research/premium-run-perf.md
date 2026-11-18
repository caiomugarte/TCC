# Onde vai o tempo do run Premium e se o GA pode ser compartilhado por faixa de perfil

**Pesquisa:** ticket #90 do mapa #67. **Data:** 2026-11-17. **Código:** `develop` em `94e42f6`.
**Escopo:** só mede e estima. Nada do código de produção foi alterado. As correções testadas entram por
monkeypatch nos scripts de [`premium-run-perf/`](premium-run-perf/).

## Resposta curta

- **O gargalo é o pandas copiando o `df.attrs` a cada avaliação de fitness.** O `GeneticAlgorithm.fitness`
  faz `df.iloc[mask]`, `selected["SETOR"]` e `value_counts()` para cada um dos 300 cromossomos de cada geração.
  Toda operação dessas chama o `__finalize__` do pandas, que copia o `attrs` em profundidade. O `attrs` do
  `df_ranked` carrega `exclusion_reasons` (6 KB nas ações e 60 KB nos FIIs, em pickle). No cProfile, **95% do
  tempo está no fitness e 81% em `copy.deepcopy`**.
- **Vetorizar o fitness com numpy deixa o run Premium 30× a 50× mais rápido num núcleo, com resultado
  idêntico bit a bit.** Medido na configuração de produção (150 runs, adaptativo, população 300, 400 gerações):
  conservador 98 s, moderado 196 s, arrojado 230 s. Pela baseline eram 81, 157 e 117 min. O consumo do RNG não
  muda, então a carteira também não muda e a política não precisa de versão nova.
- **Paralelizar rende menos e custa mais:** estimo 3,5× a 4,5× com 6 processos, ocupando 6 núcleos por run,
  e os FIIs perdem o determinismo no ponto de parada adaptativo. Depois de vetorizar, não compensa.
- **Compartilhar o GA por faixa de perfil é viável, mas hoje não compensa.** A zona de proteção e as
  quantidades já rodam depois do GA. Mesmo assim, compartilhar exige quantizar o perfil (política nova, que
  muda a carteira de todos), separar o GA do consenso nos dois seletores e criar um cache com chave de
  proveniência. Com o run em 2 a 4 min, isso só se paga com dezenas de contas por manifesto.

**Recomendação:** abrir um ticket para vetorizar a avaliação de fitness do `GeneticAlgorithm`
(`py/core/optimizer.py`), com teste de equivalência bit a bit e regeneração da baseline de performance. Não
paralelizar nem compartilhar o GA agora.

## Método

| Item | Valor |
| --- | --- |
| Snapshot | `dev-2026-10-29-ibov`, o mesmo da baseline, validado pelo `verify_snapshot` do `scripts/benchmark_premium.py` |
| Política | `build_policy` do benchmark (perfil sintético no score âncora, restrição `nenhuma`), seed explícita 42 |
| Máquina | Ryzen 5 7600X (6 núcleos/12 threads), a mesma da baseline; Python 3.14, pandas 3.0.6, numpy 2.5.4 |
| Ferramenta | cProfile (py-spy não está instalado) e `time.perf_counter` |

Cada número abaixo vem marcado como **medido** (rodou) ou **estimado** (conta sobre o que rodou). Algumas
medições rodaram em paralelo (3 a 6 processos em 6 núcleos), então os tempos absolutos delas têm algum ruído
de contenção. As razões de ganho vêm de rodadas sem contenção.

Os scripts estão em [`premium-run-perf/`](premium-run-perf/), com os resultados em `results/`:

- `prof_run.py`: run reduzido com ou sem cProfile;
- `vec_proto.py`: GA original contra o vetorizado, com as mesmas seeds;
- `e2e.py`: pipeline inteiro (`run_premium_optimization`) com o GA trocado por monkeypatch;
- `band_sens.py`: efeito da seed e do passo de score no consenso;
- `bitwise.py`: fitness original contra o vetorizado em cromossomos aleatórios.

## Frente 1: onde vai o tempo

### Perfil de CPU (medido)

O run foi o moderado, com 2 runs por classe e o GA de produção. Com o profiler ligado, o wall total foi de
450 s. Arquivo: `results/cprofile-moderado-2runs.txt`.

| Função | Tempo cumulativo (s) | % do total |
| --- | ---: | ---: |
| `GeneticAlgorithm.fitness` (`py/core/optimizer.py:120`) | 429 | 95% |
| `NDFrame.__finalize__` → `copy.deepcopy` (copia o `attrs`) | 367 | 81% |
| `hhi_sector` (`py/core/metrics.py:18`), incluindo o deepcopy | 128 | 28% |
| `value_counts` | 41 | 9% |
| `GeneticAlgorithm.mutate` | 2 | < 1% |

A causa está na fonte do pandas instalado: `pandas/core/generic.py` documenta no `attrs` que "Copies are always
deep" e o `__finalize__` (linha 6119) propaga o `attrs` em cada operação que gera um objeto novo. O
`df_ranked` herda o `attrs` de `preprocess_profile`, `build_scores` e da preparação dos FIIs
(`exclusion_reasons`, `eligibility_filters`; `py/core/preprocessing.py:125` e `:482`,
`py/core/scoring.py:251`, `py/fii_selection.py:361`). O FII sai mais caro por geração que a ação, mesmo com
universo menor (57 contra 186 papéis), porque o `exclusion_reasons` dele é 10× maior.

### Custo por geração e ganho das correções (medido)

GA isolado, no perfil moderado, com 2 ou 3 seeds por classe e sem carga concorrente
(`results/vec_mod.json` e `results/vec_mod2.json`):

| Variante | Ações (s/geração) | FIIs (s/geração) | Ganho em ações | Ganho em FIIs | Carteira igual à original? |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | 0,28 | 0,455 | 1× | 1× | n/a |
| Original com `df.attrs = {}` antes do GA | 0,094 | 0,12 | 3,0× | 3,8× | sim (2 seeds/classe) |
| Fitness vetorizado (numpy em lote) | 0,0082 | 0,0050 | 34,8× | 90× | sim (3 seeds/classe) |

"Igual" quer dizer mesmos tickers, mesmo fitness (`==`) e mesmo número de gerações.

O fitness vetorizado (`fitness_batch` em `e2e.py`) monta a matriz da população, soma `SCORE[idx]` linha a linha,
conta os setores com `np.add.at` e calcula o HHI somando `(contagem/n)²` dos setores não vazios em ordem
decrescente de contagem, a mesma ordem do `value_counts(normalize=True)`. Com isso o float sai igual bit a bit.
Os tetos `max_per_segment` e `hhi_max` viram máscaras. O resto do GA (seleção, crossover, mutação) não muda,
então o RNG é consumido na mesma ordem.

**Equivalência bit a bit (medido, `bitwise.py`):** 20.000 cromossomos aleatórios por caso (10% com tamanho
errado), em ações e FIIs, cada um sem teto, com teto de 2 por segmento e com `hhi_max` 0,25. Nos 6 casos (120
mil avaliações, entre 1.904 e 18.544 inviáveis por caso) os dois fitness deram arrays idênticos, com diferença
máxima de 0,0.

### Run completo na configuração de produção (medido)

Os três perfis rodaram com 150 runs, adaptativo, `min_runs` 40, população 300 e 400 gerações, com o fitness
vetorizado, em paralelo em 6 núcleos (`results/full_vec_*.json`):

| Perfil | Wall total | Ações | FIIs | Gerações (ações / FIIs) | Baseline (#8, `a7c7d38`) |
| --- | ---: | ---: | ---: | --- | ---: |
| conservador | 98 s (1,6 min) | 87 s | 11 s (40 runs, `stability_target`) | 12.680 / 2.058 | 81,1 min |
| moderado | 196 s (3,3 min) | 111 s | 85 s | 13.017 / 13.041 | 156,6 min |
| arrojado | 230 s (3,8 min) | 118 s | 111 s | 11.232 / 13.248 | 116,8 min |

- **Estimado:** o moderado no código atual, sem a correção, levaria 13.017 × 0,28 + 13.041 × 0,455 ≈ 9.580 s ≈
  160 min, perto dos 157 min da baseline. O ganho de ponta a ponta fica em **≈ 49×**.
- As gerações diferem das da baseline porque o código mudou desde `a7c7d38` (por exemplo, o scoring v2 das
  ações). A vetorização não muda as gerações: a equivalência por run foi provada acima.
- **Medido:** fora do GA sobram ≈ 4 s fixos por run (`e2e_vec10.json`: 15,2 s de wall, dos quais 11,0 s são
  GA). Esse resto é quase todo o `groupby(...).transform` da padronização por setor, que também paga o
  deepcopy do `attrs`.
- **Medido, depois de vetorizar:** o maior pedaço que sobra no GA é o `mutate`, que sorteia um
  `random.random()` por gene em Python. Cortar isso exige mudar a ordem de consumo do RNG, o que muda a carteira
  e pede versão nova do modelo. Para ganhar no máximo mais 2× sobre 3 min, não vale.

### Paralelizar (estimado)

- Os runs são independentes: cada um deriva a própria seed com `derive_run_seed(base, run_id)`. O ramo
  `parallel=True` já existe nos dois seletores (`mp.Pool` em lotes de 10).
- O ganho fica limitado por `min(workers, 10)` vezes a eficiência da barreira de cada lote, porque o lote espera
  o run mais lento. As gerações por run variam de 51 a 206, com média de 87. Estimo uma eficiência de 0,6 a
  0,75, o que dá **≈ 3,5× a 4,5× com 6 workers**, ocupando 6 núcleos por run.
- **Determinismo:** nas ações a parada adaptativa é checada a cada 10 runs nos dois ramos, então o resultado não
  muda. Nos FIIs o ramo sequencial checa a cada run e o paralelo a cada lote: quando o alvo de estabilidade é
  atingido fora de um múltiplo de 10, o `n_runs` efetivo muda e o consenso também.
- **Operação:** o run roda numa thread do processo da API (`PremiumExecutor`). Abrir um `mp.Pool` por fork num
  processo com threads é frágil, e com `spawn` cada worker reimporta o pacote. Com mais de uma conta na fila,
  paralelizar dentro do run não aumenta a vazão.
- **Conclusão:** a vetorização dá 30× a 50× em um núcleo, sem processo novo e sem tocar no determinismo.
  Paralelizar depois disso levaria 3 min para cerca de 1 min, ocupando 6 núcleos.

## Frente 2: compartilhar o GA por faixa de perfil

### Como a seed e a política são resolvidas hoje (código)

- **Seed por conta.** `derive_premium_seed(account_id, profile_id, manifest_id, policy_version)`
  (`api/app/services/premium_recommendation.py:111`) é um SHA-256 truncado. `PremiumRecommendationService.create_run`
  grava a seed em `policy.provenance.random_seed` e na proveniência do run. O adapter usa essa seed (`_seed`,
  `api/app/adapters/premium_optimization.py:144`) e cada run do GA deriva a sua com
  `derive_run_seed(seed, run_id, "stock"|"fii")`.
- **Perfil contínuo.** `resolve_premium_policy` calcula
  `selector_score = 0,40·score + 0,15·(apetite + capacidade + liquidez + conhecimento)`, arredondado a 6 casas
  (`api/app/services/premium_policy.py:306`). `_selector_policy` interpola linearmente entre as âncoras
  conservador, moderado e arrojado: pesos dos fatores, filtros `cap_min`/`size_min`/`liq_min`, `lambda_hhi` e
  `n_assets` arredondado. Duas restrições mexem no seletor: `evitar_illiquidez` (liga os filtros) e
  `limitar_concentracao` (`hhi_max` 0,25 e `lambda` ≥ 0,25). As outras só mexem na alocação.
- **O que depende da conta acontece depois do GA.** A zona de proteção é aplicada dentro de
  `build_consensus_portfolio` (`py/pipelines/multi_run.py:283`) e de `_consensus` (`py/fii_selection.py:605`),
  sobre as frequências dos runs e os scores. Os runs do GA não veem os incumbentes. As quantidades inteiras
  (`_buy_list`) e o capital entram só na composição.

Por isso, a entrada do GA de uma classe é função de quatro coisas: os dados do manifesto, a seção do seletor
na política (sem a `protection_zone`), a versão do modelo e a seed. A conta entra só pela seed.

### O que mudaria para compartilhar

1. **Quantizar o perfil do seletor.** Seria arredondar o `selector_score` para uma grade, por exemplo passo de
   0,05 (21 pontos). Isso exige uma política nova (`premium-policy-v5`) e muda a carteira de quem não está em
   cima da grade. Chaves possíveis por manifesto e por classe: 21 pontos × 4 combinações de restrições = 84.
2. **Seed sem a conta.** A seed passaria a ser `hash(chave da faixa, manifest_id, policy_version)`. O
   determinismo continua: a mesma chave dá o mesmo resultado. Duas contas na mesma faixa passam a receber a
   mesma base e só diferem pela zona de proteção (incumbentes) e pelas quantidades (capital).
3. **Separar o GA do consenso.** Hoje o consenso e a zona rodam dentro das engines. Elas teriam de devolver os
   runs (tickers, fitness e gerações por run, mais as métricas de estabilidade), e o consenso com os
   incumbentes da conta passaria para fora. O `df_ranked` é determinístico a partir do snapshot e da política,
   então não precisa ir para o cache. O checkpoint de ações (`.checkpoint.json`) já tem quase esse formato.
   O payload por chave tem ~150 × 15 tickers, algumas dezenas de KB.
4. **Cache com proveniência.** A chave seria o SHA-256 de `manifest_sha256`, `policy_version`, `model_versions`,
   a seção do seletor sem a zona, o `system_ga_config` e a seed. O run guardaria na proveniência a chave, a seed
   compartilhada e o id do run que calculou o GA. Se o código do GA mudar sem subir o `model_versions`, o
   cache serve um resultado velho. A chave precisa falhar fechada nesse caso, como o `code_fingerprint` do gate
   da #8.

### Custo e benefício (estimado sobre o medido)

| Cenário | Custo por manifesto (mês) |
| --- | --- |
| Hoje, por conta, sem vetorizar | ≈ 1,4 a 2,6 h por conta |
| Vetorizado, por conta | ≈ 2 a 4 min por conta |
| Compartilhado e pré-calculado, sem vetorizar | 84 chaves × ≈ 2 h ≈ 170 h: inviável |
| Compartilhado e pré-calculado, vetorizado | 84 × ≈ 3 min ≈ 4 h num núcleo (≈ 45 min em 6) |
| Compartilhado sob demanda (cache preguiçoso) | ≤ o custo por conta; só ganha quando 2+ contas caem na mesma chave |

Com o GA vetorizado, compartilhar só reduz custo de forma relevante com muitas contas por manifesto (dezenas ou
mais). No uso pessoal (1 conta, ciclo mensal da #82) o ganho é zero e o custo de implementação é alto: política
nova, refatoração dos dois seletores e um armazenamento novo.

### Quanto a seed e a faixa mexem na carteira (medido)

Perfil moderado, configuração de produção completa e GA vetorizado (`results/band_*.json`). Trocas = papéis da
carteira de referência (score 0,50, seed 42) que saem na outra:

| Comparação | Ações (12 papéis): trocas, Jaccard | FIIs (5): trocas, Jaccard |
| --- | --- | --- |
| Só a seed muda (42 contra 7) | 3, 0,60 | 0, 1,00 |
| Só a seed muda (42 contra 1234) | 4, 0,50 | 0, 1,00 |
| Só a seed muda (7 contra 1234) | 4 | 0 |
| Score 0,50 → 0,55 (seed 42) | 3, 0,60 | 0, 1,00 |
| Score 0,50 → 0,45 (seed 42) | 2, 0,71 | 1, 0,67 |
| Score 0,50 → 0,60 (seed 42; 13 ações) | 2, 0,67 | 1, 0,67 |

O que os números mostram:

- **Nas ações, a seed por conta já move 3 a 4 de 12 papéis**: duas contas com o mesmo perfil recebem listas
  diferentes por puro ruído do GA. A estabilidade confirma isso: Jaccard entre runs de 0,14 contra o alvo de
  0,75, e as ações sempre param por `run_count`. Um passo de 0,05 no score move 2 a 3 papéis, a mesma ordem do
  ruído. Faixas de 0,05 perderiam pouca informação de perfil.
- **Nos FIIs, a seed não muda nada, mas o score muda 0 a 1 de 5.** Nos FIIs o passo da faixa é o que
  perde resolução.
- São 6 runs num perfil e num manifesto: serve de ordem de grandeza, não de distribuição.

Uma opção mais barata que a faixa, que fica registrada para decisão, seria tirar só a conta da seed:
`hash(perfil resolvido, manifest_id, policy_version)`. Contas com o mesmo perfil passariam a receber a mesma
base, sem quantizar nada e sem cache. Mesmo assim muda a carteira de todos, então pede política nova. Essa é
uma decisão de produto (dispersão arbitrária entre contas iguais), não de performance.

## Recomendação

1. **Fazer: vetorizar o fitness do `GeneticAlgorithm`** (`py/core/optimizer.py`), avaliando a população em
   lote como no `fitness_batch` de `e2e.py`.
   - Ganho medido: 30× a 50× no run inteiro, de 1,4–2,6 h para 2–4 min num núcleo, com a mesma carteira bit a
     bit. Não precisa de política nem de `model_versions` novos.
   - Testes: equivalência do fitness em lote contra o original em cromossomos aleatórios (com e sem
     `max_per_segment`/`hhi_max`) e um run curto com seed fixa comparando os tickers.
   - Depois, regenerar `docs/validation/premium-performance-baseline.*`. O gate sai com código 6 porque o
     `code_fingerprint` muda. Vale ajustar o timeout de 4 h e a espera de 5,2 h do modelo de capacidade.
   - Alternativa mínima, se a vetorização demorar: limpar o `df.attrs` antes do GA (uma linha) já dá 3× a 4×,
     também com a mesma carteira.
2. **Não fazer agora: paralelizar.** Depois do item 1 o ganho é marginal (3 min para cerca de 1 min), ocupa 6
   núcleos, complica a execução numa thread da API e muda o ponto de parada dos FIIs.
3. **Não fazer agora: compartilhar o GA por faixa.** Com o item 1, o motivo que levou à #90 (a nota da #82 de
   que "fila não resolve 2 h de GA por conta") deixa de existir. Revisitar com dezenas de contas por manifesto.
   O desenho está acima: zona e quantidades já são pós-GA, falta quantizar o perfil, tirar a conta da seed,
   separar o GA do consenso e criar um cache com chave que inclua `model_versions`.
4. **Decisão de produto à parte, sem relação com performance:** a seed por conta faz contas com o mesmo perfil
   receberem 3 a 4 ações diferentes por ruído. Tirar a conta da seed resolve isso, mas pede política nova.

## Limitações

- O profiling cobre um perfil (moderado), com 2 runs por classe sob cProfile. O overhead do profiler infla os
  tempos absolutos, mas não a proporção entre fitness e deepcopy.
- O tempo "original" do run completo no código atual é estimado (custo por geração medido × gerações medidas).
  O original completo, de ~2,7 h, não foi rodado de novo.
- A equivalência foi provada por run (3 seeds por classe) e por avaliação (120 mil cromossomos), não com dois
  runs completos de 150 comparados entre si.
- Os runs completos e o estudo de faixa rodaram em paralelo (3 a 6 processos em 6 núcleos), com algum ruído de
  contenção nos tempos absolutos.
- O estudo de seed contra faixa tem 6 runs num perfil e num manifesto (o de outubro, não o `cvm-b3-2026-11-17`).
- Os scripts apontam para o registry local (`/home/caio/Documents/projetos/TCC/snapshots`). Rodam a partir do
  venv do clone principal.
