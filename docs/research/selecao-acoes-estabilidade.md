# Estabilidade e qualidade da seleção de ações com o GA rápido

**Pesquisa:** ticket #100 do mapa #67. **Data:** 2026-11-19. **Código:** `develop` em `8fd8c12` (já com o fitness
vetorizado da #95).
**Escopo:** só mede. Nada do código de produção foi alterado. Os scripts estão em
[`selecao-acoes-estabilidade/`](selecao-acoes-estabilidade/), com os resultados em `results/`.

## Resposta curta

- **Cada run do GA de ações fica longe do ótimo, e o consenso compensa quase tudo.** O ótimo exato do fitness do
  GA no moderado é **7,144**, e é a carteira dos 12 maiores scores sem duas classes da mesma empresa. Os 3.000
  runs medidos ficaram em **4,65 de média (65% do ótimo)**, o melhor em 6,09, nenhum no ótimo, e não houve dois
  runs com a mesma carteira. O consenso por frequência de 150 runs chega a **7,04 ± 0,05 (98,5%)**.
- **(a)** Entre os consensos das 10 seeds, o fitness varia de **6,985 a 7,144** (dp 0,051). Dentro de um
  consenso, os 150 runs que o formam variam de ~4,1 a ~5,6 (dp 0,26, cinco vezes mais). Os 8 papéis de maior
  score estão nos 10 consensos. As trocas acontecem entre papéis de score 0,42 a 0,54, mas não são
  equivalentes: o pior consenso fica **0,158 abaixo do ótimo** (2,2%), e só 1 seed em 10 entrega o ótimo.
- **(b)** Com 50, 150 e 300 runs, o Jaccard médio entre os consensos de seeds diferentes é **0,62, 0,72 e
  0,82**, com **2,9, 2,0 e 1,2 papéis trocados** em média. Não há platô até 300, e nem com 3.000 runs juntos o
  consenso para no ótimo: do 9º ao 16º papel, as frequências ficam entre 0,19 e 0,28, com diferenças de 0,002
  a 0,01 entre vizinhos. O fitness sobe pouco depois de ~100 runs (7,03 com 100, 7,10 com 300).
- **(c)** A referência sem sorteio (top-12 por score, sem duas classes da mesma empresa) **é o ótimo exato**:
  7,144. Ela sai de uma programação dinâmica em 0,6 ms. O consenso de 150 runs fica, em média, 0,107 abaixo
  dela.
- **(d)** Runs mais longos sobem pouco: a mediana do fitness por run vai de 4,71 (produção) para 4,93 com as
  400 gerações inteiras, 5,18 com 2.000, 5,49 com 10.000 gerações e 5,35 com população 3.000. Fica sempre
  entre 66% e 77% do ótimo, e o Jaccard entre runs vai de 0,17 para no máximo 0,23. **O GA de hoje otimiza
  pouco, e não é por parar cedo.** Uma sondagem fora do escopo mostra que a taxa de mutação é o que segura: com
  0,001 em vez de 0,02 (e 2.000 gerações), a mediana vai a 6,93 (97%) e o Jaccard entre runs a 0,59.

## Método

| Item | Valor |
| --- | --- |
| Snapshot | `cvm-b3-2026-11-17` (registry local, fora do git), ações de `data/selectors-2026-11-17/stocks.csv` |
| Política | `benchmark_premium.build_policy("moderado", …)`: perfil sintético no score âncora, restrição `nenhuma`, `premium-policy` atual |
| Seletor | 12 papéis, λ = 0,25, sem `hhi_max` e sem `max_per_segment`, população 300, 400 gerações, mutação 0,02, crossover 0,8, paciência 50 (fixa no código) |
| Universo | 169 ações depois do preparo, 10 setores, nenhum SETOR ou SCORE vazio |
| Seeds | seeds base 1 a 10; cada run usa `derive_run_seed(seed, run_id, "stock")`, como o run Premium |
| Máquina | Ryzen 5 7600X (6 núcleos), venv do clone principal, 5 ou 6 processos em paralelo |

O `common.py` monta o `df_ranked` e o GA como o pipeline de produção. O `verify_pipeline.py` confirma que 3 runs
do `run_multi_execution_profile` e 3 runs montados aqui dão os mesmos tickers e o mesmo fitness (`==`). O GA é
determinístico: os 20 runs da variante `base` do item (d) repetem bit a bit os 20 primeiros runs da seed 1.

O consenso com R runs é o consenso dos R primeiros runs de uma seed. Isso é igual a rodar com `run_count` = R
sem parada adaptativa. Na produção a parada adaptativa nunca dispara nas ações, então o run Premium de hoje é o
caso R = 150. O consenso é o `build_consensus_portfolio` de produção, sem titulares (sem zona de proteção). O
fitness de qualquer carteira é o `GeneticAlgorithm.fitness` de produção. Ele está em unidades de score:
`soma(SCORE) − λ·HHI·n`, e o SCORE do universo tem desvio-padrão de 0,37.

Scripts:

- `common.py`: entradas do GA, fitness de uma lista de tickers e consenso de produção;
- `verify_pipeline.py`: o `common.py` contra o pipeline de produção;
- `reference.py`: ótimo exato (programação dinâmica) e top-N por score;
- `run_seeds.py`: 300 runs por seed, para 10 seeds;
- `analyze.py`: itens (a), (b) e (c) sobre os runs;
- `pooled.py`: consenso com runs de várias seeds juntos e as trocas de cada consenso contra o ótimo;
- `long_runs.py` e `analyze_long.py`: item (d).

### Por que dá para calcular o ótimo exato

Com `HHI = Σ (c_s/n)²`, o fitness fica `Σ_s [soma dos c_s papéis escolhidos no setor s − (λ/n)·c_s²]`. É
separável por setor, e, fixado quantos papéis entram num setor, os melhores são sempre os de maior score. Uma
programação dinâmica sobre (setor, papéis já escolhidos) acha o máximo global (`reference.py`). O valor da
programação dinâmica bate com o `GeneticAlgorithm.fitness` da carteira resultante. Com `hhi_max` ou
`max_per_segment`, como nos FIIs ou com a restrição `limitar_concentracao`, a mesma programação dinâmica vale
com um teto por setor e uma dimensão a mais para `Σ c_s²`. Isso não foi implementado aqui.

## (a) Diferença entre consensos contra a dispersão dentro de um consenso (medido)

R = 150, 10 seeds (`results/analysis.json`):

| Seed | Fitness do consenso | Runs do consenso: média ± dp (mín–máx) | Melhor run | Jaccard run a run | Abaixo do ótimo |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 6,988 | 4,662 ± 0,258 (4,121–5,339) | 5,339 | 0,152 | 0,156 |
| 2 | 7,039 | 4,637 ± 0,252 (4,048–5,544) | 5,544 | 0,146 | 0,104 |
| 3 | 6,985 | 4,620 ± 0,277 (4,100–5,685) | 5,685 | 0,145 | 0,158 |
| 4 | 6,998 | 4,665 ± 0,287 (4,023–5,687) | 5,687 | 0,155 | 0,145 |
| 5 | 7,144 | 4,683 ± 0,276 (4,170–5,626) | 5,626 | 0,150 | 0 |
| 6 | 7,082 | 4,668 ± 0,249 (4,042–5,559) | 5,559 | 0,151 | 0,061 |
| 7 | 7,049 | 4,692 ± 0,276 (4,112–5,402) | 5,402 | 0,150 | 0,095 |
| 8 | 6,995 | 4,639 ± 0,292 (4,030–5,497) | 5,497 | 0,148 | 0,149 |
| 9 | 6,998 | 4,647 ± 0,249 (4,124–5,381) | 5,381 | 0,148 | 0,145 |
| 10 | 7,092 | 4,646 ± 0,266 (4,191–5,453) | 5,453 | 0,144 | 0,052 |

- **Entre consensos:** média 7,037, dp 0,051, amplitude 0,159.
- **Dentro de um consenso:** dp médio de 0,266 entre os runs, com amplitude média de ~1,4. O consenso (~7,0)
  fica muito acima do melhor run que o forma (5,3 a 5,7). O consenso não é "um run bom": é o voto que filtra o
  ruído de runs ruins.
- Gerações por run: média 82, de 51 a 219. Quase todos param pela paciência de 50.

Papéis nos 10 consensos de R = 150:

| Papel | Em quantos consensos | Score | No ótimo? |
| --- | ---: | ---: | --- |
| ABCB4, FIQE3, EUCA4, BRAV3, RDOR3, ISAE4, CURY3, BEEF3 | 10 | 0,52 a 0,90 | sim |
| BBSE3 | 8 | 0,525 | sim |
| POMO3 | 8 | 0,515 | sim |
| MILS3 | 6 | 0,537 | sim |
| PGMN3 | 5 | 0,453 | não |
| LAVV3 | 4 | 0,480 | sim |
| MOTV3, RECV3, USIM5 | 2 cada | 0,42 a 0,46 | não |
| POMO4, DIRR3, PETR4 | 1 cada | 0,33 a 0,49 | não |

O que sai e o que entra contra o ótimo, por seed (`results/pooled.json`):

| Seed | Sai | Entra | Abaixo do ótimo |
| ---: | --- | --- | ---: |
| 1 | BBSE3 (0,525), LAVV3 (0,480) | MOTV3 (0,460), RECV3 (0,430) | 0,156 |
| 2 | LAVV3, MILS3 (0,537) | MOTV3, PGMN3 (0,453) | 0,104 |
| 3 | MILS3 | DIRR3 (0,421) | 0,158 |
| 4 | LAVV3, MILS3 | PGMN3, USIM5 (0,419) | 0,145 |
| 5 | nada | nada | 0 |
| 6 | POMO3 (0,515) | PGMN3 | 0,061 |
| 7 | BBSE3 | RECV3 | 0,095 |
| 8 | LAVV3 | PETR4 (0,331) | 0,149 |
| 9 | LAVV3, MILS3 | PGMN3, USIM5 | 0,145 |
| 10 | LAVV3, POMO3 | PGMN3, POMO4 (0,490) | 0,052 |

**Leitura.** Os papéis que trocam por ruído ficam todos na cauda da carteira (scores de 0,42 a 0,54, contra 0,52
a 0,90 do núcleo fixo). Uma troca custa de 0,03 a 0,16 de fitness, o equivalente a trocar um papel por outro
0,1 a 0,4 desvio-padrão de score abaixo. Isso é pequeno perto da dispersão dos runs, mas não é zero: 9 de 10
seeds entregam uma carteira pior que o ótimo, e a diferença vem de onde a seed cai. A troca de MILS3 (o 8º maior
score) por DIRR3 na seed 3 mostra que o voto pode tirar um papel bem colocado.

**Confiança:** alta para este perfil e este manifesto (10 seeds × 150 runs, GA determinístico, pipeline conferido
contra a produção). Não foi medido em outros perfis nem com `hhi_max`.

## (b) Estabilidade do consenso com o número de runs (medido)

Para cada R, o consenso dos R primeiros runs de cada uma das 10 seeds (`results/analysis.json`, tabela completa
de 10 em 10 runs no `analyze.py`):

| R | Fitness do consenso (média ± dp) | Mín–máx | Jaccard entre seeds | Papéis trocados entre seeds | Jaccard com o ótimo | Consensos distintos (de 10) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 10 | 6,620 ± 0,289 | 5,883–7,007 | 0,46 | 4,6 | 0,54 | 10 |
| 50 | 6,962 ± 0,112 | 6,679–7,072 | 0,62 | 2,9 | 0,72 | 9 |
| 100 | 7,034 ± 0,054 | 6,955–7,144 | 0,74 | 1,9 | 0,81 | 9 |
| 150 | 7,037 ± 0,051 | 6,985–7,144 | 0,72 | 2,0 | 0,80 | 9 |
| 200 | 7,079 ± 0,045 | 6,998–7,144 | 0,79 | 1,4 | 0,86 | 9 |
| 250 | 7,092 ± 0,042 | 7,022–7,144 | 0,79 | 1,4 | 0,84 | 7 |
| 300 | 7,101 ± 0,031 | 7,034–7,144 | 0,82 | 1,2 | 0,86 | 6 |

- **A margem de frequência entre o 12º e o 13º papel fica em ~0,01 em todo R** (de 0,003 a 0,017): um ou dois
  runs em 150 decidem quem entra. Por isso o consenso de uma mesma seed ainda muda de R para R + 10 em 1 a 5 das
  10 seeds, mesmo depois de 200 runs.
- **Juntando runs de várias seeds** (`pooled.py`), o consenso com 600 runs troca LAVV3 por MOTV3 (7,082), com
  1.500 bate o ótimo e com 3.000 troca LAVV3 por PGMN3 (7,117). Com 3.000 runs, as frequências do 9º ao 16º
  papel são 0,256, 0,256, 0,242, 0,218 (POMO4, pulado por ser a mesma empresa que POMO3), 0,213 (PGMN3), 0,211
  (LAVV3), 0,199 e 0,194. A frequência não segue o score: MILS3 (0,537) aparece menos que POMO3 (0,515), e
  PGMN3 (0,453) empata com LAVV3 (0,480).
- **A métrica de parada de hoje não depende de R.** O Jaccard run a run é ~0,15 com qualquer número de runs,
  porque mede runs individuais, que nunca se repetem. O CV do fitness dos runs é ~0,059 (0,27/4,65). Com o alvo
  de Jaccard ≥ 0,75 e CV ≤ 0,02, a parada adaptativa não dispara com nenhum R.

**Leitura.** O top-N para de mudar com a seed? Não até 300 runs: ainda sobra 1,2 papel de diferença entre seeds
e 4 em 10 seeds sem o ótimo. Há um platô de fitness: de 100 para 300 runs o consenso médio sobe só 0,07 (de
98,5% para 99,4% do ótimo), contra +0,34 de 10 para 50. Não há platô de composição: o Jaccard entre seeds segue
subindo devagar (0,74 → 0,82), e o limite do voto não é o ótimo, porque a ordem das frequências na cauda é um
empate que o sorteio desfaz.

**Confiança:** alta nos números. O ponto "platô ≈ 100 runs" é uma leitura da curva com 10 seeds, não um teste
formal. O limite com 3.000 runs vem de uma única amostra de 3.000.

## (c) Referência sem sorteio (medido)

`results/reference.json`:

| Carteira | Fitness | Diferença para o ótimo |
| --- | ---: | ---: |
| Ótimo exato do fitness do GA (programação dinâmica) | 7,1436 | 0 |
| Top-12 por score, sem duas classes da mesma empresa | 7,1436 | 0 (é a mesma carteira) |
| Top-12 por score, com POMO3 e POMO4 | 7,1115 | −0,032 |
| Consenso de 150 runs, média das 10 seeds | 7,037 | −0,107 |
| Consenso de 150 runs, pior seed | 6,985 | −0,158 |
| Carteira mais votada entre os 10 consensos de 150 | 7,117 | −0,027 |
| Melhor run individual em 3.000 | 6,086 | −1,057 |
| Run individual médio | 4,653 | −2,491 |

O ótimo é ABCB4, BBSE3, BEEF3, BRAV3, CURY3, EUCA4, FIQE3, ISAE4, LAVV3, MILS3, POMO3 e RDOR3, com 2 papéis em
Financeiro, Consumo Cíclico e Bens Industriais e 1 em cada um dos outros 6 setores. Com λ = 0,25 e 12 papéis,
a penalidade de HHI de um segundo papel no mesmo setor é só 3/48 ≈ 0,06. Ela não muda nenhuma escolha do
top-12 neste universo. O que a programação dinâmica "vê" além do top-12 é só a regra de uma classe por
empresa.

**Leitura.** Neste perfil, o GA mais o consenso são uma aproximação ruidosa de "pegue os 12 maiores scores".
A referência sem sorteio é determinística, custa 0,6 ms e é pelo menos tão boa quanto qualquer consenso medido.

**Confiança:** alta. É uma conta exata, conferida contra o `GeneticAlgorithm.fitness`. Ela só vale para o caso
sem `hhi_max`/`max_per_segment`. Com tetos, a carteira ótima pode deixar de ser o top-N, embora a programação
dinâmica continue exata.

## (d) Runs mais longos (medido)

Mesmo GA, mudando população, gerações e paciência. Runs com as seeds da seed base 1. Tempo por run com 5 ou 6
processos em paralelo, então tem contenção (`results/long_analysis.json`):

| Variante | Pop. | Gerações | Paciência | Runs | Fitness por run: mediana (mín–máx) | % do ótimo | Gerações rodadas | s/run | Jaccard run a run | Jaccard com o ótimo |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| produção | 300 | 400 | 50 | 20 | 4,707 (4,268–5,205) | 66% | 101 | 0,8 | 0,17 | 0,26 |
| paciência 200 | 300 | 400 | 200 | 20 | 4,852 (4,571–5,205) | 68% | 298 | 2,4 | 0,16 | 0,27 |
| 400 gerações inteiras | 300 | 400 | 400 | 20 | 4,926 (4,571–5,205) | 69% | 400 | 3,2 | 0,18 | 0,29 |
| 2.000 gerações | 300 | 2.000 | 2.000 | 10 | 5,181 (4,964–5,508) | 73% | 2.000 | 15,9 | 0,19 | 0,34 |
| 10.000 gerações | 300 | 10.000 | 10.000 | 6 | 5,487 (5,227–5,647) | 77% | 10.000 | 80,9 | 0,23 | 0,38 |
| população 1.000 | 1.000 | 400 | 50 | 20 | 4,741 (4,277–5,110) | 66% | 72 | 2,1 | 0,16 | 0,27 |
| população 1.000, 400 gerações | 1.000 | 400 | 400 | 10 | 5,119 (4,759–5,394) | 72% | 400 | 11,3 | 0,18 | 0,32 |
| população 3.000, 400 gerações | 3.000 | 400 | 400 | 6 | 5,350 (4,990–5,549) | 75% | 400 | 41,3 | 0,22 | 0,34 |

**Leitura.** Chegam a fitness maior, mas muito pouco: 100 vezes mais gerações (10.000 contra ~100) levam a
mediana de 66% para 77% do ótimo, e o Jaccard entre runs continua em 0,16 a 0,23. Não há várias carteiras
empatadas no topo. O ótimo é único e fica 0,03 acima da segunda melhor carteira medida. O Jaccard não sobe
porque cada run termina num ponto diferente, longe do topo. A paciência de 50 corta pouco: deixar as 400
gerações inteiras ganha 0,22 de mediana.

### Sondagem fora do escopo: taxa de mutação

Os números acima pedem uma explicação. Pela leitura de `GeneticAlgorithm.mutate` e `optimize`
(`py/core/optimizer.py`), com 169 genes e mutação de 0,02 por gene, cada filho sofre ~3,4 flips, quase todos de
0 para 1 (há 157 zeros e 12 uns). O reparo então tira ~3 dos 12 papéis ao acaso e completa com papéis ao acaso.
Em cada geração, cada filho troca uns 3 dos 12 papéis por sorteio, e não há elitismo: a população inteira é
substituída e o melhor só fica guardado de fora. Para testar essa hipótese, rodei a mesma configuração mudando só
a taxa de mutação:

| Variante | Pop. | Gerações | Paciência | Mutação | Runs | Mediana (mín–máx) | % do ótimo | s/run | Jaccard run a run | Jaccard com o ótimo | Fitness do consenso dos runs |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| mutação 0,005 | 300 | 400 | 50 | 0,005 | 20 | 5,587 (5,021–5,890) | 78% | 0,7 | 0,23 | 0,31 | 6,799 |
| mutação 0,005, 400 gerações | 300 | 400 | 400 | 0,005 | 20 | 5,818 (5,589–6,260) | 81% | 2,7 | 0,26 | 0,37 | 6,997 |
| mutação 0,001, 2.000 gerações | 300 | 2.000 | 2.000 | 0,001 | 10 | 6,934 (6,809–7,066) | 97% | 12,2 | 0,59 | 0,72 | 7,144 |

Com mutação 0,001, o run individual chega a 97% do ótimo, e o consenso de só 10 runs bate o ótimo. Isso é uma
sondagem (10 a 20 runs, uma seed base), não uma proposta: mudar a mutação muda o consumo do RNG e a carteira, e
pede versão nova do modelo.

**Confiança:** alta para "runs mais longos com os operadores de hoje não chegam perto do ótimo" (8 variantes,
até 100× mais gerações e 10× mais população). Média para o papel da mutação: o efeito é grande e consistente,
mas foi medido com poucos runs e sem procurar a melhor combinação.

## O que isso decide

Fatos para os dois grillings. Não são recomendações.

### Para [Quando vale trocar um papel titular](https://github.com/caiomugarte/TCC/issues/101)

- **O fitness dá para comparar carteiras em unidades de score, e de forma exata.** O fitness é
  `soma(SCORE) − (λ/n)·Σ_s c_s²`. A diferença de fitness entre a carteira com o titular e a carteira com o
  candidato é a diferença de score dos dois, menos a variação da penalidade de setor. Ela não depende de seed nem
  de runs e custa microssegundos.
- **Ordem de grandeza do ruído, em fitness:** neste perfil, o consenso de 150 runs fica de 0 a 0,158 abaixo do
  ótimo (dp 0,051 entre seeds). Uma troca de ruído custa de 0,03 a 0,16. Um limiar de "melhor o bastante" abaixo
  de ~0,05 trocaria papéis dentro da faixa que o sorteio já move. Acima de ~0,16, nenhuma troca medida aqui por
  ruído passaria.
- **A frequência nos runs não separa os papéis da cauda.** Do 9º ao 16º papel, as frequências ficam entre 0,19
  e 0,28 e os vizinhos diferem de 0,002 a 0,01. Elas não seguem o score (MILS3, de score 0,537, aparece menos
  que POMO3, de 0,515). Com o δ de hoje (0,11), toda a cauda cabe na zona: a zona por frequência protege o
  titular de qualquer papel da cauda, seja ele melhor ou pior em score.
- **Escala do score:** desvio-padrão de 0,37 no universo; o núcleo fixo da carteira vai de 0,52 a 0,90 e a cauda
  disputada de 0,42 a 0,54.
- Só ações do moderado. Nos FIIs a seed não muda a carteira (#90), e o limiar de lá não sai desta medida.

### Para [Critério de parada do GA no platô](https://github.com/caiomugarte/TCC/issues/102)

- **O critério de hoje não dispara com nenhum número de runs.** Jaccard run a run de ~0,15 e CV do fitness
  dos runs de ~0,059 são propriedades de runs individuais, que não convergem. Eles não melhoram com R.
- **O consenso tem platô de fitness, mas não de composição.** O fitness médio do consenso é 6,96 com 50 runs,
  7,03 com 100, 7,04 com 150 e 7,10 com 300 (ótimo 7,14). A composição entre seeds segue mudando: 2,9, 1,9, 2,0
  e 1,2 papéis trocados. Nem 3.000 runs fixam a cauda, porque as frequências do 12º ao 14º empatam a ±0,002.
- **Custo medido por run do GA de produção:** ~0,67 s com 5 processos em paralelo (300 runs por seed em ~200 s).
  150 runs ≈ 100 s e 300 runs ≈ 200 s num núcleo.
- **Runs mais longos com os operadores de hoje rendem pouco:** 10.000 gerações levam o run de 66% para 77% do
  ótimo, a 80 s por run. População 3.000: 75%, a 41 s por run.
- **O que mais mexeu foi a taxa de mutação (sondagem):** com 0,001, o run chega a 97% do ótimo e o consenso de
  10 runs bate o ótimo, a ~12 s por run. Isso muda o modelo e a carteira.
- **Existe uma referência exata e determinística:** neste perfil, o ótimo do fitness é calculável por
  programação dinâmica em 0,6 ms, sem seed e sem runs, e é o top-12 por score sem duas classes da mesma
  empresa. Com `hhi_max`/`max_per_segment` (FIIs, `limitar_concentracao`), a mesma programação dinâmica continua
  exata com mais uma dimensão. Qualquer critério de parada pode ser medido contra esse valor.

## Limitações

- Um perfil (moderado, no score âncora, sem restrições) e um manifesto. O moderado não tem `hhi_max` nem
  `max_per_segment`. Com tetos, o ótimo pode deixar de ser o top-N e as conclusões de (c) precisam ser medidas
  de novo.
- O consenso foi medido sem titulares (sem zona de proteção), que é o caso de uma conta nova.
- As variantes longas de (d) têm de 6 a 20 runs de uma seed base. A sondagem de mutação não procurou a melhor
  combinação.
- Os tempos por run foram medidos com 5 ou 6 processos em paralelo em 6 núcleos e têm contenção.
- Não deu para isolar a rede: `unshare -rn` não é permitido nesta máquina. O caminho medido só lê o CSV e o
  manifesto locais. Nenhum script chama `refresh_inputs` ou a rede.
- Os scripts apontam para o registry local (`PRUMO_SNAPSHOTS`, padrão
  `/home/caio/Documents/projetos/TCC/snapshots`) e rodam a partir do venv do clone principal.

## Como reproduzir

Da pasta `docs/research/selecao-acoes-estabilidade/`, com o venv do clone principal (`$PY`):

```bash
$PY verify_pipeline.py results/verify_pipeline.json
$PY reference.py results/reference.json
$PY run_seeds.py 300 5 1 2 3 4 5 6 7 8 9 10          # ~7 min com 5 processos
$PY analyze.py results/analysis.json
$PY pooled.py results/pooled.json
$PY long_runs.py 5 base:300:400:50:20 pat200:300:400:200:20 full400:300:400:400:20 \
    gen2000:300:2000:2000:10 pop1000:1000:400:50:20 pop1000full:1000:400:400:10   # ~2 min
$PY long_runs.py 6 gen10000:300:10000:10000:6 pop3000full:3000:400:400:6 \
    mut005:300:400:50:20:0.005 mut005full:300:400:400:20:0.005 mut001full:300:2000:2000:10:0.001  # ~3 min
$PY analyze_long.py results/long_analysis.json
```
