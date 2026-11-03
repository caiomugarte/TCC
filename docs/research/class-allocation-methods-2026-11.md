# Pesquisa: métodos de alocação entre classes

> **Validação point-in-time:** os números históricos deste documento não têm fonte point-in-time auditada. Veredito da alocação entre classes: VALIDATED; sobreposição com o período da escolha do método: full (ver [docs/validation/class-allocation-oos.md](../validation/class-allocation-oos.md)).

**Datas:** 29 de outubro a 1º de novembro de 2026
**Dados:** snapshot `dev-2026-10-29`, com corte em 2026-10-28
**Escopo:** entender por que a alocação Premium saía com ~65–75% em renda
fixa e sem cripto nem FIIs, avaliar se as âncoras dos perfis são
conservadoras demais e escolher o método que divide o capital entre as cinco
classes.
**Decisão:** [ADR 0003](../adr/0003-equal-weight-risky-classes-with-fixed-income.md).
**Artefatos:** [`class-allocation-2026-11/`](class-allocation-2026-11/).

## Conclusão

Dividir igualmente entre as quatro classes de risco e completar com renda fixa
até os tetos de volatilidade e de queda máxima do perfil (1/N + renda fixa)
venceu todos os otimizadores testados fora da amostra. Teve o melhor retorno
por unidade de risco, a menor queda máxima e quase nenhum giro. Isso
reproduz DeMiguel, Garlappi & Uppal (2009): os otimizadores perseguem
retornos passados e, com cinco classes e poucos anos de dados, essas
estimativas são majoritariamente ruído.

O porém: o 1/N dá a cripto o mesmo peso das outras classes de risco. Com a
covariância de 10 anos, cripto responde por ~72% do risco da carteira em
todos os perfis, acima do `crypto_risk_contribution_cap` atual (25% / ~42% /
50%). O conservador também recebe cripto (7%). Ver ADR 0003.

## 1. Diagnóstico do resultado original

A execução Premium `8d6ed43c` (perfil v2, score 0,53) entregou 75% renda
fixa, 15% internacional, 10% ações brasileiras, 0% FIIs e 0% cripto. O perfil
v3 (score 0,58) entregou 65 / 20 / 15 / 0 / 0. Causas encontradas:

1. **Histórico curto.** A classe FIIs usava a cesta fixa `caio_fiis`
   (`carteira_fii_caio_consensus.json`), com dados só a partir de
   2024-06-13. A alocação usa a interseção das datas de todas as classes,
   então todas foram cortadas para ~2,3 anos (580 dias), um período de Selic
   muito alta. O CDI rendeu 13,5% a.a. com 0,2% de volatilidade; o BTC,
   8,2% a.a. com 47%; os FIIs, 5,6% a.a.
2. **Cestas fixas em vez de índices.** Ações brasileiras usavam a cesta fixa
   `caio_stocks`, não o IBOV. Em 10 anos a cesta rendeu 21,9% a.a. contra
   10,5% a.a. do IBOV. IBOV e IFIX já eram baixados, mas o carregador Premium
   não os usava de propósito.
3. **Uma única janela de 3 anos.** `run_allocation` otimiza só os últimos
   `training_years = 3`. Os horizontes de 10 e 5 anos alimentam walk-forward
   e relatórios de robustez, não a carteira recomendada.
4. **Sem piso por classe.** A política Premium define `minimum_class_weight`
   como 0,0, sobrescrevendo o 0,05 da configuração de pesquisa, então classes
   dominadas vão a zero.
5. **O score é o mínimo entre apetite, capacidade e liquidez.** O apetite do
   usuário era 0,95, mas a capacidade (patrimônio até 50 mil, esta carteira
   sendo 60–80% do patrimônio) segurou o score em 0,58. Com essas respostas o
   score não passa de ~0,58, e a alocação de janela única só inclui cripto a
   partir de ~0,75.

A alocação entre classes é uma busca exaustiva em grade
(`py/pipelines/asset_allocation.py`, `optimize_window`), não um GA. O GA
seleciona ações e FIIs dentro das classes.

## 2. Âncoras dos perfis

Âncoras atuais (`ALLOCATION_PROFILE_ANCHORS`): conservador com volatilidade
≤ 10% e queda ≤ 15%, moderado com 15% / 25%, arrojado com 20% / 35%.

Com 3 anos completos de dados (IFIX como série de FIIs), o score 0,58 já
recebe 15% ações, 10% FIIs, 25% internacional, 45% renda fixa e 5% cripto com
as âncoras atuais. Um conjunto mais amplo (arrojado com 25% / 45%) só mudou
os scores a partir de 0,75 e gerou carteiras de canto, como 0% renda fixa,
70% internacional e 25% cripto. **Manter as âncoras atuais.**

Carteiras-modelo brasileiras (fontes primárias encontradas para três
instituições):

| Instituição | Renda fixa no arrojado | Cripto/alternativos no arrojado | Fonte |
|---|---|---|---|
| Santander | 48% + 20% multimercado | 7% alternativos (FIIs, ouro, notas; sem cripto) | [jul/2025](https://cms.santander.com.br/sites/WPS/documentos/url-arq-carteira-modelo-jul2025/25-06-27_125803_07-2025-recomendacao-mensal-de-investimento-completo.pdf) |
| Inter | 52% + 10% multiestratégia | 5% | [mai/2026](https://d9hzq3shxgvwu.cloudfront.net/news/files/cfdafdeb02674a4c99963fa1dadc58cc_investment-strategy-asset-allocation-maio-2026.pdf) |
| XP | ao menos 50% (fonte secundária) | não publicado | [InfoMoney](https://www.infomoney.com.br/?p=2735733) |

Os conservadores de Santander e Inter têm 0% de cripto. A Itaú Asset sugere
1–3% de bitcoin "proporcional ao perfil de risco" (fonte secundária:
[The Block, dez/2025](https://www.theblock.co/news/business/2025-12-14-brazils-largest-bank-recommends-a-3-bitcoin-portfolio-allocation-real-diversification-382459)).
Nenhuma instituição publicou metas de queda máxima; a "meta de oscilação" do
perfil mais arrojado da XP é 8% ([BPMoney, set/2026](https://bpmoney.com.br/economia/xp-aumenta-aposta-em-prefixados-com-expectativa-de-selic-mais-baixa/)).

A [Resolução CVM 30](https://conteudo.cvm.gov.br/legislacao/resolucoes/resol030.html)
exige avaliar objetivos, situação financeira e conhecimento, mas deixa a
forma de combinar as dimensões com a instituição. A única regra obrigatória
encontrada (orientação CVM/ANBIMA,
[fev/2021](https://www.anbima.com.br/pt_br/noticias/cvm-e-anbima-orientam-sobre-metodologia-de-classificacao-do-perfil-dos-investidores.htm))
é que um cliente avesso a risco e que precisa de liquidez fica no perfil mais
conservador. A regra do mínimo em `compute_profile` é escolha de produto, não
exigência regulatória.

## 3. Literatura e softwares

| Trabalho | Relevância |
|---|---|
| Michaud 1998; Michaud & Michaud 2008 | Reamostragem: tirar a média dos pesos ótimos de entradas reamostradas gera carteiras mais estáveis e diversificadas. |
| [Markowitz & Usmen, JOIM 2003](https://joim.com/article/resampled-frontiers-vs-diffuse-bayes-an-experiment) | A reamostragem de Michaud venceu uma abordagem bayesiana difusa numa simulação com 8 classes. |
| [Fletcher & Hillier, FAJ 2001](https://rpc.cfainstitute.org/research/financial-analysts-journal/2001/an-examination-of-resampled-portfolio-efficiency) | Algum ganho de Sharpe fora da amostra com reamostragem, possivelmente não significativo. |
| [Scherer, FAJ 2002](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/portfolio-resampling-review-and-critique) | Crítica: sem base em teoria da decisão; diversifica demais; ainda depende das entradas. |
| [Harvey, Liechty & Liechty, JOIM 2008](https://joim.com/article/bayes-vs-resampling-a-rematch) | Um ajuste pequeno faz o bayesiano vencer nos perfis de menor risco (lido só em resumo). |
| [DeMiguel, Garlappi & Uppal, RFS 2009](https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html) | Nenhum modelo otimizado venceu o 1/N de forma consistente fora da amostra. A rodada 2 reproduziu isso. |
| Ledoit & Wolf 2003/2004; Chekhlov, Uryasev & Zabarankin 2005; Politis & Romano 1994 | Encolhimento da covariância; CDaR; bootstrap estacionário por blocos. Citados de memória pelo agente de pesquisa, sem conferência nos artigos. |

A patente de Michaud US 6,003,018 aparece como expirada (2018) no
[Google Patents](https://patents.google.com/patent/US6003018A/en), assim como
as patentes relacionadas. "Resampled Efficiency" provavelmente é marca
registrada (não verificado). Não é parecer jurídico.

Bibliotecas Python (PyPI, nov/2026): **skfolio** (BSD-3, 1.12.5; MeanRisk com
limites de volatilidade, CDaR e queda máxima, walk-forward, bootstrap),
Riskfolio-Lib (BSD-3, 7.4.0), PyPortfolioOpt (MIT, 1.6.0), arch (NCSA,
bootstrap por blocos). O cvxportfolio é GPLv3; evitar. Não foi encontrado
estudo brasileiro cobrindo IBOV, IFIX, CDI, BTC e S&P juntos.

## 4. Experimentos

Todas as rodadas usam IBOV, IFIX, S&P 500 retorno total × PTAX, CDI e
BTC × PTAX de 2016-10-31 a 2026-10-28, âncoras atuais, sem custos nem
impostos. O IBOV vem do Yahoo e é um índice de retorno total
([B3](https://b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm)),
assim como o IFIX. Teste fora da amostra: todo 1º de novembro, de 2019 a
2025, cada método decide só com dados passados e segura a carteira por um
ano (score 0,58).

### Rodada 1: reamostragem × paridade de risco

- **Reamostragem por janelas** (média de 14 janelas móveis de 3 anos): escada
  coerente entre perfis e pesos estáveis.
- **Paridade de risco** (risco igual entre as classes de risco, completada
  com CDI até os tetos): 45–64% em FIIs, porque o IFIX tem volatilidade
  baixa, e a mesma carteira para os scores 0,75 e 1,0. Descartada.

### Rodada 2: todos os métodos

Fora da amostra, score 0,58, de 2019-11 a 2026-10:

| Método | Retorno a.a. | Volatilidade | Queda máxima | Giro por ano |
|---|---|---|---|---|
| Hoje (janela única de 3 anos) | 15,7% | 13,9% | −28,8% | 28% |
| Reamostragem por janelas | 16,4% | 13,6% | −28,8% | 10% |
| **1/N + renda fixa** | **17,5%** | **10,7%** | **−18,6%** | **2,5%** |
| Bootstrap por blocos + busca (40 amostras) | 15,6% | 12,7% | −27,5% | 10% |
| skfolio MeanRisk, limite de pior queda | 21,2% | 17,8% | −28,8% | 22% |
| skfolio MeanRisk, limite de CDaR | 21,2% | 17,9% | −28,8% | 22% |
| skfolio CDaR + bootstrap por blocos (300 amostras) | 20,7% | 17,5% | −28,4% | 11% |

Carteiras decididas com dados até 2026-10 (renda fixa / cripto):

| Score | Hoje | Reamostragem por janelas | 1/N + RF | Bootstrap + busca | skfolio CDaR + bootstrap |
|---|---|---|---|---|---|
| 0,00 | 60 / 0 | 74 / 0 | 71 / 7 | 65 / 0 | 49 / 9 |
| 0,25 | 50 / 0 | 56 / 3 | 62 / 10 | 50 / 2 | 41 / 12 |
| 0,50 | 45 / 5 | 52 / 5 | 52 / 12 | 41 / 5 | 34 / 14 |
| 0,58 | 45 / 5 | 52 / 5 | 49 / 13 | 37 / 6 | 32 / 15 |
| 0,75 | 35 / 10 | 45 / 9 | 43 / 14 | 28 / 8 | 28 / 17 |
| 1,00 | 10 / 20 | 33 / 12 | 32 / 17 | 19 / 10 | 23 / 19 |

Resultados:

- **1/N + renda fixa** foi o único método que ficou dentro do teto de queda
  do perfil 0,58 (26,6%) fora da amostra, inclusive em março de 2020.
- **skfolio** maximiza o retorno até os tetos, sem penalidade de
  concentração: 0% de renda fixa já no moderado, 9% de cripto até no
  conservador, e a maioria dos perfis acima do teto de volatilidade em 10
  anos. **O CDaR não mudou nada**: quem limitou foi a volatilidade, não a
  queda.
- **A reamostragem** (por janelas ou bootstrap) respeitou os tetos e escalou
  bem entre os perfis, mas não venceu o 1/N.
- Todos os métodos, exceto o 1/N, passaram do teto de queda do perfil 0,58
  fora da amostra. Os tetos são checados no passado e não são garantia.

### Ressalvas

- **Teste curto.** Sete anos e um único caminho de mercado, com uma alta
  forte de cripto que favorece qualquer peso fixo em cripto. Diferenças de
  1–2 pontos ao ano podem ser sorte.
- **Simplificações.** O 1/N checa os tetos em até 10 anos de dados passados,
  enquanto a busca checa na própria janela. As carteiras fora da amostra não
  são rebalanceadas dentro do ano.
