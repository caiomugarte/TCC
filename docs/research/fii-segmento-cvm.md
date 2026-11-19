# Dá para derivar o segmento de cada FII no ponto no tempo com a CVM?

**Medido em:** 18 de novembro de 2026
**Issue:** [#105](https://github.com/caiomugarte/TCC/issues/105), para o λ·concentração do GA de FIIs
([#99](https://github.com/caiomugarte/TCC/issues/99), decisão 6).
**Base:** o relatório da #98 (`docs/research/fii-pit-indicators.md`, branch `research/fii-pit-indicators`), com as
fontes, as URLs e as armadilhas dos informes da CVM.

## Resposta curta

Sim. Uma regra em cascata, só com os informes mensal e trimestral da CVM entregues até a data, dá um segmento para
**94% do universo líquido em 2019, 98% em 2022 e 97% em 2026** (ficam sem 4, 3 e 5 fundos). Ela concorda com o
segmento do Status Invest de 05/11/2026 em **81% dos fundos comparáveis (100 de 124)**. Sobram 5 a 7 segmentos
com pelo menos 5 fundos em cada data, o que deixa o λ·concentração separar carteiras que o tipo trata como iguais
(cinco tijolos de logística × cinco tijolos de segmentos diferentes). Dá para trocar o tipo pelo segmento no
λ·concentração, mantendo o tipo no z-score e no teto por tipo.

Três coisas que a issue sugeria **não** servem sozinhas:

- **`Segmento_Atuacao` sozinho:** concorda muito (88%), mas só dá segmento a 77% (2019), 69% (2022) e **36% (2026)**
  do universo. A CVM trocou a taxonomia nos informes com `Data_Referencia` desde ago/2025 (ver Armadilhas) e quase
  dois terços dos fundos passaram a "Multicategoria" ou "Outros".
- **Setor dos inquilinos (`imovel_renda_acabado_inquilino.Setor_Atuacao`):** é texto livre (212 valores distintos no
  universo de 2026), com 18–44% das linhas sendo modelo vazio ("Setor1", "SetorN", sempre com 0%), e descreve o
  **negócio do inquilino**, não o uso do imóvel: um galpão alugado ao Magazine Luiza aparece como "Varejo". Tomado
  como segmento, concorda com o Status Invest em só 59% (59 de 100).
- **Carteira de CRIs:** não diz o segmento do lastro. O `Emissor` é a securitizadora e não há devedor nem setor
  (#98). Papel fica como um segmento só, "Papel", que é também o que o Status Invest faz.

## A regra

Para cada fundo e data de decisão `d`, com o último informe mensal entregue até `d` (no máximo 270 dias de idade) e
o último trimestral entregue até `d`:

1. **Tipo** pela regra da #99: blocos de `ativo_passivo` do mensal sobre `Total_Investido`. Imóveis =
   `Direitos_Bens_Imoveis + Acoes_Sociedades_Atividades_FII + Cotas_Sociedades_Atividades_FII`; papel =
   `CRI + CRI_CRA + LCI + LCI_LCA + LIG + Letras_Hipotecarias`; FoF = `FII`. Um bloco com ≥ 60% define o tipo; senão,
   híbrido.
2. **Papel** → segmento "Papel".
3. **FoF ou híbrido** → o `Segmento_Atuacao` do mensal, se for específico; senão, a palavra-chave do nome do fundo;
   senão, o próprio tipo ("FoF" ou "Híbrido"). Isso trata os fundos que têm os imóveis por meio de cotas de FII
   exclusivos (RBRL11, GARE11, HOFC11: FoF pela composição, logística ou escritórios na prática).
4. **Tijolo** → a primeira fonte que der um segmento:
   1. `Segmento_Atuacao` do mensal, se específico (tabela abaixo);
   2. os **imóveis de renda** do trimestral (`imovel` com `Classe` "Imóveis para renda ..."): o uso de cada imóvel por
      palavra-chave no `Nome_Imovel` e em `Outras_Caracteristicas_Relevantes` (galpão, CD, logístic... → Logística;
      edifício, torre, corporate... → Escritórios; shopping → Shoppings; loja, agência, atacarejo... → Renda urbana;
      hospital, hotel, escola, fazenda...), com o setor do inquilino só como desempate onde o negócio coincide com o
      uso (hotelaria, educação, saúde, agência bancária). Pondera por `Percentual_Receitas_FII` (normalizado, porque
      mistura fração e percentual). Se pelo menos 50% da receita tem uso reconhecido e um uso tem ≥ 60% dela, é esse
      segmento; senão, "Misto";
   3. a palavra-chave do nome do fundo no informe (LOGÍSTICA, SHOPPING, RENDA URBANA, CORPORATE...).

Correspondência do `Segmento_Atuacao` (as duas taxonomias, antes e depois de ago/2025):

| CVM | segmento |
| --- | --- |
| Logística | Logística |
| Lajes Corporativas, Escritórios | Escritórios |
| Shoppings | Shoppings |
| Varejo | Renda urbana |
| Hospital, Hotel, Residencial, Educacional | o mesmo nome |
| Títulos e Val. Mob. | Papel (só confirma o tipo) |
| Híbrido, Multicategoria, Outros, vazio | não diz o segmento |

Peso de cada fonte na decisão em 2026: tipo 61 fundos (40 papel, 21 FoF/híbrido), `Segmento_Atuacao` 52, imóveis 27,
nome 4, sem segmento 5. Em 2019 os imóveis decidem 8 e o nome 1; o `Segmento_Atuacao` ainda carregava a maior parte.

## (a) Cobertura no universo líquido

Universo pela regra da #60/#98: patrimônio ≥ R$ 200 mi no último informe mensal entregue até `d` e liquidez média de 21
pregões ≥ R$ 50 mil/dia (COTAHIST, `CODBDI` 12, ticker terminado em 11 ou 11B), ligados pelo ISIN.

| regra | 05/11/2019 (66) | 05/11/2022 (131) | 05/11/2026 (149) | concorda com o Status Invest (2026) |
| --- | --- | --- | --- | --- |
| só `Segmento_Atuacao` | 51 (77%) | 91 (69%) | 53 (36%) | 42 de 48 (88%) |
| tipo + `Segmento_Atuacao` | 55 (83%) | 103 (79%) | 116 (78%) | 77 de 97 (79%) |
| tipo + setor dos inquilinos | 57 (86%) | 111 (85%) | 117 (79%) | 59 de 100 (59%) |
| tipo + `Segmento_Atuacao` + imóveis | 63 (95%) | 122 (93%) | 143 (96%) | 93 de 123 (76%) |
| **a regra acima** | **62 (94%)** | **128 (98%)** | **144 (97%)** | **100 de 124 (81%)** |

Ficam sem segmento: TGAR11, CARE11, ALZR11, MFII11 (2019); MFII11, ALZR11, TGAR11 (2022); VGRI11, MFII11, BPML11,
TGAR11, GGRC11 (2026). São tijolos com `Segmento_Atuacao` genérico e imóveis com nome que não diz o uso (GGRC11 tem 47
imóveis chamados "EXTREMA1", "QUEI 13638", "HERING"). Para esses, o λ pode usar o tipo (Tijolo) como segmento, o que
equivale ao comportamento de hoje.

O universo deu 66, 131 e 149 fundos, contra 70, 135 e 160 da #98: alguns fundos líquidos não declaram ISIN em nenhum
informe (PSEC11 e PMLL11, por exemplo) e não se ligam ao COTAHIST. Ligar pelo FNET (que traz ticker e ISIN) recupera
esses.

Distribuição em 05/11/2026: Papel 40, Logística 26, Escritórios 24, Shoppings 18, FoF 14, Híbrido 7, Renda urbana 7,
Hospital 2, Misto 2, Residencial 1, Hotel 1, Educacional 1, Agro 1, sem 5.

## (b) Concordância com o Status Invest e o Funds Explorer

Duas medições:

- **Universo inteiro de 05/11/2026 contra o CSV do Status Invest de 05/11/2026** já no repo
  (`docs/validation/proofs/cvm-b3-2026-11-17/fiis.csv`, coluna `SETOR`): **100 de 124 (81%)**. Ficam fora 20 fundos
  com "Indefinido", "Serviços Financeiros Diversos" ou fora do CSV, e os 5 sem segmento. "Misto" do site conta como
  acerto quando a regra dá "Misto" ou "Híbrido".
- **Os 30 fundos mais líquidos de 05/11/2026, na página de cada fundo dos dois sites, acessada em 18/11/2026**
  (os dois responderam ao acesso automatizado; o `robots.txt` deles não proíbe essas páginas): concorda com o Status
  Invest em **20 de 28** e com o Funds Explorer em **20 de 28**; com pelo menos um dos dois em 20 de 30; discorda dos
  dois em 8; 2 sem comparação (GGRC11 sem segmento nosso, KNHF11 "Indefinido" nos sites).

**Os dois sites não são referências independentes:** nos 30 fundos, o segmento do Funds Explorer é idêntico ao do
Status Invest em 30 de 30, com a mesma taxonomia ("Imóveis Industriais e Logísticos", "Imóveis Comerciais - Outros",
"Fundo de Desenvolvimento"...), e a página do Status Invest de hoje bate com o CSV de 05/11 em 30 de 30. O critério de
aceite da #99 ("concordar com pelo menos um dos sites") vira, na prática, concordar com o Status Invest.

Os 8 casos em que a regra discorda dos dois sites, para o Caio arbitrar (decisão 3 da #99):

| fundo | regra | sites | leitura |
| --- | --- | --- | --- |
| KNRI11 | Escritórios | Misto | ~64% da receita reconhecida em edifícios, o resto em CDs; limiar de 60% |
| JSRE11 | Escritórios | Misto | todos os imóveis são torres de escritório no informe |
| HGRU11 | Renda urbana | Misto | lojas e agências; o site não tem "renda urbana" |
| RZTR11 | Agro | Misto | fazendas; o site não tem segmento agro |
| TRXB11, VXXV11 | Renda urbana, Shoppings | Fundo de Desenvolvimento | o site separa desenvolvimento; a regra olha o uso do imóvel |
| SNEL11 | Híbrido | Fundo de Desenvolvimento | idem |
| CPTS11 | FoF | Papéis | 76% do ativo em cotas de FII no informe; é a regra de tipo da #99, não a de segmento |

No universo inteiro, as 24 discordâncias se dividem em: o site diz "Misto" e a regra dá um segmento (9: os de cima,
mais ADSH11, CPLG11, CPUR11, ALZR11, HGBL11); FoF ou híbrido pela composição que o site chama de papel, shopping ou
desenvolvimento (7: CPTS11, SNEL11, JSCR11, ITRI11, HSAF11, JCCJ11, URPR11; HSAF11 se chama "HSI RENDA LOGISTICA CDI"
no informe e o nome leva a regra para Logística); papel que o site chama de "Misto" ou "Fundo de Fundos" (3);
"Fundo de Desenvolvimento" (2); outros (3: VCRR11, em que a CVM diz Logística e os imóveis são hotéis e residências de
curta estadia, LIFE11 e TOPP11, que o site chama de "Imóveis Comerciais - Outros").

## (c) Quantos segmentos sobram e se o λ·concentração tem efeito

O fitness do GA é `Σ SCORE − λ · HHI · n`, com o HHI por **contagem** de fundos por `SETOR` (`py/core/optimizer.py`).
Na política, λ = 0,10 a 0,50 e n = 4 a 15 (`api/app/services/premium_policy.py`).

| data | tipo: grupos | segmento: grupos | segmentos com ≥ 5 fundos |
| --- | --- | --- | --- |
| 05/11/2019 | 4 | 10 | 5: Escritórios 18, Shoppings 13, Papel 13, Logística 6, FoF 5 |
| 05/11/2022 | 4 (3 com ≥ 5) | 12 | 6: Papel 38, Escritórios 23, Logística 21, FoF 18, Shoppings 12, Renda urbana 5 |
| 05/11/2026 | 4 | 13 | 7: Papel 40, Logística 26, Escritórios 24, Shoppings 18, FoF 14, Híbrido 7, Renda urbana 7 |

Menor HHI que o GA consegue (fundos espalhados ao máximo entre os grupos disponíveis):

| data | n | tipo | segmento (todos) | segmento (só os grupos com ≥ 5) |
| --- | --- | --- | --- | --- |
| 2019 | 5 / 10 / 15 | 0,28 / 0,26 / 0,25 | 0,20 / 0,10 / 0,11 | 0,20 / 0,20 / 0,20 |
| 2022 | 5 / 10 / 15 | 0,28 / 0,26 / 0,25 | 0,20 / 0,10 / 0,09 | 0,20 / 0,18 / 0,17 |
| 2026 | 5 / 10 / 15 | 0,28 / 0,26 / 0,25 | 0,20 / 0,10 / 0,08 | 0,20 / 0,16 / 0,15 |

Com o tipo, o HHI não cai abaixo de ~0,25 (4 grupos, número efetivo ≤ 4) e não distingue cinco tijolos de logística de
cinco tijolos de segmentos diferentes. Com o segmento, o número efetivo vai a 5–7 usando só os segmentos grandes e o
GA ganha um gradiente dentro do tijolo. Exemplo com n = 10 e λ = 0,5: 5 logísticas + 5 papéis e 5 tijolos de
segmentos diferentes + 5 papéis têm o mesmo HHI pelo tipo (0,50); pelo segmento, 0,50 contra 0,30, uma diferença de
λ·n·ΔHHI = 1,0 no fitness. Se 1,0 pesa contra a escala do `SCORE` do seletor novo não foi medido (a escala depende
dos z-scores da #99, que ainda não existem).

Dois limites: em 2019 o universo tem só 66 fundos e 5 segmentos grandes, então com n = 15 o piso é 0,20; e Papel
continua um bloco único de 38–40 fundos, sem subsegmento (high grade × high yield, indexador) porque a CVM não dá isso
de forma estruturada.

**Estabilidade no tempo:** dos fundos com segmento em 2022 e 2026, 82 de 89 mantêm o segmento; de 2019 para 2022,
52 de 54. As mudanças são fundos que de fato mudaram de composição ou passaram de tipo (híbrido ↔ FoF).

## Armadilhas

- **A CVM trocou a taxonomia de `Segmento_Atuacao` com `Data_Referencia` de ago/2025**, no mesmo mês em que o
  `Mandato` esvaziou (Resolução 175). Some "Títulos e Val. Mob." e "Híbrido"; entram "Multicategoria", "Escritórios"
  (no lugar de "Lajes Corporativas"), "Varejo" e "Educacional". Os papéis passaram a "Multicategoria" ou "Outros":
  em 2026, 64 dos 149 fundos do universo são "Multicategoria" e 32 "Outros". A regra precisa mapear as duas taxonomias.
- **`Setor_Atuacao` do inquilino é o negócio do inquilino**, não o uso do imóvel, e as linhas "Setor1"... "SetorN"
  são o modelo do formulário, sempre com 0%.
- **O nome do imóvel às vezes é o nome do inquilino:** MCLO11 tem galpões chamados "JUNDIAÍ - CASA BAHIA", que a
  palavra-chave de varejo pega. Por isso o `Segmento_Atuacao` específico vem antes dos imóveis.
- **O informe mensal às vezes lança imóveis de renda como "para venda":** MCLO11 declara 100% em
  `Imoveis_Venda_Acabados`. Uma regra de "desenvolvimento" pela composição (venda + terrenos) errou mais do que acertou
  e ficou de fora.
- **`Percentual_Receitas_FII` mistura fração e percentual** (#98): normalizar por fundo (se a soma passar de 1,5,
  dividir por 100). Às vezes a soma é zero; aí a ponderação cai para `Percentual_Imovel_Total_Investido` e, por fim,
  peso igual.
- **ISIN ausente no informe mais recente:** 15% das linhas do `geral` de 2026 vêm sem `Codigo_ISIN`. Ligar o COTAHIST
  pelo conjunto de ISINs que o CNPJ declarou em qualquer informe, e pelo prefixo de 9 caracteres (`BRXXXXCTF`) quando
  a série muda.
- **FoF pela composição inclui fundos que têm imóveis ou CRIs por meio de FIIs exclusivos** (RBRL11, GARE11, HOFC11,
  CPTS11). A regra de segmento contorna com o `Segmento_Atuacao`; a regra de tipo da #99 não.
- **Status Invest e Funds Explorer dão o mesmo segmento** (30 de 30): contar os dois como duas conferências é contar
  a mesma uma vez a mais.

## Como foi medido

Scripts descartáveis em Python (não commitados), sobre os informes mensal e trimestral de FII da CVM de 2018, 2019,
2021, 2022, 2025 e 2026 baixados de `https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/` e
`.../INF_TRIMESTRAL/DADOS/` em 18/11/2026, e o COTAHIST da B3 de 2019, 2022 e 2026. Para cada data (05/11 de 2019,
2022 e 2026), vale o último informe por CNPJ com `Data_Entrega` ≤ `d` (última versão), com `CNPJ_Fundo_Classe`
juntado a `CNPJ_Fundo` dos anos antigos; os detalhes do trimestral (`imovel`, `imovel_renda_acabado_inquilino`) entram
pela chave (CNPJ, `Data_Referencia`, `Versao`) do `geral` trimestral escolhido. A conferência usou o CSV do Status
Invest de 05/11/2026 do repo e as páginas de fundo do Status Invest e do Funds Explorer dos 30 fundos mais líquidos,
baixadas uma a uma com 1,5 s de intervalo; do Funds Explorer, este relatório cita só a concordância agregada e
exemplos, sem reproduzir a classificação.
