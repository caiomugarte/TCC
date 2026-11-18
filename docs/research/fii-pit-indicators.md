# O que a CVM e a B3 dão por FII no ponto no tempo

**Acessado em:** 18 de novembro de 2026
**Issue:** [#98](https://github.com/caiomugarte/TCC/issues/98), para a escolha dos indicadores do seletor de FIIs
([#94](https://github.com/caiomugarte/TCC/issues/94)). Este documento não decide quais indicadores entram.
**Base:** o relatório da #60 (`docs/validation/pit-rebuild-fiis.md`) e o pacote `py/pit_rebuild/`.

## Resposta curta

- **Proventos por cota têm fonte primária com mais de 3 anos:** os avisos estruturados de "Rendimentos e
  Amortizações" do **FNET** da B3 (Fundos.NET, onde o administrador publica o aviso aos cotistas). Cada aviso é um XML
  com ISIN, ticker, data-base (data-com), data de pagamento e `ValorProventoCota`, e a listagem traz a data de entrega
  com hora. O primeiro aviso é de 30/11/2016; a entrega sai no mesmo dia da data-base (mediana 0 dia, p99 4 dias).
  No universo líquido de 05/11/2023, 136 de 140 fundos têm aviso na janela de 12 meses, entregue até a data; no de
  05/11/2026, 148 de 160 (parte dos 12 que faltam é erro de ligação por ISIN, ver Armadilhas). Os valores batem com a
  página de fundos da B3 (199 de 199 iguais) e, onde diferem do Yahoo, o erro é do Yahoo (ajuste para trás,
  provento faltando). Isso substitui o Yahoo da #60.
- **Reserva dentro da CVM:** o informe mensal tem `Percentual_Dividend_Yield_Mes` (fração, não percentual).
  `Percentual_Dividend_Yield_Mes × Valor_Patrimonial_Cotas` reproduz o provento por cota do mês da data-com com
  diferença de até 5% em 91% dos casos contra a página da B3 (652 pares) e em 69% contra o FNET (478), mas chega
  com ~45 dias de atraso e arredondado. Serve como checagem ou reserva, não como fonte principal.
- **Patrimônio, P/VP, cotistas, taxa de administração, alavancagem, segmento e tipo de gestão** saem do informe
  mensal da CVM (desde set/2016, entregue ~15 dias após o mês). **Preço, liquidez e volatilidade** saem do COTAHIST.
- **Dos candidatos próprios,** existem com histórico desde 2016: vacância e inadimplência por imóvel, % da receita por
  imóvel, setor dos inquilinos relevantes, vencimento e indexador dos contratos (todos no informe trimestral, ~45 dias
  de atraso, só para fundos com imóveis), alavancagem e tijolo × papel (mensal), carteira de CRIs por emissão
  (trimestral). **Não existem** em fonte estruturada: vacância financeira, nome do inquilino, indexador, LTV e devedor
  de cada CRI.

## Fontes

| fonte | o que é | onde | desde | atraso de entrega |
| --- | --- | --- | --- | --- |
| Informe mensal de FII (CVM) | 3 tabelas: `geral`, `complemento`, `ativo_passivo` | `https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_AAAA.zip` | referência set/2016 | `Data_Entrega` − `Data_Referencia` (1º dia do mês): mediana 44 dias, p90 47, p99 137 (desde 2023). Ou seja, ~15 dias após o fim do mês. 11,5% dos (fundo, mês) têm reapresentação |
| Informe trimestral de FII (CVM) | 15 tabelas, entre elas `imovel`, `imovel_renda_acabado_inquilino`, `imovel_renda_acabado_contrato`, `complemento`, `ativo`, `resultado_contabil_financeiro` | `.../FII/DOC/INF_TRIMESTRAL/DADOS/inf_trimestral_fii_AAAA.zip` | referência 31/12/2016 | mediana 45 dias após o fim do trimestre, p90 49, p99 364 (desde 2023). 15,9% com reapresentação |
| Informe anual de FII (CVM) | 13 tabelas, entre elas `distribuicao_cotistas`, `ativo_valor_contabil`, `complemento` (gestor, política de remuneração) | `.../FII/DOC/INF_ANUAL/DADOS/inf_anual_fii_AAAA.zip` | referência 2016 | mediana 119 dias após `Data_Referencia` (~90 dias após o fim do exercício). 12% com reapresentação |
| Dicionários dos três informes | descrição dos campos | `.../INF_*/META/meta_inf_*_fii.zip` | | |
| COTAHIST (B3) | cotação diária por ticker e ISIN | `https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_AAAA.ZIP` | décadas antes de 2016 (baixado e lido aqui para 2019 e 2022–2026) | D+1 |
| FNET (B3), aviso estruturado "Rendimentos e Amortizações" | um XML por aviso de provento | listagem: `https://fnet.bmfbovespa.com.br/fnet/publico/pesquisarGerenciadorDocumentosDados?d=1&s=0&l=200&tipoFundo=1&idCategoriaDocumento=14&idTipoDocumento=41&idEspecieDocumento=0&cnpjFundo=<CNPJ só dígitos>`; documento: `https://fnet.bmfbovespa.com.br/fnet/publico/downloadDocumento?id=<id>` | data-base 30/11/2016 (35.938 avisos de FII no total em nov/2026) | `dataEntrega` − data-base: mediana 0 dia, p90 0, p99 4 (13.444 avisos do universo) |
| Página de fundos da B3 (`fundsProxy/GetListedSupplementFunds`) | proventos recentes, ISIN, quantidade de cotas | já usada na #60 | só ~12 meses | não tem data de entrega |

Prazos legais dos informes (mensal 15 dias após o mês, trimestral 45 dias após o trimestre, anual 90 dias após o
exercício) vêm da Instrução CVM 472, art. 39, hoje no Anexo Normativo III da Resolução CVM 175. Os atrasos medidos
acima batem com eles.

## Validação dos proventos do FNET

Amostra: 25 fundos sorteados (semente 98) entre os que estão nos universos de 05/11/2023 e de 05/11/2026; 516 XMLs
baixados de 17 fundos (o servidor limita a vazão, a amostra parou aí), com data-base de set/2022 a nov/2026.

| comparação | resultado |
| --- | --- |
| FNET × página de fundos da B3 (mesmo ISIN e data-base = `lastDatePrior`) | 199 pares de 17 fundos, **100% iguais** (diferença ≤ 0,1%) |
| FNET × Yahoo, soma de 12 meses até 05/11/2023 | 13 fundos: 8 dentro de 5% sem ajuste; 10 depois de desfazer o desdobramento que o Yahoo aplica para trás. Nos 3 que sobram, o erro é do Yahoo: RECT11 aparece com todos os valores × 0,765 (ajuste do Yahoo que o FNET e a B3 não têm), RBRL11 sem o provento de abr/2023, ALZR11 com a janela cortada por um desdobramento |
| FNET × CVM `Percentual_Dividend_Yield_Mes` × `Valor_Patrimonial_Cotas` (mês da data-base) | 478 (fundo, mês): 44% dentro de 1%, 69% dentro de 5% |
| FNET × CVM `Rendimentos_Distribuir` / `Cotas_Emitidas` (mesmo mês) | 59% dentro de 5%: o passivo acumula rendimento de mais de um mês e não serve como provento por cota |
| Página da B3 × CVM `Percentual_Dividend_Yield_Mes` × `Valor_Patrimonial_Cotas` (cache da #60, 62 fundos) | 652 pares: 57% dentro de 0,5%, 70% dentro de 1%, 91% dentro de 5% |

Formato do XML: até 08/09/2022, um `Rendimento` por aviso, com `DadosGerais/CodISINCota`, `CodNegociacaoCota` e
`ValorProventoCota`. A partir de 16/09/2022, blocos `InformeRendimentos/Provento`, cada um com `CodISIN`,
`CodNegociacao` e `Rendimento/ValorProvento`; um aviso pode trazer vários tickers (ALZR11, ALZR13 ... ALZR19:
recibos e direitos de subscrição, às vezes com valor diferente). Filtre pelo ticker ou ISIN da cota principal
(79 linhas de outro ticker na amostra).

## Checklist 1: os indicadores que o Caio usa

Universo líquido: patrimônio ≥ R$ 200 mi (informe mensal conhecido em `d`) e liquidez média de 21 pregões ≥ R$ 50
mil/dia (COTAHIST), cotas com ticker terminado em 11 ou 11B, ligados pelo ISIN. Tamanho medido: 70 fundos em
05/11/2019, 135 em 05/11/2022, 140 em 05/11/2023, 161 em 05/11/2024, 160 em 05/11/2026 (a #60 achou 162 e 166 com a
lista do Status Invest).

"Cobertura no universo" é a dos fundos do universo da data que têm o campo preenchido no informe daquele ano. A
cobertura por ano reflete sobretudo a idade dos fundos (um fundo de 2021 não tem informe de 2018), não falha da fonte.

| indicador | fonte e campo exato | fórmula point-in-time em `d` | ano mais antigo | atraso | cobertura no universo | veredito |
| --- | --- | --- | --- | --- | --- | --- |
| média de yield dos últimos 12 meses | FNET: `InformeRendimentos/Rendimento/ValorProventoCota`, `DataBase`; `dataEntrega` da listagem. Preço: COTAHIST | soma de `ValorProventoCota` dos avisos com data-base em (d − 365, d] e entregues até d (por (fundo, data-base) vale o último aviso não cancelado), sobre o fechamento do último pregão até d. Para "média mensal", dividir por 12 ou pela média dos preços de fim de mês | data-base nov/2016 | 0 dia (mediana) | 136 de 140 em 2023-11-05; 148 de 160 em 2026-11-05 (≥ 10 datas-base na janela: 130 e 140) | reconstruível |
| (reserva) yield pelo informe mensal | CVM mensal `complemento.Percentual_Dividend_Yield_Mes` × `Valor_Patrimonial_Cotas` | soma dos 12 últimos informes entregues até d; é yield sobre VP, não sobre preço | set/2016 | ~45 dias | 146 de 147 fundos do universo atual em 2023, 162 em 2026 | com ressalvas (arredondado; janela atrasada 1–2 meses) |
| patrimônio | CVM mensal `complemento.Patrimonio_Liquido` | do último informe com `Data_Entrega` ≤ d (já é a regra da #60) | set/2016 | ~45 dias | todo fundo com informe | reconstruível (#60: com ressalvas por ligação e defasagem) |
| P/VP | COTAHIST fechamento ÷ CVM mensal `complemento.Valor_Patrimonial_Cotas` | preço em d ÷ VP do último informe entregue até d | set/2016 | ~45 dias (VP) | todo fundo com informe | reconstruível (#60: 87%) |
| liquidez média diária | COTAHIST `VOLTOT`, CODBDI 12 | média dos 21 últimos pregões até d (#60) | antes de 2016 | D+1 | todo fundo negociado | reconstruível |
| volatilidade | COTAHIST `PREULT` | desvio-padrão dos retornos diários dos últimos 252 pregões até d, anualizado; precisa ajustar desdobramentos (salto de `Cotas_Emitidas` no informe, já em `fii_informes.split_factor`) e, se for retorno total, somar os proventos do FNET na data-ex | antes de 2016 | D+1 | 124 de 140 em 2023-11-05 e 135 de 160 em 2026-11-05 têm ≥ 200 pregões nos 400 dias anteriores | reconstruível |
| número de cotistas | CVM mensal `complemento.Total_Numero_Cotistas` (há também a quebra por tipo de investidor e, no anual, `distribuicao_cotistas` por faixa de participação) | do último informe entregue até d | set/2016 | ~45 dias | todo fundo com informe | reconstruível (#60: 74–90%, a referência do Status Invest é de outra data) |

## Checklist 2: candidatos próprios de FII

| candidato | existe? informe e campo | fórmula possível | desde | atraso | cobertura no universo de 2026 | observação |
| --- | --- | --- | --- | --- | --- | --- |
| vacância física | sim. Trimestral `imovel.Percentual_Vacancia` por imóvel (fração 0–1), com `Area` | média de `Percentual_Vacancia` ponderada por `Area` nos imóveis da classe "Imóveis para renda acabados" | 2016-12 | ~45 dias | 96–99 fundos (só os que têm imóvel) | um fundo pode informar o imóvel como "Imóveis para venda"; área às vezes zero |
| vacância financeira | não há campo. Aproximação: `Percentual_Vacancia` ponderada por `imovel.Percentual_Receitas_FII` | | 2016-12 | ~45 dias | idem | a ponderação por receita subestima: imóvel vago tem pouca receita |
| inadimplência | sim. Trimestral `imovel.Percentual_Inadimplencia` por imóvel | média ponderada por `Percentual_Receitas_FII` | 2016-12 | ~45 dias | 96–99 fundos | unidades misturadas (fração e percentual; há valores como 160000): exige limpeza |
| alavancagem (passivo/PL) | sim. Mensal `ativo_passivo.Total_Passivo` e `complemento.Patrimonio_Liquido` (também `Obrigacoes_Aquisicao_Imoveis`, `Obrigacoes_Securitizacao_Recebiveis`) | `Total_Passivo / Patrimonio_Liquido` do último informe entregue até d | set/2016 | ~45 dias | todo fundo com informe (mediana 2,3%, p75 15,5% em ago/2026) | o passivo inclui `Rendimentos_Distribuir`; separar a dívida de verdade |
| concentração por imóvel | sim. Trimestral `imovel.Percentual_Receitas_FII` por imóvel (e `Percentual_Imovel_Total_Investido`) | maior % ou HHI das receitas por imóvel | 2016-12 | ~45 dias | 91 fundos em jun/2026 (mediana 4 imóveis) | a soma por fundo às vezes passa de 1 (unidade em %) |
| concentração por inquilino | em parte. Trimestral `imovel_renda_acabado_inquilino`: `Setor_Atuacao`, `Percentual_Receita_Imovel`, `Percentual_Receitas_FII` por inquilino relevante, **sem o nome** | HHI por setor de inquilino; o maior inquilino só como "linha de maior %" | 2016-12 | ~45 dias | 90–95 fundos | sem nome não dá para somar o mesmo inquilino em imóveis diferentes |
| prazo dos contratos | sim, por faixa. Trimestral `complemento.Percentual_Vencimento_Receita_FII_Faixa_*` (até 3 meses ... acima de 36 meses, indeterminado) e `Percentual_Indexador_Receita_FII_{IGPM,IPCA,INPC,INCC}` | prazo médio aproximado pelos pontos médios das faixas; % da receita vencendo em 12 meses | 2016-12 | ~45 dias | 75–91 fundos com faixas; 85–106 com indexador | `imovel_renda_acabado_contrato.Caracteristicas_Contratuais` é texto livre ("Não há") |
| tijolo × papel | sim. Mensal `geral.Mandato` (Renda, Títulos e Valores Mobiliários, Híbrido, Desenvolvimento) e a composição do ativo (`Imoveis_Renda_Acabados`, `CRI_CRA`, `FII` sobre `Total_Investido`) | pela composição: % em imóveis vs % em CRI/CRA vs % em cotas de FII | set/2016 | ~45 dias | composição: todos; `Mandato`: vazio desde ago/2025 | `Mandato` sumiu do informe com a adaptação à Resolução 175: para datas recentes, só a composição |
| segmento (`Segmento_Atuacao`) | sim. Mensal, trimestral e anual `geral.Segmento_Atuacao` | do último informe entregue até d | set/2016 | ~45 dias | todos | 62% do universo é "Multicategoria" ou "Outros" em 2026; a #60 achou 27–28% de concordância com o Status Invest |
| carteira de CRIs: lista e concentração por emissão | sim. Trimestral `ativo` com `Tipo` (CRI, CRI/CRA, FII, LCI...), `Emissor` (securitizadora), `Emissao`, `Serie`, `Quantidade`, `Valor`, `Data_Vencimento` (muitas vezes vazio) | concentração por série ou por securitizadora; % da carteira em CRI | 2016-12 | ~45 dias | 158 fundos com alguma linha em `ativo` | o emissor é a securitizadora, não o devedor |
| CRIs: indexador, LTV, devedor | **não existe** em nenhum informe estruturado da CVM | | | | | só nos relatórios gerenciais (PDF no FNET), fora de dado estruturado |
| taxa de administração | sim. Mensal `complemento.Percentual_Despesas_Taxa_Administracao` (fração do PL no mês; mediana 0,056%/mês) e trimestral `resultado_contabil_financeiro.Taxa_Administracao_Financeiro` (R$) | soma dos 12 últimos meses entregues até d | set/2016 | ~45 dias | todos | é a despesa incorrida, não a taxa do regulamento |
| tipo de gestão | sim. Mensal `geral.Tipo_Gestao` (Ativa, Definida) | do último informe | set/2016 | ~45 dias | todos | a #60 não teve referência para comparar |

Outros campos com histórico que podem virar candidato: `Percentual_Rentabilidade_Patrimonial_Mes` e
`Percentual_Rentabilidade_Efetiva_Mes` (mensal), `Percentual_Amortizacao_Cotas_Mes` (mensal), lucro e resultado
financeiro do trimestre (`resultado_contabil_financeiro`), e valor justo × contábil por ativo (anual
`ativo_valor_contabil`).

## Armadilhas

- **`Percentual_Dividend_Yield_Mes` é fração, não percentual** (mediana 0,0068 em ago/2026). O mesmo vale para os
  `Percentual_*` do trimestral (vacância 0–1), mas `Percentual_Inadimplencia` e `Percentual_Receitas_FII` misturam as
  duas unidades entre fundos.
- **`Data_Referencia` do informe mensal é o 1º dia do mês**, não o último. O do trimestral é o último dia do trimestre.
- **O FNET indexa pelo CNPJ do fundo**; o informe da CVM, desde a Resolução 175, por `CNPJ_Fundo_Classe`. Os anos
  antigos do informe mensal usam a coluna `CNPJ_Fundo` (é preciso juntar as duas colunas, senão aparecem 19 mil linhas
  sem CNPJ e chaves duplicadas).
- **`Codigo_ISIN` do informe não é único**: fundos diferentes declaram o mesmo ISIN (ex.: `BRSPTWCTF002` aparece em
  `43.867.361/0001-42` e `43.951.785/0001-90`). Parte dos 12 fundos sem aviso no FNET em 2026 é ligação errada por
  ISIN, não falta de dado. O XML do FNET traz `CodISINCota` e `CodNegociacaoCota`, que servem de chave mais limpa.
- **Avisos cancelados e reapresentados no FNET:** a listagem traz `status` AC (ativo, 12.655 no universo), IC (inativo,
  571, versões substituídas) e CC (cancelado, 218). Por (fundo, data-base) vale o último aviso entregue até d que não
  foi cancelado.
- **A data-base do aviso é a data-com**; a data-ex é o pregão seguinte (a mesma regra da #60 para o `lastDatePrior`
  da B3).
- **Desdobramentos:** o `ValorProventoCota` é por cota da época (ex.: MXRF11 pagou R$ 0,95 em abr/2017, antes do
  desdobramento). O COTAHIST também não é ajustado, então o yield de 12 meses só quebra se a janela atravessar um
  desdobramento; ajustar pelo salto de `Cotas_Emitidas`.
- **`Mandato` está vazio desde ago/2025** (adaptação dos informes à Resolução 175). Tijolo × papel recente precisa da
  composição do ativo.
- **O Yahoo ajusta proventos para trás** (desdobramentos e outros eventos: RECT11 × 0,765) e perde registros; não serve
  de referência para o FNET.
- **Dois formatos de XML no FNET** (antes e depois de set/2022) e avisos com vários tickers; ver Validação.
- **A listagem do FNET recusa `l` > 200** (responde 302); pagine com `s`. Em rajada, o servidor devolve corpo vazio ou
  "error code: 522"; com 0,3–1 s entre chamadas funcionou.
- **Cobertura por ano é viés de sobrevivência e idade:** medir no universo de cada data, não no de hoje.

## Como foi medido

Scripts descartáveis em Python sobre os brutos (não commitados): os informes mensais do cache
`~/.cache/tcc-pit-rebuild`, os trimestrais e anuais baixados da CVM em 18/11/2026, o COTAHIST de 2019 e 2022–2026, a
listagem do FNET de 191 CNPJs (os universos de 2023 e 2026) e 516 XMLs de avisos. Os universos foram montados com a
regra da #60 (último informe entregue até `d`, no máximo 270 dias de idade; 21 pregões do COTAHIST).
