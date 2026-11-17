# Análise de Alocação da Carteira

Este contexto define a análise de alocação percentual entre classes de ativos para perfis de investimento.

## Linguagem

**Carteira**:
O conjunto total de capital analisado para um perfil, distribuído entre classes de ativos.
_Evitar_: wallet

**Classe de ativo**:
Uma categoria de investimento usada como bloco de alocação: ações brasileiras, FIIs, exposição internacional, renda fixa ou criptoativos.
_Evitar_: ativo individual, ticker

**Alocação entre classes**:
Os percentuais da carteira atribuídos a cada classe de ativo, cuja soma representa 100% do capital.
_Evitar_: seleção de ativos

**Perfil de alocação Caio**:
O perfil usado para calcular a distribuição da carteira de Caio, calibrado pelos parâmetros e resultados já existentes para sua carteira de ações e complementado por restrições próprias de alocação entre classes.
_Evitar_: copiar parâmetros de seleção de ações como se fossem percentuais de classes

**Calibração do perfil**:
O uso dos parâmetros e resultados atuais de Caio como evidência de suas preferências e tolerância observadas, sem executar novamente a seleção de ativos individuais na análise de alocação.
_Evitar_: ignorar o perfil atual, reutilizar o otimizador de ações

**Parâmetros de referência**:
Os parâmetros atuais de Caio permanecem responsáveis por produzir a carteira de ações de referência; a alocação entre classes usa controles próprios e não transfere pesos fundamentalistas para classes incompatíveis.
_Evitar_: aplicar parâmetros de ações diretamente a renda fixa ou criptoativos

**Referência retrospectiva**:
Uma simulação que leva a carteira Caio de referência atual para trás no tempo para medir seu comportamento histórico, sem apresentá-la como uma estratégia ex-ante disponível naquele passado.
_Evitar_: chamar a carteira estática de estratégia histórica sem viés

**Data de execução**:
Os pesos decididos no fechamento de uma data entram na carteira somente no próximo dia de negociação disponível.
_Evitar_: usar o mesmo fechamento para decidir e executar

**Política adaptativa**:
Uma regra que reotimiza os pesos anualmente durante o backtest walk-forward, usando apenas os 3 anos anteriores a cada decisão.
_Evitar_: confundir a política histórica com o alvo atual de Caio

**Alocação atual**:
Os pesos produzidos pela última janela de treino disponível, usados como o alvo vigente para Caio; por padrão, será o ponto de joelho da fronteira retorno-diversificação.
_Evitar_: chamar o melhor peso histórico de recomendação atual

**Retorno nominal**:
O retorno total bruto medido em BRL correntes; o retorno ajustado pelo IPCA será apenas uma métrica secundária.
_Evitar_: misturar retorno nominal e real no objetivo principal

**Carteira Caio de referência**:
O resultado registrado em `outputs/carteira_caio_consensus.json` e suas métricas de backtest, usados como referência empírica para a classe de ações brasileiras na alocação entre classes.
Seus 10 tickers permanecem fixos e são rebalanceados anualmente para pesos iguais.
_Evitar_: nova seleção de ações dentro da análise de alocação

**Exposição internacional**:
A participação em mercados ou ativos estrangeiros, acessada por BDRs, ETFs ou outro instrumento local, tratada como uma exposição econômica única para evitar dupla contagem.
Na primeira versão, será representada pelo S&P 500 Total Return convertido de USD para BRL como proxy de ações americanas.
_Evitar_: classe ETF, afirmar que o S&P 500 representa todo o mercado global

**Instrumento de acesso**:
O produto usado para obter uma exposição de carteira, como um BDR ou ETF; o instrumento não cria uma classe de alocação adicional.
_Evitar_: tratar BDR e ETF como classes independentes

**Horizonte de investimento**:
O período usado para avaliar a alocação: 10 anos como horizonte principal e 5 anos como janela de robustez e estresse.
_Evitar_: escolher a alocação com base em uma única janela histórica

**Rebalanceamento**:
A restauração periódica dos pesos-alvo da carteira; a regra inicial é rebalanceamento anual no aniversário de 12 meses do aporte.
_Evitar_: tratar os pesos iniciais como permanentes em uma análise de longo prazo

**Objetivo de alocação**:
Maximizar o retorno anualizado bruto, antes de custos e impostos, respeitando no cenário-base 20% de volatilidade anual e 30% de drawdown máximo; Sharpe, Sortino e Calmar servem como diagnósticos ou desempates.
_Evitar_: chamar o resultado bruto de retorno líquido, maximizar apenas o retorno histórico sem limites de risco

**Peso de classe**:
O percentual da carteira atribuído a uma classe de ativo; qualquer classe pode ter peso zero e não há limite máximo específico por classe, desde que os pesos não sejam negativos e somem 100%.
_Evitar_: impor diversificação nominal antes de avaliar o resultado

**Benchmark de classe**:
Uma série única de retorno total que representa o comportamento de uma classe de ativo para comparação e otimização.
_Evitar_: ativo escolhido

**Universo de alocação Caio**:
As cinco classes elegíveis para a carteira: ações brasileiras, FIIs, exposição internacional, renda fixa e criptoativos.
_Evitar_: separar BDRs e ETFs como classes adicionais

**Renda fixa de referência**:
A exposição pós-fixada representada pela Taxa DI diária da B3, frequentemente chamada de CDI, usada como proxy para CDBs e Tesouro Selic na primeira versão.
_Evitar_: misturar nessa classe títulos prefixados, indexados à inflação ou crédito privado

**Benchmark de FIIs**:
O IFIX da B3, usado como índice de retorno total incluindo as distribuições dos fundos, para representar a classe de FIIs.
_Evitar_: usar apenas a variação de preço do índice

**Benchmark de criptoativos**:
O retorno do Bitcoin convertido para BRL na primeira versão, escolhido pela extensão e liquidez de seu histórico.
_Evitar_: tratar uma cesta ampla de criptoativos como disponível sem uma série histórica consistente

**Fonte de criptoativos**:
Uma única fonte de mercado documentada fornece a série diária BTC/USD; o snapshot e o provedor ficam registrados na execução.
_Evitar_: combinar cotações de exchanges diferentes na mesma série

**Metodologia comparável**:
O uso da mesma janela, frequência, regra de rebalanceamento e tratamento de distribuições para comparar todas as classes; a primeira versão não aplica custos nem impostos.
_Evitar_: comparar séries calculadas sob regras diferentes

**Validação walk-forward**:
O processo de otimizar nos 3 anos anteriores, avaliar no ano seguinte não usado no ajuste e reotimizar anualmente, repetido ao longo da janela disponível.
_Evitar_: escolher pesos pelo melhor resultado dentro de toda a amostra histórica

**Janela sem solução factível**:
Uma janela em que nenhum vetor de pesos satisfaz simultaneamente as restrições de volatilidade e drawdown definidas para o treino.
_Evitar_: relaxar limites silenciosamente para fabricar uma solução

**Janela comum**:
O mesmo intervalo de datas usado por todas as classes em cada execução da comparação, limitado pela disponibilidade do benchmark mais curto.
_Evitar_: comparar classes com períodos históricos diferentes

**Cenário de aporte único**:
Uma simulação que investe o capital inicial uma vez e não inclui novos aportes ou retiradas durante o horizonte.
_Evitar_: interpretar a primeira versão como um plano de fluxo de caixa pessoal

**Escopo fiscal**:
Os impostos ficam fora da primeira versão; o resultado será reportado antes de impostos.
_Evitar_: apresentar o retorno pré-impostos como retorno líquido fiscal

**Retorno total bruto**:
O retorno da carteira considerando valorização e distribuições, antes de custos e impostos.
_Evitar_: retorno de preço, retorno líquido

**Série de retorno total**:
Uma série diária que combina valorização e distribuições reinvestidas para representar o retorno econômico da classe.
_Evitar_: usar apenas preços de fechamento sem distribuições

**Tratamento de dados ausentes**:
As séries são alinhadas pelas datas de negociação comuns, sem preenchimento artificial de preços; janelas com histórico insuficiente ou lacunas materiais são excluídas e reportadas.
_Evitar_: forward-fill de preços como se fosse retorno observado

**Histórico incompleto da referência**:
Se qualquer ticker fixo da carteira Caio de referência não tiver histórico suficiente em uma janela, essa janela fica incompleta e não recebe substituição.
_Evitar_: preencher a carteira com outro ticker ou ignorar silenciosamente o problema

**Snapshot de benchmark**:
Uma cópia dos dados de benchmark usada em uma execução, acompanhada da fonte e da data de obtenção para permitir reprodução dos resultados.
_Evitar_: depender apenas do estado atual de uma fonte externa

**Data de corte**:
O último dia comum disponível em todos os snapshots usados na execução, registrado como o limite temporal da análise.
_Evitar_: usar a data do computador como se fosse a última observação de mercado

**Moeda de análise**:
O BRL é a moeda-base da carteira; exposições estrangeiras são convertidas para BRL antes da otimização.
_Evitar_: misturar retornos em BRL e USD na mesma comparação

**Conversão cambial**:
A conversão diária de USD para BRL usa a cotação PTAX do Banco Central do Brasil, alinhada à data de observação disponível.
_Evitar_: converter cada benchmark com uma fonte cambial diferente

**Escopo de custos**:
Custos operacionais, taxas de administração, spreads e custos de rebalanceamento ficam fora da primeira versão.
_Evitar_: interpretar o resultado como retorno realizável líquido

**Métricas de decisão**:
Retorno anualizado bruto, volatilidade anual, drawdown máximo e Calmar são usados para avaliar a alocação; Sharpe fica fora do núcleo até existir uma série variável de taxa livre de risco.
_Evitar_: usar o proxy fixo de 10% como critério principal

**Diversificação entre classes**:
A dispersão dos pesos-alvo entre as cinco classes, medida pelo HHI de classes; HHI menor indica menor concentração, sem impor um teto fixo a uma classe.
_Evitar_: confundir diversificação entre classes com diversificação setorial de ações

**Fronteira retorno-diversificação**:
O conjunto de alocações não dominadas que mostra quanto retorno, risco e drawdown são trocados por diferentes níveis de concentração entre classes.
_Evitar_: apresentar uma única alocação como universalmente melhor

**Alocação de joelho**:
O ponto da fronteira em que a carteira já obteve grande parte do benefício de reduzir a concentração, e diversificar ainda mais passa a custar retorno de forma desproporcional; é um candidato a alvo equilibrado, não uma verdade universal.
_Evitar_: tratar o joelho como média igual entre as classes

**Seleção do joelho**:
O ponto da fronteira com maior distância normalizada da linha que conecta a alocação factível de maior retorno à alocação factível de menor HHI.
_Evitar_: escolher o joelho manualmente após observar os resultados

**Estabilidade da alocação**:
A variação dos pesos selecionados entre as janelas walk-forward, resumida por mediana, faixa e dispersão para cada classe.
_Evitar_: interpretar o último vetor de pesos como robusto sem comparar as janelas anteriores

**Baselines de alocação**:
As referências fixas comparadas sob as mesmas regras: 100% na carteira Caio de ações, pesos iguais nas cinco classes e 100% na Taxa DI.
_Evitar_: avaliar o otimizador sem um cenário simples de comparação

**Benchmark de contexto**:
O Ibovespa é uma referência externa para contextualizar o desempenho das ações brasileiras, mas não pertence ao universo de cinco classes da alocação.
_Evitar_: tratar o Ibovespa como uma sexta classe

**Varredura de diversificação**:
A execução da alocação sob diferentes intensidades de penalização do HHI de classes para revelar a fronteira retorno-diversificação.
_Evitar_: escolher uma penalização única sem mostrar seu efeito
