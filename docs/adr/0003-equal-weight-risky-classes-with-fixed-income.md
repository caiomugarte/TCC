# ADR 0003: Pesos iguais entre as classes de risco, completados com renda fixa

## Status

Aceita. Substitui as partes de alocação entre classes da ADR 0001 (cesta fixa
de ações e fronteira determinística como alvo do Premium).

## Contexto

A alocação Premium entre classes fazia uma busca em grade (passos de 5% e 1%)
nos últimos três anos de dados, com tetos de volatilidade e de queda máxima
por perfil. Na prática o resultado era instável e dominado pelo ciclo de
juros mais recente: um perfil moderado recebia 65–75% em renda fixa e nada de
cripto ou FIIs, e os pesos mudavam até 40 pontos de um ano para o outro. As
séries das classes também usavam cestas fixas da carteira do fundador, não
índices de mercado, e o histórico curto da cesta de FIIs cortava todas as
classes para ~2,3 anos.

As execuções Premium devem ser refeitas a cada ~30 dias. Um método que
embaralha a carteira a cada execução é um problema de produto, não só
estatístico.

A pesquisa em
[`docs/research/class-allocation-methods-2026-11.md`](../research/class-allocation-methods-2026-11.md)
comparou sete métodos fora da amostra, de 2019 a 2026. Pesos iguais entre as
classes de risco, completados com renda fixa, renderam 17,5% a.a. com 10,7%
de volatilidade, queda máxima de −18,6% e 2,5% de giro anual. O melhor
otimizador por risco rendeu 16,4% a.a. com 13,6% de volatilidade e −28,8%; o
melhor por retorno, 21,2% a.a. com 17,8% e −28,8%.

## Decisão

- **Séries das classes:** a etapa de alocação usa índices de mercado. IBOV
  para ações brasileiras, IFIX para FIIs, S&P 500 retorno total × PTAX para
  internacional, CDI para renda fixa e BTC × PTAX para cripto. As cestas
  fixas `caio_stocks` e `caio_fiis` saem da etapa de alocação. IBOV e IFIX
  são índices de retorno total, então já incluem proventos.
- **Método:** dividir o orçamento de risco igualmente entre as classes de
  risco permitidas ao perfil (ações brasileiras, FIIs, internacional e
  cripto) e completar com renda fixa. A parcela de risco é a maior, em passos
  de 1 ponto, cuja volatilidade e queda máxima históricas ficam dentro dos
  tetos interpolados do perfil.
- **Restrições do formulário continuam valendo.** "Evitar cripto" e "evitar
  exterior" tiram a classe da divisão (1/N entre as classes restantes);
  "priorizar renda" mantém o mínimo de 40% em renda fixa.
- **Iniciantes e cripto:** o formulário passa a sugerir a restrição "evitar
  cripto" quando as respostas indicarem um investidor iniciante ou
  conservador. A sugestão é do formulário; quem decide é o usuário.
- **Âncoras:** manter as `ALLOCATION_PROFILE_ANCHORS` atuais.
- **Sem mudança:** o GA continua selecionando ações e FIIs dentro das classes.

## Consequências

Positivas:

- Melhor retorno por unidade de risco e menor queda máxima entre os métodos
  testados fora da amostra.
- Carteiras quase estáticas: as execuções mensais só mudam os pesos quando o
  risco da parcela de risco muda, o que combina com a cadência de 30 dias do
  Premium.
- Fácil de explicar: "partes iguais em cada classe de risco, o resto em renda
  fixa de acordo com o seu perfil".
- Rápido: sem busca em grade, a etapa de alocação leva bem menos de um
  segundo.

Negativas:

- O método deixa de usar retornos esperados; o perfil só move a parcela de
  renda fixa.
- Cripto recebe o mesmo peso das outras classes de risco: 7% no score 0,0,
  13% no 0,58 e 17% no 1,0. Com a covariância de 10 anos, cripto responde por
  ~72% do risco da carteira em todos os perfis.
- Parte da vantagem fora da amostra pode vir da alta de cripto de 2019–2026.
  O teste cobre sete anos e um único caminho de mercado.

## Questões em aberto

### 1. Limite de participação de cripto no risco

**O que é:** cada perfil tem hoje um `crypto_risk_contribution_cap`, a fatia
máxima do risco total da carteira que pode vir de cripto: 25% no
conservador, ~42% no score 0,58 e 50% no arrojado. Esse limite foi criado
para o otimizador antigo. No 1/N, cripto responde por ~72% do risco em todos
os perfis, então o limite sempre seria violado.

**Opções:**

- **Manter o limite:** o 1/N teria que reduzir cripto até caber, deixando de
  ser 1/N. O resultado testado não vale mais.
- **Remover o limite:** cripto fica com peso igual, e o risco total continua
  contido pelos tetos de volatilidade e de queda do perfil, que já
  consideram cripto ao definir a parcela de risco.
- **Trocar por um teto de peso** (por exemplo, cripto no máximo 10%): meio
  termo simples, mas também não é o que foi testado.

**Recomendação: remover o limite no método 1/N.** O que protege o usuário é
o risco total da carteira, e ele continua limitado pelos tetos do perfil. Em
números: com 13% em cripto, uma queda de 75% do BTC (como em 2022) tira ~10%
da carteira, dentro do teto de queda do perfil. O teste fora da amostra já
incluiu esse efeito. Quem não quer cripto usa a restrição "evitar cripto".

### 2. Janela usada para checar os tetos

**O que é:** a parcela de risco é escolhida para que a volatilidade e a queda
máxima históricas fiquem dentro dos tetos do perfil. É preciso decidir em
qual período medir isso.

**Opções:**

- **3 anos:** reflete o momento recente, mas pode não incluir nenhuma crise
  (a janela atual não tem a COVID) e libera mais risco do que o perfil
  aguentaria numa crise de verdade.
- **10 anos (todo o histórico disponível):** inclui a COVID em 2020 e o
  inverno cripto de 2022, então a medição é mais prudente e mais estável
  mês a mês.

**Recomendação: 10 anos**, como na pesquisa. Foi com essa janela que o 1/N
ficou dentro do teto de queda fora da amostra. Uma janela que inclui crises
é a forma honesta de dizer "esta carteira aguentaria uma queda como a de
2020".

### 3. Regra para sugerir "evitar cripto" no formulário

**O que é:** falta definir quais respostas disparam a sugestão.

**Recomendação:** sugerir quando ao menos uma destas valer: experiência
"nenhuma" ou "básica"; produtos já usados "nenhum" ou "renda fixa e fundos";
ou perfil resultante conservador (score abaixo de 0,25). Mostrar um texto
curto explicando o motivo e deixar o usuário desmarcar.

### Resolvida

- **Proventos no IBOV:** o Ibovespa é um índice de retorno total
  ([B3](https://b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm)),
  e a série do Yahoo replica o índice oficial, então já inclui proventos.
