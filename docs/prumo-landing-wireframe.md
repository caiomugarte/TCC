# Prumo landing page wireframe

> Portuguese copy and low-fidelity structure for the public MVP page

**Status:** Draft  
**Related spec:** `.specs/features/landing-page-mvp/spec.md`  
**Audience:** Brazilian individual investors  
**Primary conversion:** Create an account

## Content direction

- Language: Brazilian Portuguese.
- Tone: calm, precise, transparent, and human.
- Positioning: portfolio decision assistant, not broker, custodian, or
  guaranteed-return service.
- Product promise: make profile, allocation, and portfolio drift easier to
  understand.
- Primary CTA: **Criar conta**.
- Secondary CTA: **Ver como funciona** or **Ver planos**.

## Low-fidelity wireframe

```text
┌────────────────────────────────────────────────────────────────────┐
│ prumo          Como funciona   Planos   Metodologia       Entrar   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  MAIS CLAREZA PARA DECIDIR SUA CARTEIRA.      ┌─────────────────┐  │
│  O Prumo organiza seu perfil, mostra uma       │  PREVIEW         │  │
│  alocação-alvo e ajuda você a revisar          │  Alocação-alvo   │  │
│  os desvios da sua carteira.                   │  Carteira atual  │  │
│                                                │  Desvios         │  │
│  [Criar conta]  [Ver como funciona]           └─────────────────┘  │
│  Sem custódia. Sem execução de ordens.                             │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  COMO FUNCIONA                                                     │
│  1. Entenda seu perfil   2. Defina seu alvo   3. Revise seus desvios│
├────────────────────────────────────────────────────────────────────┤
│  VEJA O QUE IMPORTA                                                │
│  Product preview: target allocation, current allocation, drift,     │
│  and contribution-first review.                                    │
├────────────────────────────────────────────────────────────────────┤
│  ESCOLHA COMO COMEÇAR                                              │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐ │
│  │ BASIC               │  │ PREMIUM                             │ │
│  │ Generic profiles    │  │ Customized profile                  │ │
│  │ Class allocation    │  │ Brazilian-stock analysis            │ │
│  │ [Começar grátis]    │  │ Exact targets and drift review      │ │
│  └─────────────────────┘  │ [Conhecer Premium]                 │ │
│                            └─────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│  TRANSPARÊNCIA ANTES DE PROMESSA                                  │
│  Methodology · assumptions · risks · data cutoff · no order         │
│  execution · legal/compliance boundary                             │
├────────────────────────────────────────────────────────────────────┤
│  FAQ                                                                │
│  O Prumo é uma corretora?                                          │
│  O Prumo executa ordens?                                           │
│  O resultado garante retorno?                                      │
│  O que muda entre Basic e Premium?                                 │
├────────────────────────────────────────────────────────────────────┤
│  SUA PRÓXIMA DECISÃO PODE COMEÇAR COM MAIS CONTEXTO.               │
│                         [Criar conta]                              │
├────────────────────────────────────────────────────────────────────┤
│ prumo · Privacidade · Termos · Contato · Avisos e riscos           │
└────────────────────────────────────────────────────────────────────┘
```

## Proposed copy

### Header

- Brand: `prumo`
- Navigation: **Como funciona**, **Planos**, **Metodologia**
- Utility action: **Entrar**

### Hero

Eyebrow:

> DECISÕES PARA SUA CARTEIRA

Headline:

> Mais clareza para decidir sua carteira.

Supporting copy:

> O Prumo ajuda você a entender seu perfil, definir uma alocação entre
> diferentes classes de ativos e acompanhar a distância entre sua carteira
> atual e seu objetivo.

Primary CTA: **Criar conta**  
Secondary CTA: **Ver como funciona**

Trust line:

> Sem custódia. Sem execução automática de ordens. Sem promessa de retorno.

### Product journey

Section heading:

> Uma decisão mais clara começa por três passos.

Step 1 — `Entenda seu perfil`

> Organize objetivos, horizonte, liquidez, capacidade financeira,
> conhecimento e restrições em um perfil compreensível.

Step 2 — `Defina seu alvo`

> Visualize uma alocação entre ações brasileiras, FIIs, exposição
> internacional, renda fixa e criptoativos.

Step 3 — `Revise seus desvios`

> Compare sua carteira atual com o alvo e priorize novos aportes antes de
> considerar mudanças mais difíceis.

### Product preview

Section heading:

> Veja o que importa antes de tomar a próxima decisão.

Supporting copy:

> O Prumo coloca alocação-alvo, carteira atual e desvios na mesma visão.
> Cada resultado mostra as premissas, o período dos dados e os limites do
> modelo.

Preview labels:

- `Alocação-alvo`
- `Carteira atual`
- `Desvio`
- `Próximo aporte`
- `Exemplo de interface`

Required preview disclaimer:

> Exemplo de interface. Não representa uma recomendação personalizada.

### Plans

Section heading:

> Comece com o nível de clareza que faz sentido para você.

Basic card:

> **Basic**  
> Para conhecer modelos de alocação e entender sua carteira em nível de
> classes.

Included copy:

- Perfis conservador, moderado e agressivo.
- Alocação entre cinco classes.
- Diagnóstico básico da carteira.

CTA: **Começar grátis**

Premium card:

> **Premium**  
> Para acompanhar uma visão personalizada da sua carteira e seus próximos
> ajustes.

Included copy:

- Perfil personalizado e contínuo.
- Análise aprofundada de ações brasileiras.
- Valores-alvo por classe e ação.
- Revisão de desvios e rebalanceamento.

CTA: **Conhecer Premium**

Pricing note: do not show a number until monthly price, annual price, and
trial policy are decided.

### Methodology and trust

Section heading:

> Transparência antes de promessa.

Supporting copy:

> O Prumo mostra como cada resultado foi produzido. Você vê as premissas,
> os riscos, a data de corte dos dados e a versão do modelo usada na análise.

Trust points:

- Dados e período de análise identificados.
- Premissas e limitações visíveis.
- Perfil e resultado preservados para comparação.
- Sem custódia e sem execução automática de ordens.
- Recomendações personalizadas dependem de revisão jurídica e operacional
  antes de uma oferta pública.

### FAQ

Question: `O Prumo é uma corretora?`

Answer:

> Não. O Prumo é uma ferramenta de apoio à decisão. Ele não mantém a
> custódia dos seus investimentos.

Question: `O Prumo executa ordens?`

Answer:

> Não. O MVP mostra alvos, desvios e ações para sua revisão. A decisão e a
> execução permanecem com você.

Question: `O resultado garante retorno?`

Answer:

> Não. Resultados dependem dos dados, das premissas, do período analisado e
> das condições futuras do mercado. Nenhum resultado representa garantia.

Question: `O que muda entre Basic e Premium?`

Answer:

> Basic oferece modelos genéricos e uma visão por classes. Premium adiciona
> perfil personalizado, análise aprofundada de ações brasileiras, valores
> exatos e revisão de desvios.

### Final CTA

Headline:

> Sua próxima decisão pode começar com mais contexto.

Supporting copy:

> Crie sua conta e conheça a visão do Prumo sobre sua carteira.

CTA: **Criar conta**

### Footer

- `Prumo`
- `Privacidade`
- `Termos de uso`
- `Avisos e riscos`
- `Contato`

Do not publish placeholder legal links as if they were final legal coverage.

## Motion cues

- Hero preview: one-time opacity and upward transform reveal.
- Journey steps: optional staggered reveal when they enter the viewport.
- Buttons and links: short hover and keyboard-focus transitions.
- Plan cards: no automatic movement; focus and hover states only.
- Reduced motion: remove entrance and scroll reveals; preserve content order.
- Never animate numbers as if they were live performance or guaranteed growth.

## Review checklist

- [ ] Hero explains product, audience, and next action.
- [ ] Copy distinguishes decision support from brokerage and advice claims.
- [ ] Three-step journey matches `LAND-02`.
- [ ] Preview is labeled and uses no unsupported result.
- [ ] Basic/Premium copy matches `LAND-04`.
- [ ] Prices and trial rules remain absent until decided.
- [ ] Methodology and compliance boundary appear before final CTA.
- [ ] Motion remains optional and reduced-motion safe.
- [ ] Layout works in `pt-BR` on mobile and desktop.
