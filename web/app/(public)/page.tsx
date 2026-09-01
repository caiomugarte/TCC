import { Show, UserButton } from "@clerk/nextjs";
import Link from "next/link";

import { LandingMotion } from "@/components/landing-motion";

export default function HomePage() {
  return (
    <div className="landing-page" data-landing-page>
      <a className="skip-link" href="#main">
        Pular para o conteúdo
      </a>

      <header className="site-header shell">
        <Link className="brand" href="#top" aria-label="Prumo, página inicial">
          <svg className="logo-mark" viewBox="0 0 64 64" aria-hidden="true">
            <path
              d="M22 8h14c11 0 20 9 20 20s-9 20-20 20H30v8a8 8 0 1 1-16 0V16a8 8 0 0 1 8-8Z"
              fill="currentColor"
            />
            <path d="M30 22h6a6 6 0 0 1 0 12h-6V22Z" fill="var(--paper)" />
          </svg>
          <span>prumo</span>
        </Link>
        <nav className="site-nav" aria-label="Navegação principal">
          <a href="#how">Como funciona</a>
          <a href="#plans">Planos</a>
          <a href="#trust">Metodologia</a>
          <Show
            when="signed-in"
            fallback={
              <Link className="button small secondary" href="/login">
                Entrar
              </Link>
            }
          >
            <div className="landing-account">
              <Link className="button small secondary" href="/app/onboarding">
                Abrir aplicação
              </Link>
              <UserButton />
            </div>
          </Show>
        </nav>
      </header>

      <main id="main">
        <section className="hero shell" id="top">
          <div className="hero-copy">
            <p className="eyebrow">Decisões para sua carteira</p>
            <h1>Mais clareza para decidir sua carteira.</h1>
            <p>
              O Prumo é um assistente de decisão para investidores individuais.
              Entenda seu perfil, defina uma alocação entre diferentes classes
              de ativos e acompanhe a distância entre sua carteira atual e seu
              objetivo.
            </p>
            <div className="hero-actions">
              <Show
                when="signed-in"
                fallback={
                  <Link className="button primary" href="/signup">
                    Criar conta
                  </Link>
                }
              >
                <Link className="button primary" href="/app/onboarding">
                  Abrir aplicação
                </Link>
              </Show>
              <a className="button secondary" href="#how">
                Ver como funciona
              </a>
            </div>
            <p className="trust-line">
              Sem custódia. Sem execução automática de ordens. Sem promessa de
              retorno.
            </p>
          </div>

          <div className="hero-visual">
            <div className="preview-card" aria-label="Exemplo de interface do Prumo">
              <div className="preview-top">
                <span>Exemplo de interface</span>
                <span className="preview-badge">Visão mensal</span>
              </div>
              <div className="preview-total">
                <div>
                  <span>Alocação-alvo</span>
                  <small>Exemplo ilustrativo</small>
                </div>
                <strong>100%</strong>
              </div>
              <div className="allocation-list">
                <div className="preview-row">
                  <div className="preview-row-head">
                    <span>
                      <i className="dot" aria-hidden="true" />Ações brasileiras
                    </span>
                    <strong>32%</strong>
                  </div>
                  <div className="bar">
                    <span style={{ width: "32%" }} />
                  </div>
                </div>
                <div className="preview-row">
                  <div className="preview-row-head">
                    <span>
                      <i className="dot green" aria-hidden="true" />FIIs
                    </span>
                    <strong>12%</strong>
                  </div>
                  <div className="bar">
                    <span className="green" style={{ width: "12%" }} />
                  </div>
                </div>
                <div className="preview-row">
                  <div className="preview-row-head">
                    <span>
                      <i className="dot violet" aria-hidden="true" />Exterior
                    </span>
                    <strong>24%</strong>
                  </div>
                  <div className="bar">
                    <span className="violet" style={{ width: "24%" }} />
                  </div>
                </div>
                <div className="preview-row">
                  <div className="preview-row-head">
                    <span>
                      <i className="dot blue" aria-hidden="true" />Renda fixa
                    </span>
                    <strong>20%</strong>
                  </div>
                  <div className="bar">
                    <span className="blue" style={{ width: "20%" }} />
                  </div>
                </div>
                <div className="preview-row">
                  <div className="preview-row-head">
                    <span>
                      <i className="dot gray" aria-hidden="true" />Criptoativos
                    </span>
                    <strong>12%</strong>
                  </div>
                  <div className="bar">
                    <span className="gray" style={{ width: "12%" }} />
                  </div>
                </div>
              </div>
              <div className="drift-box">
                <span>Revisão de desvios</span>
                <div className="drift-row">
                  <strong>+4,2 p.p.</strong>
                  <span className="drift-state">Dentro da faixa</span>
                </div>
              </div>
              <p className="preview-note">
                Dados ilustrativos. Não representa uma recomendação personalizada.
              </p>
            </div>
          </div>
        </section>

        <section className="section section-muted" id="how">
          <div className="shell">
            <div className="section-heading" data-reveal>
              <p className="eyebrow">Como funciona</p>
              <h2>Uma decisão mais clara começa por três passos.</h2>
            </div>
            <div className="steps" data-reveal>
              <article className="step">
                <p className="step-number">01</p>
                <h3>Entenda seu perfil</h3>
                <p>
                  Organize objetivos, horizonte, liquidez, capacidade financeira,
                  conhecimento e restrições em um perfil compreensível.
                </p>
              </article>
              <article className="step">
                <p className="step-number">02</p>
                <h3>Defina seu alvo</h3>
                <p>
                  Visualize uma alocação entre ações brasileiras, FIIs, exposição
                  internacional, renda fixa e criptoativos.
                </p>
              </article>
              <article className="step">
                <p className="step-number">03</p>
                <h3>Revise seus desvios</h3>
                <p>
                  Com dados inseridos manualmente, compare sua carteira atual com
                  o alvo e revise os desvios mensalmente, priorizando novos
                  aportes.
                </p>
              </article>
            </div>
          </div>
        </section>

        <section className="section shell preview-section" id="preview">
          <div data-reveal>
            <p className="eyebrow">Uma visão para decidir</p>
            <h2>Veja o que importa antes de tomar a próxima decisão.</h2>
            <p>
              O Prumo coloca alocação-alvo, carteira atual e desvios na mesma
              visão. Cada resultado mostra as premissas, o período dos dados e os
              limites do modelo.
            </p>
            <div className="callout">
              Exemplo de interface. Não representa uma recomendação personalizada.
            </div>
          </div>
          <div className="preview-card" aria-label="Exemplo de revisão de carteira" data-reveal>
            <div className="preview-top">
              <span>Carteira principal</span>
              <span className="preview-badge">Exemplo</span>
            </div>
            <div className="preview-total">
              <div>
                <span>Valor acompanhado</span>
                <small>Atualização manual</small>
              </div>
              <strong>—</strong>
            </div>
            <div className="allocation-list">
              <div className="preview-row">
                <div className="preview-row-head">
                  <span>
                    <i className="dot" aria-hidden="true" />Ações brasileiras
                  </span>
                  <strong>Alvo 32%</strong>
                </div>
                <div className="bar">
                  <span style={{ width: "32%" }} />
                </div>
              </div>
              <div className="preview-row">
                <div className="preview-row-head">
                  <span>
                    <i className="dot green" aria-hidden="true" />FIIs
                  </span>
                  <strong>Alvo 12%</strong>
                </div>
                <div className="bar">
                  <span className="green" style={{ width: "12%" }} />
                </div>
              </div>
              <div className="preview-row">
                <div className="preview-row-head">
                  <span>
                    <i className="dot violet" aria-hidden="true" />Exterior
                  </span>
                  <strong>Alvo 24%</strong>
                </div>
                <div className="bar">
                  <span className="violet" style={{ width: "24%" }} />
                </div>
              </div>
            </div>
            <div className="drift-box">
              <span>Próximo aporte</span>
              <div className="drift-row">
                <strong>Itens abaixo do alvo</strong>
                <span className="drift-state">Revisar</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section section-muted" id="plans">
          <div className="shell">
            <div className="section-heading" data-reveal>
              <p className="eyebrow">Planos</p>
              <h2>Comece com o nível de clareza que faz sentido para você.</h2>
            </div>
            <div className="plans" data-reveal>
              <article className="plan">
                <p className="plan-label">Basic</p>
                <h3>Modelos claros para começar.</h3>
                <p>
                  Para conhecer modelos de alocação e entender sua carteira em
                  nível de classes.
                </p>
                <ul>
                  <li>Perfis conservador, moderado e agressivo.</li>
                  <li>Alocação entre cinco classes.</li>
                  <li>Diagnóstico básico da carteira.</li>
                </ul>
                <Link className="button secondary" href="/signup">
                  Começar grátis
                </Link>
              </article>
              <article className="plan featured">
                <p className="plan-label">Premium</p>
                <h3>Uma visão personalizada da sua carteira.</h3>
                <p>
                  Para acompanhar seus alvos, sua composição e seus próximos
                  ajustes.
                </p>
                <ul>
                  <li>Perfil personalizado e contínuo.</li>
                  <li>Análise aprofundada de ações brasileiras.</li>
                  <li>Valores-alvo por classe e ação.</li>
                  <li>Revisão de desvios e rebalanceamento.</li>
                </ul>
                <Link className="button primary" href="/signup">
                  Conhecer Premium
                </Link>
              </article>
            </div>
          </div>
        </section>

        <section className="section shell" id="trust">
          <div className="section-heading" data-reveal>
            <p className="eyebrow">Metodologia e limites</p>
            <h2>Transparência antes de promessa.</h2>
            <p>
              O Prumo mostra como cada resultado foi produzido. Você vê as
              premissas, os riscos, a data de corte dos dados e a versão do modelo
              usada na análise.
            </p>
          </div>
          <div className="trust-grid" data-reveal>
            <div className="trust-item">
              <strong>Dados identificados</strong>
              <p>Período e fonte entram na leitura do resultado.</p>
            </div>
            <div className="trust-item">
              <strong>Premissas visíveis</strong>
              <p>Limites e escolhas do modelo ficam explicados.</p>
            </div>
            <div className="trust-item">
              <strong>Sem execução</strong>
              <p>O Prumo não mantém custódia nem executa ordens.</p>
            </div>
            <div className="trust-item">
              <strong>Revisão responsável</strong>
              <p>Personalização depende de revisão jurídica e operacional.</p>
            </div>
          </div>
        </section>

        <section className="section section-muted" id="faq">
          <div className="shell faq">
            <div className="section-heading" data-reveal>
              <p className="eyebrow">Dúvidas comuns</p>
              <h2>Perguntas importantes antes de começar.</h2>
            </div>
            <div data-reveal>
              <details>
                <summary>O Prumo é uma corretora?</summary>
                <p>
                  Não. O Prumo é uma ferramenta de apoio à decisão. Ele não mantém
                  a custódia dos seus investimentos.
                </p>
              </details>
              <details>
                <summary>O Prumo executa ordens?</summary>
                <p>
                  Não. O MVP mostra alvos, desvios e ações para sua revisão. A
                  decisão e a execução permanecem com você.
                </p>
              </details>
              <details>
                <summary>O resultado garante retorno?</summary>
                <p>
                  Não. Resultados dependem dos dados, das premissas, do período
                  analisado e das condições futuras do mercado.
                </p>
              </details>
              <details>
                <summary>O que muda entre Basic e Premium?</summary>
                <p>
                  Basic oferece modelos genéricos e uma visão por classes. Premium
                  adiciona perfil personalizado, análise aprofundada de ações
                  brasileiras, valores exatos e revisão de desvios.
                </p>
              </details>
            </div>
          </div>
        </section>

        <section className="final-cta" id="final-cta">
          <div className="shell">
            <h2>Sua próxima decisão pode começar com mais contexto.</h2>
            <Link className="button" href="/signup">
              Criar conta
            </Link>
          </div>
        </section>
      </main>

      <footer id="footer">
        <div className="shell footer-inner">
          <span>prumo · decisões mais claras para sua carteira</span>
          <nav className="footer-links" aria-label="Links do rodapé">
            <a href="#trust">Avisos e riscos</a>
            <a href="#faq">Perguntas</a>
            <span>Privacidade, termos e contato antes do lançamento.</span>
          </nav>
        </div>
      </footer>
      <LandingMotion />
    </div>
  );
}
