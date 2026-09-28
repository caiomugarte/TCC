"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  ApiClientError,
  createRecommendation,
  getLatestCompletedRecommendation,
  getLatestRecommendation,
  getMe,
  getRecommendation,
  startPremiumRecommendation,
} from "@/lib/api-client";
import type { Plan, Recommendation, RecommendationStatus } from "@/lib/api-types";

const currency = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
});
const percentage = new Intl.NumberFormat("pt-BR", {
  style: "percent",
  maximumFractionDigits: 1,
});
const dateTime = new Intl.DateTimeFormat("pt-BR", {
  dateStyle: "short",
  timeStyle: "short",
});

function statusInvestRetrievedAt(recommendation: Recommendation): string {
  const sources = recommendation.provenance?.selector_sources;
  const values: string[] = [];
  for (const [label, value] of ([
    ["Ações", sources?.stocks?.retrieved_at],
    ["FIIs", sources?.fiis?.retrieved_at],
  ] as const)) {
    if (!value) continue;
    const parsed = new Date(value);
    const formatted = Number.isNaN(parsed.getTime()) ? value : dateTime.format(parsed);
    values.push(`${label}: ${formatted}`);
  }
  return values.length > 0 ? values.join(" · ") : "Horário indisponível";
}

function errorMessage(cause: unknown): string {
  return cause instanceof ApiClientError
    ? cause.message
    : "Não foi possível carregar sua recomendação.";
}

export function RecommendationView() {
  const { getToken } = useAuth();
  const [currentRun, setCurrentRun] = useState<Recommendation | null>(null);
  const [latestCompleted, setLatestCompleted] = useState<Recommendation | null>(null);
  const [plan, setPlan] = useState<Plan>("basic");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const requestRef = useRef(0);

  async function pollRecommendation(
    id: string,
    token: string | null,
    active: () => boolean,
  ): Promise<void> {
    let delayMs = 500;
    while (active()) {
      await new Promise((resolve) => window.setTimeout(resolve, delayMs));
      if (!active()) return;
      try {
        const next = await getRecommendation(id, token);
        if (!active()) return;
        setCurrentRun(next);
        setError(null);
        if (next.status === "completed") setLatestCompleted(next);
        if (next.status === "completed" || next.status === "failed") return;
        delayMs = Math.min(delayMs * 2, 5_000);
      } catch (cause) {
        if (active()) setError(errorMessage(cause));
        delayMs = Math.min(delayMs * 2, 5_000);
      }
    }
  }

  async function loadRecommendation(): Promise<void> {
    const requestId = ++requestRef.current;
    const active = () => requestRef.current === requestId;
    setLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const [account, current, completed] = await Promise.all([
        getMe(token),
        getLatestRecommendation(token),
        getLatestCompletedRecommendation(token),
      ]);
      if (!active()) return;
      setPlan(account.plan);
      setCurrentRun(current);
      setLatestCompleted(
        completed ?? (current?.status === "completed" ? current : null),
      );
      if (current?.status === "queued" || current?.status === "running") {
        void pollRecommendation(current.id, token, active);
      }
    } catch (cause) {
      if (active()) setError(errorMessage(cause));
    } finally {
      if (active()) setLoading(false);
    }
  }

  async function startNewRun(retryPremium = false): Promise<void> {
    const requestId = ++requestRef.current;
    const active = () => requestRef.current === requestId;
    setStarting(true);
    setError(null);
    try {
      const token = await getToken();
      const account = await getMe(token);
      if (!active()) return;
      setPlan(account.plan);
      const run = account.plan === "premium"
        ? await startPremiumRecommendation(retryPremium ? { force: true } : {}, token)
        : await createRecommendation({}, token);
      if (!active()) return;
      setCurrentRun(run);
      if (run.status === "completed") setLatestCompleted(run);
      if (run.status === "queued" || run.status === "running") {
        void pollRecommendation(run.id, token, active);
      }
    } catch (cause) {
      if (active()) setError(errorMessage(cause));
    } finally {
      if (active()) setStarting(false);
    }
  }

  useEffect(() => {
    void loadRecommendation();
    return () => {
      requestRef.current += 1;
    };
  }, [getToken]);

  const inProgress = currentRun?.status === "queued" || currentRun?.status === "running";
  const failed = currentRun?.status === "failed";
  const completed = latestCompleted ?? (currentRun?.status === "completed" ? currentRun : null);
  const statusLabel: Record<RecommendationStatus, string> = {
    queued: "Na fila",
    running: "Em processamento",
    completed: "Concluída",
    failed: "Indisponível",
  };

  if (loading && !currentRun && !latestCompleted) {
    return <section className="card" aria-live="polite"><p>Carregando sua alocação…</p></section>;
  }

  return (
    <section className="stack" aria-labelledby="recommendation-title">
      <div>
        <p className="eyebrow">{plan === "premium" ? "Plano Premium" : "Plano Basic"}</p>
        <h1 id="recommendation-title">Sua recomendação de investimentos</h1>
        <p>Consulte seu resultado mais recente ou solicite uma nova execução. Não é uma ordem de compra.</p>
      </div>

      {loading && <p aria-live="polite">Carregando recomendações salvas…</p>}
      {error && <p role="alert">{error}</p>}

      {inProgress && currentRun && (
        <section className="card" aria-live="polite">
          <p className="eyebrow">{currentRun.plan === "premium" ? "Plano Premium" : "Plano Basic"}</p>
          <h2>Preparando sua recomendação</h2>
          <p>{statusLabel[currentRun.status]}. A execução continua no servidor enquanto o resultado é preparado.</p>
          <div className="metadata-grid">
            <div><span>Status</span><strong>{statusLabel[currentRun.status]}</strong></div>
            <div><span>Execução</span><strong>{currentRun.id}</strong></div>
          </div>
        </section>
      )}

      {failed && currentRun && (
        <section className="card" aria-live="assertive">
          <p className="eyebrow">Execução atual · {statusLabel[currentRun.status]}</p>
          <h2>Esta execução não produziu um resultado</h2>
          <p>{currentRun.failureMessage ?? "A execução não produziu um resultado."}</p>
          <button className="button" type="button" disabled={starting} onClick={() => void startNewRun(true)}>
            Tentar novamente
          </button>
        </section>
      )}

      {completed && (
        <section className="stack" aria-labelledby="completed-title">
          <div>
            <p className="eyebrow">Resultado concluído · {completed.plan === "premium" ? "Premium" : "Basic"}</p>
            <h2 id="completed-title">Sua alocação por classe</h2>
          </div>

          <div className="allocation-grid">
            {completed.classes.map((item) => (
              <article className="allocation-card" key={item.key}>
                <p>{item.label}</p>
                <strong>{percentage.format(item.targetWeight)}</strong>
                <span>{currency.format(item.targetAmountBrl)}</span>
              </article>
            ))}
          </div>

          <div className="card metadata-grid">
            <div><span>Status</span><strong>{statusLabel[completed.status]}</strong></div>
            <div><span>Perfil</span><strong>v{completed.profileVersion}</strong></div>
            <div><span>Modelo</span><strong>{completed.modelVersion}</strong></div>
            <div><span>Dados até</span><strong>{completed.snapshotCutoff}</strong></div>
            <div><span>Status Invest</span><strong>{statusInvestRetrievedAt(completed)}</strong></div>
          </div>

          <div className="card split-card">
            <div>
              <h3>Ações brasileiras</h3>
              <ul>
                {completed.stocks.map((item) => (
                  <li key={item.ticker}>{item.ticker}: {percentage.format(item.portfolioWeight)} ({currency.format(item.targetAmountBrl)})</li>
                ))}
              </ul>
            </div>
            <div>
              <h3>FIIs</h3>
              <ul>
                {completed.fiis.map((item) => (
                  <li key={item.ticker}>{item.ticker}: {percentage.format(item.portfolioWeight)} ({currency.format(item.targetAmountBrl)})</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="card split-card">
            <div><h3>Premissas</h3><ul>{completed.assumptions.map((item) => <li key={item}>{item}</li>)}</ul></div>
            <div><h3>Riscos</h3><ul>{completed.risks.map((item) => <li key={item}>{item}</li>)}</ul></div>
          </div>
        </section>
      )}

      {!loading && !error && !completed && !inProgress && !failed && (
        <section className="card">
          <h2>Nenhuma recomendação concluída</h2>
          <p>Gere uma recomendação para atualizar seus dados e calcular sua alocação.</p>
        </section>
      )}

      <div className="actions">
        <button
          className="button"
          type="button"
          disabled={loading || starting || Boolean(inProgress)}
          onClick={() => void startNewRun(Boolean(latestCompleted))}
        >
          {starting ? "Enviando…" : completed ? "Gerar nova recomendação" : "Gerar recomendação"}
        </button>
        {error && <button className="button secondary" type="button" onClick={() => void loadRecommendation()}>Consultar novamente</button>}
        <Link className="button secondary" href="/app/portfolio">Informar minha carteira</Link>
        <Link className="button secondary" href="/app/onboarding">Revisar perfil</Link>
      </div>
    </section>
  );
}
