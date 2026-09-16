"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  ApiClientError,
  createRecommendation,
  getMe,
  getRecommendation,
  getLatestRecommendation,
  startPremiumRecommendation,
} from "@/lib/api-client";
import type { Recommendation, RecommendationStatus } from "@/lib/api-types";

const currency = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
});
const percentage = new Intl.NumberFormat("pt-BR", {
  style: "percent",
  maximumFractionDigits: 1,
});

export function RecommendationView() {
  const { getToken } = useAuth();
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const requestRef = useRef(0);

  async function loadRecommendation(retryPremium = false) {
    const requestId = ++requestRef.current;
    const active = () => requestRef.current === requestId;
    setLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const account = await getMe(token);
      if (!active()) return;
      const stored = await getLatestRecommendation(token);
      if (!active()) return;

      if (account.plan !== "premium") {
        const basic = stored ?? (await createRecommendation({}, token));
        if (!active()) return;
        setRecommendation(basic);
        return;
      }

      const initial = !retryPremium && stored?.plan === "premium"
        ? stored
        : await startPremiumRecommendation({}, token);
      if (!active()) return;
      setRecommendation(initial);
      if (initial.status === "queued" || initial.status === "running") {
        await pollPremiumRecommendation(initial.id, token, active);
      }
    } catch (cause) {
      if (!active()) return;
      setError(
        cause instanceof ApiClientError
          ? cause.message
          : "Não foi possível carregar sua recomendação.",
      );
    } finally {
      if (active()) setLoading(false);
    }
  }

  async function pollPremiumRecommendation(
    id: string,
    token: string | null,
    active: () => boolean,
  ): Promise<void> {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 500));
      if (!active()) return;
      const next = await getRecommendation(id, token);
      if (!active()) return;
      setRecommendation(next);
      if (next.status === "completed" || next.status === "failed") return;
    }
    throw new Error("A recomendação Premium demorou mais que o esperado.");
  }

  useEffect(() => {
    void loadRecommendation();
    return () => {
      requestRef.current += 1;
    };
  }, [getToken]);

  const premium = recommendation?.plan === "premium";
  const status = recommendation?.status ?? "completed";
  const stocks = recommendation?.stocks ?? [];
  const fiis = recommendation?.fiis ?? [];
  const statusLabel: Record<RecommendationStatus, string> = {
    queued: "Na fila",
    running: "Em processamento",
    completed: "Concluída",
    failed: "Indisponível",
  };

  if (loading) {
    return <section className="card" aria-live="polite"><p>Carregando sua alocação…</p></section>;
  }

  if (error || !recommendation) {
    return (
      <section className="card" aria-live="assertive">
        <p className="eyebrow">{premium ? "Premium" : "Basic"}</p>
        <h1>Recomendação indisponível</h1>
        <p>{error ?? "Complete seu perfil para continuar."}</p>
        <div className="actions">
          <button className="button" type="button" onClick={() => void loadRecommendation(premium)}>Tentar novamente</button>
          <Link className="button secondary" href="/app/onboarding">Revisar perfil</Link>
        </div>
      </section>
    );
  }

  if (status === "queued" || status === "running") {
    return (
      <section className="card" aria-live="polite">
        <p className="eyebrow">Premium</p>
        <h1>Preparando sua recomendação</h1>
        <p>{statusLabel[status]}. Esta página atualizará o resultado automaticamente.</p>
        <div className="metadata-grid">
          <div><span>Status</span><strong>{statusLabel[status]}</strong></div>
          <div><span>Execução</span><strong>{recommendation.id}</strong></div>
        </div>
      </section>
    );
  }

  if (status === "failed") {
    return (
      <section className="card" aria-live="assertive">
        <p className="eyebrow">Premium</p>
        <h1>Recomendação indisponível</h1>
        <p>{recommendation.failureMessage ?? "A execução não produziu um resultado."}</p>
        <div className="actions">
          <button className="button" type="button" onClick={() => void loadRecommendation(true)}>Tentar novamente</button>
          <Link className="button secondary" href="/app/onboarding">Revisar perfil</Link>
        </div>
      </section>
    );
  }

  return (
    <section className="stack" aria-labelledby="recommendation-title">
      <div>
        <p className="eyebrow">{premium ? "Plano Premium" : "Plano Basic"}</p>
        <h1 id="recommendation-title">Sua alocação por classe</h1>
        <p>
          {premium
            ? "Uma alocação personalizada com sleeves selecionados. Não é uma ordem de compra."
            : "Uma referência de distribuição para o seu perfil. Não é uma ordem de compra."}
        </p>
      </div>

      <div className="allocation-grid">
        {recommendation.classes.map((item) => (
          <article className="allocation-card" key={item.key}>
            <p>{item.label}</p>
            <strong>{percentage.format(item.targetWeight)}</strong>
            <span>{currency.format(item.targetAmountBrl)}</span>
          </article>
        ))}
      </div>

      <div className="card metadata-grid">
        <div><span>Status</span><strong>{statusLabel[status]}</strong></div>
        <div><span>Perfil</span><strong>v{recommendation.profileVersion}</strong></div>
        <div><span>Modelo</span><strong>{recommendation.modelVersion}</strong></div>
        <div><span>Dados até</span><strong>{recommendation.snapshotCutoff}</strong></div>
      </div>

      {premium && (stocks.length > 0 || fiis.length > 0) && (
        <div className="card split-card">
          <div>
            <h2>Ações brasileiras</h2>
            <ul>
              {stocks.map((item) => (
                <li key={item.ticker}>{item.ticker}: {percentage.format(item.portfolioWeight)} ({currency.format(item.targetAmountBrl)})</li>
              ))}
            </ul>
          </div>
          <div>
            <h2>FIIs</h2>
            <ul>
              {fiis.map((item) => (
                <li key={item.ticker}>{item.ticker}: {percentage.format(item.portfolioWeight)} ({currency.format(item.targetAmountBrl)})</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="card split-card">
        <div><h2>Premissas</h2><ul>{recommendation.assumptions.map((item) => <li key={item}>{item}</li>)}</ul></div>
        <div><h2>Riscos</h2><ul>{recommendation.risks.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>

      <div className="actions">
        <Link className="button" href="/app/portfolio">Informar minha carteira</Link>
        <Link className="button secondary" href="/app/onboarding">Revisar perfil</Link>
      </div>
    </section>
  );
}
