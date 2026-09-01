"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiClientError,
  createRecommendation,
  getLatestRecommendation,
} from "@/lib/api-client";
import type { Recommendation } from "@/lib/api-types";

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

  async function loadRecommendation() {
    setLoading(true);
    setError(null);
    try {
      const token = await getToken();
      const stored = await getLatestRecommendation(token);
      setRecommendation(stored ?? (await createRecommendation({}, token)));
    } catch (cause) {
      setError(
        cause instanceof ApiClientError
          ? cause.message
          : "Não foi possível carregar sua recomendação.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadRecommendation();
  }, [getToken]);

  if (loading) {
    return <section className="card" aria-live="polite"><p>Carregando sua alocação…</p></section>;
  }

  if (error || !recommendation) {
    return (
      <section className="card" aria-live="assertive">
        <p className="eyebrow">Basic</p>
        <h1>Recomendação indisponível</h1>
        <p>{error ?? "Complete seu perfil para continuar."}</p>
        <div className="actions">
          <button className="button" type="button" onClick={() => void loadRecommendation()}>Tentar novamente</button>
          <Link className="button secondary" href="/app/onboarding">Revisar perfil</Link>
        </div>
      </section>
    );
  }

  return (
    <section className="stack" aria-labelledby="recommendation-title">
      <div>
        <p className="eyebrow">Plano Basic</p>
        <h1 id="recommendation-title">Sua alocação por classe</h1>
        <p>Uma referência de distribuição para o seu perfil. Não é uma ordem de compra.</p>
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
        <div><span>Perfil</span><strong>v{recommendation.profileVersion}</strong></div>
        <div><span>Modelo</span><strong>{recommendation.modelVersion}</strong></div>
        <div><span>Dados até</span><strong>{recommendation.snapshotCutoff}</strong></div>
      </div>

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
