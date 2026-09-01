"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiClientError, getReview } from "@/lib/api-client";
import type { Review } from "@/lib/api-types";

const percentage = new Intl.NumberFormat("pt-BR", {
  style: "percent",
  maximumFractionDigits: 1,
});
const currency = new Intl.NumberFormat("pt-BR", {
  style: "currency",
  currency: "BRL",
});
const labels: Record<string, string> = {
  brazilian_stocks: "Ações brasileiras",
  fiis: "FIIs",
  international: "Exposição internacional",
  fixed_income: "Renda fixa",
  crypto: "Criptoativos",
};

export function ReviewView() {
  const { getToken } = useAuth();
  const [review, setReview] = useState<Review | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadReview() {
    setLoading(true);
    setError(null);
    try {
      setReview(await getReview(await getToken()));
    } catch (cause) {
      setError(cause instanceof ApiClientError ? cause.message : "Não foi possível calcular a revisão.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadReview(); }, [getToken]);

  if (loading) return <section className="card" aria-live="polite"><p>Calculando sua revisão…</p></section>;

  if (error || !review) {
    return (
      <section className="card" aria-live="assertive">
        <p className="eyebrow">Revisão mensal</p>
        <h1>Revisão indisponível</h1>
        <p>{error ?? "Informe sua carteira e gere uma recomendação para continuar."}</p>
        <div className="actions">
          <button className="button" type="button" onClick={() => void loadReview()}>Tentar novamente</button>
          <Link className="button secondary" href="/app/portfolio">Informar carteira</Link>
        </div>
      </section>
    );
  }

  return (
    <section className="stack" aria-labelledby="review-title">
      <div>
        <p className="eyebrow">Revisão mensal</p>
        <h1 id="review-title">Atual versus alvo</h1>
        <p>Dentro de uma faixa de {percentage.format(review.driftBand)}, a orientação é manter. Contribuições vêm antes de qualquer revisão de venda.</p>
      </div>
      <div className="review-list">
        {review.items.map((item) => (
          <article className="card review-row" key={item.classKey}>
            <div>
              <h2>{labels[item.classKey] ?? item.classKey}</h2>
              <p className={`status status-${item.status}`}>{item.status === "within_range" ? "Dentro da faixa" : item.status === "underweight" ? "Abaixo do alvo" : "Acima do alvo"}</p>
            </div>
            <dl className="review-metrics">
              <div><dt>Atual</dt><dd>{percentage.format(item.currentWeight)}</dd></div>
              <div><dt>Alvo</dt><dd>{percentage.format(item.targetWeight)}</dd></div>
              <div><dt>Desvio</dt><dd>{percentage.format(item.drift)}</dd></div>
              <div><dt>Gap</dt><dd>{currency.format(item.valueGapBrl)}</dd></div>
            </dl>
            <p className="review-action">{item.suggestedAction === "contribute" ? "Priorizar novos aportes" : item.suggestedAction === "review_sale" ? "Revisar eventual redução" : "Manter e acompanhar"}</p>
          </article>
        ))}
      </div>
      <div className="notice">Esta tela não cria ordens nem executa operações. Ela organiza o que merece sua revisão.</div>
      <div className="actions"><Link className="button" href="/app/portfolio">Atualizar carteira</Link><Link className="button secondary" href="/app/recommendation">Ver alocação</Link></div>
    </section>
  );
}
