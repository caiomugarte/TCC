"use client";

import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";

import { ApiClientError, getPortfolio, savePortfolio } from "@/lib/api-client";
import type { AssetClassKey } from "@/lib/api-types";

const CLASSES: Array<{ key: AssetClassKey; label: string }> = [
  { key: "brazilian_stocks", label: "Ações brasileiras" },
  { key: "fiis", label: "FIIs" },
  { key: "international", label: "Exposição internacional" },
  { key: "fixed_income", label: "Renda fixa" },
  { key: "crypto", label: "Criptoativos" },
];

type FormValues = Record<AssetClassKey, string>;

const EMPTY_VALUES: FormValues = {
  brazilian_stocks: "",
  fiis: "",
  international: "",
  fixed_income: "",
  crypto: "",
};

export function PortfolioForm() {
  const { getToken } = useAuth();
  const [values, setValues] = useState<FormValues>(EMPTY_VALUES);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function loadPortfolio() {
      try {
        const snapshot = await getPortfolio(await getToken());
        if (!active || !snapshot) return;
        setValues(Object.fromEntries(
          CLASSES.map(({ key }) => [key, String(snapshot.classes[key] ?? 0)]),
        ) as FormValues);
      } catch (cause) {
        if (active) setError(cause instanceof ApiClientError ? cause.message : "Não foi possível carregar a carteira.");
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadPortfolio();
    return () => { active = false; };
  }, [getToken]);

  function updateValue(key: AssetClassKey, value: string) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    const parsed = Object.fromEntries(
      CLASSES.map(({ key }) => [key, Number(values[key].replace(",", "."))]),
    ) as Record<AssetClassKey, number>;
    if (Object.values(parsed).some((value) => !Number.isFinite(value) || value < 0)) {
      setError("Informe valores válidos e não negativos para todas as classes.");
      return;
    }
    if (Object.values(parsed).every((value) => value === 0)) {
      setError("A carteira precisa ter valor maior que zero.");
      return;
    }

    setSaving(true);
    try {
      await savePortfolio({ currency: "BRL", classes: parsed }, await getToken());
      window.location.assign("/app/review");
    } catch (cause) {
      setError(cause instanceof ApiClientError ? cause.message : "Não foi possível salvar sua carteira.");
      setSaving(false);
    }
  }

  if (loading) return <section className="card" aria-live="polite"><p>Carregando sua carteira…</p></section>;

  return (
    <section className="stack" aria-labelledby="portfolio-title">
      <div>
        <p className="eyebrow">Carteira principal</p>
        <h1 id="portfolio-title">Onde seu dinheiro está hoje?</h1>
        <p>Informe o valor aproximado em cada classe. O Prumo usa isso apenas para calcular o desvio do alvo.</p>
      </div>
      <form className="card form-stack" onSubmit={submit}>
        <fieldset>
          <legend>Valores atuais em reais</legend>
          {CLASSES.map(({ key, label }) => (
            <label className="money-field" htmlFor={`portfolio-${key}`} key={key}>
              <span>{label}</span>
              <span className="money-input"><span aria-hidden="true">R$</span><input id={`portfolio-${key}`} inputMode="decimal" min="0" name={key} onChange={(event) => updateValue(key, event.target.value)} step="0.01" type="number" value={values[key]} /></span>
            </label>
          ))}
        </fieldset>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="actions">
          <button className="button" disabled={saving} type="submit">{saving ? "Salvando…" : "Salvar carteira"}</button>
          <Link className="button secondary" href="/app/recommendation">Voltar à alocação</Link>
        </div>
      </form>
    </section>
  );
}
