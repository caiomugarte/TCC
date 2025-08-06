#!/usr/bin/env python
"""report.py
Visualiza comparativos das carteiras GA versus o universo filtrado + Ibovespa.

▪ Lê `outputs/summary_ga.json` (gerado pelo build_portfolios_summary.py).
▪ Cria um gráfico de barras comparativo das métricas selecionadas (z‑scores).
   – Cada métrica tem cor conforme seu grupo (Dividendos/Value, Qualidade, Crescimento, Liquidez),
     **e** aparece individualmente na legenda.
▪ Desenha gráficos horizontais da distribuição setorial para cada perfil.
▪ Salva as figuras em `outputs/fig_*.png` e exibe‑as.

Requisitos: pandas, matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ------------------------ Config --------------------------------------- #
SUMMARY_PATH = Path("outputs/summary_ga.json")  # ajuste se necessário
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Métricas e cores agrupadas por tipo ---------------------------------- #
COLOR_MAP = {
    # Dividendos / Value (amarelos)
    "DY": "#ffc107",
    "P/VP": "#ffca28",
    "EV/EBIT": "#ffd54f",

    # Qualidade / Rentabilidade (verdes)
    "ROE": "#388e3c",
    "ROIC": "#4caf50",
    "MARGEM EBIT": "#66bb6a",
    "MARG. LIQUIDA": "#81c784",

    # Crescimento (azuis)
    "CAGR RECEITAS 5 ANOS": "#2196f3",
    "CAGR LUCROS 5 ANOS": "#42a5f5",
    "PEG RATIO": "#64b5f6",

    # Liquidez / Alavancagem (vermelhos‑rosados)
    "LIQ. CORRENTE": "#e57373",
    "DIVIDA LIQUIDA / EBIT": "#ef5350",
    "DIV. LIQ. / PATRI.": "#f06292",
}

METRICS = list(COLOR_MAP.keys())
PROFILES = ["ibovespa", "conservador", "moderado", "arrojado"]

# ------------------------ Helpers -------------------------------------- #

def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Resumo não encontrado em {path!s}. Rode build_portfolios_summary.py antes.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_metric_df(summary: dict) -> pd.DataFrame:
    """DataFrame (rows = métricas, cols = perfis) com medianas z‑score."""
    data = {
        p: [summary[p]["median_metrics"].get(m, float("nan")) for m in METRICS]
        for p in PROFILES
    }
    return pd.DataFrame(data, index=METRICS)

def build_zscore_df(summary: dict) -> pd.DataFrame:
    """DataFrame (rows = métricas, cols = perfis) com medianas z-score."""
    data = {
        p: [summary[p].get("zscore_metrics", {}).get(m, float("nan")) for m in METRICS]
        for p in PROFILES if p != "ibovespa"  # Ibovespa não tem zscore_metrics
    }
    return pd.DataFrame(data, index=METRICS)



def save_show(fig: plt.Figure, name: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_{name}.png", dpi=120)
    plt.show()

# ------------------------ Main ----------------------------------------- #

def main():
    summary = load_summary(SUMMARY_PATH)
    df_metrics = build_metric_df(summary)

    # ---------- Gráfico de barras das métricas ------------------------ #
    colors = [COLOR_MAP[m] for m in METRICS]
    fig = plt.figure(figsize=(14, 7))
    ax = plt.gca()
    df_metrics.T.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
    ax.set_ylabel("z‑score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title("Métricas fundamentais — Carteiras GA vs Ibovespa")

    # Legenda individual por métrica (13 entradas)
    legend_handles = [mpatches.Patch(color=COLOR_MAP[m], label=m) for m in METRICS]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="Indicadores (cor = grupo)")
    save_show(fig, "metrics_comparison")

    # ---------- Gráfico de barras das métricas (z-score) ------------------------ #
    zscore_df = build_zscore_df(summary)
    colors = [COLOR_MAP[m] for m in METRICS]
    fig_z = plt.figure(figsize=(14, 7))
    ax_z = plt.gca()
    zscore_df.T.plot(kind="bar", color=colors, edgecolor="black", ax=ax_z)
    ax_z.set_ylabel("z-score")
    ax_z.set_xticklabels(ax_z.get_xticklabels(), rotation=0)
    ax_z.set_title("Métricas fundamentais — Z-Score das Carteiras GA")

    # Legenda individual por métrica (13 entradas)
    legend_handles = [mpatches.Patch(color=COLOR_MAP[m], label=m) for m in METRICS]
    ax_z.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                title="Indicadores (z-score)")
    save_show(fig_z, "metrics_comparison_zscore")


    # ---------- Gráficos setoriais ----------------------------------- #
    for perfil in ("conservador", "moderado", "arrojado"):
        sector_w = summary[perfil]["sector_weights"]
        sectors = list(sector_w.keys())
        weights = [w * 100 for w in sector_w.values()]

        fig2 = plt.figure(figsize=(8, 5))
        plt.barh(sectors, weights, color="#42a5f5", edgecolor="black")
        plt.xlabel("Peso (%)")
        plt.title(f"Distribuição setorial — {perfil.capitalize()}")
        plt.gca().invert_yaxis()
        save_show(fig2, f"sector_{perfil}")

    print("➜ Gráficos prontos em", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
