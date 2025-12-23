"""analyze_stability.py
Gera visualizações das métricas de estabilidade.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT_DIR = Path("outputs")
SUMMARY_FILE = OUT_DIR / "multiple_runs_summary.json"

with open(SUMMARY_FILE, "r") as f:
    data = json.load(f)

# 1. Boxplot de Fitness por Perfil
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, perfil in enumerate(["conservador", "moderado", "arrojado", "caio"]):
    df = pd.read_csv(OUT_DIR / f"metrics_stability_{perfil}.csv")
    axes[idx].boxplot(df["fitness"])
    axes[idx].set_title(f"{perfil.capitalize()}")
    axes[idx].set_ylabel("Fitness")
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_fitness_boxplot.png", dpi=120)
plt.show()

# 2. Heatmap de Frequência de Ativos
for perfil in ["conservador", "moderado", "arrojado"]:
    results = data[perfil]["all_runs"]
    all_tickers = []
    for r in results:
        all_tickers.extend(r["tickers"])

    from collections import Counter
    ticker_freq = Counter(all_tickers)
    top_20 = dict(sorted(ticker_freq.items(), key=lambda x: x[1], reverse=True)[:20])

    plt.figure(figsize=(10, 6))
    plt.barh(list(top_20.keys()), list(top_20.values()), color='steelblue')
    plt.xlabel("Frequência de Aparição (em 30 runs)")
    plt.title(f"Top 20 Ativos Mais Frequentes - {perfil.capitalize()}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fig_asset_frequency_{perfil}.png", dpi=120)
    plt.close()

# 3. Convergência de Métricas
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, perfil in enumerate(["conservador", "moderado", "arrojado"]):
    df = pd.read_csv(OUT_DIR / f"metrics_stability_{perfil}.csv")

    # Fitness
    axes[0, idx].plot(df["run_id"], df["fitness"], marker='o', linestyle='-', alpha=0.7)
    axes[0, idx].axhline(df["fitness"].mean(), color='red', linestyle='--', label='Média')
    axes[0, idx].fill_between(
        df["run_id"],
        df["fitness"].mean() - df["fitness"].std(),
        df["fitness"].mean() + df["fitness"].std(),
        alpha=0.2, color='red'
    )
    axes[0, idx].set_title(f"Fitness - {perfil.capitalize()}")
    axes[0, idx].set_xlabel("Run ID")
    axes[0, idx].legend()
    axes[0, idx].grid(alpha=0.3)

    # HHI
    axes[1, idx].plot(df["run_id"], df["hhi"], marker='s', linestyle='-', alpha=0.7, color='green')
    axes[1, idx].axhline(df["hhi"].mean(), color='orange', linestyle='--', label='Média')
    axes[1, idx].set_title(f"HHI - {perfil.capitalize()}")
    axes[1, idx].set_xlabel("Run ID")
    axes[1, idx].legend()
    axes[1, idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_convergence_analysis.png", dpi=120)
plt.show()

print("✅ Visualizações salvas em", OUT_DIR)