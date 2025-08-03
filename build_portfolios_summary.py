"""build_portfolios_summary.py
--------------------------------------------------
Pipeline único que
1. Aplica filtros de robustez aos dados fundamentalistas
   • DY negativo -> 0
   • Mantém somente empresas com ≥ 80 % das métricas preenchidas
2. Recalcula os scores via profiles.py
3. Executa o GA (ga.py) para obter as carteiras finais
4. Gera um arquivo JSON "summary_ga.json" com:
   {
     "perfil": {
        "num_assets": int,
        "hhi": float,
        "median_metrics": {...},
        "sector_weights": {"SETOR": pct, ...}
     },
     "universe": {"median_metrics": {...}}
   }

Métricas comparadas: Dividend Yield (DY), ROE, CAGR Receitas 5 Anos,
Liquidez Corrente, Dívida Líquida / EBIT.
"""
from pathlib import Path
import json
import pandas as pd
import profiles as pf
import ga

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

METRIC_COLS = [
    "DY",
    "ROE",
    "CAGR RECEITAS 5 ANOS",
    "LIQ. CORRENTE",
    "DIVIDA LIQUIDA / EBIT",
]

summary = {}

# -------- função auxiliar para filtrar robustez ---------

def robust_filter(df: pd.DataFrame) -> pd.DataFrame:
    # DY negativo vira zero
    if "DY" in df.columns:
        df["DY"] = df["DY"].clip(lower=0)

    # Retira linhas com >20 % NaN (considera apenas colunas métricas)
    metric_subset = df.drop(columns=["TICKER", "SETOR"], errors="ignore")
    mask_quality = metric_subset.notna().mean(axis=1) >= 0.80
    return df[mask_quality].reset_index(drop=True)

# -------- universo completo (para comparação) ---------
# usa o arquivo moderado como proxy; ele já tem filtros de cap/liq aplicados
df_universe = pd.read_csv(DATA_DIR / "fundamentals_clean_moderado.csv")
df_universe = robust_filter(df_universe)
summary["universe"] = {
    "median_metrics": df_universe[METRIC_COLS].median().to_dict()
}

# -------- processa cada perfil ---------
for perfil in ("conservador", "moderado", "arrojado"):
    df = pf.load_profile_data(perfil, DATA_DIR)
    df = robust_filter(df)

    rank = pf.build_scores(df, perfil)
    cart = ga.run_ga(rank, perfil)

    # salva carteira final
    outfile = OUT_DIR / f"carteira_{perfil}_ga.json"
    cart.to_json(outfile, orient="records", indent=2, force_ascii=False)

    # resumo quantitativo
    medians = cart[METRIC_COLS].median().to_dict()
    sector_weights = (
        cart["SETOR"].value_counts(normalize=True)  # porcentagens
            .round(3)
            .to_dict()
    )
    summary[perfil] = {
        "num_assets": len(cart),
        "hhi": round(cart.attrs["hhi"], 3),
        "median_metrics": medians,
        "sector_weights": sector_weights,
    }

# -------- grava summary ---------
with open(OUT_DIR / "summary_ga.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Resumo salvo em", OUT_DIR / "summary_ga.json")
