"""build_portfolios_summary.py
--------------------------------------------------
Pipeline único que
1. Aplica filtros de robustez aos dados fundamentalistas
   • DY negativo -> 0
   • Mantém somente empresas com ≥ 80 % das métricas preenchidas
2. Recalcula os scores via profiles.py
3. Executa o GA (ga.py) para obter as carteiras finais
4. Gera um arquivo JSON "summary_ga.json" com:
   {
     "perfil": {
        "num_assets": int,
        "hhi": float,
        "median_metrics": {...},        # medianas em valores brutos
        "sector_weights": {"SETOR": pct, ...}
     },
     "ibovespa": {"median_metrics": {...}}  # medianas em valores brutos
   }

Métricas comparadas: Dividend Yield (DY), ROE, CAGR Receitas 5 Anos,
Liquidez Corrente, Dívida Líquida / EBIT.
"""
from pathlib import Path
import json
import pandas as pd
import profiles as pf
import ga
from cleaner import to_float  # função de conversão br->float
import os

DATA_DIR     = Path("data/processed")
DATA_DIR_RAW = Path("data/raw")
OUT_DIR      = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

METRIC_COLS = [
    "DY", "P/VP", "EV/EBIT",
    "ROE", "ROIC", "MARGEM EBIT", "MARG. LIQUIDA",
    "CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS", "PEG RATIO",
    "LIQ. CORRENTE", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.",
]

IBOV_LIST = {
    "ABEV3","ALOS3","ASAI3","AURE3","AZUL4","AZZA3","B3SA3","BBAS3","BBDC3","BBDC4",
    "BBSE3","BEEF3","BPAC11","BRAP4","BRAV3","BRFS3","BRKM5","CMIG4","CMIN3","COGN3",
    "CPFE3","CPLE6","CRFB3","CSAN3","CSNA3","CVCB3","CXSE3","CYRE3","DIRR3","EGIE3",
    "ELET3","ELET6","EMBR3","ENEV3","ENGI11","EQTL3","FLRY3","GGBR4","GOAU4","HAPV3",
    "HYPE3","IGTI11","IRBR3","ISAE4","ITSA4","ITUB4","JBSS3","KLBN11","LREN3","MGLU3",
    "MOTV3","MRFG3","MRVE3","MULT3","NTCO3","PCAR3","PETR3","PETR4","PETZ3","POMO4",
    "PRIO3","PSSA3","RADL3","RAIL3","RAIZ4","RDOR3","RECV3","RENT3","SANB11","SBSP3",
    "SLCE3","SMFT3","SMTO3","STBP3","SUZB3","TAEE11","TIMS3","TOTS3","UGPA3","USIM5",
    "VALE3","VAMO3","VBBR3","VIVA3","VIVT3","WEGE3","YDUQ3"
}

MULTI_RUN_SUMMARY = OUT_DIR / "multiple_runs_summary.json"


summary = {}

# -------- função auxiliar para filtrar robustez (z-scores) ---------
def robust_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "DY" in df.columns:
        df["DY"] = df["DY"].clip(lower=0)
    metric_subset = df.drop(columns=["TICKER", "SETOR"], errors="ignore")
    mask_quality = metric_subset.notna().mean(axis=1) >= 0.80
    return df[mask_quality].reset_index(drop=True)


# -------- carregamento do raw para Ibovespa e perfis ---------
df_raw = pd.read_csv(DATA_DIR_RAW / "status_invest_fundamentals.csv")
df_raw["TICKER"] = df_raw["TICKER"].str.upper()
# converte métricas brutas para float
for col in METRIC_COLS:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].apply(to_float)

# -------- universo Ibovespa (valores brutos) ---------
df_raw["IN_IBOV"] = df_raw["TICKER"].isin(IBOV_LIST)
df_ibov = df_raw[df_raw["IN_IBOV"]].copy()
summary["ibovespa"] = {
    "median_metrics": df_ibov[METRIC_COLS].median().to_dict()
}


# -------- processa cada perfil ---------
for perfil in ("conservador", "moderado", "arrojado"):
    # Carrega dados padronizados (z-scores) e filtra robustez
    df = pf.load_profile_data(perfil, DATA_DIR)
    df = robust_filter(df)

    # Ranking e GA
    rank = pf.build_scores(df, perfil)
    cart = ga.run_ga(rank, perfil)

    # Salva carteira final
    outfile = OUT_DIR / f"carteira_{perfil}_ga.json"
    cart.to_json(outfile, orient="records", indent=2, force_ascii=False)

    # Extrai tickers selecionados
    tickers = cart["TICKER"].str.upper().tolist()
    # Seleciona valores brutos do raw
    df_sel_raw = df_raw[df_raw["TICKER"].isin(tickers)].copy()

    # Medianas em valores originais
    raw_medians = {
        col: float(df_sel_raw[col].median()) if col in df_sel_raw else None
        for col in METRIC_COLS
    }
    # Medianas em z-score (direto do cart, que está padronizado)
    zscore_medians = {
        col: float(cart[col].median()) if col in cart else None
        for col in METRIC_COLS
    }


    # Pesos setoriais (mesma lógica de antes)
    sector_weights = (
        cart["SETOR"].value_counts(normalize=True)
            .round(3)
            .to_dict()
    )

    summary[perfil] = {
        "num_assets": len(cart),
        "hhi": round(cart.attrs["hhi"], 3),
        "median_metrics": raw_medians,
        "zscore_metrics": zscore_medians,  # <<<<<<<< NOVO CAMPO
        "sector_weights": sector_weights,
    }


# -------- grava summary atualizado ---------
with open(OUT_DIR / "summary_ga.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Resumo salvo em", OUT_DIR / "summary_ga.json")

if os.path.exists(MULTI_RUN_SUMMARY):
    with open(MULTI_RUN_SUMMARY, "r", encoding="utf-8") as f:
        multi_run_data = json.load(f)

    # Adiciona estatísticas de múltiplas execuções ao summary
    for perfil in ("conservador", "moderado", "arrojado"):
        if perfil in multi_run_data:
            summary[perfil]["multi_run_stats"] = multi_run_data[perfil]["stability_metrics"]
            summary[perfil]["consensus_portfolio"] = multi_run_data[perfil]["consensus_portfolio"]

    print("✓ Estatísticas de múltiplas execuções integradas ao summary")
