"""
Pré-processamento dos indicadores fundamentalistas B3
-----------------------------------------------------
1. Carrega CSVs brutos (Status Invest) já consolidados pelo seu cleaner.
2. Aplica filtros de elegibilidade (cap & liquidez) por *perfil*.
3. Winsoriza outliers setoriais (1 %).
4. Inverte métricas “quanto menor, melhor”.
5. Normaliza via z-score intra-setor.
6. Salva:
   • fundamentals_clean.parquet  (inputs p/ modelagem)
   • eligibility_filters.parquet (ticker, cap, liq) – auditoria
"""

from pathlib import Path
import pandas as pd

# ---------- Config ----------------------------------------------------------- #
RAW_PATH      = Path("data/raw/status_invest_fundamentals.csv")
OUT_PATH      = Path("data/processed")
OUT_PATH.mkdir(parents=True, exist_ok=True)

FILTERS = {
    "conservador": {"cap_min": 5_000_000_000, "liq_min": 2_000_000},
    "moderado":    {"cap_min": 1_000_000_000, "liq_min":   500_000},
    "arrojado":    {"cap_min":   200_000_000, "liq_min":    50_000},
}

INVERT_METRICS = [
    "P/L", "P/VP", "P/ATIVOS", "PSR", "DIVIDA LIQUIDA / EBIT",
]
PRICE_COLS = ["PRECO"]                     # eliminados sempre
FILTER_COLS = ["VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"]  # usados só no filtro
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
# ---------------------------------------------------------------------------- #

def load_raw(path: Path) -> pd.DataFrame:
    # tenta ; ou ,
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")

    # padroniza nome
    ticker_col = next((c for c in df.columns if c.upper().startswith("TICK")), None)
    df.rename(columns={ticker_col: "TICKER"}, inplace=True)

    # ► REMOVA OU COMENTE a linha abaixo
    # df = df[df["TICKER"].str.match(r"^[A-Z]{4}\d{1,2}[A-Z]?$", na=False)]

    return df



def apply_filters(df, cap_min, liq_min) -> pd.DataFrame:
    cond = (df["VALOR DE MERCADO"] >= cap_min) & (df["LIQUIDEZ MEDIA DIARIA"] >= liq_min)
    return df.loc[cond].copy()

def winsorize_sector(df, cols, p=0.01):
    return df.groupby("SETOR")[cols].transform(
        lambda x: x.clip(x.quantile(p), x.quantile(1 - p))
    )

def zscore_sector(df, cols):
    return df.groupby("SETOR")[cols].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

def preprocess(df, cap_min, liq_min):
    df = apply_filters(df, cap_min, liq_min)
    if df.empty:
        print(f"--> Nenhum ativo passou pelos filtros cap={cap_min:,} liq={liq_min:,}")
        return pd.DataFrame()
        # lista de candidatos (tudo que não é id, setor, preço ou filtros)
    candidates = df.columns.difference(
        ["TICKER", "SETOR"] + PRICE_COLS + FILTER_COLS
    )

    # 1) converte apenas colunas do tipo object (string)
    for col in candidates:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)                       # garante string
                .str.replace(".", "", regex=False)  # milhar
                .str.replace(",", ".", regex=False) # decimal
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) agora, selecione as realmente numéricas
    metric_cols = [c for c in candidates if df[c].dtype.kind in "fi"]
    if not metric_cols:
        raise RuntimeError("Nenhuma métrica numérica encontrada.")

    # winsorização, inversão, z-score (mesmo código de antes)
    df[metric_cols] = winsorize_sector(df, metric_cols)
    invert = [c for c in INVERT_METRICS if c in df.columns]
    df[invert] = df[invert] * -1
    df[metric_cols] = zscore_sector(df, metric_cols)

    # remove colunas proibidas
    return df.drop(columns=PRICE_COLS + FILTER_COLS, errors="ignore")



def main():
    raw = load_raw(RAW_PATH)
    raw["IN_IBOV"] = raw["TICKER"].str.upper().isin(IBOV_LIST)

    for perfil, limits in FILTERS.items():
        clean = preprocess(raw.copy(), **limits)
        outfile = OUT_PATH / f"fundamentals_clean_{perfil}"
        clean.to_csv(outfile.with_suffix(".csv"), index=False)
        clean.to_json(outfile.with_suffix(".json"), orient="records", force_ascii=False)

    # salva filtros brutos para auditoria
    raw[["TICKER", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"]] \
        .to_parquet(OUT_PATH / "eligibility_filters.parquet", index=False)

if __name__ == "__main__":
    main()
