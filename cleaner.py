import pandas as pd
import numpy as np

# Arquivos brutos organizados por setor
sector_files = {
    "Consumo Cíclico": "data/consumo_ciclico_raw.csv",
    "Consumo não Cíclico": "data/consumo_nao_ciclico_raw.csv",
    "Utilidade Pública": "data/utilidade_publica_raw.csv",
    "Bens Industriais": "data/bens_industriais_raw.csv",
    "Materiais Básicos": "data/materiais_basicos_raw.csv",
    "Financeiro e Outros": "data/financeiros_outros_raw.csv",
    "Tecnologia da Informação": "data/tecnologia_informacao_raw.csv",
    "Saúde": "data/sauda_raw.csv",
    "Petróleo, Gás e Biocombustíveis": "data/petroleo_gas_biocombustivel_raw.csv",
    "Comunicações": "data/comunicacoes_raw.csv"
}

def to_float(x):
    """Converte valores no formato brasileiro para float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    x = str(x).strip()
    if x in ['', '-', '--']:
        return np.nan
    x = x.replace('.', '').replace(',', '.')
    try:
        return float(x)
    except ValueError:
        return np.nan

def clean_sector_file(file_path, sector_name):
    df = pd.read_csv(file_path, sep=';', engine='python')
    df['SETOR'] = sector_name
    for col in df.columns:
        if col.strip().upper() not in ['TICKER', 'SETOR']:
            df[col] = df[col].apply(to_float)
    return df

def unify_tickers(df):
    """Agrupa tickers da mesma empresa (SANB3/SANB4 → SANB) e tira a média dos indicadores."""
    df['EMPRESA'] = df['TICKER'].str.extract(r'([A-Z]+)', expand=False)
    cols_to_avg = [c for c in df.columns if c not in ['TICKER', 'SETOR', 'EMPRESA']]
    df_grouped = df.groupby(['EMPRESA', 'SETOR'])[cols_to_avg].mean().reset_index()
    df_grouped.rename(columns={'EMPRESA': 'TICKER'}, inplace=True)
    return df_grouped

def remove_outliers(df):
    """Remove empresas com métricas absurdas que distorcem as médias."""
    for col in ["ROE", "ROIC", "MARGEM BRUTA", "MARGEM EBIT", "MARG. LIQUIDA"]:
        if col in df.columns:
            mask = (
                    df["P/L"].isna() |
                    ((df["P/L"] > -100) & (df["P/L"] < 100))
            )
            df = df[mask]
    if "P/L" in df.columns:
        mask = (
                df["P/L"].isna() |
                ((df["P/L"] > -20) & (df["P/L"] < 100))
        )
        df = df[mask]
    return df

def build_clean_dataset(output_path="status_invest_fundamentals.csv"):
    all_dfs = []
    for sector, file_path in sector_files.items():
        df_sector = clean_sector_file(file_path, sector)
        all_dfs.append(df_sector)

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.columns = [col.strip().upper() for col in df_final.columns]
    df_final.dropna(how='all', inplace=True)

    # Consolida por empresa e remove outliers
    #df_final = unify_tickers(df_final)
    df_final = remove_outliers(df_final)

    df_final.to_csv(output_path, index=False)
    print(f"Dataset consolidado salvo em: {output_path}")
    return df_final

if __name__ == "__main__":
    build_clean_dataset()
