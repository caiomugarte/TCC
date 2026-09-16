"""core/preprocessing.py
=============================================================================
Módulo de pré-processamento de dados fundamentalistas.

Funções principais:
- Carregamento de dados brutos
- Aplicação de filtros de elegibilidade
- Winsorização de outliers setoriais
- Inversão de métricas
- Normalização via z-score intra-setor
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Mapping, Optional

from config import (
    RAW_DATA_FILE,
    DATA_PROCESSED,
    FILTERS,
    INVERT_METRICS,
    PRICE_COLS,
    FILTER_COLS,
    IBOV_TICKERS
)


def load_raw_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Carrega dados brutos do CSV.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame com dados brutos.
    """
    # Tenta detectar separador automaticamente
    df = pd.read_csv(file_path, sep=None, engine="python")

    if df.shape[1] == 1:
        df = pd.read_csv(file_path, sep=";")

    # Padroniza nome da coluna ticker
    ticker_col = next(
        (c for c in df.columns if c.upper().startswith("TICK")),
        None
    )

    if ticker_col:
        df.rename(columns={ticker_col: "TICKER"}, inplace=True)

    # Adiciona flag Ibovespa
    df["IN_IBOV"] = df["TICKER"].str.upper().isin(IBOV_TICKERS)

    return df


def apply_eligibility_filters(
    df: pd.DataFrame,
    cap_min: float,
    liq_min: float,
) -> pd.DataFrame:
    """
    Aplica filtros de elegibilidade (valor de mercado e liquidez).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados brutos.
    cap_min : float
        Valor de mercado mínimo.
    liq_min : float
        Liquidez média diária mínima.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    numeric = df.copy()
    for column in ("VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"):
        if column not in numeric.columns:
            raise ValueError(f"coluna de filtro ausente: {column}")
        if numeric[column].dtype == "object":
            numeric[column] = pd.to_numeric(
                numeric[column].astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            )
    condition = (
        (numeric["VALOR DE MERCADO"] >= cap_min)
        & (numeric["LIQUIDEZ MEDIA DIARIA"] >= liq_min)
    )
    result = df.loc[condition].copy()
    ticker_values = df.get("TICKER", pd.Series(df.index, index=df.index)).astype(str)
    reasons: Dict[str, List[str]] = {}
    for index in df.index[~condition]:
        ticker = ticker_values.loc[index].strip().upper()
        reason = []
        if pd.isna(numeric.loc[index, "VALOR DE MERCADO"]) or numeric.loc[index, "VALOR DE MERCADO"] < cap_min:
            reason.append("market_cap_below_minimum")
        if pd.isna(numeric.loc[index, "LIQUIDEZ MEDIA DIARIA"]) or numeric.loc[index, "LIQUIDEZ MEDIA DIARIA"] < liq_min:
            reason.append("liquidity_below_minimum")
        reasons[ticker] = reason or ["eligibility_filter"]
    result.attrs["exclusion_reasons"] = reasons
    result.attrs["eligibility_filters"] = {
        "cap_min": float(cap_min),
        "liq_min": float(liq_min),
    }
    return result


def _normalise_filter_config(
    filters: Mapping[str, object],
) -> Dict[str, float]:
    """Normalize the explicit stock liquidity/size filter seam."""

    if not isinstance(filters, Mapping):
        raise ValueError("stock filters must be an object")
    aliases = {
        "cap_min": ("cap_min", "market_cap_min", "min_market_cap", "market_cap_floor"),
        "liq_min": ("liq_min", "liquidity_min", "min_liquidity", "liquidity_floor"),
    }
    result = {}
    for target, names in aliases.items():
        raw = next((filters[name] for name in names if name in filters), None)
        if raw is None:
            raise ValueError(f"stock filters missing {target}")
        value = float(raw)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"stock filter {target} must be finite and non-negative")
        result[target] = value
    return result


def winsorize_sector(
    df: pd.DataFrame,
    cols: List[str],
    percentile: float = 0.01
) -> pd.DataFrame:
    """
    Winsoriza outliers por setor.

    Limita valores extremos aos percentis definidos, por setor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados.
    cols : List[str]
        Colunas para winsorizar.
    percentile : float
        Percentil para corte (default: 1%).

    Returns
    -------
    pd.DataFrame
        DataFrame com valores winzorizados.
    """
    result = df.copy()
    result[cols] = df.groupby("SETOR")[cols].transform(
        lambda x: x.clip(x.quantile(percentile), x.quantile(1 - percentile))
    )
    return result


def zscore_sector(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normaliza métricas via z-score intra-setor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados.
    cols : List[str]
        Colunas para normalizar.

    Returns
    -------
    pd.DataFrame
        DataFrame com valores normalizados.
    """
    result = df.copy()
    result[cols] = df.groupby("SETOR")[cols].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    return result


def preprocess_profile(
    df_raw: pd.DataFrame,
    profile: Optional[str] = None,
    filters: Optional[Mapping[str, object]] = None,
    config: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """
    Pipeline completo de pré-processamento para um perfil.

    Steps:
    1. Aplica filtros de elegibilidade
    2. Converte colunas string para numeric
    3. Winsoriza outliers setoriais
    4. Inverte métricas (quanto menor, melhor)
    5. Normaliza via z-score intra-setor
    6. Remove colunas proibidas (preço, etc)

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame com dados brutos.
    profile : str, optional
        Named profile retained for Basic/offline callers.
    filters : Mapping[str, object], optional
        Explicit ``cap_min``/``liq_min`` stock eligibility filters.
    config : Mapping[str, object], optional
        Explicit selector config; reads ``liquidity_and_size_filters``.

    Returns
    -------
    pd.DataFrame
        DataFrame pré-processado.
    """
    if filters is not None and config is not None:
        raise ValueError("pass only one of filters or config")
    if config is not None:
        filters = config.get("liquidity_and_size_filters") or config.get("filters")
        if filters is None:
            raise ValueError("stock config must contain liquidity_and_size_filters")
    if filters is None and profile not in FILTERS:
        raise ValueError(f"Perfil desconhecido: {profile}")

    # 1. Aplica filtros
    limits = _normalise_filter_config(
        filters if filters is not None else FILTERS[profile]
    )
    df = apply_eligibility_filters(
        df_raw,
        cap_min=limits["cap_min"],
        liq_min=limits["liq_min"]
    )

    if df.empty:
        print(
            f"Nenhum ativo passou pelos filtros do perfil '{profile or 'explicit'}' "
            f"(cap >= {limits['cap_min']:,}, liq >= {limits['liq_min']:,})"
        )
        empty = df.iloc[0:0].copy()
        empty.attrs.update(df.attrs)
        return empty

    # 2. Identifica colunas candidatas a métricas
    candidates = df.columns.difference(
        ["TICKER", "SETOR"] + PRICE_COLS + FILTER_COLS
    )

    # 3. Converte colunas string para numeric
    for col in candidates:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)  # Remove separador de milhar
                .str.replace(",", ".", regex=False)  # Troca vírgula por ponto
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Seleciona apenas colunas numéricas
    metric_cols = [c for c in candidates if df[c].dtype.kind in "fi"]

    if not metric_cols:
        raise RuntimeError("Nenhuma métrica numérica encontrada após conversão.")

    # 5. Winsoriza outliers
    df = winsorize_sector(df, metric_cols)

    # 6. Inverte métricas "quanto menor, melhor"
    invert = [c for c in INVERT_METRICS if c in df.columns]
    df[invert] = df[invert] * -1

    # 7. Normaliza via z-score intra-setor
    df = zscore_sector(df, metric_cols)

    # 8. Remove colunas proibidas
    df = df.drop(columns=PRICE_COLS + FILTER_COLS, errors="ignore")

    df.attrs["eligibility_filters"] = limits
    return df


def preprocess_explicit(
    df_raw: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Explicit-config spelling used by Premium stock adapters."""

    return preprocess_profile(df_raw, config=config)


def preprocess_all_profiles(save: bool = True) -> dict:
    """
    Pré-processa dados para todos os perfis.

    Parameters
    ----------
    save : bool
        Se True, salva CSVs e JSONs processados.

    Returns
    -------
    dict
        Dicionário {profile: DataFrame}.
    """
    df_raw = load_raw_data()
    results = {}

    for profile in FILTERS.keys():
        print(f"Pré-processando perfil: {profile}")
        df_clean = preprocess_profile(df_raw, profile)

        if save and not df_clean.empty:
            outfile = DATA_PROCESSED / f"fundamentals_clean_{profile}"
            df_clean.to_csv(outfile.with_suffix(".csv"), index=False)
            df_clean.to_json(
                outfile.with_suffix(".json"),
                orient="records",
                force_ascii=False
            )
            print(f"  ✓ Salvo em {outfile}")

        results[profile] = df_clean

    # Salva filtros brutos para auditoria
    if save:
        df_raw[["TICKER", "VALOR DE MERCADO", "LIQUIDEZ MEDIA DIARIA"]].to_parquet(
            DATA_PROCESSED / "eligibility_filters.parquet",
            index=False
        )

    return results


def load_processed_data(profile: str) -> pd.DataFrame:
    """
    Carrega dados já processados de um perfil.

    Parameters
    ----------
    profile : str
        Perfil do investidor.

    Returns
    -------
    pd.DataFrame
        DataFrame processado.
    """
    file_path = DATA_PROCESSED / f"fundamentals_clean_{profile}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo processado não encontrado: {file_path}. "
            f"Execute o pré-processamento primeiro."
        )

    return pd.read_csv(file_path)


def apply_robustness_filter(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Filtra ativos com pelo menos `threshold`% das métricas preenchidas.

    Também força DY negativo para 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados processados.
    threshold : float
        Percentual mínimo de métricas preenchidas (0-1).

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    df = df.copy()

    # DY negativo -> 0
    if "DY" in df.columns:
        df["DY"] = df["DY"].clip(lower=0)

    # Filtra ativos com ≥ threshold% das métricas
    metric_subset = df.drop(columns=["TICKER", "SETOR"], errors="ignore")
    quality_mask = metric_subset.notna().mean(axis=1) >= threshold

    result = df[quality_mask].reset_index(drop=True)
    reasons = {
        key: list(value)
        for key, value in df.attrs.get("exclusion_reasons", {}).items()
    }
    if "TICKER" in df.columns:
        for index in df.index[~quality_mask]:
            ticker = str(df.loc[index, "TICKER"]).strip().upper()
            reasons.setdefault(ticker, []).append("insufficient_metric_coverage")
    result.attrs["exclusion_reasons"] = reasons
    result.attrs["eligibility_filters"] = dict(
        df.attrs.get("eligibility_filters", {})
    )
    return result
