"""core/scoring.py
=============================================================================
Módulo de scoring fundamentalista por perfil de investidor.

Calcula pontuações ponderadas baseadas em grupos de métricas contábeis,
com pesos específicos para cada perfil (conservador, moderado, arrojado).
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, List

from config import METRIC_GROUPS, PROFILE_WEIGHTS


def add_dynamic_columns(df: pd.DataFrame) -> None:
    """
    Adiciona colunas dinâmicas aos grupos se existirem no DataFrame.

    Por exemplo, adiciona "EV/EBITDA" ao grupo "value" se a coluna existir.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com métricas.
    """
    if "EV/EBITDA" in df.columns and "EV/EBITDA" not in METRIC_GROUPS["value"]:
        METRIC_GROUPS["value"].append("EV/EBITDA")


def calculate_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a média de cada grupo de métricas.

    Cria colunas AVG_LIQUIDEZ, AVG_RENT, AVG_VALUE, AVG_GROWTH, AVG_DIV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com métricas padronizadas (z-scores).

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas de médias por grupo.
    """
    df = df.copy()
    add_dynamic_columns(df)

    for group, cols in METRIC_GROUPS.items():
        # Seleciona apenas colunas que existem no DataFrame
        existing = [c for c in cols if c in df.columns]

        if not existing:
            df[f"AVG_{group.upper()}"] = float("nan")
        else:
            # Média ignorando NaNs
            df[f"AVG_{group.upper()}"] = df[existing].mean(axis=1, skipna=True)

    return df


def calculate_weighted_score(
    df: pd.DataFrame,
    profile: str
) -> pd.DataFrame:
    """
    Calcula score final ponderado para um perfil específico.

    Score = Σ (peso_grupo × média_grupo)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com médias de grupos já calculadas.
    profile : str
        Perfil do investidor (conservador, moderado, arrojado, caio).

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna SCORE adicionada.
    """
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(
            f"Perfil desconhecido: {profile}. "
            f"Perfis disponíveis: {list(PROFILE_WEIGHTS.keys())}"
        )

    weights = PROFILE_WEIGHTS[profile]
    df = df.copy()

    # Score final = soma ponderada das médias de grupo
    df["SCORE"] = sum(
        weights[group] * df[f"AVG_{group.upper()}"]
        for group in weights
    )

    return df


def build_scores(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """
    Pipeline completo de cálculo de scores.

    1. Calcula médias por grupo
    2. Aplica pesos do perfil
    3. Ordena por score (maior para menor)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados fundamentalistas padronizados.
    profile : str
        Perfil do investidor.

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado por SCORE (descendente).

    Examples
    --------
    >>> df_clean = load_processed_data("conservador")
    >>> df_ranked = build_scores(df_clean, "conservador")
    >>> print(df_ranked[["TICKER", "SCORE"]].head())
    """
    df = calculate_group_scores(df)
    df = calculate_weighted_score(df, profile)

    # Ordena do maior para o menor score
    return df.sort_values("SCORE", ascending=False).reset_index(drop=True)


def get_top_stocks(
    df_ranked: pd.DataFrame,
    n: int = 20,
    columns: List[str] = None
) -> pd.DataFrame:
    """
    Retorna os N ativos com maior score.

    Parameters
    ----------
    df_ranked : pd.DataFrame
        DataFrame já ordenado por score.
    n : int
        Número de ativos para retornar.
    columns : List[str], optional
        Colunas para exibir. Se None, retorna todas.

    Returns
    -------
    pd.DataFrame
        Top N ativos.
    """
    if columns:
        return df_ranked[columns].head(n)
    return df_ranked.head(n)


def compare_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula scores para todos os perfis e retorna comparação.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados padronizados.

    Returns
    -------
    pd.DataFrame
        DataFrame com scores de todos os perfis.
    """
    results = []

    for profile in PROFILE_WEIGHTS.keys():
        df_scored = build_scores(df.copy(), profile)
        df_scored = df_scored[["TICKER", "SETOR", "SCORE"]].copy()
        df_scored.rename(columns={"SCORE": f"SCORE_{profile.upper()}"}, inplace=True)
        results.append(df_scored)

    # Merge todos os resultados
    comparison = results[0]
    for df_result in results[1:]:
        comparison = comparison.merge(
            df_result,
            on=["TICKER", "SETOR"],
            how="outer"
        )

    return comparison
