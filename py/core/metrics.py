"""core/metrics.py
=============================================================================
Funções de métricas compartilhadas entre diferentes módulos.

Inclui:
- Herfindahl-Hirschman Index (HHI) para concentração setorial
- Índice de Jaccard para similaridade de carteiras
- Outras métricas de diversificação e risco
=============================================================================
"""

from typing import Set
import pandas as pd
import numpy as np


def hhi_sector(df: pd.DataFrame) -> float:
    """
    Calcula o Herfindahl-Hirschman Index (HHI) para concentração setorial.

    O HHI mede a concentração de uma carteira entre setores. Valores mais
    altos indicam maior concentração (menos diversificação).

    Fórmula: HHI = Σ(peso_setor_i)²

    Intervalo: [1/n_setores, 1]
    - 1/n_setores: perfeitamente diversificado
    - 1: totalmente concentrado em um setor

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'SETOR' representando a carteira.
        Assume peso equiponderado (1/n) para cada ativo.

    Returns
    -------
    float
        Valor do HHI entre 0 e 1.

    Examples
    --------
    >>> df = pd.DataFrame({'SETOR': ['Financeiro', 'Financeiro', 'Tecnologia']})
    >>> hhi_sector(df)
    0.555...
    """
    if df.empty or "SETOR" not in df.columns:
        return np.nan

    weights = df["SETOR"].value_counts(normalize=True)
    return float((weights ** 2).sum())


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calcula o Índice de Jaccard entre dois conjuntos.

    O índice de Jaccard mede a similaridade entre dois conjuntos:
    J(A,B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set1 : Set
        Primeiro conjunto (ex: tickers de uma carteira).
    set2 : Set
        Segundo conjunto (ex: tickers de outra carteira).

    Returns
    -------
    float
        Índice de Jaccard entre 0 (totalmente diferentes) e 1 (idênticos).

    Examples
    --------
    >>> jaccard_similarity({'PETR4', 'VALE3'}, {'PETR4', 'ITUB4'})
    0.333...
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def portfolio_overlap(tickers1: list, tickers2: list) -> float:
    """
    Calcula o percentual de sobreposição entre duas carteiras.

    Diferente do Jaccard, este métrica considera apenas os ativos
    do primeiro conjunto que também aparecem no segundo.

    Parameters
    ----------
    tickers1 : list
        Lista de tickers da primeira carteira.
    tickers2 : list
        Lista de tickers da segunda carteira.

    Returns
    -------
    float
        Percentual de overlap (0 a 1).
    """
    if not tickers1:
        return 0.0

    set1 = set(tickers1)
    set2 = set(tickers2)

    return len(set1.intersection(set2)) / len(set1)


def coefficient_of_variation(values: list | np.ndarray) -> float:
    """
    Calcula o Coeficiente de Variação (CV).

    CV = std / mean

    Mede a variabilidade relativa. Útil para comparar dispersão
    entre distribuições com médias diferentes.

    Parameters
    ----------
    values : list or np.ndarray
        Valores para calcular o CV.

    Returns
    -------
    float
        Coeficiente de Variação.
    """
    arr = np.array(values)
    mean = np.mean(arr)

    if mean == 0:
        return 0.0

    return float(np.std(arr) / mean)


def sector_distribution(df: pd.DataFrame) -> dict:
    """
    Retorna a distribuição setorial de uma carteira.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'SETOR'.

    Returns
    -------
    dict
        Dicionário {setor: peso} normalizado.
    """
    if df.empty or "SETOR" not in df.columns:
        return {}

    return df["SETOR"].value_counts(normalize=True).to_dict()


def effective_number_of_sectors(df: pd.DataFrame) -> float:
    """
    Calcula o número efetivo de setores usando o inverso do HHI.

    Esta métrica dá uma interpretação mais intuitiva da diversificação
    setorial. Por exemplo, se ENS = 3.5, significa que a carteira está
    diversificada "como se tivesse 3.5 setores equiponderados".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'SETOR'.

    Returns
    -------
    float
        Número efetivo de setores.
    """
    hhi = hhi_sector(df)
    return 1.0 / hhi if hhi > 0 else 0.0
