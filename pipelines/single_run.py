"""pipelines/single_run.py
=============================================================================
Pipeline para execução única do algoritmo genético.

Executa o fluxo completo:
1. Pré-processamento (com cache)
2. Cálculo de scores
3. Otimização via GA
4. Geração de relatórios
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm

from config import OUTPUTS_DIR, PROFILES, DATA_PROCESSED, RAW_DATA_FILE, METRIC_COLS
from core.preprocessing import (
    load_raw_data,
    preprocess_profile,
    apply_robustness_filter
)
from core.scoring import build_scores
from core.optimizer import optimize_portfolio
from core.metrics import hhi_sector
from utils.cache import CacheManager
from cleaner import to_float


def run_single_portfolio(
    profile: str,
    use_cache: bool = True,
    robustness_filter: bool = True,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Executa pipeline completo para um único perfil.

    Parameters
    ----------
    profile : str
        Perfil do investidor.
    use_cache : bool
        Se True, usa cache para etapas intermediárias.
    robustness_filter : bool
        Se True, aplica filtro de qualidade (≥80% métricas preenchidas).
    random_seed : int, optional
        Seed para reprodutibilidade do GA.

    Returns
    -------
    pd.DataFrame
        Carteira otimizada.
    """
    cache = CacheManager()

    # 1. Carrega dados brutos
    print(f"[{profile}] Carregando dados brutos...")
    df_raw = load_raw_data()

    # 2. Pré-processamento (com cache)
    print(f"[{profile}] Pré-processando dados...")
    cache_key = f"preprocessing_{profile}"

    if use_cache:
        df_clean = cache.get_or_compute(
            key=cache_key,
            compute_fn=lambda: preprocess_profile(df_raw, profile),
            dependencies=[str(RAW_DATA_FILE)],
            format="csv"
        )
    else:
        df_clean = preprocess_profile(df_raw, profile)

    if df_clean.empty:
        raise ValueError(f"Nenhum ativo disponível para perfil {profile}")

    # 3. Filtro de robustez (opcional)
    if robustness_filter:
        print(f"[{profile}] Aplicando filtro de robustez...")
        df_clean = apply_robustness_filter(df_clean)

    # 4. Calcula scores
    print(f"[{profile}] Calculando scores...")
    df_ranked = build_scores(df_clean, profile)

    # 5. Otimiza carteira via GA
    print(f"[{profile}] Executando Algoritmo Genético...")
    portfolio = optimize_portfolio(df_ranked, profile, random_seed=random_seed)

    print(f"[{profile}] ✓ Carteira otimizada: {len(portfolio)} ativos")
    print(f"[{profile}]   Fitness: {portfolio.attrs['fitness']:.2f}")
    print(f"[{profile}]   HHI: {portfolio.attrs['hhi']:.3f}")

    return portfolio


def run_all_profiles(
    use_cache: bool = True,
    robustness_filter: bool = True,
    save_outputs: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Executa pipeline para todos os perfis.

    Parameters
    ----------
    use_cache : bool
        Se True, usa cache.
    robustness_filter : bool
        Se True, aplica filtro de robustez.
    save_outputs : bool
        Se True, salva carteiras e summary em outputs/.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dicionário {profile: portfolio}.
    """
    print("=" * 70)
    print("PIPELINE: Execução Única do Algoritmo Genético")
    print("=" * 70)

    portfolios = {}

    for profile in tqdm(PROFILES, desc="Processando perfis"):
        portfolio = run_single_portfolio(
            profile=profile,
            use_cache=use_cache,
            robustness_filter=robustness_filter
        )
        portfolios[profile] = portfolio

    if save_outputs:
        print("\nSalvando outputs...")
        save_portfolios(portfolios)
        save_summary(portfolios)

    print("\n" + "=" * 70)
    print("✓ Pipeline concluído com sucesso!")
    print("=" * 70)

    return portfolios


def save_portfolios(portfolios: Dict[str, pd.DataFrame]):
    """
    Salva carteiras individuais em JSON.

    Parameters
    ----------
    portfolios : Dict[str, pd.DataFrame]
        Dicionário de carteiras por perfil.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for profile, portfolio in portfolios.items():
        outfile = OUTPUTS_DIR / f"carteira_{profile}_ga.json"
        portfolio.to_json(
            outfile,
            orient="records",
            indent=2,
            force_ascii=False
        )
        print(f"  ✓ {outfile}")


def save_summary(portfolios: Dict[str, pd.DataFrame]):
    """
    Gera e salva summary consolidado.

    Parameters
    ----------
    portfolios : Dict[str, pd.DataFrame]
        Dicionário de carteiras por perfil.
    """
    # Carrega dados raw para métricas brutas
    df_raw = load_raw_data()

    # Converte métricas brutas para float
    for col in METRIC_COLS:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(to_float)

    # Benchmark: Ibovespa
    df_ibov = df_raw[df_raw["IN_IBOV"]].copy()

    summary = {
        "ibovespa": {
            "median_metrics": df_ibov[METRIC_COLS].median().to_dict()
        }
    }

    # Cada perfil
    for profile, portfolio in portfolios.items():
        tickers = portfolio["TICKER"].str.upper().tolist()
        df_sel_raw = df_raw[df_raw["TICKER"].isin(tickers)].copy()

        # Medianas em valores brutos
        raw_medians = {
            col: float(df_sel_raw[col].median()) if col in df_sel_raw else None
            for col in METRIC_COLS
        }

        # Medianas em z-score
        zscore_medians = {
            col: float(portfolio[col].median()) if col in portfolio else None
            for col in METRIC_COLS
        }

        # Distribuição setorial
        sector_weights = (
            portfolio["SETOR"]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )

        summary[profile] = {
            "num_assets": len(portfolio),
            "hhi": round(portfolio.attrs["hhi"], 3),
            "fitness": round(portfolio.attrs["fitness"], 2),
            "median_metrics": raw_medians,
            "zscore_metrics": zscore_medians,
            "sector_weights": sector_weights,
        }

    # Salva
    summary_file = OUTPUTS_DIR / "summary_ga.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {summary_file}")
