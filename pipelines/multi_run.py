"""pipelines/multi_run.py
=============================================================================
Pipeline para múltiplas execuções do GA (análise de robustez).

Executa o GA N vezes com diferentes seeds para avaliar:
- Estabilidade das soluções
- Variação de fitness e HHI
- Similaridade entre carteiras (Jaccard)
- Carteira consenso
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from config import (
    OUTPUTS_DIR,
    PROFILES,
    N_RUNS,
    GA_CONFIG,
    METRIC_COLS,
    DATA_PROCESSED
)
from core.preprocessing import load_processed_data, apply_robustness_filter
from core.scoring import build_scores
from core.optimizer import optimize_portfolio
from core.metrics import hhi_sector, jaccard_similarity, coefficient_of_variation
from cleaner import to_float


def run_single_execution(
    df_ranked: pd.DataFrame,
    profile: str,
    run_id: int
) -> Dict:
    """
    Executa uma única rodada do GA com seed específico.

    Parameters
    ----------
    df_ranked : pd.DataFrame
        DataFrame com scores ranqueados.
    profile : str
        Perfil do investidor.
    run_id : int
        ID da execução (usado como seed offset).

    Returns
    -------
    Dict
        Resultados da execução.
    """
    seed_value = 42 + run_id
    portfolio = optimize_portfolio(df_ranked, profile, random_seed=seed_value)

    return {
        "run_id": run_id,
        "seed": seed_value,
        "tickers": sorted(portfolio["TICKER"].tolist()),
        "fitness": float(portfolio.attrs["fitness"]),
        "hhi": float(portfolio.attrs["hhi"]),
        "score_median": float(portfolio["SCORE"].median()),
        "score_mean": float(portfolio["SCORE"].mean()),
        "score_std": float(portfolio["SCORE"].std()),
        "sectors": portfolio["SETOR"].value_counts().to_dict(),
    }


def analyze_stability(results: List[Dict]) -> Dict:
    """
    Calcula métricas de estabilidade das execuções.

    Parameters
    ----------
    results : List[Dict]
        Lista de resultados de cada execução.

    Returns
    -------
    Dict
        Métricas de estabilidade.
    """
    fitness_values = [r["fitness"] for r in results]
    hhi_values = [r["hhi"] for r in results]

    # Jaccard médio entre todos os pares
    ticker_sets = [set(r["tickers"]) for r in results]
    jaccard_scores = []
    for i in range(len(ticker_sets)):
        for j in range(i + 1, len(ticker_sets)):
            jaccard_scores.append(
                jaccard_similarity(ticker_sets[i], ticker_sets[j])
            )

    return {
        "fitness": {
            "mean": float(np.mean(fitness_values)),
            "median": float(np.median(fitness_values)),
            "std": float(np.std(fitness_values)),
            "min": float(np.min(fitness_values)),
            "max": float(np.max(fitness_values)),
            "cv": coefficient_of_variation(fitness_values),
            "ci_95": [
                float(np.percentile(fitness_values, 2.5)),
                float(np.percentile(fitness_values, 97.5))
            ]
        },
        "hhi": {
            "mean": float(np.mean(hhi_values)),
            "median": float(np.median(hhi_values)),
            "std": float(np.std(hhi_values)),
            "min": float(np.min(hhi_values)),
            "max": float(np.max(hhi_values)),
            "cv": coefficient_of_variation(hhi_values),
        },
        "portfolio_similarity": {
            "jaccard_mean": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_std": float(np.std(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_min": float(np.min(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_max": float(np.max(jaccard_scores)) if jaccard_scores else 1.0,
        }
    }


def build_consensus_portfolio(
    results: List[Dict],
    df_ranked: pd.DataFrame,
    profile: str
) -> pd.DataFrame:
    """
    Constrói carteira consenso baseada na frequência de aparição.

    Parameters
    ----------
    results : List[Dict]
        Resultados de todas as execuções.
    df_ranked : pd.DataFrame
        DataFrame com dados ranqueados.
    profile : str
        Perfil do investidor.

    Returns
    -------
    pd.DataFrame
        Carteira consenso.
    """
    cfg = GA_CONFIG[profile]
    n_assets = cfg["n_assets"]

    # Conta frequência de cada ticker
    all_tickers = []
    for r in results:
        all_tickers.extend(r["tickers"])

    ticker_counts = Counter(all_tickers)

    # Seleciona os N mais frequentes
    most_common = ticker_counts.most_common(n_assets)
    consensus_tickers = [ticker for ticker, _ in most_common]

    # Constrói DataFrame da carteira consenso
    consensus_portfolio = df_ranked[
        df_ranked["TICKER"].isin(consensus_tickers)
    ].copy()

    # Adiciona coluna de frequência
    consensus_portfolio["FREQUENCY"] = consensus_portfolio["TICKER"].map(
        lambda t: ticker_counts[t] / len(results)
    )

    # Calcula métricas
    hhi = hhi_sector(consensus_portfolio)
    avg_score = consensus_portfolio["SCORE"].mean()

    consensus_portfolio.attrs["hhi"] = hhi
    consensus_portfolio.attrs["avg_score"] = avg_score
    consensus_portfolio.attrs["method"] = "consensus_frequency"

    return consensus_portfolio.sort_values("FREQUENCY", ascending=False)


def run_multi_execution_profile(
    profile: str,
    n_runs: int = N_RUNS,
    use_cache: bool = True,
    parallel: bool = True
) -> Dict:
    """
    Executa múltiplas rodadas do GA para um perfil.

    Parameters
    ----------
    profile : str
        Perfil do investidor.
    n_runs : int
        Número de execuções.
    use_cache : bool
        Se True, usa dados pré-processados do cache.
    parallel : bool
        Se True, executa runs em paralelo.

    Returns
    -------
    Dict
        Resultados consolidados.
    """
    print(f"\n{'='*70}")
    print(f"Perfil: {profile.upper()}")
    print(f"{'='*70}")

    # Carrega dados
    if use_cache:
        df = load_processed_data(profile)
    else:
        from core.preprocessing import load_raw_data, preprocess_profile
        df_raw = load_raw_data()
        df = preprocess_profile(df_raw, profile)

    df = apply_robustness_filter(df)
    df_ranked = build_scores(df, profile)

    # Executa múltiplas rodadas
    results = []

    if parallel:
        # Paralelização
        with mp.Pool() as pool:
            run_func = partial(
                run_single_execution,
                df_ranked,
                profile
            )
            results = list(tqdm(
                pool.imap(run_func, range(n_runs)),
                total=n_runs,
                desc=f"Executando {n_runs} rodadas"
            ))
    else:
        # Sequencial
        for run_id in tqdm(range(n_runs), desc=f"Executando {n_runs} rodadas"):
            result = run_single_execution(df_ranked, profile, run_id)
            results.append(result)

    # Análise de estabilidade
    stability = analyze_stability(results)

    # Carteira consenso
    consensus = build_consensus_portfolio(results, df_ranked, profile)

    # Salva carteira consenso
    consensus_file = OUTPUTS_DIR / f"carteira_{profile}_consensus.json"
    consensus.to_json(
        consensus_file,
        orient="records",
        indent=2,
        force_ascii=False
    )

    # Salva métricas detalhadas
    df_metrics = pd.DataFrame(results)
    metrics_file = OUTPUTS_DIR / f"metrics_stability_{profile}.csv"
    df_metrics.to_csv(metrics_file, index=False)

    print(f"\n  📊 Fitness: {stability['fitness']['mean']:.2f} ± {stability['fitness']['std']:.2f}")
    print(f"  📊 HHI: {stability['hhi']['mean']:.3f} ± {stability['hhi']['std']:.3f}")
    print(f"  📊 Jaccard Médio: {stability['portfolio_similarity']['jaccard_mean']:.3f}")
    print(f"  💾 Carteira consenso: {consensus_file}")
    print(f"  💾 Métricas: {metrics_file}")

    return {
        "n_runs": len(results),
        "stability_metrics": stability,
        "consensus_portfolio": {
            "tickers": consensus["TICKER"].tolist(),
            "hhi": float(consensus.attrs["hhi"]),
            "avg_score": float(consensus.attrs["avg_score"]),
            "frequency_mean": float(consensus["FREQUENCY"].mean()),
        },
        "all_runs": results
    }


def run_multi_execution_all_profiles(
    n_runs: int = N_RUNS,
    parallel: bool = True,
    save_summary: bool = True
) -> Dict:
    """
    Executa múltiplas rodadas para todos os perfis.

    Parameters
    ----------
    n_runs : int
        Número de execuções por perfil.
    parallel : bool
        Se True, paraleliza execuções.
    save_summary : bool
        Se True, salva summary consolidado.

    Returns
    -------
    Dict
        Resultados de todos os perfis.
    """
    print("=" * 70)
    print("PIPELINE: Múltiplas Execuções do Algoritmo Genético")
    print(f"Número de execuções por perfil: {n_runs}")
    print("=" * 70)

    all_results = {}

    for profile in PROFILES:
        results = run_multi_execution_profile(
            profile=profile,
            n_runs=n_runs,
            parallel=parallel
        )
        all_results[profile] = results

    if save_summary:
        summary_file = OUTPUTS_DIR / "multiple_runs_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*70}")
        print(f"✅ Análise completa salva em: {summary_file}")
        print(f"{'='*70}\n")

    # Relatório de interpretação
    print("\n📋 INTERPRETAÇÃO DOS RESULTADOS:")
    print("-" * 70)
    for profile, data in all_results.items():
        stab = data["stability_metrics"]
        print(f"\n{profile.upper()}:")
        print(f"  • CV do Fitness: {stab['fitness']['cv']:.2%}")
        print(f"    → {'BAIXA variabilidade' if stab['fitness']['cv'] < 0.05 else 'ALTA variabilidade'}")
        print(f"  • Índice Jaccard: {stab['portfolio_similarity']['jaccard_mean']:.3f}")
        print(f"    → {'ALTA similaridade' if stab['portfolio_similarity']['jaccard_mean'] > 0.7 else 'MODERADA similaridade'}")

    return all_results
