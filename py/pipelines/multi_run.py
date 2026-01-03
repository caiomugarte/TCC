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

# Configura matplotlib para backend não-interativo (evita problemas com multiprocessing)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run_single_execution(
    df_ranked: pd.DataFrame,
    profile: str,
    run_id: int
) -> tuple:
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
    tuple
        (Dict com resultados, DataFrame do portfolio)
    """
    seed_value = 42 + run_id
    portfolio = optimize_portfolio(df_ranked, profile, random_seed=seed_value)

    result = {
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

    return result, portfolio


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


def get_best_individual_portfolio(
    results: List[Dict],
    portfolios: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Retorna o portfolio com maior fitness de todas as execuções.

    Parameters
    ----------
    results : List[Dict]
        Resultados de todas as execuções.
    portfolios : List[pd.DataFrame]
        Portfolios de todas as execuções.

    Returns
    -------
    pd.DataFrame
        Portfolio com maior fitness.
    """
    # Encontra o índice do melhor fitness
    best_idx = max(range(len(results)), key=lambda i: results[i]["fitness"])

    best_portfolio = portfolios[best_idx].copy()
    best_portfolio.attrs["method"] = "best_individual"
    best_portfolio.attrs["run_id"] = results[best_idx]["run_id"]
    best_portfolio.attrs["seed"] = results[best_idx]["seed"]

    return best_portfolio


def compare_portfolios(
    consensus: pd.DataFrame,
    best_individual: pd.DataFrame,
    profile: str
) -> Dict:
    """
    Compara as métricas entre carteira consensual e melhor indivíduo.

    Parameters
    ----------
    consensus : pd.DataFrame
        Carteira consensual.
    best_individual : pd.DataFrame
        Melhor indivíduo.
    profile : str
        Perfil do investidor.

    Returns
    -------
    Dict
        Comparação detalhada entre as carteiras.
    """
    # Tickers
    consensus_tickers = set(consensus["TICKER"].tolist())
    best_tickers = set(best_individual["TICKER"].tolist())

    # Overlap (Jaccard similarity)
    overlap = jaccard_similarity(consensus_tickers, best_tickers)
    common_tickers = list(consensus_tickers.intersection(best_tickers))
    only_consensus = list(consensus_tickers - best_tickers)
    only_best = list(best_tickers - consensus_tickers)

    # Fitness e HHI
    consensus_fitness = consensus["SCORE"].sum() - GA_CONFIG[profile]["lambda"] * consensus.attrs["hhi"] * GA_CONFIG[profile]["n_assets"]
    best_fitness = best_individual.attrs["fitness"]

    # Composição setorial
    consensus_sectors = consensus["SETOR"].value_counts().to_dict()
    best_sectors = best_individual["SETOR"].value_counts().to_dict()

    return {
        "profile": profile,
        "overlap": {
            "jaccard_index": float(overlap),
            "common_tickers": sorted(common_tickers),
            "n_common": len(common_tickers),
            "only_consensus": sorted(only_consensus),
            "n_only_consensus": len(only_consensus),
            "only_best_individual": sorted(only_best),
            "n_only_best": len(only_best),
        },
        "fitness": {
            "consensus": float(consensus_fitness),
            "best_individual": float(best_fitness),
            "difference": float(best_fitness - consensus_fitness),
            "percent_difference": float((best_fitness - consensus_fitness) / abs(consensus_fitness) * 100) if consensus_fitness != 0 else 0.0,
        },
        "hhi": {
            "consensus": float(consensus.attrs["hhi"]),
            "best_individual": float(best_individual.attrs["hhi"]),
            "difference": float(best_individual.attrs["hhi"] - consensus.attrs["hhi"]),
        },
        "score_stats": {
            "consensus": {
                "mean": float(consensus["SCORE"].mean()),
                "median": float(consensus["SCORE"].median()),
                "std": float(consensus["SCORE"].std()),
            },
            "best_individual": {
                "mean": float(best_individual["SCORE"].mean()),
                "median": float(best_individual["SCORE"].median()),
                "std": float(best_individual["SCORE"].std()),
            },
        },
        "sector_composition": {
            "consensus": consensus_sectors,
            "best_individual": best_sectors,
        },
    }


def run_backtest_comparison(
    consensus: pd.DataFrame,
    best_individual: pd.DataFrame,
    profile: str,
    period_years: int = 5
) -> Dict:
    """
    Executa backtest comparativo entre carteira consensual e melhor indivíduo.

    Parameters
    ----------
    consensus : pd.DataFrame
        Carteira consensual.
    best_individual : pd.DataFrame
        Melhor indivíduo.
    profile : str
        Perfil do investidor.
    period_years : int
        Número de anos para backtest.

    Returns
    -------
    Dict
        Resultados do backtest comparativo.
    """
    from datetime import datetime, timedelta
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠ yfinance não instalado. Backtest não será executado.")
        return {}

    print(f"\n  🔍 Executando backtest comparativo ({period_years} anos)...")

    # Datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365 + 30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Função auxiliar para buscar dados
    def fetch_portfolio_data(tickers):
        tickers_yf = [f"{t}.SA" for t in tickers]
        try:
            data = yf.download(
                tickers_yf,
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True,
                threads=True
            )
            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data

            # Remove colunas com muitos NaN
            valid_cols = [col for col in prices.columns
                         if prices[col].count() > len(prices) * 0.5]
            prices = prices[valid_cols]
            prices = prices.ffill(limit=5)
            prices = prices.dropna(how='all')

            return prices
        except Exception as e:
            print(f"    ✗ Erro ao buscar dados: {e}")
            return pd.DataFrame()

    # Função para calcular retornos
    def calculate_portfolio_performance(prices):
        if prices.empty or len(prices.columns) == 0:
            return None

        # Retornos normalizados (início = 100)
        normalized = prices / prices.iloc[0] * 100
        portfolio_value = normalized.mean(axis=1)

        # Métricas
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        annual_return = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / period_years) - 1) * 100

        # Volatilidade
        returns = portfolio_value.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe (usando 10% como taxa livre de risco)
        sharpe = (annual_return - 10) / volatility if volatility > 0 else 0

        # Drawdown
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        return {
            "values": portfolio_value,
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "volatility_pct": round(volatility, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "n_assets": len(prices.columns),
        }

    # Backtest consenso
    consensus_tickers = consensus["TICKER"].tolist()
    consensus_prices = fetch_portfolio_data(consensus_tickers)
    consensus_results = calculate_portfolio_performance(consensus_prices)

    # Backtest melhor indivíduo
    best_tickers = best_individual["TICKER"].tolist()
    best_prices = fetch_portfolio_data(best_tickers)
    best_results = calculate_portfolio_performance(best_prices)

    if consensus_results is None or best_results is None:
        print("  ✗ Dados insuficientes para backtest")
        return {}

    # Comparação
    comparison = {
        "period_years": period_years,
        "start_date": start_str,
        "end_date": end_str,
        "consensus": consensus_results,
        "best_individual": best_results,
        "winner": "consensus" if consensus_results["total_return_pct"] > best_results["total_return_pct"] else "best_individual",
        "return_difference_pct": round(
            consensus_results["total_return_pct"] - best_results["total_return_pct"], 2
        ),
    }

    print(f"    • Consenso: Retorno={consensus_results['annual_return_pct']:.2f}% aa, "
          f"Sharpe={consensus_results['sharpe_ratio']:.2f}, "
          f"Drawdown={consensus_results['max_drawdown_pct']:.2f}%")
    print(f"    • Melhor Ind: Retorno={best_results['annual_return_pct']:.2f}% aa, "
          f"Sharpe={best_results['sharpe_ratio']:.2f}, "
          f"Drawdown={best_results['max_drawdown_pct']:.2f}%")
    print(f"    • Vencedor: {comparison['winner'].upper()} "
          f"(diferença: {comparison['return_difference_pct']:+.2f}%)")

    return comparison


def plot_comparison_charts(
    comparison: Dict,
    backtest_comparison: Dict,
    profile: str
):
    """
    Gera gráficos comparativos entre carteira consensual e melhor indivíduo.

    Parameters
    ----------
    comparison : Dict
        Comparação de métricas.
    backtest_comparison : Dict
        Resultados do backtest.
    profile : str
        Perfil do investidor.
    """
    if not backtest_comparison:
        print("  ⚠ Backtest não disponível. Gráficos de backtest não serão gerados.")
        return

    print(f"\n  📊 Gerando gráficos comparativos...")

    # Configurações de estilo
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ===== GRÁFICO 1: Fitness e HHI (Barras) =====
    ax1 = axes[0, 0]

    categories = ['Fitness', 'HHI']
    consensus_vals = [
        comparison['fitness']['consensus'],
        comparison['hhi']['consensus']
    ]
    best_vals = [
        comparison['fitness']['best_individual'],
        comparison['hhi']['best_individual']
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, consensus_vals, width, label='Consenso', color='#2E7D32', alpha=0.8)
    bars2 = ax1.bar(x + width/2, best_vals, width, label='Melhor Indivíduo', color='#D32F2F', alpha=0.8)

    ax1.set_ylabel('Valor', fontweight='bold')
    ax1.set_title('Comparação: Fitness e HHI', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ===== GRÁFICO 2: Composição Setorial (Barras Horizontais) =====
    ax2 = axes[0, 1]

    consensus_sectors = comparison['sector_composition']['consensus']
    best_sectors = comparison['sector_composition']['best_individual']

    all_sectors = set(consensus_sectors.keys()).union(set(best_sectors.keys()))
    sectors_list = sorted(all_sectors)

    consensus_counts = [consensus_sectors.get(s, 0) for s in sectors_list]
    best_counts = [best_sectors.get(s, 0) for s in sectors_list]

    y = np.arange(len(sectors_list))
    width = 0.35

    ax2.barh(y - width/2, consensus_counts, width, label='Consenso', color='#2E7D32', alpha=0.8)
    ax2.barh(y + width/2, best_counts, width, label='Melhor Indivíduo', color='#D32F2F', alpha=0.8)

    ax2.set_xlabel('Número de Ativos', fontweight='bold')
    ax2.set_title('Composição Setorial', fontweight='bold', fontsize=12)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sectors_list, fontsize=9)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    # ===== GRÁFICO 3: Métricas de Backtest (Barras) =====
    ax3 = axes[1, 0]

    metrics = ['Retorno\nAnual (%)', 'Sharpe\nRatio', 'Volatilidade\n(%)', 'Max Drawdown\n(%)']
    consensus_metrics = [
        backtest_comparison['consensus']['annual_return_pct'],
        backtest_comparison['consensus']['sharpe_ratio'] * 10,  # Escala para visualização
        backtest_comparison['consensus']['volatility_pct'],
        abs(backtest_comparison['consensus']['max_drawdown_pct'])
    ]
    best_metrics = [
        backtest_comparison['best_individual']['annual_return_pct'],
        backtest_comparison['best_individual']['sharpe_ratio'] * 10,
        backtest_comparison['best_individual']['volatility_pct'],
        abs(backtest_comparison['best_individual']['max_drawdown_pct'])
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, consensus_metrics, width, label='Consenso', color='#2E7D32', alpha=0.8)
    bars2 = ax3.bar(x + width/2, best_metrics, width, label='Melhor Indivíduo', color='#D32F2F', alpha=0.8)

    ax3.set_ylabel('Valor', fontweight='bold')
    ax3.set_title('Métricas de Backtest (5 anos)', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Nota sobre Sharpe
    ax3.text(1, max(consensus_metrics[1], best_metrics[1]) * 1.1,
            '*Sharpe x10', ha='center', fontsize=8, style='italic', color='gray')

    # ===== GRÁFICO 4: Evolução do Backtest =====
    ax4 = axes[1, 1]

    if 'values' in backtest_comparison['consensus'] and 'values' in backtest_comparison['best_individual']:
        consensus_values = backtest_comparison['consensus']['values']
        best_values = backtest_comparison['best_individual']['values']

        ax4.plot(consensus_values.index, consensus_values, label='Consenso',
                color='#2E7D32', linewidth=2.5, alpha=0.9)
        ax4.plot(best_values.index, best_values, label='Melhor Indivíduo',
                color='#D32F2F', linewidth=2.5, alpha=0.9, linestyle='--')

        ax4.set_xlabel('Data', fontweight='bold')
        ax4.set_ylabel('Valor Normalizado (Base 100)', fontweight='bold')
        ax4.set_title('Evolução do Portfólio (5 anos)', fontweight='bold', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax4.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Dados de evolução\nnão disponíveis',
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, color='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])

    # Título geral
    fig.suptitle(f'Comparação: Carteira Consensual vs Melhor Indivíduo - {profile.upper()}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Salva
    output_file = OUTPUTS_DIR / f"comparison_charts_{profile}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráficos salvos: {output_file.name}")


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
    portfolios = []

    if parallel:
        # Paralelização
        with mp.Pool() as pool:
            run_func = partial(
                run_single_execution,
                df_ranked,
                profile
            )
            outputs = list(tqdm(
                pool.imap(run_func, range(n_runs)),
                total=n_runs,
                desc=f"Executando {n_runs} rodadas"
            ))
            # Separa resultados e portfolios
            results = [output[0] for output in outputs]
            portfolios = [output[1] for output in outputs]
    else:
        # Sequencial
        for run_id in tqdm(range(n_runs), desc=f"Executando {n_runs} rodadas"):
            result, portfolio = run_single_execution(df_ranked, profile, run_id)
            results.append(result)
            portfolios.append(portfolio)

    # Análise de estabilidade
    stability = analyze_stability(results)

    # Carteira consenso
    consensus = build_consensus_portfolio(results, df_ranked, profile)

    # Melhor indivíduo (maior fitness)
    best_individual = get_best_individual_portfolio(results, portfolios)

    # Comparação entre carteiras
    comparison = compare_portfolios(consensus, best_individual, profile)

    # Backtest comparativo
    backtest_comparison = run_backtest_comparison(consensus, best_individual, profile, period_years=5)

    # Salva carteira consenso
    consensus_file = OUTPUTS_DIR / f"carteira_{profile}_consensus.json"
    consensus.to_json(
        consensus_file,
        orient="records",
        indent=2,
        force_ascii=False
    )

    # Salva melhor indivíduo
    best_individual_file = OUTPUTS_DIR / f"carteira_{profile}_best_individual.json"
    best_individual.to_json(
        best_individual_file,
        orient="records",
        indent=2,
        force_ascii=False
    )

    # Salva comparação (inclui métricas e backtest)
    # Remove as séries temporais do backtest para tornar serializável
    backtest_comparison_serializable = backtest_comparison.copy()
    if backtest_comparison_serializable:
        for key in ["consensus", "best_individual"]:
            if key in backtest_comparison_serializable and backtest_comparison_serializable[key]:
                if "values" in backtest_comparison_serializable[key]:
                    del backtest_comparison_serializable[key]["values"]

    full_comparison = {
        "metrics_comparison": comparison,
        "backtest_comparison": backtest_comparison_serializable
    }
    comparison_file = OUTPUTS_DIR / f"comparison_{profile}.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, ensure_ascii=False, indent=2)

    # Gera gráficos comparativos
    plot_comparison_charts(comparison, backtest_comparison, profile)

    # Salva métricas detalhadas
    df_metrics = pd.DataFrame(results)
    metrics_file = OUTPUTS_DIR / f"metrics_stability_{profile}.csv"
    df_metrics.to_csv(metrics_file, index=False)

    print(f"\n  📊 Fitness: {stability['fitness']['mean']:.2f} ± {stability['fitness']['std']:.2f}")
    print(f"  📊 HHI: {stability['hhi']['mean']:.3f} ± {stability['hhi']['std']:.3f}")
    print(f"  📊 Jaccard Médio: {stability['portfolio_similarity']['jaccard_mean']:.3f}")
    print(f"\n  🏆 COMPARAÇÃO CONSENSO vs MELHOR INDIVÍDUO:")
    print(f"     • Overlap (Jaccard): {comparison['overlap']['jaccard_index']:.3f}")
    print(f"     • Fitness Consenso: {comparison['fitness']['consensus']:.2f}")
    print(f"     • Fitness Melhor: {comparison['fitness']['best_individual']:.2f}")
    print(f"     • Diferença: {comparison['fitness']['difference']:+.2f} ({comparison['fitness']['percent_difference']:+.1f}%)")
    print(f"  💾 Carteira consenso: {consensus_file}")
    print(f"  💾 Melhor indivíduo: {best_individual_file}")
    print(f"  💾 Comparação: {comparison_file}")
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
        "best_individual_portfolio": {
            "tickers": best_individual["TICKER"].tolist(),
            "fitness": float(best_individual.attrs["fitness"]),
            "hhi": float(best_individual.attrs["hhi"]),
            "run_id": int(best_individual.attrs["run_id"]),
            "seed": int(best_individual.attrs["seed"]),
        },
        "comparison": comparison,
        "backtest_comparison": backtest_comparison,
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
