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
from typing import Dict, List, Optional, Sequence
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


def normalize_tickers(tickers: Optional[Sequence[str]]) -> set[str]:
    return {
        str(ticker).strip().upper()
        for ticker in (tickers or ())
        if str(ticker).strip()
    }


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
        "generations_run": portfolio.attrs.get("generations_run", 0),
        "converged_early": portfolio.attrs.get("converged_early", False),
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

    # Seleciona os N mais frequentes com critério de desempate por SCORE
    # Cria lista de (ticker, frequência, score) para ordenação
    ticker_data = []
    for ticker, count in ticker_counts.items():
        score = df_ranked[df_ranked["TICKER"] == ticker]["SCORE"].iloc[0]
        ticker_data.append((ticker, count, score))

    # Ordena por: 1) frequência (desc), 2) score (desc), 3) ticker (alfabético)
    ticker_data.sort(key=lambda x: (-x[1], -x[2], x[0]))

    # Filtra duplicatas de mesma empresa (ex: POMO3 e POMO4)
    # Remove sufixos 3, 4, 5, 6, 11 para identificar empresa base
    def get_base_ticker(ticker):
        """Remove sufixo de classe de ação para identificar empresa base."""
        for suffix in ['11', '3', '4', '5', '6']:
            if ticker.endswith(suffix):
                return ticker[:-len(suffix)]
        return ticker

    # Seleciona ativos evitando duplicatas de mesma empresa
    consensus_tickers = []
    seen_companies = set()
    skipped_duplicates = []

    for ticker, freq, score in ticker_data:
        base_ticker = get_base_ticker(ticker)

        # Se ainda não temos ativo dessa empresa, adiciona
        if base_ticker not in seen_companies:
            consensus_tickers.append(ticker)
            seen_companies.add(base_ticker)

            # Para quando atingir o número desejado
            if len(consensus_tickers) == n_assets:
                break
        else:
            # Registra duplicata pulada
            existing_ticker = [t for t in consensus_tickers if get_base_ticker(t) == base_ticker][0]
            skipped_duplicates.append((ticker, existing_ticker, freq, score))

    # Informa sobre duplicatas removidas
    if skipped_duplicates:
        print(f"\n  ⚠️  Duplicatas de mesma empresa removidas da carteira consensual:")
        for skipped, kept, freq, score in skipped_duplicates:
            if skipped in [t for t, _, _ in ticker_data[:n_assets * 2]]:  # Só mostra se estava entre os candidatos
                print(f"      • {skipped} (freq: {freq/len(results):.1%}, score: {score:.3f}) "
                      f"→ já tem {kept} da mesma empresa")

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
    portfolios: List[pd.DataFrame] = None,
    df_ranked: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Retorna o portfolio com maior fitness de todas as execuções.

    Parameters
    ----------
    results : List[Dict]
        Resultados de todas as execuções.
    portfolios : List[pd.DataFrame], optional
        Portfolios de todas as execuções. Se None, reconstrói a partir de results e df_ranked.
    df_ranked : pd.DataFrame, optional
        DataFrame ranqueado para reconstruir portfolio. Necessário se portfolios=None.

    Returns
    -------
    pd.DataFrame
        Portfolio com maior fitness.
    """
    # Encontra o índice do melhor fitness
    best_idx = max(range(len(results)), key=lambda i: results[i]["fitness"])

    # Se portfolios foi fornecido, usa diretamente
    if portfolios is not None and best_idx < len(portfolios):
        best_portfolio = portfolios[best_idx].copy()
    # Caso contrário, reconstrói a partir dos tickers
    elif df_ranked is not None:
        best_tickers = results[best_idx]["tickers"]
        best_portfolio = df_ranked[df_ranked["TICKER"].isin(best_tickers)].copy()
        best_portfolio.attrs["fitness"] = results[best_idx]["fitness"]
        best_portfolio.attrs["hhi"] = results[best_idx]["hhi"]
    else:
        raise ValueError("É necessário fornecer 'portfolios' ou 'df_ranked' para reconstruir o portfolio")

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


def compare_fundamental_metrics(
    consensus: pd.DataFrame,
    best_individual: pd.DataFrame,
    profile: str
) -> Dict:
    """
    Compara métricas fundamentalistas entre carteira consensual e melhor indivíduo.

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
        Comparação de métricas fundamentalistas.
    """
    from config import METRIC_COLS

    # Métricas disponíveis em ambas as carteiras
    available_metrics = [col for col in METRIC_COLS if col in consensus.columns]

    comparison = {
        "profile": profile,
        "metrics": {},
        "summary": {}
    }

    for metric in available_metrics:
        consensus_values = consensus[metric].dropna()
        best_values = best_individual[metric].dropna()

        if len(consensus_values) == 0 or len(best_values) == 0:
            continue

        comparison["metrics"][metric] = {
            "consensus": {
                "mean": float(consensus_values.mean()),
                "median": float(consensus_values.median()),
                "std": float(consensus_values.std()),
                "min": float(consensus_values.min()),
                "max": float(consensus_values.max()),
            },
            "best_individual": {
                "mean": float(best_values.mean()),
                "median": float(best_values.median()),
                "std": float(best_values.std()),
                "min": float(best_values.min()),
                "max": float(best_values.max()),
            },
            "difference": {
                "mean": float(consensus_values.mean() - best_values.mean()),
                "median": float(consensus_values.median() - best_values.median()),
            }
        }

    # Análise resumida: qual carteira é "melhor" em cada grupo de métricas
    metric_groups = {
        "valuation": ["P/L", "P/VP", "EV/EBIT", "PSR"],
        "profitability": ["ROE", "ROA", "ROIC", "MARG. LIQUIDA", "MARGEM EBIT"],
        "growth": ["CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS"],
        "dividend": ["DY"],
        "liquidity": ["LIQ. CORRENTE", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI."]
    }

    for group_name, metrics in metric_groups.items():
        consensus_wins = 0
        best_wins = 0

        for metric in metrics:
            if metric not in comparison["metrics"]:
                continue

            # Para métricas de valuation, menor é melhor
            # Para outras, maior é melhor
            is_lower_better = metric in ["P/L", "P/VP", "EV/EBIT", "PSR", "DIVIDA LIQUIDA / EBIT"]

            consensus_mean = comparison["metrics"][metric]["consensus"]["mean"]
            best_mean = comparison["metrics"][metric]["best_individual"]["mean"]

            if is_lower_better:
                if consensus_mean < best_mean:
                    consensus_wins += 1
                else:
                    best_wins += 1
            else:
                if consensus_mean > best_mean:
                    consensus_wins += 1
                else:
                    best_wins += 1

        comparison["summary"][group_name] = {
            "consensus_wins": consensus_wins,
            "best_individual_wins": best_wins,
            "winner": "consensus" if consensus_wins > best_wins else ("best_individual" if best_wins > consensus_wins else "tie")
        }

    # Vencedor geral
    total_consensus_wins = sum(g["consensus_wins"] for g in comparison["summary"].values())
    total_best_wins = sum(g["best_individual_wins"] for g in comparison["summary"].values())

    comparison["overall_winner"] = "consensus" if total_consensus_wins > total_best_wins else ("best_individual" if total_best_wins > total_consensus_wins else "tie")
    comparison["overall_score"] = {
        "consensus": total_consensus_wins,
        "best_individual": total_best_wins
    }

    return comparison


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

            # Keep requested order and expose missing downloads as all-NaN columns.
            prices = prices.reindex(columns=tickers_yf)

            # ponytail: endpoint + 95% coverage gate; replace with exchange-calendar
            # validation if suspended assets need separate treatment.
            incomplete = []
            latest_first_date = pd.Timestamp(start_str) + pd.Timedelta(days=7)
            earliest_last_date = pd.Timestamp(end_str) - pd.Timedelta(days=7)
            for ticker in tickers_yf:
                series = prices[ticker].dropna()
                if (
                    series.empty
                    or series.index[0] > latest_first_date
                    or series.index[-1] < earliest_last_date
                    or len(series) < len(prices) * 0.95
                ):
                    incomplete.append(ticker)

            if incomplete:
                print(
                    "    ⚠ Backtest descartado: histórico incompleto para "
                    + ", ".join(incomplete)
                )
                return pd.DataFrame()

            # Align all assets on observed trading dates; never invent prices.
            prices = prices.dropna()

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
    parallel: bool = True,
    save_interval: int = 10,
    adaptive_mode: bool = False,
    min_runs: int = 30,
    target_cv: float = 0.03,
    target_jaccard: float = 0.70,
    exclude_tickers: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Executa múltiplas rodadas do GA para um perfil.

    Parameters
    ----------
    profile : str
        Perfil do investidor.
    n_runs : int
        Número máximo de execuções.
    use_cache : bool
        Se True, usa dados pré-processados do cache.
    parallel : bool
        Se True, executa runs em paralelo.
    save_interval : int
        Salva checkpoint a cada N runs (default: 10).
    adaptive_mode : bool
        Se True, para automaticamente quando atingir estabilidade (default: False).
    min_runs : int
        Número mínimo de runs antes de verificar convergência (default: 30).
    target_cv : float
        CV alvo do fitness para convergência (default: 0.03 = 3%).
    target_jaccard : float
        Jaccard médio alvo para convergência (default: 0.70 = 70%).
    exclude_tickers : Sequence[str], optional
        Tickers removed before scoring and genetic optimization.

    Returns
    -------
    Dict
        Resultados consolidados.
    """
    print(f"\n{'='*70}")
    print(f"Perfil: {profile.upper()}")
    print(f"{'='*70}")

    excluded = normalize_tickers(exclude_tickers)
    checkpoint_suffix = f"_excluding_{'-'.join(sorted(excluded))}" if excluded else ""
    checkpoint_file = OUTPUTS_DIR / f".checkpoint_{profile}_runs{checkpoint_suffix}.json"

    # Tenta carregar checkpoint existente
    results = []
    portfolios = []
    start_run = 0

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                if checkpoint.get("n_runs") == n_runs and checkpoint.get("profile") == profile:
                    print(f"\n  ℹ️  Checkpoint encontrado com {len(checkpoint['results'])} runs completados")
                    resume = input("  Deseja retomar de onde parou? (s/n) [s]: ").strip().lower()
                    if resume != "n":
                        results = checkpoint["results"]
                        start_run = len(results)
                        print(f"  ✓ Retomando a partir do run {start_run}")
                else:
                    print(f"  ⚠️  Checkpoint incompatível (runs diferentes). Ignorando...")
        except Exception as e:
            print(f"  ⚠️  Erro ao carregar checkpoint: {e}")

    # Carrega dados
    if use_cache:
        df = load_processed_data(profile)
    else:
        from core.preprocessing import load_raw_data, preprocess_profile
        df_raw = load_raw_data()
        df = preprocess_profile(df_raw, profile)

    if excluded:
        available = normalize_tickers(df["TICKER"].tolist())
        unknown = excluded - available
        if unknown:
            raise ValueError(f"tickers not found for exclusion: {sorted(unknown)}")
        df = df[~df["TICKER"].astype(str).str.upper().isin(excluded)].copy()
        print(f"Excluding tickers: {', '.join(sorted(excluded))}")

    df = apply_robustness_filter(df)
    df_ranked = build_scores(df, profile)

    # Executa múltiplas rodadas
    # Para otimizar memória, mantém apenas o melhor portfolio de cada batch
    best_portfolio_so_far = None
    best_fitness_so_far = -np.inf

    if parallel:
        # Paralelização com salvamento incremental
        remaining_runs = n_runs - start_run
        batch_size = min(save_interval, remaining_runs)

        for batch_start in range(start_run, n_runs, batch_size):
            batch_end = min(batch_start + batch_size, n_runs)
            batch_range = range(batch_start, batch_end)

            with mp.Pool() as pool:
                run_func = partial(
                    run_single_execution,
                    df_ranked,
                    profile
                )
                batch_outputs = list(tqdm(
                    pool.imap(run_func, batch_range),
                    total=len(batch_range),
                    desc=f"Batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end}/{n_runs})"
                ))

                # Separa resultados e portfolios
                batch_results = [output[0] for output in batch_outputs]
                batch_portfolios = [output[1] for output in batch_outputs]

                results.extend(batch_results)

                # Mantém apenas o melhor portfolio do batch (otimização de memória)
                for i, portfolio in enumerate(batch_portfolios):
                    if batch_results[i]["fitness"] > best_fitness_so_far:
                        best_fitness_so_far = batch_results[i]["fitness"]
                        best_portfolio_so_far = portfolio.copy()

                # Limpa portfolios do batch para liberar memória
                del batch_portfolios

            # Salva checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "profile": profile,
                    "n_runs": n_runs,
                    "completed_runs": len(results),
                    "results": results
                }, f, indent=2)

            print(f"  💾 Checkpoint salvo: {len(results)}/{n_runs} runs completados")

            # Verifica convergência no modo adaptativo
            if adaptive_mode and len(results) >= min_runs:
                stability = analyze_stability(results)
                current_cv = stability['fitness']['cv']
                current_jaccard = stability['portfolio_similarity']['jaccard_mean']

                print(f"\n  📊 Verificação de Convergência:")
                print(f"     • CV Fitness: {current_cv:.4f} (alvo: {target_cv:.4f})")
                print(f"     • Jaccard Médio: {current_jaccard:.4f} (alvo: {target_jaccard:.4f})")

                if current_cv <= target_cv and current_jaccard >= target_jaccard:
                    print(f"\n  ✅ Convergência atingida após {len(results)} runs!")
                    print(f"     • Parando antecipadamente (economia de {n_runs - len(results)} runs)")
                    break
    else:
        # Sequencial com salvamento incremental
        for run_id in tqdm(range(start_run, n_runs), desc=f"Executando {n_runs} rodadas"):
            result, portfolio = run_single_execution(df_ranked, profile, run_id)
            results.append(result)

            # Mantém apenas o melhor portfolio (otimização de memória)
            if result["fitness"] > best_fitness_so_far:
                best_fitness_so_far = result["fitness"]
                best_portfolio_so_far = portfolio.copy()

            # Salva checkpoint periodicamente
            if (run_id + 1) % save_interval == 0 or (run_id + 1) == n_runs:
                with open(checkpoint_file, "w") as f:
                    json.dump({
                        "profile": profile,
                        "n_runs": n_runs,
                        "completed_runs": len(results),
                        "results": results
                    }, f, indent=2)
                print(f"  💾 Checkpoint salvo: {len(results)}/{n_runs} runs completados")

                # Verifica convergência no modo adaptativo
                if adaptive_mode and len(results) >= min_runs:
                    stability = analyze_stability(results)
                    current_cv = stability['fitness']['cv']
                    current_jaccard = stability['portfolio_similarity']['jaccard_mean']

                    print(f"\n  📊 Verificação de Convergência:")
                    print(f"     • CV Fitness: {current_cv:.4f} (alvo: {target_cv:.4f})")
                    print(f"     • Jaccard Médio: {current_jaccard:.4f} (alvo: {target_jaccard:.4f})")

                    if current_cv <= target_cv and current_jaccard >= target_jaccard:
                        print(f"\n  ✅ Convergência atingida após {len(results)} runs!")
                        print(f"     • Parando antecipadamente (economia de {n_runs - len(results)} runs)")
                        break

    # Remove checkpoint após conclusão bem-sucedida
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"  🗑️  Checkpoint removido (execução completa)")

    # Análise de estabilidade
    stability = analyze_stability(results)

    # Estatísticas de early stopping
    generations_run = [r.get("generations_run", 0) for r in results if "generations_run" in r]
    if generations_run:
        avg_generations = np.mean(generations_run)
        early_stopped = sum(1 for r in results if r.get("converged_early", False))
        print(f"\n  ⚡ Early Stopping Stats:")
        print(f"     • Gerações médias: {avg_generations:.1f}")
        print(f"     • Convergência antecipada: {early_stopped}/{len(results)} runs ({early_stopped/len(results)*100:.1f}%)")

    # Carteira consenso
    consensus = build_consensus_portfolio(results, df_ranked, profile)

    # Melhor indivíduo (maior fitness)
    # Usa o melhor portfolio salvo ou reconstrói a partir dos results
    if best_portfolio_so_far is not None:
        best_individual = best_portfolio_so_far.copy()
        best_idx = max(range(len(results)), key=lambda i: results[i]["fitness"])
        best_individual.attrs["method"] = "best_individual"
        best_individual.attrs["run_id"] = results[best_idx]["run_id"]
        best_individual.attrs["seed"] = results[best_idx]["seed"]
    else:
        best_individual = get_best_individual_portfolio(results, portfolios=None, df_ranked=df_ranked)

    # Comparação entre carteiras
    comparison = compare_portfolios(consensus, best_individual, profile)

    # Comparação de métricas fundamentalistas
    fundamental_comparison = compare_fundamental_metrics(consensus, best_individual, profile)

    # Backtest comparativo (5 e 10 anos)
    backtest_5y = run_backtest_comparison(consensus, best_individual, profile, period_years=5)
    backtest_10y = run_backtest_comparison(consensus, best_individual, profile, period_years=10)

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

    # Salva comparação (inclui métricas, fundamentals e backtest)
    # Remove as séries temporais do backtest para tornar serializável
    def make_backtest_serializable(backtest_data):
        backtest_copy = backtest_data.copy()
        if backtest_copy:
            for key in ["consensus", "best_individual"]:
                if key in backtest_copy and backtest_copy[key]:
                    if "values" in backtest_copy[key]:
                        del backtest_copy[key]["values"]
        return backtest_copy

    backtest_5y_serializable = make_backtest_serializable(backtest_5y)
    backtest_10y_serializable = make_backtest_serializable(backtest_10y)

    full_comparison = {
        "metrics_comparison": comparison,
        "fundamental_comparison": fundamental_comparison,
        "backtest_5y": backtest_5y_serializable,
        "backtest_10y": backtest_10y_serializable
    }
    comparison_file = OUTPUTS_DIR / f"comparison_{profile}.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, ensure_ascii=False, indent=2)

    # Gera gráficos comparativos
    plot_comparison_charts(comparison, backtest_5y, profile)

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

    print(f"\n  📈 MÉTRICAS FUNDAMENTALISTAS:")
    print(f"     • Vencedor: {fundamental_comparison['overall_winner'].upper()}")
    print(f"     • Score: Consenso {fundamental_comparison['overall_score']['consensus']} x {fundamental_comparison['overall_score']['best_individual']} Melhor Indivíduo")
    for group, stats in fundamental_comparison['summary'].items():
        print(f"     • {group.capitalize()}: {stats['winner'].upper()} ({stats['consensus_wins']}-{stats['best_individual_wins']})")

    print(f"\n  💾 Carteira consenso: {consensus_file}")
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
        "fundamental_comparison": fundamental_comparison,
        "backtest_5y": backtest_5y,
        "backtest_10y": backtest_10y,
        "all_runs": results
    }


def run_multi_execution_all_profiles(
    n_runs: int = N_RUNS,
    parallel: bool = True,
    adaptive_mode: bool = False,
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
    adaptive_mode : bool
        Se True, para automaticamente quando atingir estabilidade.
    save_summary : bool
        Se True, salva summary consolidado.

    Returns
    -------
    Dict
        Resultados de todos os perfis.
    """
    print("=" * 70)
    print("PIPELINE: Múltiplas Execuções do Algoritmo Genético")
    print(f"Número máximo de execuções por perfil: {n_runs}")
    if adaptive_mode:
        print("Modo Adaptativo: ATIVADO (para quando convergir)")
    print("=" * 70)

    all_results = {}

    for profile in PROFILES:
        results = run_multi_execution_profile(
            profile=profile,
            n_runs=n_runs,
            parallel=parallel,
            adaptive_mode=adaptive_mode
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
