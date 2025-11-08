"""backtest_analysis.py
===============================================================================
Backtesting de Carteiras Otimizadas por Algoritmo Genético

Simula performance histórica das carteiras GA nos últimos 5 e 10 anos,
comparando com o Ibovespa (benchmark). Calcula métricas de risco-retorno
e gera visualizações para apresentação em TCC.

Estratégia: Buy-and-Hold equiponderado (sem rebalanceamento)
Data Source: yfinance (preços ajustados por dividendos e splits)

Autor: Análise Quantitativa para TCC
Data: 2025
===============================================================================
"""

from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Suprime warnings do yfinance
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance não encontrado. Instale com: pip install yfinance"
    )

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

PERIODS = {
    "5anos": 5,
    "10anos": 10
}

PROFILES = ["conservador", "moderado", "arrojado"]

INITIAL_CAPITAL = 10_000.0  # Capital inicial hipotético
SELIC_PROXY = 0.10  # 10% aa como taxa livre de risco simplificada
TRADING_DAYS = 252  # Dias úteis por ano

# Configurações de visualização
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

COLORS = {
    "conservador": "#2E7D32",  # Verde escuro
    "moderado": "#1976D2",     # Azul
    "arrojado": "#D32F2F",     # Vermelho
    "ibovespa": "#757575"      # Cinza
}

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================

def load_portfolio(perfil: str) -> List[str]:
    """
    Carrega lista de tickers de uma carteira GA.

    Args:
        perfil: Nome do perfil (conservador, moderado, arrojado)

    Returns:
        Lista de tickers (ex: ['WEGE3', 'ITUB4', ...])
    """
    portfolio_file = OUTPUTS_DIR / f"carteira_{perfil}_consensus.json"

    if not portfolio_file.exists():
        raise FileNotFoundError(
            f"Carteira não encontrada: {portfolio_file}\n"
            f"Execute build_portfolios_summary.py primeiro."
        )

    with open(portfolio_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickers = [asset["TICKER"] for asset in data]
    print(f"  ✓ {perfil.capitalize()}: {len(tickers)} ativos carregados")

    return tickers


def fetch_historical_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        benchmark: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Busca dados históricos do yfinance para múltiplos tickers.

    Args:
        tickers: Lista de tickers brasileiros (sem .SA)
        start_date: Data inicial (formato 'YYYY-MM-DD')
        end_date: Data final
        benchmark: Se True, não adiciona .SA (para ^BVSP)

    Returns:
        Tuple (DataFrame com preços ajustados, lista de tickers faltantes)
    """
    if benchmark:
        tickers_yf = tickers
    else:
        # Adiciona .SA para tickers brasileiros
        tickers_yf = [f"{t}.SA" for t in tickers]

    print(f"    Buscando dados de {start_date} a {end_date}...")

    try:
        # Para ticker único, use método diferente
        if len(tickers_yf) == 1:
            ticker = tickers_yf[0]
            data = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )

            if data.empty:
                print(f"    ✗ Nenhum dado retornado para {ticker}")
                return pd.DataFrame(), tickers_yf

            # Extrai coluna Close
            prices = pd.DataFrame({ticker: data['Close']})

        else:
            # Para múltiplos tickers
            data = yf.download(
                tickers_yf,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=True
            )

            if data.empty:
                raise ValueError("Nenhum dado retornado pelo yfinance")

            # Extrai Close
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data

        # Identifica tickers com dados faltantes
        missing = []
        valid_columns = []

        for ticker in tickers_yf:
            if ticker not in prices.columns:
                missing.append(ticker)
            elif prices[ticker].isna().all():
                missing.append(ticker)
            elif prices[ticker].count() < len(prices) * 0.5:  # < 50% de dados
                missing.append(ticker)
            else:
                valid_columns.append(ticker)

        # Filtra apenas colunas válidas
        prices = prices[valid_columns]

        # Forward-fill para preencher gaps de até 5 dias
        prices = prices.fillna(method='ffill', limit=5)

        # Remove linhas onde TODOS os ativos estão NaN
        prices = prices.dropna(how='all')

        if missing:
            print(f"    ⚠ {len(missing)} ativo(s) sem dados suficientes: {missing[:3]}{'...' if len(missing) > 3 else ''}")

        print(f"    ✓ {len(valid_columns)} ativos com dados completos")

        return prices, missing

    except Exception as e:
        print(f"    ✗ Erro ao buscar dados: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), tickers


# ============================================================================
# CÁLCULO DE PERFORMANCE DA CARTEIRA
# ============================================================================

def calculate_portfolio_value(
        prices: pd.DataFrame,
        initial_capital: float = INITIAL_CAPITAL
) -> pd.Series:
    """
    Calcula o valor da carteira ao longo do tempo (buy-and-hold equiponderado).

    Args:
        prices: DataFrame com preços diários dos ativos
        initial_capital: Capital inicial em R$

    Returns:
        Series com valor da carteira ao longo do tempo
    """
    n_assets = len(prices.columns)

    if n_assets == 0:
        return pd.Series(dtype=float)

    # Alocação equiponderada
    allocation_per_asset = initial_capital / n_assets

    # Preços normalizados (base 100 no primeiro dia)
    normalized_prices = prices / prices.iloc[0]

    # Valor de cada posição ao longo do tempo
    positions_value = normalized_prices * allocation_per_asset

    # Valor total da carteira (soma de todas as posições)
    portfolio_value = positions_value.sum(axis=1)

    return portfolio_value


def calculate_returns(values: pd.Series) -> pd.Series:
    """
    Calcula retornos percentuais diários.

    Args:
        values: Series com valores da carteira

    Returns:
        Series com retornos diários (%)
    """
    return values.pct_change().dropna()


def calculate_drawdown(values: pd.Series) -> pd.Series:
    """
    Calcula drawdown (queda acumulada desde o pico).

    Args:
        values: Series com valores da carteira

    Returns:
        Series com drawdown (%) em cada momento
    """
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100
    return drawdown


# ============================================================================
# MÉTRICAS DE PERFORMANCE
# ============================================================================

def calculate_metrics(
        values: pd.Series,
        returns: pd.Series,
        years: float
) -> Dict[str, float]:
    """
    Calcula métricas de performance da carteira.

    Args:
        values: Series com valores da carteira
        returns: Series com retornos diários
        years: Número de anos do período

    Returns:
        Dicionário com todas as métricas
    """
    if len(values) == 0 or len(returns) == 0:
        return {
            "retorno_total_pct": 0.0,
            "retorno_anualizado_pct": 0.0,
            "volatilidade_anual_pct": 0.0,
            "sharpe_ratio": 0.0,
            "drawdown_maximo_pct": 0.0,
            "calmar_ratio": 0.0,
            "retorno_mensal_medio_pct": 0.0
        }

    # Retorno total
    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100

    # Retorno anualizado (CAGR)
    annualized_return = ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100

    # Volatilidade anualizada
    volatility = returns.std() * np.sqrt(TRADING_DAYS) * 100

    # Sharpe Ratio (usando Selic como proxy de taxa livre de risco)
    excess_return = annualized_return - SELIC_PROXY * 100
    sharpe = excess_return / volatility if volatility > 0 else 0.0

    # Drawdown máximo
    drawdown = calculate_drawdown(values)
    max_drawdown = drawdown.min()

    # Calmar Ratio (retorno anualizado / drawdown máximo absoluto)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # Retorno mensal médio
    monthly_returns = values.resample('M').last().pct_change().dropna() * 100
    avg_monthly_return = monthly_returns.mean()

    return {
        "retorno_total_pct": round(total_return, 2),
        "retorno_anualizado_pct": round(annualized_return, 2),
        "volatilidade_anual_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 3),
        "drawdown_maximo_pct": round(max_drawdown, 2),
        "calmar_ratio": round(calmar, 3),
        "retorno_mensal_medio_pct": round(avg_monthly_return, 2)
    }


def calculate_benchmark_metrics(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Calcula métricas relativas ao benchmark (Alpha, Beta, Information Ratio).

    Args:
        portfolio_returns: Retornos diários da carteira
        benchmark_returns: Retornos diários do benchmark

    Returns:
        Dicionário com métricas relativas
    """
    # Alinha séries temporais
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    aligned.columns = ['portfolio', 'benchmark']
    aligned = aligned.dropna()

    if len(aligned) < 30:  # Dados insuficientes
        return {
            "beta": 0.0,
            "alpha_anualizado_pct": 0.0,
            "information_ratio": 0.0
        }

    # Beta (regressão linear)
    covariance = aligned['portfolio'].cov(aligned['benchmark'])
    benchmark_variance = aligned['benchmark'].var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

    # Alpha anualizado (Jensen's Alpha)
    portfolio_mean = aligned['portfolio'].mean() * TRADING_DAYS
    benchmark_mean = aligned['benchmark'].mean() * TRADING_DAYS
    alpha = (portfolio_mean - SELIC_PROXY - beta * (benchmark_mean - SELIC_PROXY)) * 100

    # Information Ratio (tracking error)
    excess_returns = aligned['portfolio'] - aligned['benchmark']
    tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS)
    info_ratio = excess_returns.mean() * TRADING_DAYS / tracking_error if tracking_error > 0 else 0.0

    return {
        "beta": round(beta, 3),
        "alpha_anualizado_pct": round(alpha, 2),
        "information_ratio": round(info_ratio, 3)
    }


# ============================================================================
# PIPELINE DE BACKTESTING
# ============================================================================

def run_backtest(
        perfil: str,
        period_years: int,
        end_date: Optional[datetime] = None
) -> Dict:
    """
    Executa backtesting completo para uma carteira.

    Args:
        perfil: Nome do perfil (conservador, moderado, arrojado, ibovespa)
        period_years: Número de anos para simular
        end_date: Data final (default: hoje)

    Returns:
        Dicionário com valores, retornos e métricas
    """
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=period_years * 365 + 30)  # Margem

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Ibovespa (benchmark)
    if perfil == "ibovespa":
        print("  Carregando Ibovespa (^BVSP)...")
        prices, missing = fetch_historical_data(
            ["^BVSP"],
            start_str,
            end_str,
            benchmark=True
        )

        if prices.empty:
            print("    ✗ Erro: Ibovespa não disponível")
            return {
                "values": pd.Series(dtype=float),
                "returns": pd.Series(dtype=float),
                "metrics": {},
                "benchmark_metrics": {},
                "missing_tickers": ["^BVSP"]
            }

        values = calculate_portfolio_value(prices)

    # Carteiras GA
    else:
        tickers = load_portfolio(perfil)
        prices, missing = fetch_historical_data(tickers, start_str, end_str)

        if prices.empty or len(prices.columns) == 0:
            print(f"    ✗ Erro: Nenhum ativo disponível para {perfil}")
            return {
                "values": pd.Series(dtype=float),
                "returns": pd.Series(dtype=float),
                "metrics": {},
                "benchmark_metrics": {},
                "missing_tickers": missing
            }

        values = calculate_portfolio_value(prices)

    # Cálculos
    returns = calculate_returns(values)
    metrics = calculate_metrics(values, returns, period_years)

    print(f"    📊 Retorno: {metrics['retorno_anualizado_pct']:.2f}% aa | "
          f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
          f"Drawdown: {metrics['drawdown_maximo_pct']:.2f}%")

    return {
        "values": values,
        "returns": returns,
        "metrics": metrics,
        "benchmark_metrics": {},
        "missing_tickers": missing if perfil != "ibovespa" else []
    }


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

def plot_evolution(results: Dict, period: str):
    """
    Gráfico 1: Evolução do capital ao longo do tempo.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for perfil in PROFILES + ["ibovespa"]:
        values = results[perfil]["values"]
        if not values.empty:
            ax.plot(
                values.index,
                values,
                label=perfil.capitalize(),
                color=COLORS[perfil],
                linewidth=2.5 if perfil != "ibovespa" else 2,
                linestyle='-' if perfil != "ibovespa" else '--'
            )

    ax.set_xlabel("Data", fontsize=12, fontweight='bold')
    ax.set_ylabel("Valor da Carteira (R$)", fontsize=12, fontweight='bold')
    ax.set_title(
        f"Performance Histórica - Carteiras GA vs. Ibovespa ({period})",
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Formatação do eixo X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    # Linha horizontal no capital inicial
    ax.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5, label='Capital Inicial')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"backtest_evolution_{period}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: backtest_evolution_{period}.png")


def plot_returns_comparison(results: Dict, period: str):
    """
    Gráfico 2: Comparação de retornos totais (barras).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = []
    returns = []
    colors_list = []

    for perfil in PROFILES + ["ibovespa"]:
        ret = results[perfil]["metrics"].get("retorno_total_pct", 0)
        labels.append(perfil.capitalize())
        returns.append(ret)
        colors_list.append('green' if ret > 0 else 'red')

    bars = ax.bar(labels, returns, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Annotations
    for i, (bar, ret) in enumerate(zip(bars, returns)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (5 if height > 0 else -15),
            f'{ret:.1f}%',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=12,
            fontweight='bold'
        )

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Retorno Total (%)", fontsize=12, fontweight='bold')
    ax.set_title(
        f"Comparação de Retornos Totais ({period})",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"backtest_returns_{period}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: backtest_returns_{period}.png")


def plot_risk_return(results: Dict, period: str):
    """
    Gráfico 3: Risco vs. Retorno (scatter plot).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for perfil in PROFILES + ["ibovespa"]:
        metrics = results[perfil]["metrics"]
        vol = metrics.get("volatilidade_anual_pct", 0)
        ret = metrics.get("retorno_anualizado_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        # Tamanho proporcional ao Sharpe
        size = max(100, abs(sharpe) * 300)

        ax.scatter(
            vol,
            ret,
            s=size,
            color=COLORS[perfil],
            alpha=0.6,
            edgecolors='black',
            linewidth=2,
            label=perfil.capitalize()
        )

        # Annotation
        ax.annotate(
            perfil.capitalize(),
            (vol, ret),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS[perfil], alpha=0.3)
        )

    ax.set_xlabel("Volatilidade Anualizada (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Retorno Anualizado (%)", fontsize=12, fontweight='bold')
    ax.set_title(
        f"Risco vs. Retorno - Tamanho = Sharpe Ratio ({period})",
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Legenda customizada
    legend_elements = [
        Patch(facecolor=COLORS[p], label=p.capitalize(), alpha=0.6)
        for p in PROFILES + ["ibovespa"]
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"backtest_risk_return_{period}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: backtest_risk_return_{period}.png")


def plot_drawdowns(results: Dict, period: str):
    """
    Gráfico 4: Drawdown ao longo do tempo (subplots).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, perfil in enumerate(PROFILES + ["ibovespa"]):
        values = results[perfil]["values"]

        if not values.empty:
            drawdown = calculate_drawdown(values)

            axes[idx].fill_between(
                drawdown.index,
                0,
                drawdown,
                color='red',
                alpha=0.3,
                label='Drawdown'
            )

            axes[idx].plot(
                drawdown.index,
                drawdown,
                color='darkred',
                linewidth=1.5
            )

            axes[idx].set_title(
                f"{perfil.capitalize()} - Drawdown Máximo: {drawdown.min():.2f}%",
                fontsize=12,
                fontweight='bold'
            )
            axes[idx].set_ylabel("Drawdown (%)", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(
        f"Análise de Drawdown - {period}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"backtest_drawdowns_{period}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: backtest_drawdowns_{period}.png")


def plot_radar_metrics(results: Dict, period: str):
    """
    Gráfico 5: Radar chart comparativo de métricas normalizadas.
    """
    from math import pi

    # Métricas para comparar (4 dimensões)
    metrics_keys = [
        "retorno_anualizado_pct",
        "sharpe_ratio",
        "calmar_ratio",
        "information_ratio"
    ]

    labels = [
        "Retorno\nAnualizado",
        "Sharpe\nRatio",
        "Calmar\nRatio",
        "Info\nRatio"
    ]

    # Normalização 0-10 para cada métrica
    def normalize(values_list):
        min_val = min(values_list)
        max_val = max(values_list)
        range_val = max_val - min_val
        if range_val == 0:
            return [5.0] * len(values_list)
        return [((v - min_val) / range_val) * 10 for v in values_list]

    # Coleta dados
    data = {perfil: [] for perfil in PROFILES + ["ibovespa"]}

    for key in metrics_keys:
        values = []
        for perfil in PROFILES + ["ibovespa"]:
            if key == "information_ratio":
                val = results[perfil].get("benchmark_metrics", {}).get(key, 0)
            else:
                val = results[perfil]["metrics"].get(key, 0)
            values.append(val)

        normalized = normalize(values)
        for perfil, norm_val in zip(PROFILES + ["ibovespa"], normalized):
            data[perfil].append(norm_val)

    # Plotagem
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for perfil in PROFILES + ["ibovespa"]:
        values = data[perfil]
        values += values[:1]

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            label=perfil.capitalize(),
            color=COLORS[perfil]
        )
        ax.fill(angles, values, alpha=0.15, color=COLORS[perfil])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='gray')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title(
        f"Comparação Multidimensional de Métricas ({period})\n(Escala normalizada 0-10)",
        fontsize=14,
        fontweight='bold',
        pad=30
    )

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"backtest_radar_{period}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: backtest_radar_{period}.png")


def plot_all(results: Dict, period: str):
    """
    Gera todos os gráficos.
    """
    print(f"\n  Gerando gráficos para {period}...")
    plot_evolution(results, period)
    plot_returns_comparison(results, period)
    plot_risk_return(results, period)
    plot_drawdowns(results, period)
    plot_radar_metrics(results, period)


# ============================================================================
# SALVAMENTO DE MÉTRICAS
# ============================================================================

def save_metrics(results: Dict, period: str):
    """
    Salva métricas em JSON.
    """
    end_date = datetime.now()
    period_years = PERIODS[period]
    start_date = end_date - timedelta(days=period_years * 365)

    output = {
        "periodo": period,
        "anos": period_years,
        "data_inicio": start_date.strftime("%Y-%m-%d"),
        "data_fim": end_date.strftime("%Y-%m-%d"),
        "capital_inicial": INITIAL_CAPITAL
    }

    for perfil in PROFILES + ["ibovespa"]:
        output[perfil] = {
            **results[perfil]["metrics"],
            **results[perfil].get("benchmark_metrics", {}),
            "ativos_faltantes": results[perfil].get("missing_tickers", [])
        }

    output_file = OUTPUTS_DIR / f"backtest_metrics_{period}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Métricas salvas: backtest_metrics_{period}.json")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """
    Executa análise completa de backtesting para todos os perfis e períodos.
    """
    print("\n" + "="*70)
    print(" BACKTESTING DE CARTEIRAS OTIMIZADAS POR ALGORITMO GENÉTICO")
    print("="*70)

    for period_name, years in PERIODS.items():
        print(f"\n{'='*70}")
        print(f" PERÍODO: {period_name.upper()} ({years} anos)")
        print(f"{'='*70}\n")

        results = {}

        # Executa backtesting para cada perfil
        for perfil in PROFILES:
            print(f"\n📊 Processando: {perfil.upper()}")
            results[perfil] = run_backtest(perfil, years)

        # Adiciona Ibovespa como benchmark
        print(f"\n📊 Processando: IBOVESPA (Benchmark)")
        results["ibovespa"] = run_backtest("ibovespa", years)

        # Calcula métricas relativas ao benchmark
        print("\n  Calculando métricas relativas ao benchmark...")
        for perfil in PROFILES:
            if not results[perfil]["returns"].empty and not results["ibovespa"]["returns"].empty:
                bench_metrics = calculate_benchmark_metrics(
                    results[perfil]["returns"],
                    results["ibovespa"]["returns"]
                )
                results[perfil]["benchmark_metrics"] = bench_metrics

                print(f"    {perfil.capitalize()}: "
                      f"Beta={bench_metrics['beta']:.2f}, "
                      f"Alpha={bench_metrics['alpha_anualizado_pct']:.2f}%, "
                      f"IR={bench_metrics['information_ratio']:.2f}")

        # Gera visualizações
        plot_all(results, period_name)

        # Salva métricas
        save_metrics(results, period_name)

    print("\n" + "="*70)
    print(" ✅ ANÁLISE COMPLETA!")
    print(" 📁 Todos os arquivos salvos em: outputs/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()