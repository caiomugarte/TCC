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

# Configura matplotlib para backend não-interativo (evita problemas com Tkinter)
import matplotlib
matplotlib.use('Agg')
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

# Usa o mesmo caminho que config.py (raiz do projeto)
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

PERIODS = {
    "5anos": 5,
    "10anos": 10
}

PROFILES = ["conservador", "moderado", "arrojado", "caio", "caio2"]

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
    "ibovespa": "#757575",      # Cinza
    "caio": "#3503ff",
    "caio2": "#3CDBD3"
}

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================


def analyze_asset_performance(
        tickers: List[str],
        start_date: str,
        end_date: str,
        period_years: int
) -> pd.DataFrame:
    """
    Analisa performance individual de cada ativo com e sem dividendos.

    Args:
        tickers: Lista de tickers brasileiros (sem .SA)
        start_date: Data inicial
        end_date: Data final
        period_years: Número de anos

    Returns:
        DataFrame com métricas por ativo
    """
    results = []

    for ticker in tickers:
        ticker_yf = f"{ticker}.SA"

        try:
            # 1. COM DIVIDENDOS (auto_adjust=True)
            data_adjusted = yf.Ticker(ticker_yf).history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )

            if data_adjusted.empty or len(data_adjusted) < 30:
                print(f"    ⚠ {ticker}: Dados insuficientes")
                continue

            # 2. SEM DIVIDENDOS (auto_adjust=False)
            data_raw = yf.Ticker(ticker_yf).history(
                start=start_date,
                end=end_date,
                auto_adjust=False
            )

            # 3. DIVIDENDOS PAGOS
            dividends = yf.Ticker(ticker_yf).dividends
            dividends_period = dividends[
                (dividends.index >= start_date) &
                (dividends.index <= end_date)
                ]

            # Cálculos - COM DIVIDENDOS
            price_initial_adj = data_adjusted['Close'].iloc[0]
            price_final_adj = data_adjusted['Close'].iloc[-1]

            total_return_adj = (price_final_adj / price_initial_adj - 1) * 100
            annual_return_adj = ((price_final_adj / price_initial_adj) ** (1 / period_years) - 1) * 100

            # Cálculos - SEM DIVIDENDOS
            price_initial_raw = data_raw['Close'].iloc[0]
            price_final_raw = data_raw['Close'].iloc[-1]

            total_return_raw = (price_final_raw / price_initial_raw - 1) * 100
            annual_return_raw = ((price_final_raw / price_initial_raw) ** (1 / period_years) - 1) * 100

            # Efeito dos dividendos
            dividend_effect_total = total_return_adj - total_return_raw
            dividend_effect_annual = annual_return_adj - annual_return_raw

            # Total de dividendos pagos
            total_dividends = dividends_period.sum()
            dividend_yield_period = (total_dividends / price_initial_raw) * 100 if price_initial_raw > 0 else 0

            # Volatilidade
            returns_adj = data_adjusted['Close'].pct_change().dropna()
            volatility = returns_adj.std() * np.sqrt(TRADING_DAYS) * 100

            results.append({
                'TICKER': ticker,
                'preco_inicial': round(price_initial_raw, 2),
                'preco_final': round(price_final_raw, 2),
                'retorno_total_sem_div_pct': round(total_return_raw, 2),
                'retorno_total_com_div_pct': round(total_return_adj, 2),
                'retorno_anual_sem_div_pct': round(annual_return_raw, 2),
                'retorno_anual_com_div_pct': round(annual_return_adj, 2),
                'efeito_dividendos_total_pct': round(dividend_effect_total, 2),
                'efeito_dividendos_anual_pct': round(dividend_effect_annual, 2),
                'dividendos_totais_pagos': round(total_dividends, 2),
                'dividend_yield_periodo_pct': round(dividend_yield_period, 2),
                'volatilidade_anual_pct': round(volatility, 2),
                'n_dias_negociacao': len(data_adjusted)
            })

        except Exception as e:
            print(f"    ✗ {ticker}: Erro - {e}")
            continue

    df = pd.DataFrame(results)

    # Ordena por retorno total com dividendos (decrescente)
    if not df.empty:
        df = df.sort_values('retorno_total_com_div_pct', ascending=False)

    return df

def plot_asset_comparison(
        asset_data: pd.DataFrame,
        perfil: str,
        period: str
):
    """
    Gráfico: Comparação de retornos por ativo (com vs. sem dividendos).
    """
    if asset_data.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ===== GRÁFICO 1: Retorno Total (Barras Comparativas) =====
    ax1 = axes[0]

    x = np.arange(len(asset_data))
    width = 0.35

    bars1 = ax1.bar(
        x - width/2,
        asset_data['retorno_total_sem_div_pct'],
        width,
        label='Sem Dividendos (Preço)',
        color='steelblue',
        alpha=0.7
    )

    bars2 = ax1.bar(
        x + width/2,
        asset_data['retorno_total_com_div_pct'],
        width,
        label='Com Dividendos (Total Return)',
        color='darkgreen',
        alpha=0.7
    )

    ax1.set_xlabel('Ativo', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Retorno Total (%)', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'Retorno Total por Ativo - {perfil.capitalize()} ({period})',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(asset_data['TICKER'], rotation=45, ha='right')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Annotations nos valores
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        # Só anota se houver espaço
        if abs(height2 - height1) > 5:
            ax1.text(
                bar2.get_x() + bar2.get_width() / 2,
                height2,
                f'{height2:.0f}%',
                ha='center',
                va='bottom' if height2 > 0 else 'top',
                fontsize=8,
                fontweight='bold'
            )

    # ===== GRÁFICO 2: Efeito dos Dividendos (Barras Empilhadas) =====
    ax2 = axes[1]

    bars_price = ax2.barh(
        asset_data['TICKER'],
        asset_data['retorno_total_sem_div_pct'],
        label='Valorização de Preço',
        color='steelblue',
        alpha=0.7
    )

    bars_div = ax2.barh(
        asset_data['TICKER'],
        asset_data['efeito_dividendos_total_pct'],
        left=asset_data['retorno_total_sem_div_pct'],
        label='Efeito dos Dividendos',
        color='orange',
        alpha=0.7
    )

    ax2.set_xlabel('Retorno Total (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ativo', fontsize=11, fontweight='bold')
    ax2.set_title(
        f'Decomposição: Preço vs. Dividendos - {perfil.capitalize()} ({period})',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(axis='x', alpha=0.3)

    # Annotations no total
    for i, (ticker, total, div_effect) in enumerate(zip(
            asset_data['TICKER'],
            asset_data['retorno_total_com_div_pct'],
            asset_data['efeito_dividendos_total_pct']
    )):
        ax2.text(
            total + 2,
            i,
            f'{total:.1f}% (div: {div_effect:.1f}%)',
            va='center',
            fontsize=8,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(
        OUTPUTS_DIR / f"backtest_assets_{perfil}_{period}.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    print(f"  ✓ Gráfico de ativos salvo: backtest_assets_{perfil}_{period}.png")

def plot_dividend_contribution(
        asset_data: pd.DataFrame,
        perfil: str,
        period: str
):
    """
    Gráfico: Contribuição dos dividendos para o retorno total (pizza).
    """
    if asset_data.empty:
        return

    # Calcula contribuição percentual dos dividendos
    asset_data['contrib_dividendos'] = (
            asset_data['efeito_dividendos_total_pct'] /
            asset_data['retorno_total_com_div_pct'] * 100
    ).fillna(0)

    # Limita entre 0 e 100%
    asset_data['contrib_dividendos'] = asset_data['contrib_dividendos'].clip(0, 100)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Top 10 ativos por contribuição de dividendos
    top_10 = asset_data.nlargest(10, 'efeito_dividendos_total_pct')

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10)))

    wedges, texts, autotexts = ax.pie(
        top_10['efeito_dividendos_total_pct'],
        labels=top_10['TICKER'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )

    ax.set_title(
        f'Contribuição dos Dividendos por Ativo\n{perfil.capitalize()} ({period})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUTS_DIR / f"backtest_dividends_{perfil}_{period}.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    print(f"  ✓ Gráfico de dividendos salvo: backtest_dividends_{perfil}_{period}.png")

def save_asset_metrics(
        asset_data: pd.DataFrame,
        perfil: str,
        period: str
):
    """
    Salva métricas individuais de ativos em CSV e JSON.
    """
    if asset_data.empty:
        return

    # CSV (para análise em Excel)
    csv_file = OUTPUTS_DIR / f"backtest_assets_{perfil}_{period}.csv"
    asset_data.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV de ativos salvo: {csv_file.name}")

    # JSON (para integração)
    json_file = OUTPUTS_DIR / f"backtest_assets_{perfil}_{period}.json"
    asset_data.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f"  ✓ JSON de ativos salvo: {json_file.name}")

    # Estatísticas resumidas
    summary = {
        'perfil': perfil,
        'periodo': period,
        'n_ativos': len(asset_data),
        'estatisticas': {
            'retorno_medio_com_div_pct': round(asset_data['retorno_total_com_div_pct'].mean(), 2),
            'retorno_medio_sem_div_pct': round(asset_data['retorno_total_sem_div_pct'].mean(), 2),
            'efeito_dividendos_medio_pct': round(asset_data['efeito_dividendos_total_pct'].mean(), 2),
            'melhor_ativo_com_div': asset_data.iloc[0]['TICKER'],
            'melhor_retorno_com_div_pct': round(asset_data.iloc[0]['retorno_total_com_div_pct'], 2),
            'pior_ativo_com_div': asset_data.iloc[-1]['TICKER'],
            'pior_retorno_com_div_pct': round(asset_data.iloc[-1]['retorno_total_com_div_pct'], 2),
            'maior_pagador_dividendos': asset_data.nlargest(1, 'dividendos_totais_pagos').iloc[0]['TICKER'],
            'dividendos_totais_periodo': round(asset_data['dividendos_totais_pagos'].sum(), 2),
        }
    }

    return summary

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

    # Verifica se o arquivo não está vazio
    if portfolio_file.stat().st_size == 0:
        raise ValueError(
            f"Arquivo de carteira está vazio: {portfolio_file}\n"
            f"Execute build_portfolios_summary.py novamente."
        )

    try:
        with open(portfolio_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Debug: mostra primeiros caracteres
            if not content.strip():
                raise ValueError(
                    f"Arquivo contém apenas espaços em branco: {portfolio_file}"
                )

            # Tenta fazer parse do JSON
            data = json.loads(content)

            if not data:
                raise ValueError(
                    f"JSON válido mas vazio: {portfolio_file}\n"
                    f"Execute build_portfolios_summary.py novamente."
                )

            # Extrai tickers
            if isinstance(data, list):
                tickers = [asset["TICKER"] for asset in data]
            elif isinstance(data, dict):
                # Caso o formato seja diferente
                tickers = [asset["TICKER"] for asset in data.get("ativos", [])]
            else:
                raise ValueError(
                    f"Formato JSON não reconhecido em {portfolio_file}\n"
                    f"Esperado: lista de objetos com campo 'TICKER'"
                )

            if not tickers:
                raise ValueError(
                    f"Nenhum ticker encontrado em {portfolio_file}\n"
                    f"Execute build_portfolios_summary.py novamente."
                )

            print(f"  ✓ {perfil.capitalize()}: {len(tickers)} ativos carregados")
            return tickers

    except json.JSONDecodeError as e:
        print(f"\n❌ ERRO: Arquivo JSON corrompido ou inválido")
        print(f"   Arquivo: {portfolio_file}")
        print(f"   Erro: {e}")
        print(f"\n💡 SOLUÇÃO: Execute os seguintes comandos na ordem:")
        print(f"   1. python data_preprocessing.py")
        print(f"   2. python build_portfolios_summary.py")
        print(f"   3. python backtest_analysis.py")
        raise

    except KeyError as e:
        print(f"\n❌ ERRO: Campo obrigatório não encontrado no JSON")
        print(f"   Arquivo: {portfolio_file}")
        print(f"   Campo faltando: {e}")
        print(f"\n💡 SOLUÇÃO: Execute build_portfolios_summary.py novamente")
        raise

    except Exception as e:
        print(f"\n❌ ERRO inesperado ao carregar carteira")
        print(f"   Arquivo: {portfolio_file}")
        print(f"   Erro: {e}")
        raise


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

def get_ibovespa_tickers() -> List[str]:
    """
    Retorna lista de tickers que compõem o Ibovespa.

    Returns:
        Lista de tickers (sem .SA)
    """
    # Composição do Ibovespa (atualizada 2024)
    # Fonte: B3 - http://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/ibovespa.htm
    ibov_tickers = [
        "ABEV3", "ALPA4", "AMER3", "ASAI3", "AZUL4", "B3SA3", "BBAS3", "BBDC3",
        "BBDC4", "BBSE3", "BEEF3", "BPAC11", "BRAP4", "BRFS3", "BRKM5", "CASH3",
        "CCRO3", "CIEL3", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CPLE6", "CRFB3",
        "CSAN3", "CSNA3", "CVCB3", "CYRE3", "DXCO3", "ECOR3", "EGIE3", "ELET3",
        "ELET6", "EMBR3", "ENBR3", "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3",
        "GGBR4", "GOAU4", "GOLL4", "HAPV3", "HYPE3", "IGTI11", "IRBR3", "ITSA4",
        "ITUB4", "JBSS3", "KLBN11", "LREN3", "LWSA3", "MGLU3", "MRFG3", "MRVE3",
        "MULT3", "NTCO3", "PCAR3", "PETR3", "PETR4", "PETZ3", "PRIO3", "RADL3",
        "RAIL3", "RAIZ4", "RDOR3", "RECV3", "RENT3", "RRRP3", "SANB11", "SBSP3",
        "SLCE3", "SMTO3", "SOMA3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3",
        "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVT3", "WEGE3", "YDUQ3"
    ]

    return ibov_tickers

# ============================================================================
# PIPELINE DE BACKTESTING
# ============================================================================

def run_backtest(
        perfil: str,
        period_years: int,
        end_date: Optional[datetime] = None,
        analyze_assets: bool = True
) -> Dict:
    """
    Executa backtesting completo para uma carteira.

    Args:
        perfil: Nome do perfil (conservador, moderado, arrojado, ibovespa)
        period_years: Número de anos para simular
        end_date: Data final (default: hoje)
        analyze_assets: Se True, analisa ativos individualmente

    Returns:
        Dicionário com valores, retornos, métricas e análise de ativos
    """
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=period_years * 365 + 30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # ============ MODIFICAÇÃO AQUI ============
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
                "missing_tickers": ["^BVSP"],
                "asset_analysis": pd.DataFrame()
            }

        values = calculate_portfolio_value(prices)

        # ===== NOVO: Análise dos ativos do Ibovespa =====
        if analyze_assets:
            print(f"    Analisando composição do Ibovespa...")
            ibov_tickers = get_ibovespa_tickers()  # Lista dos ativos do índice
            asset_analysis = analyze_asset_performance(
                ibov_tickers,
                start_str,
                end_str,
                period_years
            )
        else:
            asset_analysis = pd.DataFrame()

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
                "missing_tickers": missing,
                "asset_analysis": pd.DataFrame()
            }

        values = calculate_portfolio_value(prices)

        # Análise individual de ativos
        if analyze_assets:
            print(f"    Analisando {len(tickers)} ativos individualmente...")
            asset_analysis = analyze_asset_performance(
                tickers,
                start_str,
                end_str,
                period_years
            )
        else:
            asset_analysis = pd.DataFrame()

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
        "missing_tickers": missing if perfil != "ibovespa" else [],
        "asset_analysis": asset_analysis
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
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
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

    # Oculta o subplot vazio (temos 5 perfis em 6 subplots)
    if len(PROFILES + ["ibovespa"]) < len(axes):
        for idx in range(len(PROFILES + ["ibovespa"]), len(axes)):
            axes[idx].set_visible(False)

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
    Gráfico 5: Radar chart comparativo de métricas normalizadas INDIVIDUALMENTE.
    """
    from math import pi

    # Métricas para comparar
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

    # Ranges de referência por métrica (mercado brasileiro)
    metric_ranges = {
        "retorno_anualizado_pct": {"min": -10, "max": 30},
        "sharpe_ratio": {"min": -0.5, "max": 2.0},
        "calmar_ratio": {"min": -0.5, "max": 1.5},
        "information_ratio": {"min": -1.0, "max": 1.0}
    }

    def normalize_individual(value, metric_key):
        """Normaliza valor individual baseado no range da métrica."""
        range_def = metric_ranges.get(metric_key, {"min": 0, "max": 10})
        min_val = range_def["min"]
        max_val = range_def["max"]

        # Clip para evitar valores fora do range
        value = max(min_val, min(value, max_val))

        # Normaliza 0-10
        if max_val == min_val:
            return 5.0
        normalized = ((value - min_val) / (max_val - min_val)) * 10
        return normalized

    # Coleta e normaliza dados
    data = {perfil: [] for perfil in PROFILES + ["ibovespa"]}

    for key in metrics_keys:
        for perfil in PROFILES + ["ibovespa"]:
            if key == "information_ratio":
                val = results[perfil].get("benchmark_metrics", {}).get(key, 0)
            else:
                val = results[perfil]["metrics"].get(key, 0)

            # Normaliza individualmente por métrica
            normalized = normalize_individual(val, key)
            data[perfil].append(normalized)

    # Plotagem
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for perfil in PROFILES + ["ibovespa"]:
        values = data[perfil]
        values += values[:1]

        linewidth = 2.5 if perfil != "ibovespa" else 2
        linestyle = '-' if perfil != "ibovespa" else '--'
        alpha_fill = 0.15 if perfil != "ibovespa" else 0.05

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=linewidth,
            linestyle=linestyle,
            label=perfil.capitalize(),
            color=COLORS[perfil],
            markersize=8 if perfil != "ibovespa" else 6
        )
        ax.fill(angles, values, alpha=alpha_fill, color=COLORS[perfil])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='gray')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Linha de referência (valor neutro = 5)
    ax.plot(angles, [5]*len(angles), 'k:', alpha=0.3, linewidth=1)

    ax.set_title(
        f"Comparação Multidimensional de Métricas ({period})\n"
        f"(Normalização individual por métrica - Ranges: Sharpe [-0.5, 2.0], Calmar [-0.5, 1.5], IR [-1, 1], Retorno [-10%, 30%])",
        fontsize=12,
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
    print(f"\n  Gerando gráficos consolidados para {period}...")
    plot_evolution(results, period)
    plot_returns_comparison(results, period)
    plot_risk_return(results, period)
    plot_drawdowns(results, period)
    plot_radar_metrics(results, period)

    # ===== NOVO: Gráfico comparativo de ativos =====
    plot_portfolios_vs_ibov_assets(results, period)


# ============================================================================
# SALVAMENTO DE MÉTRICAS
# ============================================================================

def save_metrics(results: Dict, period: str, asset_summaries: Dict = None):
    """
    Salva métricas em JSON (agora inclui resumo de ativos).
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

        # Adiciona resumo de ativos (se disponível)
        if asset_summaries and perfil in asset_summaries:
            output[perfil]["analise_ativos"] = asset_summaries[perfil]

    output_file = OUTPUTS_DIR / f"backtest_metrics_{period}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Métricas salvas: backtest_metrics_{period}.json")

def verify_portfolio_files():
    """Verifica se os arquivos de carteira existem e são válidos."""
    print("\n🔍 Verificando arquivos de carteira...\n")

    all_ok = True

    for perfil in PROFILES:
        arquivo = OUTPUTS_DIR / f"carteira_{perfil}_consensus.json"

        if not arquivo.exists():
            print(f"❌ {perfil}: Arquivo não encontrado")
            all_ok = False
        elif arquivo.stat().st_size == 0:
            print(f"❌ {perfil}: Arquivo vazio")
            all_ok = False
        else:
            try:
                with open(arquivo, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data:
                        print(f"✅ {perfil}: {len(data)} ativos encontrados")
                    else:
                        print(f"❌ {perfil}: JSON vazio")
                        all_ok = False
            except Exception as e:
                print(f"❌ {perfil}: Erro ao ler - {e}")
                all_ok = False

    if not all_ok:
        print("\n" + "="*70)
        print("⚠️  ATENÇÃO: Alguns arquivos de carteira estão faltando ou inválidos")
        print("="*70)
        print("\nExecute os seguintes comandos na ordem:")
        print("  1. python data_preprocessing.py")
        print("  2. python build_portfolios_summary.py")
        print("  3. python backtest_analysis.py")
        print("\n" + "="*70 + "\n")
        return False

    print("\n✅ Todos os arquivos de carteira estão OK!\n")
    return True


def main():
    """
    Executa análise completa de backtesting para todos os perfis e períodos.
    """
    print("\n" + "="*70)
    print(" BACKTESTING DE CARTEIRAS OTIMIZADAS POR ALGORITMO GENÉTICO")
    print("="*70)

    # Verifica arquivos antes de começar
    if not verify_portfolio_files():
        return  # Interrompe se houver problema

    # ... resto do código ...
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

    # Verifica arquivos antes de começar
    if not verify_portfolio_files():
        return

    for period_name, years in PERIODS.items():
        print(f"\n{'='*70}")
        print(f" PERÍODO: {period_name.upper()} ({years} anos)")
        print(f"{'='*70}\n")

        results = {}
        asset_summaries = {}

        # Executa backtesting para cada perfil
        for perfil in PROFILES:
            print(f"\n📊 Processando: {perfil.upper()}")
            results[perfil] = run_backtest(
                perfil,
                years,
                analyze_assets=True
            )

            # Gera gráficos e salva dados de ativos
            asset_data = results[perfil]["asset_analysis"]
            if not asset_data.empty:
                print(f"\n  📈 Gerando análise de ativos para {perfil}...")
                plot_asset_comparison(asset_data, perfil, period_name)
                plot_dividend_contribution(asset_data, perfil, period_name)
                asset_summaries[perfil] = save_asset_metrics(asset_data, perfil, period_name)

        # ============ MODIFICAÇÃO AQUI ============
        # Adiciona Ibovespa como benchmark (COM análise de ativos)
        print(f"\n📊 Processando: IBOVESPA (Benchmark)")
        results["ibovespa"] = run_backtest(
            "ibovespa",
            years,
            analyze_assets=True  # ← MUDOU DE False PARA True
        )

        # ===== NOVO: Gera gráficos dos ativos do Ibovespa =====
        asset_data_ibov = results["ibovespa"]["asset_analysis"]
        if not asset_data_ibov.empty:
            print(f"\n  📈 Gerando análise de ativos do Ibovespa...")

            # Limita aos top 15 ativos para visualização
            top_15_ibov = asset_data_ibov.head(15)

            plot_asset_comparison(top_15_ibov, "ibovespa_top15", period_name)
            plot_dividend_contribution(top_15_ibov, "ibovespa_top15", period_name)
            asset_summaries["ibovespa"] = save_asset_metrics(asset_data_ibov, "ibovespa", period_name)

            print(f"  📊 Top 5 ativos do Ibovespa por retorno:")
            for i, row in asset_data_ibov.head(5).iterrows():
                print(f"    {i+1}. {row['TICKER']}: {row['retorno_total_com_div_pct']:.2f}% "
                      f"(dividendos: {row['efeito_dividendos_total_pct']:.2f}%)")

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

        # Gera visualizações consolidadas
        plot_all(results, period_name)

        # Salva métricas
        save_metrics(results, period_name, asset_summaries)

    print("\n" + "="*70)
    print(" ✅ ANÁLISE COMPLETA!")
    print(" 📁 Todos os arquivos salvos em: outputs/")
    print("="*70 + "\n")

def plot_portfolios_vs_ibov_assets(
        results: Dict,
        period: str
):
    """
    Gráfico comparativo: Melhores ativos de cada carteira vs. Ibovespa.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    profiles_to_compare = PROFILES + ["ibovespa"]

    for idx, perfil in enumerate(profiles_to_compare):
        asset_data = results[perfil]["asset_analysis"]

        if asset_data.empty:
            continue

        # Top 10 ativos por retorno total
        top_10 = asset_data.head(10)

        ax = axes[idx]

        # Barras empilhadas: Preço + Dividendos
        y_pos = np.arange(len(top_10))

        bars_price = ax.barh(
            y_pos,
            top_10['retorno_total_sem_div_pct'],
            label='Valorização de Preço',
            color='steelblue',
            alpha=0.7
        )

        bars_div = ax.barh(
            y_pos,
            top_10['efeito_dividendos_total_pct'],
            left=top_10['retorno_total_sem_div_pct'],
            label='Contribuição de Dividendos',
            color='orange',
            alpha=0.7
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10['TICKER'])
        ax.set_xlabel('Retorno Total (%)', fontsize=10, fontweight='bold')
        ax.set_title(
            f'{perfil.capitalize()} - Top 10 Ativos',
            fontsize=12,
            fontweight='bold'
        )
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        # Annotations
        for i, (price_ret, div_ret) in enumerate(zip(
                top_10['retorno_total_sem_div_pct'],
                top_10['efeito_dividendos_total_pct']
        )):
            total = price_ret + div_ret
            ax.text(
                total + 5,
                i,
                f'{total:.0f}%',
                va='center',
                fontsize=8,
                fontweight='bold'
            )

    # Oculta o subplot vazio (temos 5 perfis em 6 subplots)
    if len(profiles_to_compare) < len(axes):
        for idx in range(len(profiles_to_compare), len(axes)):
            axes[idx].set_visible(False)

    fig.suptitle(
        f'Comparação: Melhores Ativos por Carteira ({period})',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUTS_DIR / f"backtest_comparison_all_assets_{period}.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    print(f"  ✓ Gráfico comparativo salvo: backtest_comparison_all_assets_{period}.png")

if __name__ == "__main__":
    main()