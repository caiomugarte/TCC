"""regenerate_dividend_plots.py
===============================================================================
Script para regenerar gráficos de dividendos com melhor visualização

Este script lê os dados já processados (arquivos CSV de ativos) e regenera
apenas os gráficos de contribuição de dividendos com visualização melhorada,
SEM recalcular nenhum dado ou alterar os resultados do backtest original.

Uso:
    python regenerate_dividend_plots.py

Autor: Análise Quantitativa para TCC
Data: 2025
===============================================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

OUTPUTS_DIR = Path("outputs")

PERIODS = ["5anos", "10anos"]
PROFILES = ["conservador", "moderado", "arrojado", "ibovespa", "ibovespa_top15"]

# Configurações de visualização
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ============================================================================
# FUNÇÃO DE PLOTAGEM MELHORADA
# ============================================================================

def plot_dividend_contribution_improved(
        asset_data: pd.DataFrame,
        perfil: str,
        period: str,
        output_suffix: str = "_v2"
):
    """
    Gráfico melhorado: Contribuição dos dividendos (barras horizontais).

    Melhorias em relação à versão original (pizza):
    - Usa barras horizontais ao invés de pizza (mais legível)
    - Labels dos tickers diretamente visíveis no eixo Y
    - Cores em gradiente para destacar maiores contribuições
    - Anotações com valores exatos nas barras
    - Sem risco de sobreposição de textos

    Args:
        asset_data: DataFrame com dados dos ativos
        perfil: Nome do perfil
        period: Período do backtest
        output_suffix: Sufixo para o nome do arquivo (default: "_v2")
    """
    if asset_data.empty:
        print(f"  ⚠ {perfil} ({period}): Sem dados disponíveis")
        return

    # Calcula contribuição percentual dos dividendos
    asset_data['contrib_dividendos'] = (
            asset_data['efeito_dividendos_total_pct'] /
            asset_data['retorno_total_com_div_pct'] * 100
    ).fillna(0)

    # Limita entre 0 e 100%
    asset_data['contrib_dividendos'] = asset_data['contrib_dividendos'].clip(0, 100)

    # Top 10 ativos por contribuição de dividendos
    top_10 = asset_data.nlargest(10, 'efeito_dividendos_total_pct').reset_index(drop=True)

    if len(top_10) == 0:
        print(f"  ⚠ {perfil} ({period}): Nenhum ativo com dividendos")
        return

    # Calcula contribuição percentual para o total de dividendos
    total_dividendos = top_10['efeito_dividendos_total_pct'].sum()
    top_10['contrib_pct_total'] = (top_10['efeito_dividendos_total_pct'] / total_dividendos * 100)

    # Inverte ordem para o maior ficar em cima
    top_10 = top_10.iloc[::-1].reset_index(drop=True)

    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Cores em gradiente (verde escuro para o maior, claro para o menor)
    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(top_10)))[::-1]

    # Cria barras horizontais
    y_pos = np.arange(len(top_10))
    bars = ax.barh(
        y_pos,
        top_10['contrib_pct_total'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Configura eixos
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['TICKER'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Contribuição para o Total de Dividendos (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ativo', fontsize=12, fontweight='bold')

    # Adiciona valores nas barras
    for i, (bar, contrib_pct) in enumerate(zip(bars, top_10['contrib_pct_total'])):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{contrib_pct:.1f}%',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold',
            color='black'
        )

    # Título
    ax.set_title(
        f'Contribuição dos Dividendos por Ativo\n{perfil.capitalize()} ({period})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grade apenas no eixo X
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Ajusta limites
    ax.set_xlim(0, top_10['contrib_pct_total'].max() * 1.15)

    plt.tight_layout()

    output_file = OUTPUTS_DIR / f"backtest_dividends_{perfil}_{period}{output_suffix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico salvo: {output_file.name}")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """
    Regenera todos os gráficos de dividendos a partir dos dados existentes.
    """
    print("\n" + "="*70)
    print(" REGENERAÇÃO DE GRÁFICOS DE DIVIDENDOS - VERSÃO MELHORADA")
    print("="*70)
    print("\nEste script NÃO recalcula dados, apenas melhora a visualização")
    print("dos gráficos usando os arquivos CSV já existentes.\n")

    if not OUTPUTS_DIR.exists():
        print(f"❌ Erro: Diretório {OUTPUTS_DIR} não encontrado")
        print("Execute o backtest_analysis.py primeiro para gerar os dados.\n")
        return

    total_gerados = 0
    total_faltando = 0

    for period in PERIODS:
        print(f"\n{'='*70}")
        print(f" PERÍODO: {period.upper()}")
        print(f"{'='*70}\n")

        for perfil in PROFILES:
            # Busca arquivo CSV de ativos
            csv_file = OUTPUTS_DIR / f"backtest_assets_{perfil}_{period}.csv"

            if not csv_file.exists():
                print(f"  ⚠ {perfil}: Arquivo não encontrado ({csv_file.name})")
                total_faltando += 1
                continue

            # Carrega dados
            try:
                asset_data = pd.read_csv(csv_file, encoding='utf-8-sig')

                if asset_data.empty:
                    print(f"  ⚠ {perfil}: Arquivo vazio")
                    continue

                # Gera gráfico melhorado
                plot_dividend_contribution_improved(
                    asset_data,
                    perfil,
                    period,
                    output_suffix="_v2"
                )
                total_gerados += 1

            except Exception as e:
                print(f"  ✗ {perfil}: Erro ao processar - {e}")
                continue

    print("\n" + "="*70)
    print(f" ✅ CONCLUÍDO!")
    print(f" 📊 Gráficos gerados: {total_gerados}")
    if total_faltando > 0:
        print(f" ⚠ Arquivos não encontrados: {total_faltando}")
        print(f"\n💡 Execute backtest_analysis.py para gerar os dados faltantes")
    print(f" 📁 Arquivos salvos em: {OUTPUTS_DIR}/")
    print(f" 🔍 Procure por: backtest_dividends_*_v2.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
