"""analyze_comparison.py
=============================================================================
Script para analisar comparação entre carteira consensual e melhor indivíduo.

Mostra de forma clara:
- Comparação de backtest (5 e 10 anos)
- Comparação de métricas fundamentalistas
- Recomendação de qual carteira usar

Uso: python analyze_comparison.py <perfil>
Exemplo: python analyze_comparison.py conservador
=============================================================================
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def print_section(title):
    """Imprime cabeçalho de seção."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze_backtest(backtest_data, period):
    """Analisa e imprime resultados de backtest."""
    if not backtest_data:
        print(f"  ⚠️ Dados de backtest {period} anos não disponíveis")
        return

    print(f"\n  📊 BACKTEST {period} ANOS")
    print(f"  Período: {backtest_data['start_date']} a {backtest_data['end_date']}")
    print(f"\n  Carteira Consensual:")
    print(f"    • Retorno Total: {backtest_data['consensus']['total_return_pct']:.2f}%")
    print(f"    • Retorno Anual: {backtest_data['consensus']['annual_return_pct']:.2f}% a.a.")
    print(f"    • Volatilidade: {backtest_data['consensus']['volatility_pct']:.2f}%")
    print(f"    • Sharpe Ratio: {backtest_data['consensus']['sharpe_ratio']:.3f}")
    print(f"    • Max Drawdown: {backtest_data['consensus']['max_drawdown_pct']:.2f}%")

    print(f"\n  Melhor Indivíduo:")
    print(f"    • Retorno Total: {backtest_data['best_individual']['total_return_pct']:.2f}%")
    print(f"    • Retorno Anual: {backtest_data['best_individual']['annual_return_pct']:.2f}% a.a.")
    print(f"    • Volatilidade: {backtest_data['best_individual']['volatility_pct']:.2f}%")
    print(f"    • Sharpe Ratio: {backtest_data['best_individual']['sharpe_ratio']:.3f}")
    print(f"    • Max Drawdown: {backtest_data['best_individual']['max_drawdown_pct']:.2f}%")

    print(f"\n  🏆 Vencedor: {backtest_data['winner'].upper()}")
    print(f"     Diferença de retorno: {backtest_data['return_difference_pct']:+.2f}%")


def analyze_fundamentals(fundamental_data):
    """Analisa e imprime comparação de fundamentals."""
    if not fundamental_data:
        print("  ⚠️ Dados de comparação fundamentalista não disponíveis")
        return

    print(f"\n  📈 MÉTRICAS FUNDAMENTALISTAS")
    print(f"\n  Score Geral:")
    print(f"    • Consenso: {fundamental_data['overall_score']['consensus']} vitórias")
    print(f"    • Melhor Indivíduo: {fundamental_data['overall_score']['best_individual']} vitórias")
    print(f"    • 🏆 Vencedor: {fundamental_data['overall_winner'].upper()}")

    print(f"\n  Por Grupo de Métricas:")
    for group, stats in fundamental_data['summary'].items():
        winner_emoji = "🏆" if stats['winner'] != "tie" else "🤝"
        print(f"    {winner_emoji} {group.capitalize()}: {stats['winner'].upper()} "
              f"(Consenso: {stats['consensus_wins']} | Melhor: {stats['best_individual_wins']})")


def print_recommendation(comparison_data, backtest_5y, backtest_10y, fundamental_data):
    """Imprime recomendação final de qual carteira usar."""
    print_section("RECOMENDAÇÃO FINAL")

    # Contagem de vitórias
    consensus_wins = 0
    best_wins = 0

    # Backtest 5 anos
    if backtest_5y and backtest_5y.get('winner') == 'consensus':
        consensus_wins += 1
    elif backtest_5y:
        best_wins += 1

    # Backtest 10 anos
    if backtest_10y and backtest_10y.get('winner') == 'consensus':
        consensus_wins += 1
    elif backtest_10y:
        best_wins += 1

    # Fundamentals
    if fundamental_data and fundamental_data.get('overall_winner') == 'consensus':
        consensus_wins += 1
    elif fundamental_data and fundamental_data.get('overall_winner') != 'tie':
        best_wins += 1

    # Fitness (GA)
    if comparison_data['fitness']['consensus'] > comparison_data['fitness']['best_individual']:
        consensus_wins += 1
    else:
        best_wins += 1

    print(f"\n  Análise Consolidada:")
    print(f"    • Backtest 5 anos: {backtest_5y.get('winner', 'N/A').upper() if backtest_5y else 'N/A'}")
    print(f"    • Backtest 10 anos: {backtest_10y.get('winner', 'N/A').upper() if backtest_10y else 'N/A'}")
    print(f"    • Fundamentals: {fundamental_data.get('overall_winner', 'N/A').upper() if fundamental_data else 'N/A'}")
    print(f"    • Fitness (GA): {'CONSENSUS' if comparison_data['fitness']['consensus'] > comparison_data['fitness']['best_individual'] else 'BEST_INDIVIDUAL'}")

    print(f"\n  Score Final:")
    print(f"    • Consenso: {consensus_wins} vitórias")
    print(f"    • Melhor Indivíduo: {best_wins} vitórias")

    if consensus_wins > best_wins:
        print(f"\n  🏆 RECOMENDAÇÃO: Use a CARTEIRA CONSENSUAL")
        print(f"     Motivo: Venceu em {consensus_wins}/{consensus_wins + best_wins} critérios")
        print(f"     A carteira consensual tende a ser mais robusta e estável.")
    elif best_wins > consensus_wins:
        print(f"\n  🏆 RECOMENDAÇÃO: Use o MELHOR INDIVÍDUO")
        print(f"     Motivo: Venceu em {best_wins}/{consensus_wins + best_wins} critérios")
        print(f"     O melhor indivíduo apresentou performance superior.")
    else:
        print(f"\n  🤝 EMPATE: Ambas as carteiras têm méritos")
        print(f"     Considere usar a CONSENSUAL para maior estabilidade")
        print(f"     ou o MELHOR INDIVÍDUO para maior potencial de retorno.")


def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_comparison.py <perfil>")
        print("Exemplo: python analyze_comparison.py conservador")
        print("\nPerfis disponíveis: conservador, moderado, arrojado, caio")
        sys.exit(1)

    profile = sys.argv[1]
    comparison_file = OUTPUTS_DIR / f"comparison_{profile}.json"

    if not comparison_file.exists():
        print(f"❌ Arquivo não encontrado: {comparison_file}")
        print(f"\nExecute primeiro: python main.py --multi-run --profile {profile}")
        sys.exit(1)

    # Carrega dados
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extrai seções
    comparison_data = data.get('metrics_comparison', {})
    fundamental_data = data.get('fundamental_comparison', {})
    backtest_5y = data.get('backtest_5y', {})
    backtest_10y = data.get('backtest_10y', {})

    # Exibe análises
    print_section(f"ANÁLISE DE COMPARAÇÃO - PERFIL: {profile.upper()}")

    print(f"\n  Carteiras Analisadas:")
    print(f"    • Consensual: {len(comparison_data.get('overlap', {}).get('common_tickers', []) + comparison_data.get('overlap', {}).get('only_consensus', []))} ativos")
    print(f"    • Melhor Indivíduo: {len(comparison_data.get('overlap', {}).get('common_tickers', []) + comparison_data.get('overlap', {}).get('only_best_individual', []))} ativos")
    print(f"    • Sobreposição (Jaccard): {comparison_data.get('overlap', {}).get('jaccard_index', 0):.1%}")

    analyze_backtest(backtest_5y, 5)
    analyze_backtest(backtest_10y, 10)
    analyze_fundamentals(fundamental_data)
    print_recommendation(comparison_data, backtest_5y, backtest_10y, fundamental_data)

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
