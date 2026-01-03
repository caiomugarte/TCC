#!/usr/bin/env python3
"""
Script de teste para validar a comparação entre carteira consensual
e melhor indivíduo.
"""

import sys
from pathlib import Path

# Adiciona diretório py ao path
sys.path.insert(0, str(Path(__file__).parent / "py"))

from pipelines.multi_run import run_multi_execution_profile

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE: Comparação Consenso vs Melhor Indivíduo")
    print("=" * 70)
    print("\nExecutando 5 rodadas do GA para perfil 'conservador'...")
    print("Isso deve gerar:")
    print("  1. Carteira consensual")
    print("  2. Carteira do melhor indivíduo")
    print("  3. Comparação de métricas (JSON)")
    print("  4. Backtest comparativo")
    print("  5. Gráficos comparativos")
    print("=" * 70 + "\n")

    try:
        result = run_multi_execution_profile(
            profile="conservador",
            n_runs=5,
            use_cache=True,
            parallel=False  # Sequencial para evitar problemas
        )

        print("\n" + "=" * 70)
        print("RESULTADO DO TESTE")
        print("=" * 70)

        # Verifica se os dados foram gerados
        if result:
            print("\n✅ Teste concluído com sucesso!")

            # Mostra resumo
            print("\n📊 RESUMO DA COMPARAÇÃO:")
            print(f"\n  Carteira Consensual:")
            print(f"    • Tickers: {result['consensus_portfolio']['tickers']}")
            print(f"    • HHI: {result['consensus_portfolio']['hhi']:.3f}")

            print(f"\n  Melhor Indivíduo:")
            print(f"    • Tickers: {result['best_individual_portfolio']['tickers']}")
            print(f"    • Fitness: {result['best_individual_portfolio']['fitness']:.2f}")
            print(f"    • HHI: {result['best_individual_portfolio']['hhi']:.3f}")
            print(f"    • Run ID: {result['best_individual_portfolio']['run_id']}")

            print(f"\n  Overlap:")
            print(f"    • Jaccard Index: {result['comparison']['overlap']['jaccard_index']:.3f}")
            print(f"    • Ativos comuns: {result['comparison']['overlap']['n_common']}")

            if result.get('backtest_comparison'):
                print(f"\n  Backtest (5 anos):")
                bt = result['backtest_comparison']
                print(f"    • Vencedor: {bt.get('winner', 'N/A').upper()}")
                print(f"    • Diferença de retorno: {bt.get('return_difference_pct', 0):+.2f}%")

            print("\n📁 Arquivos gerados em: outputs/")
            print("   • carteira_conservador_consensus.json")
            print("   • carteira_conservador_best_individual.json")
            print("   • comparison_conservador.json")
            print("   • comparison_charts_conservador.png")

        else:
            print("\n❌ Teste falhou. Verifique os logs acima.")

    except Exception as e:
        print(f"\n❌ ERRO durante o teste:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70 + "\n")
