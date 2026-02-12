#!/usr/bin/env python3
"""run_optimized.py
=============================================================================
Script de exemplo para rodar o GA com as otimizações implementadas.

Este script demonstra as diferentes formas de usar as melhorias:
1. Modo adaptativo (recomendado)
2. Early stopping
3. Salvamento incremental com checkpoints

Uso:
    python run_optimized.py                    # Modo interativo
    python run_optimized.py --quick            # Teste rápido (20 runs)
    python run_optimized.py --production       # Produção (até 100 runs, adaptativo)
    python run_optimized.py --max-quality      # Máxima qualidade (até 150 runs)
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório py ao path
sys.path.insert(0, str(Path(__file__).parent / "py"))

from pipelines.multi_run import run_multi_execution_profile


def quick_test(profile: str = "caio"):
    """Teste rápido: 20 runs em paralelo."""
    print("\n" + "=" * 70)
    print("MODO: TESTE RÁPIDO")
    print("=" * 70)
    print("Configuração:")
    print("  • Máximo de runs: 20")
    print("  • Modo: Paralelo")
    print("  • Adaptativo: Não (poucas runs)")
    print("=" * 70 + "\n")

    return run_multi_execution_profile(
        profile=profile,
        n_runs=20,
        parallel=True,
        adaptive_mode=False,
        save_interval=5
    )


def production_mode(profile: str = "caio"):
    """Modo produção: até 100 runs com modo adaptativo."""
    print("\n" + "=" * 70)
    print("MODO: PRODUÇÃO (RECOMENDADO)")
    print("=" * 70)
    print("Configuração:")
    print("  • Máximo de runs: 100")
    print("  • Modo: Paralelo")
    print("  • Adaptativo: SIM (para quando CV < 3% e Jaccard > 70%)")
    print("  • Checkpoint: A cada 10 runs")
    print("  • Estimativa: ~30-60 runs (convergência esperada)")
    print("=" * 70 + "\n")

    return run_multi_execution_profile(
        profile=profile,
        n_runs=100,
        parallel=True,
        adaptive_mode=True,
        min_runs=30,
        target_cv=0.03,
        target_jaccard=0.70,
        save_interval=10
    )


def max_quality_mode(profile: str = "caio"):
    """Máxima qualidade: até 150 runs."""
    print("\n" + "=" * 70)
    print("MODO: MÁXIMA QUALIDADE")
    print("=" * 70)
    print("Configuração:")
    print("  • Máximo de runs: 150")
    print("  • Modo: Sequencial (mais estável, menos memória)")
    print("  • Adaptativo: SIM (para quando CV < 2% e Jaccard > 75%)")
    print("  • Checkpoint: A cada 10 runs")
    print("  • Estimativa: ~40-80 runs")
    print("  ⚠️  ATENÇÃO: Modo sequencial é mais lento!")
    print("=" * 70 + "\n")

    confirm = input("Este modo pode levar várias horas. Continuar? (s/n): ").strip().lower()
    if confirm != "s":
        print("Operação cancelada.")
        return None

    return run_multi_execution_profile(
        profile=profile,
        n_runs=150,
        parallel=False,  # Sequencial para economizar memória
        adaptive_mode=True,
        min_runs=40,
        target_cv=0.02,      # Critério mais rigoroso
        target_jaccard=0.75,  # Critério mais rigoroso
        save_interval=10
    )


def custom_mode():
    """Modo customizado: usuário escolhe parâmetros."""
    print("\n" + "=" * 70)
    print("MODO: CUSTOMIZADO")
    print("=" * 70)

    # Perfil
    print("\nPerfis disponíveis: conservador, moderado, arrojado, caio")
    profile = input("Digite o perfil [caio]: ").strip() or "caio"

    # Número de runs
    try:
        n_runs = int(input("Número máximo de runs [100]: ").strip() or "100")
    except ValueError:
        print("Valor inválido. Usando 100.")
        n_runs = 100

    # Modo adaptativo
    adaptive = input("Usar modo adaptativo? (s/n) [s]: ").strip().lower() != "n"

    # Paralelização
    parallel = input("Usar paralelização? (s/n) [s]: ").strip().lower() != "n"

    # Checkpoint interval
    try:
        save_interval = int(input("Salvar checkpoint a cada N runs [10]: ").strip() or "10")
    except ValueError:
        save_interval = 10

    print(f"\n{'='*70}")
    print("RESUMO DA CONFIGURAÇÃO:")
    print(f"  • Perfil: {profile}")
    print(f"  • Máximo de runs: {n_runs}")
    print(f"  • Modo adaptativo: {'SIM' if adaptive else 'NÃO'}")
    print(f"  • Paralelização: {'SIM' if parallel else 'NÃO'}")
    print(f"  • Checkpoint: A cada {save_interval} runs")
    print(f"{'='*70}\n")

    confirm = input("Confirmar e executar? (s/n): ").strip().lower()
    if confirm != "s":
        print("Operação cancelada.")
        return None

    return run_multi_execution_profile(
        profile=profile,
        n_runs=n_runs,
        parallel=parallel,
        adaptive_mode=adaptive,
        save_interval=save_interval
    )


def interactive_mode():
    """Modo interativo: menu de opções."""
    print("\n" + "=" * 70)
    print("ALGORITMO GENÉTICO - OTIMIZADO")
    print("=" * 70)
    print("\nEscolha um modo de execução:\n")
    print("  [1] 🚀 Teste Rápido (20 runs, ~5-10 min)")
    print("  [2] ⭐ Produção - RECOMENDADO (até 100 runs adaptativo, ~30-60 min)")
    print("  [3] 💎 Máxima Qualidade (até 150 runs sequencial, ~2-3h)")
    print("  [4] ⚙️  Customizado (você escolhe os parâmetros)")
    print("  [0] 🚪 Sair\n")

    choice = input("Escolha: ").strip()

    if choice == "0":
        print("\n👋 Até logo!")
        return

    profile = input("\nPerfil [caio]: ").strip() or "caio"

    if choice == "1":
        result = quick_test(profile)
    elif choice == "2":
        result = production_mode(profile)
    elif choice == "3":
        result = max_quality_mode(profile)
    elif choice == "4":
        result = custom_mode()
    else:
        print("Opção inválida.")
        return

    if result:
        print("\n" + "=" * 70)
        print("EXECUÇÃO CONCLUÍDA!")
        print("=" * 70)
        print(f"\nResultados salvos em:")
        print(f"  • outputs/carteira_{profile}_consensus.json")
        print(f"  • outputs/carteira_{profile}_best_individual.json")
        print(f"  • outputs/comparison_{profile}.json")
        print(f"  • outputs/metrics_stability_{profile}.csv")
        print("\nConsulte OPTIMIZATION_GUIDE.md para interpretar os resultados.")


def main():
    parser = argparse.ArgumentParser(
        description="Algoritmo Genético Otimizado - Portfolio Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python run_optimized.py                    # Modo interativo
  python run_optimized.py --quick            # Teste rápido
  python run_optimized.py --production       # Modo produção (recomendado)
  python run_optimized.py --max-quality      # Máxima qualidade

Para mais detalhes, consulte: OPTIMIZATION_GUIDE.md
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Teste rápido (20 runs)"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Modo produção (até 100 runs adaptativo) - RECOMENDADO"
    )
    parser.add_argument(
        "--max-quality",
        action="store_true",
        help="Máxima qualidade (até 150 runs)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="caio",
        choices=["conservador", "moderado", "arrojado", "caio"],
        help="Perfil de investidor (default: caio)"
    )

    args = parser.parse_args()

    # Se nenhum argumento, modo interativo
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        if args.quick:
            quick_test(args.profile)
        elif args.production:
            production_mode(args.profile)
        elif args.max_quality:
            max_quality_mode(args.profile)
        else:
            print("Use --quick, --production ou --max-quality")


if __name__ == "__main__":
    main()
