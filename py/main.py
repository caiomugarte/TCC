#!/usr/bin/env python3
"""main.py
=============================================================================
Ponto de entrada principal do sistema de otimização de carteiras.

CLI interativa para executar diferentes pipelines:
1. Pré-processamento de dados
2. Execução única do GA
3. Múltiplas execuções (análise de robustez)
4. Backtest de carteiras
5. Limpeza de cache

Uso:
    python main.py              # Modo interativo
    python main.py --all        # Executa tudo
    python main.py --preprocess # Apenas pré-processa
    python main.py --single     # Execução única
    python main.py --multi      # Múltiplas execuções
=============================================================================
"""

import argparse
import sys
from pathlib import Path

from config import PROFILES
from core.preprocessing import preprocess_all_profiles
from pipelines.single_run import run_all_profiles
from pipelines.multi_run import run_multi_execution_all_profiles
from utils.cache import CacheManager


def print_banner():
    """Exibe banner do sistema."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║      Sistema de Otimização de Carteiras - Algoritmo Genético         ║
║                           TCC - 2025                                  ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_menu():
    """Exibe menu de opções."""
    menu = """
Escolha uma opção:

  [1] 🔧 Pré-processar dados (todos os perfis)
  [2] 🚀 Execução única do GA (todos os perfis)
  [3] 📊 Múltiplas execuções (análise de robustez)
  [4] 📈 Backtest de carteiras (em desenvolvimento)
  [5] 🗑️  Limpar cache
  [6] ⚙️  Configurações
  [0] 🚪 Sair

"""
    print(menu)


def preprocess_menu():
    """Executa pré-processamento."""
    print("\n" + "=" * 70)
    print("PRÉ-PROCESSAMENTO DE DADOS")
    print("=" * 70)
    print("\nProcessando todos os perfis...")
    print(f"Perfis: {', '.join(PROFILES)}\n")

    preprocess_all_profiles(save=True)

    print("\n✅ Pré-processamento concluído!")
    input("\nPressione ENTER para continuar...")


def single_run_menu():
    """Executa pipeline de execução única."""
    print("\n" + "=" * 70)
    print("EXECUÇÃO ÚNICA DO ALGORITMO GENÉTICO")
    print("=" * 70)

    print("\nOpções:")
    print("  [1] Executar com cache (mais rápido)")
    print("  [2] Executar sem cache (reprocessa tudo)")
    print("  [0] Voltar")

    choice = input("\nEscolha: ").strip()

    if choice == "0":
        return

    use_cache = choice == "1"

    print(f"\nExecutando pipeline {'COM' if use_cache else 'SEM'} cache...\n")

    run_all_profiles(
        use_cache=use_cache,
        robustness_filter=True,
        save_outputs=True
    )

    print("\n✅ Pipeline concluído!")
    input("\nPressione ENTER para continuar...")


def multi_run_menu():
    """Executa pipeline de múltiplas execuções."""
    print("\n" + "=" * 70)
    print("MÚLTIPLAS EXECUÇÕES - ANÁLISE DE ROBUSTEZ")
    print("=" * 70)

    from config import N_RUNS

    print(f"\nNúmero padrão de execuções: {N_RUNS}")
    custom = input(f"Usar valor padrão? (s/n) [s]: ").strip().lower()

    n_runs = N_RUNS
    if custom == "n":
        try:
            n_runs = int(input("Digite o número de execuções: ").strip())
        except ValueError:
            print("⚠️  Valor inválido. Usando padrão.")

    print("\nOpções de paralelização:")
    print("  [1] Paralelo (mais rápido, usa múltiplos cores)")
    print("  [2] Sequencial (mais lento, mas consome menos memória)")

    parallel_choice = input("\nEscolha [1]: ").strip() or "1"
    parallel = parallel_choice == "1"

    print("\n🚀 Modo Adaptativo (NOVO!):")
    print("  Para automaticamente quando atingir estabilidade suficiente")
    print("  (economiza tempo sem perder qualidade)")
    adaptive_choice = input("\nUsar modo adaptativo? (s/n) [s]: ").strip().lower()
    adaptive = adaptive_choice != "n"

    if adaptive:
        print(f"\n  ✓ Modo adaptativo ativado")
        print(f"    • Rodará até {n_runs} execuções (ou menos se convergir)")
        print(f"    • Mínimo: 30 runs antes de verificar convergência")
        print(f"    • Alvo: CV < 3% e Jaccard > 70%")

    print(f"\nExecutando até {n_runs} rodadas em modo {'PARALELO' if parallel else 'SEQUENCIAL'}...\n")

    run_multi_execution_all_profiles(
        n_runs=n_runs,
        parallel=parallel,
        adaptive_mode=adaptive,
        save_summary=True
    )

    print("\n✅ Análise de robustez concluída!")
    input("\nPressione ENTER para continuar...")


def backtest_menu():
    """Executa backtest (em desenvolvimento)."""
    print("\n" + "=" * 70)
    print("BACKTEST DE CARTEIRAS")
    print("=" * 70)
    print("\n⚠️  Funcionalidade em desenvolvimento.")
    print("Por enquanto, use o arquivo backtest_analysis.py diretamente.")
    input("\nPressione ENTER para continuar...")


def cache_menu():
    """Menu de gerenciamento de cache."""
    print("\n" + "=" * 70)
    print("GERENCIAMENTO DE CACHE")
    print("=" * 70)

    cache = CacheManager()

    print("\nOpções:")
    print("  [1] Ver status do cache")
    print("  [2] Limpar todo o cache")
    print("  [0] Voltar")

    choice = input("\nEscolha: ").strip()

    if choice == "0":
        return
    elif choice == "1":
        if cache.metadata:
            print("\nCache atual:")
            for key, info in cache.metadata.items():
                print(f"  • {key} ({info['format']})")
        else:
            print("\n📭 Cache vazio.")
    elif choice == "2":
        confirm = input("\n⚠️  Tem certeza? (s/n): ").strip().lower()
        if confirm == "s":
            cache.clear()
            print("\n✅ Cache limpo com sucesso!")
        else:
            print("\n❌ Operação cancelada.")

    input("\nPressione ENTER para continuar...")


def config_menu():
    """Menu de configurações."""
    print("\n" + "=" * 70)
    print("CONFIGURAÇÕES")
    print("=" * 70)

    from config import GA_CONFIG, FILTERS, PROFILE_WEIGHTS

    print("\nPerfis disponíveis:")
    for profile in PROFILES:
        ga_cfg = GA_CONFIG[profile]
        filter_cfg = FILTERS[profile]
        print(f"\n  {profile.upper()}:")
        print(f"    • Ativos: {ga_cfg['n_assets']}")
        print(f"    • Gerações: {ga_cfg['generations']}")
        print(f"    • Pop. size: {ga_cfg['pop_size']}")
        print(f"    • Lambda (HHI): {ga_cfg['lambda']}")
        print(f"    • Cap. mín: R$ {filter_cfg['cap_min']:,.0f}")
        print(f"    • Liq. mín: R$ {filter_cfg['liq_min']:,.0f}")

    print("\nPara alterar configurações, edite o arquivo config.py")
    input("\nPressione ENTER para continuar...")


def interactive_mode():
    """Modo interativo (menu)."""
    while True:
        print_banner()
        print_menu()

        choice = input("Escolha uma opção: ").strip()

        if choice == "0":
            print("\n👋 Até logo!")
            sys.exit(0)
        elif choice == "1":
            preprocess_menu()
        elif choice == "2":
            single_run_menu()
        elif choice == "3":
            multi_run_menu()
        elif choice == "4":
            backtest_menu()
        elif choice == "5":
            cache_menu()
        elif choice == "6":
            config_menu()
        else:
            print("\n❌ Opção inválida. Tente novamente.")
            input("\nPressione ENTER para continuar...")


def cli_mode(args):
    """Modo CLI (não-interativo)."""
    print_banner()

    if args.preprocess:
        print("Executando pré-processamento...")
        preprocess_all_profiles(save=True)

    if args.single:
        print("Executando pipeline único...")
        run_all_profiles(
            use_cache=args.use_cache,
            robustness_filter=True,
            save_outputs=True
        )

    if args.multi:
        print(f"Executando múltiplas execuções (n={args.n_runs})...")
        if args.profile:
            # Roda apenas para o perfil especificado
            print(f"Perfil: {args.profile}")
            from pipelines.multi_run import run_multi_execution_profile
            run_multi_execution_profile(
                profile=args.profile,
                n_runs=args.n_runs,
                parallel=args.parallel
            )
        else:
            # Roda para todos os perfis
            run_multi_execution_all_profiles(
                n_runs=args.n_runs,
                parallel=args.parallel,
                save_summary=True
            )

    if args.clear_cache:
        print("Limpando cache...")
        cache = CacheManager()
        cache.clear()
        print("✅ Cache limpo!")

    if args.all:
        print("Executando pipeline completo...")
        preprocess_all_profiles(save=True)
        run_all_profiles(use_cache=True, robustness_filter=True, save_outputs=True)
        run_multi_execution_all_profiles(n_runs=args.n_runs, parallel=args.parallel)

    print("\n✅ Concluído!")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema de Otimização de Carteiras - Algoritmo Genético",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Executa pré-processamento de dados"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Executa pipeline único do GA"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Executa múltiplas execuções (robustez)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa pipeline completo"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Número de execuções para análise de robustez (default: 30)"
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Desabilita cache"
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Desabilita paralelização"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Limpa todo o cache"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["conservador", "moderado", "arrojado", "caio", "caio_new", "caio_last"],
        help="Perfil específico para executar (default: todos)"
    )

    args = parser.parse_args()

    # Se nenhum argumento foi passado, entra em modo interativo
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        cli_mode(args)


if __name__ == "__main__":
    main()
