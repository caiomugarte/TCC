"""ga_multiple_runs.py
---------------------------------------------------------------
Executa o Algoritmo Genético múltiplas vezes com diferentes seeds
para análise estatística da robustez e estabilidade das carteiras.

Saída:
  - multiple_runs_summary.json: Estatísticas consolidadas
  - carteira_{perfil}_consensus.json: Carteira representativa
  - metrics_stability.csv: Métricas por execução para análise
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import profiles as pf
import ga

# ==================== CONFIGURAÇÃO ====================
N_RUNS = 30  # Número de execuções independentes (padrão acadêmico)
DATA_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PROFILES = ["conservador", "moderado", "arrojado", "caio"]

# ==================== FUNÇÕES AUXILIARES ====================

def set_seed_for_run(run_id: int):
    """Define seed único para cada execução."""
    from random import seed
    import numpy as np
    seed_value = 42 + run_id  # Base seed + run offset
    seed(seed_value)
    np.random.seed(seed_value)
    return seed_value


def run_ga_single_execution(df_ranked: pd.DataFrame, perfil: str, run_id: int) -> Dict:
    """
    Executa uma única rodada do GA com seed específico.

    Returns:
        Dict com tickers, fitness, hhi, seed e métricas da carteira.
    """
    seed_value = set_seed_for_run(run_id)

    # IMPORTANTE: Precisamos re-importar ga para resetar o seed interno
    import importlib
    import ga as ga_module
    importlib.reload(ga_module)

    # Override do seed no módulo ga
    from random import seed
    seed(seed_value)

    # Executa GA
    carteira = ga_module.run_ga(df_ranked, perfil)

    result = {
        "run_id": run_id,
        "seed": seed_value,
        "tickers": sorted(carteira["TICKER"].tolist()),
        "fitness": float(carteira.attrs["fitness"]),
        "hhi": float(carteira.attrs["hhi"]),
        "score_median": float(carteira["SCORE"].median()),
        "score_mean": float(carteira["SCORE"].mean()),
        "score_std": float(carteira["SCORE"].std()),
        "sectors": carteira["SETOR"].value_counts().to_dict(),
    }

    return result


def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calcula Índice de Jaccard entre dois conjuntos de ativos.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def build_consensus_portfolio(results: List[Dict], df_ranked: pd.DataFrame,
                              perfil: str) -> pd.DataFrame:
    """
    Constrói carteira consenso baseada na frequência de aparição dos ativos.

    Estratégia: Seleciona os N ativos mais frequentes nas execuções,
    onde N = número de ativos do perfil (10, 12 ou 15).
    """
    cfg = ga.PERFIL_CONFIG[perfil]
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
    hhi = ga.hhi_sector(consensus_portfolio)
    avg_score = consensus_portfolio["SCORE"].mean()

    consensus_portfolio.attrs["hhi"] = hhi
    consensus_portfolio.attrs["avg_score"] = avg_score
    consensus_portfolio.attrs["method"] = "consensus_frequency"

    return consensus_portfolio.sort_values("FREQUENCY", ascending=False)


def analyze_stability(results: List[Dict]) -> Dict:
    """
    Calcula métricas de estabilidade das execuções.

    Métricas:
    - CV (Coeficiente de Variação) do fitness e HHI
    - Jaccard médio entre pares de carteiras
    - Dispersão setorial
    """
    fitness_values = [r["fitness"] for r in results]
    hhi_values = [r["hhi"] for r in results]

    # Coeficiente de Variação (CV = std/mean)
    fitness_cv = np.std(fitness_values) / np.mean(fitness_values) if np.mean(fitness_values) != 0 else 0
    hhi_cv = np.std(hhi_values) / np.mean(hhi_values) if np.mean(hhi_values) != 0 else 0

    # Jaccard médio entre todos os pares
    ticker_sets = [set(r["tickers"]) for r in results]
    jaccard_scores = []
    for i in range(len(ticker_sets)):
        for j in range(i + 1, len(ticker_sets)):
            jaccard_scores.append(calculate_jaccard_similarity(ticker_sets[i], ticker_sets[j]))

    stability_metrics = {
        "fitness": {
            "mean": float(np.mean(fitness_values)),
            "median": float(np.median(fitness_values)),
            "std": float(np.std(fitness_values)),
            "min": float(np.min(fitness_values)),
            "max": float(np.max(fitness_values)),
            "cv": float(fitness_cv),
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
            "cv": float(hhi_cv),
        },
        "portfolio_similarity": {
            "jaccard_mean": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_std": float(np.std(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_min": float(np.min(jaccard_scores)) if jaccard_scores else 0.0,
            "jaccard_max": float(np.max(jaccard_scores)) if jaccard_scores else 1.0,
        }
    }

    return stability_metrics


def generate_consensus_summary(
        all_results: Dict,
        data_dir: Path,
        raw_data_path: Path
) -> Dict:
    """
    Gera summary consolidado das carteiras consenso no formato do summary_ga.json.

    Args:
        all_results: Resultados de todas as execuções
        data_dir: Diretório com dados processados
        raw_data_path: Path para dados brutos (raw)

    Returns:
        Dicionário com estrutura idêntica ao summary_ga.json
    """
    from cleaner import to_float

    METRIC_COLS = [
        "DY", "P/VP", "EV/EBIT",
        "ROE", "ROIC", "MARGEM EBIT", "MARG. LIQUIDA",
        "CAGR RECEITAS 5 ANOS", "CAGR LUCROS 5 ANOS", "PEG RATIO",
        "LIQ. CORRENTE", "DIVIDA LIQUIDA / EBIT", "DIV. LIQ. / PATRI.",
    ]

    IBOV_LIST = {
        "ABEV3","ALOS3","ASAI3","AURE3","AZUL4","AZZA3","B3SA3","BBAS3","BBDC3","BBDC4",
        "BBSE3","BEEF3","BPAC11","BRAP4","BRAV3","BRFS3","BRKM5","CMIG4","CMIN3","COGN3",
        "CPFE3","CPLE6","CRFB3","CSAN3","CSNA3","CVCB3","CXSE3","CYRE3","DIRR3","EGIE3",
        "ELET3","ELET6","EMBR3","ENEV3","ENGI11","EQTL3","FLRY3","GGBR4","GOAU4","HAPV3",
        "HYPE3","IGTI11","IRBR3","ISAE4","ITSA4","ITUB4","JBSS3","KLBN11","LREN3","MGLU3",
        "MOTV3","MRFG3","MRVE3","MULT3","NTCO3","PCAR3","PETR3","PETR4","PETZ3","POMO4",
        "PRIO3","PSSA3","RADL3","RAIL3","RAIZ4","RDOR3","RECV3","RENT3","SANB11","SBSP3",
        "SLCE3","SMFT3","SMTO3","STBP3","SUZB3","TAEE11","TIMS3","TOTS3","UGPA3","USIM5",
        "VALE3","VAMO3","VBBR3","VIVA3","VIVT3","WEGE3","YDUQ3"
    }

    print("\n📊 Gerando summary consolidado das carteiras consenso...")

    # Carrega dados raw para valores brutos
    df_raw = pd.read_csv(raw_data_path)
    df_raw["TICKER"] = df_raw["TICKER"].str.upper()

    # Converte métricas brutas para float
    for col in METRIC_COLS:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(to_float)

    # ===== IBOVESPA (benchmark) =====
    df_raw["IN_IBOV"] = df_raw["TICKER"].isin(IBOV_LIST)
    df_ibov = df_raw[df_raw["IN_IBOV"]].copy()

    summary = {
        "ibovespa": {
            "median_metrics": df_ibov[METRIC_COLS].median().to_dict()
        }
    }

    # ===== CADA PERFIL =====
    for perfil in ["conservador", "moderado", "arrojado", "caio"]:
        if perfil not in all_results:
            continue

        # Carrega dados padronizados (z-scores)
        df_zscore = pf.load_profile_data(perfil, data_dir)

        # Tickers da carteira consenso
        consensus_tickers = all_results[perfil]["consensus_portfolio"]["tickers"]

        # Filtra carteira consenso nos dados z-score
        cart_zscore = df_zscore[df_zscore["TICKER"].isin(consensus_tickers)].copy()

        # Filtra carteira consenso nos dados raw
        cart_raw = df_raw[df_raw["TICKER"].isin(consensus_tickers)].copy()

        # Calcula HHI
        hhi = calculate_hhi(cart_zscore)

        # Medianas em valores brutos
        raw_medians = {
            col: float(cart_raw[col].median()) if col in cart_raw.columns else None
            for col in METRIC_COLS
        }

        # Medianas em z-score
        zscore_medians = {
            col: float(cart_zscore[col].median()) if col in cart_zscore.columns else None
            for col in METRIC_COLS
        }

        # Distribuição setorial
        sector_weights = (
            cart_zscore["SETOR"].value_counts(normalize=True)
            .round(3)
            .to_dict()
        )

        summary[perfil] = {
            "num_assets": len(consensus_tickers),
            "hhi": round(hhi, 3),
            "median_metrics": raw_medians,
            "zscore_metrics": zscore_medians,
            "sector_weights": sector_weights,
            "multi_run_stats": all_results[perfil]["stability_metrics"],
            "consensus_info": {
                "avg_frequency": all_results[perfil]["consensus_portfolio"]["frequency_mean"],
                "n_runs": all_results[perfil]["n_runs"]
            }
        }

        print(f"  ✓ {perfil}: {len(consensus_tickers)} ativos, HHI={hhi:.3f}")

    return summary
# ==================== PIPELINE PRINCIPAL ====================

def main():
    """Pipeline de múltiplas execuções do GA."""

    all_results = {}

    for perfil in PROFILES:
        print(f"\n{'='*60}")
        print(f"Perfil: {perfil.upper()}")
        print(f"{'='*60}")

        # Carrega dados do perfil
        df = pf.load_profile_data(perfil, DATA_DIR)
        rank = pf.build_scores(df, perfil)

        # Lista para armazenar resultados de cada execução
        results = []

        # Executa GA N_RUNS vezes
        for run_id in range(N_RUNS):
            print(f"  Run {run_id + 1}/{N_RUNS}...", end=" ")

            try:
                result = run_ga_single_execution(rank, perfil, run_id)
                results.append(result)
                print(f"✓ Fitness: {result['fitness']:.2f}, HHI: {result['hhi']:.3f}")
            except Exception as e:
                print(f"✗ Erro: {e}")
                continue

        # Análise de estabilidade
        stability = analyze_stability(results)

        # Carteira consenso
        consensus = build_consensus_portfolio(results, rank, perfil)

        # Salva carteira consenso
        consensus_file = OUT_DIR / f"carteira_{perfil}_consensus.json"
        consensus.to_json(
            consensus_file,
            orient="records",
            indent=2,
            force_ascii=False
        )

        # Armazena resultados consolidados
        all_results[perfil] = {
            "n_runs": len(results),
            "n_successful": len(results),
            "stability_metrics": stability,
            "consensus_portfolio": {
                "tickers": consensus["TICKER"].tolist(),
                "hhi": float(consensus.attrs["hhi"]),
                "avg_score": float(consensus.attrs["avg_score"]),
                "frequency_mean": float(consensus["FREQUENCY"].mean()),
            },
            "all_runs": results
        }

        # Salva métricas detalhadas em CSV
        df_metrics = pd.DataFrame(results)
        metrics_file = OUT_DIR / f"metrics_stability_{perfil}.csv"
        df_metrics.to_csv(metrics_file, index=False)

        print(f"\n  📊 Fitness: {stability['fitness']['mean']:.2f} ± {stability['fitness']['std']:.2f}")
        print(f"  📊 HHI: {stability['hhi']['mean']:.3f} ± {stability['hhi']['std']:.3f}")
        print(f"  📊 Jaccard Médio: {stability['portfolio_similarity']['jaccard_mean']:.3f}")
        print(f"  💾 Carteira consenso: {consensus_file}")
        print(f"  💾 Métricas: {metrics_file}")

    # ===== NOVO: Gera summary consolidado =====
    raw_data_path = Path("data/raw/status_invest_fundamentals.csv")

    if raw_data_path.exists():
        consensus_summary = generate_consensus_summary(
            all_results,
            DATA_DIR,
            raw_data_path
        )

        # Salva summary
        save_consensus_summary(consensus_summary, OUT_DIR)
    else:
        print(f"\n⚠️  Arquivo raw não encontrado: {raw_data_path}")
        print("   Summary consolidado não foi gerado.")

    # Salva resumo geral (mantém compatibilidade)
    summary_file = OUT_DIR / "multiple_runs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Análise completa salva em: {summary_file}")
    print(f"{'='*60}\n")

    # Relatório de Interpretação
    print("\n📋 INTERPRETAÇÃO DOS RESULTADOS:")
    print("-" * 60)
    for perfil, data in all_results.items():
        stab = data["stability_metrics"]
        print(f"\n{perfil.upper()}:")
        print(f"  • CV do Fitness: {stab['fitness']['cv']:.2%}")
        print(f"    → {'BAIXA variabilidade' if stab['fitness']['cv'] < 0.05 else 'ALTA variabilidade'}")
        print(f"  • Índice Jaccard: {stab['portfolio_similarity']['jaccard_mean']:.3f}")
        print(f"    → {'ALTA similaridade' if stab['portfolio_similarity']['jaccard_mean'] > 0.7 else 'MODERADA similaridade'}")

# ==================== GERAÇÃO DE SUMMARY CONSOLIDADO ====================


def calculate_hhi(df: pd.DataFrame) -> float:
    """Calcula Herfindahl-Hirschman Index para concentração setorial."""
    weights = df["SETOR"].value_counts(normalize=True)
    return float((weights ** 2).sum())


def save_consensus_summary(summary: Dict, output_dir: Path):
    """Salva summary consolidado em JSON."""
    output_file = output_dir / "summary_ga_consensus.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Summary consolidado salvo em: {output_file}")
    return output_file

if __name__ == "__main__":
    main()