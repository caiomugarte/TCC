"""pipelines/single_run.py
=============================================================================
Pipeline para execução única do algoritmo genético.

Executa o fluxo completo:
1. Pré-processamento (com cache)
2. Cálculo de scores
3. Otimização via GA
4. Geração de relatórios
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import pandas as pd
from typing import Dict, Mapping, Optional
from tqdm import tqdm

from config import OUTPUTS_DIR, PROFILES, DATA_PROCESSED, RAW_DATA_FILE, METRIC_COLS
from core.preprocessing import (
    load_raw_data,
    preprocess_profile,
    apply_robustness_filter
)
from core.scoring import build_scores
from core.optimizer import optimize_portfolio
from core.metrics import hhi_sector
from utils.cache import CacheManager
from cleaner import to_float


def _explicit_stock_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalize the Premium stock selector contract."""

    result = dict(config)
    factor_weights = result.get("factor_weights", result.get("weights"))
    filters = result.get("liquidity_and_size_filters", result.get("filters"))
    if not isinstance(factor_weights, Mapping):
        raise ValueError("explicit stock config requires factor_weights")
    if not isinstance(filters, Mapping):
        raise ValueError(
            "explicit stock config requires liquidity_and_size_filters"
        )
    system_ga = result.get("system_ga_config")
    if not isinstance(system_ga, Mapping):
        raise ValueError("explicit stock config requires system_ga_config")
    settings = dict(system_ga)
    for key in (
        "n_assets", "lambda", "lambda_hhi", "generations", "pop_size",
        "population", "crossover_rate", "mutation_rate",
    ):
        if key in result:
            settings[key] = result[key]
    if "population" in settings and "pop_size" not in settings:
        settings["pop_size"] = settings["population"]
    if "lambda_hhi" in settings and "lambda" not in settings:
        settings["lambda"] = settings["lambda_hhi"]
    missing = [
        key for key in ("n_assets", "lambda", "generations", "pop_size")
        if key not in settings
    ]
    if missing:
        raise ValueError(f"explicit stock config missing fields: {missing}")
    result["factor_weights"] = dict(factor_weights)
    result["liquidity_and_size_filters"] = dict(filters)
    result["system_ga_config"] = settings
    result["n_assets"] = int(settings["n_assets"])
    result["lambda_hhi"] = float(settings["lambda"])
    return result


def run_stock_selection(
    input_path: Path = RAW_DATA_FILE,
    config: Optional[Mapping[str, object]] = None,
    workspace: Optional[Path] = None,
    seed: int = 42,
    robustness_filter: bool = True,
    *,
    selection_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Run the explicit stock selector and return a serializable result shape.

    The mapping contains ``selected_tickers``, equal ``sleeve_weights``,
    ``metrics``, ``exclusion_reasons``, ``seed``, ``workspace``, and the
    intermediate DataFrames under ``portfolio`` and ``ranked`` for adapters
    that need diagnostics.  No shared profile-named output is written.
    """

    if config is not None and selection_config is not None:
        raise ValueError("pass only one of config or selection_config")
    explicit = selection_config if selection_config is not None else config
    if explicit is None:
        raise ValueError("explicit stock selector config is required")
    resolved = _explicit_stock_config(explicit)
    raw = input_path if isinstance(input_path, pd.DataFrame) else load_raw_data(Path(input_path))
    clean = preprocess_profile(raw, config=resolved)
    if clean.empty:
        raise ValueError("no stock assets passed explicit eligibility filters")
    if robustness_filter:
        clean = apply_robustness_filter(clean)
    if clean.empty:
        raise ValueError("no stock assets passed the robustness filter")
    ranked = build_scores(clean, factor_weights=resolved["factor_weights"])
    portfolio = optimize_portfolio(ranked, config=resolved, seed=seed)
    selected_tickers = sorted(str(ticker).upper() for ticker in portfolio["TICKER"])
    sleeve_weights = {
        ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers
    }

    workspace_path = Path(workspace) if workspace is not None else None
    output_path = None
    processed_path = None
    if workspace_path is not None:
        workspace_path.mkdir(parents=True, exist_ok=True)
        output_path = workspace_path / "stock_selection.json"
        processed_path = workspace_path / "stock_processed.csv"
        portfolio.to_json(output_path, orient="records", indent=2, force_ascii=False)
        clean.to_csv(processed_path, index=False)

    return {
        "selection_preset": resolved.get("selection_preset", "explicit"),
        "selected_tickers": selected_tickers,
        "sleeve_weights": sleeve_weights,
        "weights": sleeve_weights,
        "metrics": {
            "fitness": float(portfolio.attrs["fitness"]),
            "hhi": float(portfolio.attrs["hhi"]),
            "generations_run": int(portfolio.attrs.get("generations_run", 0)),
            "converged_early": bool(portfolio.attrs.get("converged_early", False)),
            "score_mean": float(portfolio["SCORE"].mean()),
            "score_median": float(portfolio["SCORE"].median()),
        },
        "exclusion_reasons": dict(clean.attrs.get("exclusion_reasons", {})),
        "config": resolved,
        "seed": int(seed),
        "workspace": str(workspace_path) if workspace_path is not None else None,
        "output_path": output_path,
        "processed_path": processed_path,
        "portfolio": portfolio,
        "ranked": ranked,
    }


def run_single_portfolio(
    profile: Optional[str] = None,
    use_cache: bool = True,
    robustness_filter: bool = True,
    random_seed: Optional[int] = None,
    input_path: Optional[Path] = None,
    config: Optional[Mapping[str, object]] = None,
    selection_config: Optional[Mapping[str, object]] = None,
    workspace: Optional[Path] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Executa pipeline completo para um único perfil.

    Parameters
    ----------
    profile : str
        Perfil do investidor.
    use_cache : bool
        Se True, usa cache para etapas intermediárias.
    robustness_filter : bool
        Se True, aplica filtro de qualidade (≥80% métricas preenchidas).
    random_seed : int, optional
        Seed para reprodutibilidade do GA.

    Returns
    -------
    pd.DataFrame
        Carteira otimizada.
    """
    explicit = selection_config if selection_config is not None else config
    if explicit is not None:
        if selection_config is not None and config is not None:
            raise ValueError("pass only one of config or selection_config")
        return run_stock_selection(
            input_path=input_path or RAW_DATA_FILE,
            config=explicit,
            workspace=workspace,
            seed=(seed if seed is not None else (random_seed if random_seed is not None else 42)),
            robustness_filter=robustness_filter,
        )["portfolio"]
    if profile is None:
        raise ValueError("profile is required for a named stock run")
    cache = CacheManager()

    # 1. Carrega dados brutos
    print(f"[{profile}] Carregando dados brutos...")
    df_raw = load_raw_data(input_path or RAW_DATA_FILE)

    # 2. Pré-processamento (com cache)
    print(f"[{profile}] Pré-processando dados...")
    cache_key = f"preprocessing_{profile}"

    if use_cache:
        df_clean = cache.get_or_compute(
            key=cache_key,
            compute_fn=lambda: preprocess_profile(df_raw, profile),
            dependencies=[str(input_path or RAW_DATA_FILE)],
            format="csv"
        )
    else:
        df_clean = preprocess_profile(df_raw, profile)

    if df_clean.empty:
        raise ValueError(f"Nenhum ativo disponível para perfil {profile}")

    # 3. Filtro de robustez (opcional)
    if robustness_filter:
        print(f"[{profile}] Aplicando filtro de robustez...")
        df_clean = apply_robustness_filter(df_clean)

    # 4. Calcula scores
    print(f"[{profile}] Calculando scores...")
    df_ranked = build_scores(df_clean, profile)

    # 5. Otimiza carteira via GA
    print(f"[{profile}] Executando Algoritmo Genético...")
    portfolio = optimize_portfolio(
        df_ranked,
        profile,
        random_seed=(seed if seed is not None else random_seed),
    )

    print(f"[{profile}] ✓ Carteira otimizada: {len(portfolio)} ativos")
    print(f"[{profile}]   Fitness: {portfolio.attrs['fitness']:.2f}")
    print(f"[{profile}]   HHI: {portfolio.attrs['hhi']:.3f}")

    return portfolio


def run_all_profiles(
    use_cache: bool = True,
    robustness_filter: bool = True,
    save_outputs: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Executa pipeline para todos os perfis.

    Parameters
    ----------
    use_cache : bool
        Se True, usa cache.
    robustness_filter : bool
        Se True, aplica filtro de robustez.
    save_outputs : bool
        Se True, salva carteiras e summary em outputs/.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dicionário {profile: portfolio}.
    """
    print("=" * 70)
    print("PIPELINE: Execução Única do Algoritmo Genético")
    print("=" * 70)

    portfolios = {}

    for profile in tqdm(PROFILES, desc="Processando perfis"):
        portfolio = run_single_portfolio(
            profile=profile,
            use_cache=use_cache,
            robustness_filter=robustness_filter
        )
        portfolios[profile] = portfolio

    if save_outputs:
        print("\nSalvando outputs...")
        save_portfolios(portfolios)
        save_summary(portfolios)

    print("\n" + "=" * 70)
    print("✓ Pipeline concluído com sucesso!")
    print("=" * 70)

    return portfolios


def save_portfolios(portfolios: Dict[str, pd.DataFrame]):
    """
    Salva carteiras individuais em JSON.

    Parameters
    ----------
    portfolios : Dict[str, pd.DataFrame]
        Dicionário de carteiras por perfil.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for profile, portfolio in portfolios.items():
        outfile = OUTPUTS_DIR / f"carteira_{profile}_ga.json"
        portfolio.to_json(
            outfile,
            orient="records",
            indent=2,
            force_ascii=False
        )
        print(f"  ✓ {outfile}")


def save_summary(portfolios: Dict[str, pd.DataFrame]):
    """
    Gera e salva summary consolidado.

    Parameters
    ----------
    portfolios : Dict[str, pd.DataFrame]
        Dicionário de carteiras por perfil.
    """
    # Carrega dados raw para métricas brutas
    df_raw = load_raw_data()

    # Converte métricas brutas para float
    for col in METRIC_COLS:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].apply(to_float)

    # Benchmark: Ibovespa
    df_ibov = df_raw[df_raw["IN_IBOV"]].copy()

    summary = {
        "ibovespa": {
            "median_metrics": df_ibov[METRIC_COLS].median().to_dict()
        }
    }

    # Cada perfil
    for profile, portfolio in portfolios.items():
        tickers = portfolio["TICKER"].str.upper().tolist()
        df_sel_raw = df_raw[df_raw["TICKER"].isin(tickers)].copy()

        # Medianas em valores brutos
        raw_medians = {
            col: float(df_sel_raw[col].median()) if col in df_sel_raw else None
            for col in METRIC_COLS
        }

        # Medianas em z-score
        zscore_medians = {
            col: float(portfolio[col].median()) if col in portfolio else None
            for col in METRIC_COLS
        }

        # Distribuição setorial
        sector_weights = (
            portfolio["SETOR"]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )

        summary[profile] = {
            "num_assets": len(portfolio),
            "hhi": round(portfolio.attrs["hhi"], 3),
            "fitness": round(portfolio.attrs["fitness"], 2),
            "median_metrics": raw_medians,
            "zscore_metrics": zscore_medians,
            "sector_weights": sector_weights,
        }

    # Salva
    summary_file = OUTPUTS_DIR / "summary_ga.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {summary_file}")
