"""Entradas do GA de ações do Premium, montadas como o pipeline de produção (pesquisa #100).

Reconstrói, sem rede, o ``df_ranked`` e a configuração do GA de ações do perfil moderado sobre o manifesto
``cvm-b3-2026-11-17``: política por ``benchmark_premium.build_policy`` (perfil sintético no score âncora,
restrição ``nenhuma``), seção do seletor por ``_selector_config`` do adapter e preparo dos dados igual ao de
``run_multi_execution_profile`` (``load_raw_data`` → ``preprocess_profile`` → ``apply_robustness_filter`` →
``build_scores``).

Variáveis: ``PRUMO_REPO`` (raiz do worktree; padrão: três níveis acima deste arquivo) e
``PRUMO_SNAPSHOTS`` (registry local; padrão ``/home/caio/Documents/projetos/TCC/snapshots``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("PRUMO_REPO", Path(__file__).resolve().parents[3]))
REGISTRY = Path(os.environ.get("PRUMO_SNAPSHOTS", "/home/caio/Documents/projetos/TCC/snapshots"))
MANIFEST_ID = "cvm-b3-2026-11-17"
for p in (REPO / "scripts", REPO / "api", REPO / "py"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402


def load_inputs(profile: str = "moderado", policy_seed: int = 42):
    """Devolve (df_ranked, selection_config, policy, manifest)."""
    import benchmark_premium as bp
    from snapshot_manifest import validate_manifest
    from app.adapters.premium_optimization import _selector_config

    manifest = validate_manifest(REGISTRY / f"{MANIFEST_ID}.json")
    policy = bp.build_policy(profile, manifest, policy_seed)
    config = _selector_config(policy["stocks"])
    source = manifest.selector_source_for("stock")
    stocks_csv = Path(source.path)
    if not stocks_csv.is_absolute():
        stocks_csv = REGISTRY / stocks_csv
    df_ranked = ranked_frame(stocks_csv, config)
    return df_ranked, config, policy, manifest


def ranked_frame(input_path: Path, config):
    """Mesmo preparo de ``run_multi_execution_profile`` no modo explícito."""
    import pipelines.multi_run as mr

    df_raw = mr.load_raw_data(Path(input_path))
    df = mr.preprocess_profile(df_raw, config=config)
    df = mr.apply_robustness_filter(df)
    weights = config.get("factor_weights")
    if weights is None:
        weights = config.get("weights")
    return mr.build_scores(df, factor_weights=weights)


def ga_settings(config):
    from core.optimizer import _explicit_ga_config

    return _explicit_ga_config(None, None, config)


def make_ga(config, seed, *, generations=None, pop_size=None, patience=None, mutation_rate=None):
    """GA igual ao de ``optimize_portfolio``; os três knobs só mudam no item (d)."""
    from core.optimizer import GeneticAlgorithm
    from config import GA_CROSSOVER_RATE, GA_MUTATION_RATE

    cfg = ga_settings(config)
    kw = {}
    if patience is not None:
        kw["early_stopping_patience"] = int(patience)
    return GeneticAlgorithm(
        n_assets=cfg["n_assets"],
        lambda_hhi=cfg.get("lambda", cfg.get("lambda_hhi", 0.0)),
        generations=int(generations or cfg["generations"]),
        pop_size=int(pop_size or cfg["pop_size"]),
        crossover_rate=cfg.get("crossover_rate", GA_CROSSOVER_RATE),
        mutation_rate=mutation_rate if mutation_rate is not None else cfg.get("mutation_rate", GA_MUTATION_RATE),
        random_seed=seed,
        hhi_max=cfg.get("hhi_max"),
        **kw,
    )


def fitness_of(df_ranked, config, tickers) -> float:
    """Fitness do GA (``GeneticAlgorithm.fitness``) de uma lista de tickers qualquer."""
    ga = make_ga(config, 0)
    mask = df_ranked["TICKER"].isin(list(tickers)).to_numpy().astype(int)
    return float(ga.fitness(df_ranked, mask))


def consensus_tickers(results, df_ranked, config):
    """Consenso de produção (frequência, desempate por score, sem zona: sem titulares)."""
    import contextlib
    import io
    from pipelines.multi_run import build_consensus_portfolio

    with contextlib.redirect_stdout(io.StringIO()):
        port = build_consensus_portfolio(results, df_ranked, None, selection_config=config)
    return sorted(port["TICKER"].tolist())
