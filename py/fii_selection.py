"""Independent FII preprocessing, scoring, and equal-weight selection."""

from __future__ import annotations

from collections import Counter
from functools import partial
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from core.metrics import (  # noqa: E402
    coefficient_of_variation,
    hhi_sector,
    jaccard_similarity,
)
from core.optimizer import GeneticAlgorithm, derive_run_seed  # noqa: E402
from fetch_status_invest_fii import (  # noqa: E402
    RAW_DATA_FILE,
    SOURCE_COLUMNS,
)


PROJECT_ROOT = PY_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

NUMERIC_COLUMNS = tuple(
    column for column in SOURCE_COLUMNS if column not in ("TICKER", "GESTAO")
)
FII_SCORE_GROUPS: Dict[str, List[str]] = {
    "liquidity": ["LIQUIDEZ MEDIA DIARIA", "N COTISTAS"],
    "size_cash": ["PATRIMONIO", "N COTAS", "PERCENTUAL EM CAIXA"],
    "value": ["P/VP"],
    "growth": ["CAGR DIVIDENDOS 3 ANOS", "CAGR VALOR COTA 3 ANOS"],
    "dividend": ["DY", "ULTIMO DIVIDENDO"],
}

# Explicit FII defaults. They preserve the existing profile proportions while
# replacing stock profitability with FII size/cash indicators.
FII_PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "conservador": {
        "liquidity": 0.30,
        "size_cash": 0.25,
        "value": 0.15,
        "growth": 0.10,
        "dividend": 0.20,
    },
    "moderado": {
        "liquidity": 0.20,
        "size_cash": 0.25,
        "value": 0.25,
        "growth": 0.20,
        "dividend": 0.10,
    },
    "arrojado": {
        "liquidity": 0.10,
        "size_cash": 0.20,
        "value": 0.20,
        "growth": 0.40,
        "dividend": 0.10,
    },
    "caio": {
        "liquidity": 0.29,
        "size_cash": 0.14,
        "value": 0.26,
        "growth": 0.05,
        "dividend": 0.26,
    },
    "caio_new": {
        "liquidity": 0.1338,
        "size_cash": 0.2169,
        "value": 0.2169,
        "growth": 0.3324,
        "dividend": 0.10,
    },
    "caio_last": {
        "liquidity": 0.25,
        "size_cash": 0.25,
        "value": 0.20,
        "growth": 0.15,
        "dividend": 0.15,
    },
}

FII_GA_CONFIG: Dict[str, Dict[str, int | float]] = {
    "conservador": {"n_assets": 10, "lambda": 0.50, "generations": 300, "pop_size": 200},
    "moderado": {"n_assets": 12, "lambda": 0.25, "generations": 400, "pop_size": 250},
    "arrojado": {"n_assets": 15, "lambda": 0.10, "generations": 500, "pop_size": 300},
    "caio": {"n_assets": 10, "lambda": 0.37, "generations": 600, "pop_size": 400},
    "caio_new": {"n_assets": 14, "lambda": 0.151, "generations": 470, "pop_size": 280},
    "caio_last": {"n_assets": 11, "lambda": 0.375, "generations": 350, "pop_size": 230},
}

HEADER_ALIASES = {
    "CAGR VALOR CORA 3 ANOS": "CAGR VALOR COTA 3 ANOS",
}
CORE_POSITIVE_COLUMNS = (
    "PRECO",
    "P/VP",
    "LIQUIDEZ MEDIA DIARIA",
    "PATRIMONIO",
    "N COTISTAS",
    "N COTAS",
)


class FiiSelectionError(ValueError):
    """Raised when FII input cannot be selected safely."""


FII_FILTER_ALIASES = {
    "liquidity_min": ("liquidity_min", "liq_min", "min_liquidity"),
    "patrimony_min": (
        "patrimony_min", "patrimonio_min", "min_patrimony", "size_min"
    ),
    "holders_min": ("holders_min", "cotistas_min", "min_holders"),
    "shares_min": ("shares_min", "cotas_min", "min_shares"),
    "price_min": ("price_min", "preco_min", "min_price"),
    "p_vp_min": ("p_vp_min", "min_p_vp"),
}


def _normalise_fii_filters(
    filters: Optional[Mapping[str, object]],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for target, aliases in FII_FILTER_ALIASES.items():
        raw = next(
            (filters[name] for name in aliases if filters and name in filters),
            None,
        )
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise FiiSelectionError(
                f"FII filter {target} must be finite and non-negative"
            )
        result[target] = value
    return result


def _canonical_column(column: str) -> str:
    normalized = str(column).replace("\ufeff", "").strip().upper()
    return HEADER_ALIASES.get(normalized, normalized)


def _to_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text.casefold() in {"", "-", "--", "nan", "null", "n/a"}:
        return np.nan
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError as exc:
        raise FiiSelectionError(f"invalid numeric value: {value!r}") from exc


def load_fii_data(
    file_path: Path = RAW_DATA_FILE,
    sector_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load a normalized FII dataset; require real segment labels."""

    if not file_path.exists():
        raise FiiSelectionError(f"FII dataset not found: {file_path}")
    df = pd.read_csv(file_path, sep=None, engine="python", dtype=str, keep_default_na=False)
    df.columns = [_canonical_column(column) for column in df.columns]
    if "SETOR" not in df.columns:
        if not sector_name:
            raise FiiSelectionError("FII dataset must contain SETOR")
        df["SETOR"] = sector_name

    required = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if required:
        raise FiiSelectionError(f"FII dataset missing columns: {', '.join(required)}")
    if df["SETOR"].astype(str).str.strip().eq("").any():
        raise FiiSelectionError("FII dataset contains a blank SETOR")

    df["TICKER"] = df["TICKER"].astype(str).str.strip().str.upper()
    if df["TICKER"].eq("").any():
        raise FiiSelectionError("FII dataset contains a blank TICKER")
    if df["TICKER"].duplicated().any():
        duplicated = df.loc[df["TICKER"].duplicated(), "TICKER"].tolist()
        raise FiiSelectionError(f"duplicate FII tickers: {duplicated[:5]}")

    for column in NUMERIC_COLUMNS:
        df[column] = df[column].map(_to_number)
    return df


def apply_fii_eligibility(
    df: pd.DataFrame,
    filters: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Keep FIIs with usable data and optional explicit size/liquidity bounds."""

    missing = [column for column in CORE_POSITIVE_COLUMNS if column not in df.columns]
    if missing:
        raise FiiSelectionError(f"FII eligibility columns missing: {', '.join(missing)}")
    normalised_filters = _normalise_fii_filters(filters)
    positive = (df[list(CORE_POSITIVE_COLUMNS)] > 0).all(axis=1)
    if "DY" in df.columns:
        positive &= df["DY"].ge(0)
    threshold_columns = {
        "liquidity_min": "LIQUIDEZ MEDIA DIARIA",
        "patrimony_min": "PATRIMONIO",
        "holders_min": "N COTISTAS",
        "shares_min": "N COTAS",
        "price_min": "PRECO",
        "p_vp_min": "P/VP",
    }
    for key, column in threshold_columns.items():
        if key in normalised_filters:
            positive &= df[column].ge(normalised_filters[key])
    result = df.loc[positive].copy()
    ticker_values = df["TICKER"].astype(str)
    reasons: Dict[str, List[str]] = {}
    for index in df.index[~positive]:
        ticker = ticker_values.loc[index].strip().upper()
        item_reasons = []
        for column in CORE_POSITIVE_COLUMNS:
            if pd.isna(df.loc[index, column]) or df.loc[index, column] <= 0:
                item_reasons.append(f"{column}:non_positive_or_missing")
        if "DY" in df.columns and (
            pd.isna(df.loc[index, "DY"]) or df.loc[index, "DY"] < 0
        ):
            item_reasons.append("DY:negative_or_missing")
        for key, column in threshold_columns.items():
            if key in normalised_filters and (
                pd.isna(df.loc[index, column])
                or df.loc[index, column] < normalised_filters[key]
            ):
                item_reasons.append(f"{column}:below_minimum")
        reasons[ticker] = item_reasons or ["eligibility_filter"]
    if result.empty:
        raise FiiSelectionError("no FIIs passed core eligibility filters")
    result.attrs["exclusion_reasons"] = reasons
    result.attrs["eligibility_filters"] = normalised_filters
    return result


def _winsorize(series: pd.Series, percentile: float = 0.01) -> pd.Series:
    values = series.dropna()
    if values.empty:
        return series
    return series.clip(values.quantile(percentile), values.quantile(1 - percentile))


def _zscore(series: pd.Series) -> pd.Series:
    values = series.dropna()
    if values.empty:
        return series
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series.where(series.isna(), 0.0)
    return (series - values.mean()) / std


def preprocess_fii(
    df_raw: pd.DataFrame,
    eligibility_filters: Optional[Mapping[str, object]] = None,
    filters: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Apply FII eligibility, segment winsorization, inversion, and z-scores."""

    if eligibility_filters is not None and filters is not None:
        raise FiiSelectionError("pass only one of eligibility_filters or filters")
    df = apply_fii_eligibility(
        df_raw,
        filters=eligibility_filters if eligibility_filters is not None else filters,
    )
    score_columns = [
        column
        for columns in FII_SCORE_GROUPS.values()
        for column in columns
        if column in df.columns
    ]
    df[score_columns] = df.groupby("SETOR")[score_columns].transform(_winsorize)
    if "P/VP" in df.columns:
        df["P/VP"] = df["P/VP"] * -1
    df[score_columns] = df.groupby("SETOR")[score_columns].transform(_zscore)
    result = df.reset_index(drop=True)
    result.attrs.update(df.attrs)
    return result


def build_fii_scores(
    df: pd.DataFrame,
    profile: Optional[str] = None,
    profile_weights: Optional[Mapping[str, float]] = None,
    factor_weights: Optional[Mapping[str, float]] = None,
    config: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Calculate FII group scores and sort assets by weighted score."""

    if config is not None:
        factor_weights = factor_weights or config.get("factor_weights")
        factor_weights = factor_weights or config.get("weights")
    explicit_weights = factor_weights or profile_weights
    if profile is None and explicit_weights is None:
        raise FiiSelectionError("profile or explicit FII factor weights is required")
    if explicit_weights is None and profile not in FII_PROFILE_WEIGHTS:
        raise FiiSelectionError(f"unknown FII profile: {profile}")
    weights = dict(explicit_weights or FII_PROFILE_WEIGHTS[profile])
    unknown_groups = set(weights) - set(FII_SCORE_GROUPS)
    missing_groups = [group for group in FII_SCORE_GROUPS if group not in weights]
    if missing_groups or unknown_groups:
        raise FiiSelectionError(
            "FII score weights must contain exactly the configured groups "
            f"(missing={missing_groups}, unknown={sorted(unknown_groups)})"
        )
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in weights.values()
    ):
        raise FiiSelectionError("FII score weights must be finite and non-negative")
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise FiiSelectionError("FII score weights must sum to 1")

    result = df.copy()
    for group, columns in FII_SCORE_GROUPS.items():
        available = [column for column in columns if column in result.columns]
        if not available:
            raise FiiSelectionError(f"FII score group has no columns: {group}")
        result[f"AVG_{group.upper()}"] = result[available].mean(axis=1, skipna=True)
    result["SCORE"] = sum(
        weights[group] * result[f"AVG_{group.upper()}"]
        for group in FII_SCORE_GROUPS
    )
    result = result.dropna(subset=["SCORE"])
    if result.empty:
        raise FiiSelectionError("no FIIs have a usable score")
    return result.sort_values(["SCORE", "TICKER"], ascending=[False, True]).reset_index(drop=True)


def optimize_fii_portfolio(
    df_ranked: pd.DataFrame,
    profile: Optional[str] = None,
    random_seed: Optional[int] = None,
    ga_config: Optional[Mapping[str, int | float]] = None,
    config: Optional[Mapping[str, object]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run the existing binary GA with explicit or named FII parameters."""

    if random_seed is not None and seed is not None and random_seed != seed:
        raise FiiSelectionError("random_seed and seed disagree")
    random_seed = seed if seed is not None else random_seed
    if config is not None:
        nested = config.get("system_ga_config")
        settings = dict(nested) if isinstance(nested, Mapping) else {}
        if "n_assets" in config:
            settings["n_assets"] = config["n_assets"]
        for key in (
            "lambda", "lambda_hhi", "generations", "pop_size", "population",
            "crossover_rate", "mutation_rate",
        ):
            if key in config:
                settings[key] = config[key]
        filters = config.get("liquidity_and_size_filters") or config.get(
            "eligibility_filters"
        )
        if isinstance(filters, Mapping) and "hhi_max" in filters:
            settings.setdefault("hhi_max", filters["hhi_max"])
    else:
        settings = dict(ga_config or {})
    if "population" in settings and "pop_size" not in settings:
        settings["pop_size"] = settings["population"]
    if not settings and config is None:
        if profile not in FII_GA_CONFIG:
            raise FiiSelectionError("profile or explicit FII GA config is required")
        settings = dict(FII_GA_CONFIG[profile])
    if "lambda_hhi" in settings and "lambda" not in settings:
        settings["lambda"] = settings["lambda_hhi"]
    required = ("n_assets", "lambda", "generations", "pop_size")
    missing = [key for key in required if key not in settings]
    if missing:
        raise FiiSelectionError(f"FII GA config missing fields: {missing}")
    n_assets = int(settings["n_assets"])
    if n_assets <= 0 or len(df_ranked) < n_assets:
        raise FiiSelectionError(
            f"FII universe has {len(df_ranked)} assets; {n_assets} required"
        )
    optimizer = GeneticAlgorithm(
        n_assets=n_assets,
        lambda_hhi=float(settings["lambda"]),
        generations=int(settings["generations"]),
        pop_size=int(settings["pop_size"]),
        crossover_rate=float(settings.get("crossover_rate", 0.8)),
        mutation_rate=float(settings.get("mutation_rate", 0.02)),
        random_seed=random_seed,
        hhi_max=settings.get("hhi_max"),
    )
    return optimizer.optimize(df_ranked)


def _run_fii_execution(
    df_ranked: pd.DataFrame,
    profile: Optional[str],
    settings: Mapping[str, int | float],
    random_seed: int,
    run_id: int,
    config: Optional[Mapping[str, object]] = None,
    workspace: Optional[Path] = None,
) -> tuple[Dict[str, object], pd.DataFrame]:
    seed = derive_run_seed(random_seed, run_id, "fii")
    portfolio = optimize_fii_portfolio(
        df_ranked,
        profile,
        random_seed=seed,
        ga_config=settings,
        config=config,
    )
    run_workspace = None
    if workspace is not None:
        run_workspace = Path(workspace) / f"run-{run_id}"
        run_workspace.mkdir(parents=True, exist_ok=True)
        portfolio.to_json(
            run_workspace / "portfolio.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
    return {
        "run_id": run_id,
        "seed": seed,
        "tickers": sorted(portfolio["TICKER"].tolist()),
        "fitness": float(portfolio.attrs["fitness"]),
        "hhi": float(portfolio.attrs["hhi"]),
        "generations_run": portfolio.attrs.get("generations_run", 0),
        "converged_early": portfolio.attrs.get("converged_early", False),
        "workspace": str(run_workspace) if run_workspace is not None else None,
    }, portfolio


def _fii_stability(results: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    if not results:
        return {"fitness_cv": 0.0, "jaccard_mean": 0.0}

    fitness_values = [float(result["fitness"]) for result in results]
    ticker_sets = [set(result["tickers"]) for result in results]
    jaccard_values = [
        jaccard_similarity(ticker_sets[i], ticker_sets[j])
        for i in range(len(ticker_sets))
        for j in range(i + 1, len(ticker_sets))
    ]
    return {
        "fitness_cv": coefficient_of_variation(fitness_values),
        "jaccard_mean": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
    }


def _consensus(
    portfolios: Sequence[pd.DataFrame],
    df_ranked: pd.DataFrame,
    n_assets: int,
) -> pd.DataFrame:
    counts = Counter(
        ticker
        for portfolio in portfolios
        for ticker in portfolio["TICKER"].tolist()
    )
    scores = df_ranked.set_index("TICKER")["SCORE"]
    selected = sorted(
        counts,
        key=lambda ticker: (-counts[ticker], -float(scores[ticker]), ticker),
    )[:n_assets]
    result = df_ranked[df_ranked["TICKER"].isin(selected)].copy()
    result["FREQUENCY"] = result["TICKER"].map(
        lambda ticker: counts[ticker] / len(portfolios)
    )
    return result.sort_values(["FREQUENCY", "SCORE", "TICKER"], ascending=[False, False, True])


def run_fii_selection(
    profile: Optional[str] = None,
    input_path: Path = RAW_DATA_FILE,
    output_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
    n_runs: int = 1,
    random_seed: int = 42,
    sector_name: Optional[str] = None,
    ga_config: Optional[Mapping[str, int | float]] = None,
    parallel: bool = False,
    adaptive_mode: bool = False,
    min_runs: int = 30,
    target_cv: float = 0.03,
    target_jaccard: float = 0.70,
    selection_config: Optional[Mapping[str, object]] = None,
    config: Optional[Mapping[str, object]] = None,
    workspace: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run FII selection and return a structured sleeve result.

    Premium callers pass ``selection_config`` (``config`` is an alias) with
    explicit factor weights, eligibility filters, asset count, HHI penalty, and
    system GA settings.  The result includes ``selected_tickers``, equal
    ``sleeve_weights``, metrics, exclusion reasons, seed, and workspace.
    Named ``profile`` calls retain the existing offline behavior.
    """

    if selection_config is not None and config is not None:
        raise FiiSelectionError("pass only one of selection_config or config")
    explicit_config = selection_config if selection_config is not None else config
    explicit_mode = explicit_config is not None
    if seed is not None:
        if random_seed != 42 and random_seed != seed:
            raise FiiSelectionError("random_seed and seed disagree")
        random_seed = seed
    if n_runs <= 0:
        raise FiiSelectionError("n_runs must be positive")
    if min_runs <= 0:
        raise FiiSelectionError("min_runs must be positive")

    if explicit_mode:
        resolved_config = dict(explicit_config)
        factor_weights = resolved_config.get("factor_weights")
        if factor_weights is None:
            factor_weights = resolved_config.get("weights")
        if factor_weights is None:
            raise FiiSelectionError("explicit FII config requires factor_weights")
        filters = resolved_config.get("liquidity_and_size_filters")
        if filters is None:
            filters = resolved_config.get("eligibility_filters")
        system_ga = resolved_config.get("system_ga_config", {})
        if not isinstance(system_ga, Mapping):
            raise FiiSelectionError("system_ga_config must be an object")
        settings = dict(system_ga)
        if ga_config:
            settings.update(dict(ga_config))
        for key in ("n_assets", "lambda", "lambda_hhi"):
            if key in resolved_config:
                settings[key] = resolved_config[key]
        if "population" in settings and "pop_size" not in settings:
            settings["pop_size"] = settings["population"]
        if "lambda_hhi" in settings and "lambda" not in settings:
            settings["lambda"] = settings["lambda_hhi"]
        required_settings = ("n_assets", "lambda", "generations", "pop_size")
        missing = [key for key in required_settings if key not in settings]
        if missing:
            raise FiiSelectionError(f"explicit FII config missing fields: {missing}")
        resolved_config["system_ga_config"] = dict(settings)
        selection_profile = None
    else:
        if profile not in FII_PROFILE_WEIGHTS:
            raise FiiSelectionError(f"unknown FII profile: {profile}")
        factor_weights = None
        filters = None
        settings = dict(ga_config or FII_GA_CONFIG[profile])
        selection_profile = profile

    raw = load_fii_data(input_path, sector_name=sector_name)
    clean = preprocess_fii(raw, eligibility_filters=filters)
    ranked = build_fii_scores(
        clean,
        selection_profile,
        factor_weights=factor_weights,
    )
    n_assets = int(settings["n_assets"])

    execution_results = []
    portfolios = []
    workspace_path = Path(workspace) if workspace is not None else None
    if workspace_path is not None:
        workspace_path.mkdir(parents=True, exist_ok=True)
    run = partial(
        _run_fii_execution,
        ranked,
        selection_profile,
        settings,
        random_seed,
        config=resolved_config if explicit_mode else None,
        workspace=workspace_path,
    )

    if parallel:
        batch_size = min(10, n_runs)
        for batch_start in range(0, n_runs, batch_size):
            batch_end = min(batch_start + batch_size, n_runs)
            with mp.Pool() as pool:
                batch = pool.starmap(run, [(run_id,) for run_id in range(batch_start, batch_end)])
            for execution_result, portfolio in batch:
                run_id = execution_result["run_id"]
                execution_result["workspace"] = (
                    str(workspace_path / f"run-{run_id}")
                    if workspace_path is not None
                    else None
                )
                execution_results.append(execution_result)
                portfolios.append(portfolio)
            stability = _fii_stability(execution_results)
            if (
                adaptive_mode
                and len(execution_results) >= min_runs
                and stability["fitness_cv"] <= target_cv
                and stability["jaccard_mean"] >= target_jaccard
            ):
                break
    else:
        for run_id in range(n_runs):
            execution_result, portfolio = run(run_id)
            execution_result["workspace"] = (
                str(workspace_path / f"run-{run_id}")
                if workspace_path is not None
                else None
            )
            execution_results.append(execution_result)
            portfolios.append(portfolio)
            stability = _fii_stability(execution_results)
            if (
                adaptive_mode
                and len(execution_results) >= min_runs
                and stability["fitness_cv"] <= target_cv
                and stability["jaccard_mean"] >= target_jaccard
            ):
                break

    selected = _consensus(portfolios, ranked, n_assets)
    selected.attrs["hhi"] = hhi_sector(selected)
    hhi_max = None
    if explicit_mode:
        filters_for_hhi = resolved_config.get("liquidity_and_size_filters")
        if isinstance(filters_for_hhi, Mapping):
            hhi_max = filters_for_hhi.get("hhi_max")
    if hhi_max is not None and selected.attrs["hhi"] > float(hhi_max) + 1e-12:
        raise FiiSelectionError("FII selection violates the configured HHI limit")

    if explicit_mode:
        if workspace_path is not None:
            processed_path = processed_path or workspace_path / "fii_clean.csv"
            output_path = output_path or workspace_path / "fii_selection.json"
    else:
        processed_path = processed_path or PROCESSED_DIR / f"fii_clean_{profile}.csv"
        output_path = output_path or OUTPUTS_DIR / f"carteira_fii_{profile}_consensus.json"
    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(processed_path, index=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_json(output_path, orient="records", indent=2, force_ascii=False)

    selected_tickers = selected["TICKER"].tolist()
    sleeve_weights = {
        ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers
    }
    resolved_output_config = {
        "selection_preset": (
            resolved_config.get("selection_preset", "explicit")
            if explicit_mode
            else profile
        ),
        "factor_weights": dict(
            factor_weights or FII_PROFILE_WEIGHTS[profile]
        ),
        "liquidity_and_size_filters": dict(_normalise_fii_filters(filters)),
        "n_assets": n_assets,
        "lambda_hhi": float(settings["lambda"]),
        "system_ga_config": dict(settings),
    }

    return {
        "profile": profile if profile is not None else "explicit",
        "n_runs": len(execution_results),
        "n_candidates": len(ranked),
        "n_selected": len(selected),
        "hhi": float(selected.attrs["hhi"]),
        "stability": _fii_stability(execution_results),
        "portfolio": selected,
        "ranked": ranked,
        "output_path": output_path,
        "processed_path": processed_path,
        "selected_tickers": selected_tickers,
        "sleeve_weights": sleeve_weights,
        "weights": sleeve_weights,
        "metrics": {
            "hhi": float(selected.attrs["hhi"]),
            "n_candidates": len(ranked),
            "n_selected": len(selected),
            "stability": _fii_stability(execution_results),
        },
        "exclusion_reasons": dict(clean.attrs.get("exclusion_reasons", {})),
        "config": resolved_output_config,
        "seed": int(random_seed),
        "workspace": str(workspace_path) if workspace_path is not None else None,
    }
