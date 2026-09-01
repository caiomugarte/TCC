"""Independent FII preprocessing, scoring, and equal-weight selection."""

from __future__ import annotations

from collections import Counter
from functools import partial
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
from core.optimizer import GeneticAlgorithm  # noqa: E402
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


def apply_fii_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Keep FIIs with usable price, valuation, size, and liquidity data."""

    missing = [column for column in CORE_POSITIVE_COLUMNS if column not in df.columns]
    if missing:
        raise FiiSelectionError(f"FII eligibility columns missing: {', '.join(missing)}")
    positive = (df[list(CORE_POSITIVE_COLUMNS)] > 0).all(axis=1)
    if "DY" in df.columns:
        positive &= df["DY"].ge(0)
    result = df.loc[positive].copy()
    if result.empty:
        raise FiiSelectionError("no FIIs passed core eligibility filters")
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


def preprocess_fii(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply FII eligibility, segment winsorization, inversion, and z-scores."""

    df = apply_fii_eligibility(df_raw)
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
    return df.reset_index(drop=True)


def build_fii_scores(
    df: pd.DataFrame,
    profile: str,
    profile_weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Calculate FII group scores and sort assets by weighted score."""

    if profile not in FII_PROFILE_WEIGHTS and profile_weights is None:
        raise FiiSelectionError(f"unknown FII profile: {profile}")
    weights = dict(profile_weights or FII_PROFILE_WEIGHTS[profile])
    missing_groups = [group for group in FII_SCORE_GROUPS if group not in weights]
    if missing_groups:
        raise FiiSelectionError(f"FII score weights missing groups: {missing_groups}")
    if not np.isclose(sum(weights.values()), 1.0):
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
    profile: str,
    random_seed: Optional[int] = None,
    ga_config: Optional[Mapping[str, int | float]] = None,
) -> pd.DataFrame:
    """Run the existing binary GA with independent FII parameters."""

    if profile not in FII_GA_CONFIG and ga_config is None:
        raise FiiSelectionError(f"unknown FII profile: {profile}")
    settings = dict(ga_config or FII_GA_CONFIG[profile])
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
        random_seed=random_seed,
    )
    return optimizer.optimize(df_ranked)


def _run_fii_execution(
    df_ranked: pd.DataFrame,
    profile: str,
    settings: Mapping[str, int | float],
    random_seed: int,
    run_id: int,
) -> tuple[Dict[str, object], pd.DataFrame]:
    seed = random_seed + run_id
    portfolio = optimize_fii_portfolio(
        df_ranked,
        profile,
        random_seed=seed,
        ga_config=settings,
    )
    return {
        "run_id": run_id,
        "seed": seed,
        "tickers": sorted(portfolio["TICKER"].tolist()),
        "fitness": float(portfolio.attrs["fitness"]),
        "hhi": float(portfolio.attrs["hhi"]),
        "generations_run": portfolio.attrs.get("generations_run", 0),
        "converged_early": portfolio.attrs.get("converged_early", False),
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
    profile: str,
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
) -> Dict[str, object]:
    """Run FII selection with optional stock-style multi-run controls."""

    if n_runs <= 0:
        raise FiiSelectionError("n_runs must be positive")
    if min_runs <= 0:
        raise FiiSelectionError("min_runs must be positive")

    raw = load_fii_data(input_path, sector_name=sector_name)
    clean = preprocess_fii(raw)
    ranked = build_fii_scores(clean, profile)
    settings = dict(ga_config or FII_GA_CONFIG[profile])
    n_assets = int(settings["n_assets"])

    execution_results = []
    portfolios = []
    run = partial(_run_fii_execution, ranked, profile, settings, random_seed)

    if parallel:
        batch_size = min(10, n_runs)
        for batch_start in range(0, n_runs, batch_size):
            batch_end = min(batch_start + batch_size, n_runs)
            with mp.Pool() as pool:
                batch = pool.starmap(run, [(run_id,) for run_id in range(batch_start, batch_end)])
            for execution_result, portfolio in batch:
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

    processed_path = processed_path or PROCESSED_DIR / f"fii_clean_{profile}.csv"
    output_path = output_path or OUTPUTS_DIR / f"carteira_fii_{profile}_consensus.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    selected.to_json(output_path, orient="records", indent=2, force_ascii=False)

    return {
        "profile": profile,
        "n_runs": len(execution_results),
        "n_candidates": len(ranked),
        "n_selected": len(selected),
        "hhi": float(selected.attrs["hhi"]),
        "stability": _fii_stability(execution_results),
        "portfolio": selected,
        "ranked": ranked,
        "output_path": output_path,
        "processed_path": processed_path,
    }
