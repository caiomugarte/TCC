#!/usr/bin/env python3
"""Backtest one optimized FII portfolio without stock or allocation sleeves."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_data import SnapshotError, load_portfolio_tickers  # noqa: E402
from backtest_analysis import (  # noqa: E402
    OUTPUTS_DIR,
    analyze_asset_performance,
    calculate_benchmark_metrics,
    calculate_drawdown,
    calculate_metrics,
    calculate_portfolio_value,
    calculate_returns,
    fetch_historical_data,
)
from fetch_allocation_snapshot import (  # noqa: E402
    SnapshotFetchError,
    fetch_ifix_levels,
)


INITIAL_CAPITAL = 10_000.0
DEFAULT_PERIODS = (5, 10)


def default_portfolio_path(profile: str) -> Path:
    """Return the optimized FII consensus artifact for a profile."""

    return OUTPUTS_DIR / f"carteira_fii_{profile}_consensus.json"


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Use comparable daily timestamps for Yahoo and B3 series."""

    if frame.empty:
        return frame
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result.loc[~result.index.duplicated(keep="last")]


def _normalise_series(series: pd.Series) -> pd.Series:
    """Normalize a dated level series index."""

    result = _normalise_index(series.to_frame("value"))["value"]
    return result.sort_index()


def _levels_to_value(levels: Mapping[date, float]) -> pd.Series:
    """Convert benchmark levels to the same starting capital as the portfolio."""

    series = _normalise_series(pd.Series(levels, dtype=float)).dropna()
    if series.empty:
        return series
    return series / series.iloc[0] * INITIAL_CAPITAL


def _period_dates(period_years: int, end_date: Optional[datetime]) -> tuple[datetime, datetime]:
    end = end_date or datetime.now()
    return end - timedelta(days=period_years * 365 + 30), end


def run_period_backtest(
    portfolio_path: Path,
    period_years: int,
    *,
    end_date: Optional[datetime] = None,
    include_ifix: bool = True,
    analyze_assets: bool = True,
) -> Dict[str, object]:
    """Run a fixed-ticker, equal-weight, buy-and-hold FII backtest."""

    if period_years <= 0:
        raise ValueError("period_years must be positive")

    start_date, end_date = _period_dates(period_years, end_date)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    tickers = load_portfolio_tickers(portfolio_path, "FII")
    prices, missing = fetch_historical_data(tickers, start_str, end_str)
    prices = _normalise_index(prices).dropna(how="any")

    if prices.empty or prices.shape[1] == 0:
        raise SnapshotError("no common historical prices for selected FII portfolio")

    values = calculate_portfolio_value(prices, INITIAL_CAPITAL)
    returns = calculate_returns(values)
    result: Dict[str, object] = {
        "period_years": period_years,
        "start_date": start_str,
        "end_date": end_str,
        "tickers": list(tickers),
        "available_tickers": [str(column).removesuffix(".SA") for column in prices.columns],
        "missing_tickers": list(missing),
        "values": values,
        "returns": returns,
        "metrics": calculate_metrics(values, returns, period_years),
        "asset_analysis": (
            analyze_asset_performance(list(tickers), start_str, end_str, period_years)
            if analyze_assets
            else pd.DataFrame()
        ),
        "benchmark_name": "IFIX",
        "benchmark_values": pd.Series(dtype=float),
        "benchmark_metrics": {},
        "relative_metrics": {},
        "benchmark_error": None,
    }

    if include_ifix:
        try:
            benchmark_values = _levels_to_value(
                fetch_ifix_levels(start_date.date(), end_date.date())
            )
            benchmark_returns = calculate_returns(benchmark_values)
            result["benchmark_values"] = benchmark_values
            result["benchmark_metrics"] = calculate_metrics(
                benchmark_values,
                benchmark_returns,
                period_years,
            )
            result["relative_metrics"] = calculate_benchmark_metrics(
                returns,
                benchmark_returns,
            )
        except SnapshotFetchError as exc:
            result["benchmark_error"] = str(exc)

    return result


def run_fii_backtest(
    portfolio_path: Path,
    periods: Sequence[int] = DEFAULT_PERIODS,
    *,
    end_date: Optional[datetime] = None,
    include_ifix: bool = True,
    analyze_assets: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Run requested periods for one optimized FII artifact."""

    return {
        f"{period}anos": run_period_backtest(
            portfolio_path,
            period,
            end_date=end_date,
            include_ifix=include_ifix,
            analyze_assets=analyze_assets,
        )
        for period in periods
    }


def _plot_evolution(result: Mapping[str, object], output_path: Path, title: str) -> None:
    values = result["values"]
    benchmark_values = result["benchmark_values"]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(values.index, values, label="Carteira FII otimizada", linewidth=2)
    if not benchmark_values.empty:
        ax.plot(benchmark_values.index, benchmark_values, label="IFIX", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Capital (R$)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_drawdown(result: Mapping[str, object], output_path: Path, title: str) -> None:
    values = result["values"]
    benchmark_values = result["benchmark_values"]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(values.index, calculate_drawdown(values), label="Carteira FII otimizada")
    if not benchmark_values.empty:
        ax.plot(benchmark_values.index, calculate_drawdown(benchmark_values), label="IFIX")
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_assets(asset_data: pd.DataFrame, output_path: Path, title: str) -> None:
    if asset_data.empty:
        return
    ordered = asset_data.sort_values("retorno_total_com_div_pct")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(y, ordered["retorno_total_sem_div_pct"], label="Preço")
    ax.barh(
        y,
        ordered["efeito_dividendos_total_pct"],
        left=ordered["retorno_total_sem_div_pct"],
        label="Distribuições",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["TICKER"])
    ax.set_xlabel("Retorno total (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary(result: Mapping[str, object]) -> Dict[str, object]:
    return {
        "period_years": result["period_years"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "tickers": result["tickers"],
        "available_tickers": result["available_tickers"],
        "missing_tickers": result["missing_tickers"],
        "metrics": result["metrics"],
        "benchmark": {
            "name": result["benchmark_name"],
            "metrics": result["benchmark_metrics"],
            "relative_metrics": result["relative_metrics"],
            "error": result["benchmark_error"],
        },
    }


def write_outputs(
    results: Mapping[str, Mapping[str, object]],
    profile: str,
    portfolio_path: Path,
    output_dir: Path = OUTPUTS_DIR,
) -> List[Path]:
    """Write FII-only metrics, time series, asset data, and charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    summaries = {
        "profile": profile,
        "portfolio_path": str(portfolio_path),
        "periods": {},
    }

    for period_name, result in results.items():
        summaries["periods"][period_name] = _summary(result)
        values = result["values"]
        benchmark_values = result["benchmark_values"]
        series = pd.DataFrame({"portfolio": values})
        if not benchmark_values.empty:
            series["ifix"] = benchmark_values
        series_path = output_dir / f"fii_backtest_series_{profile}_{period_name}.csv"
        series.to_csv(series_path, index_label="date")
        written.append(series_path)

        asset_data = result["asset_analysis"]
        if not asset_data.empty:
            asset_csv = output_dir / f"fii_backtest_assets_{profile}_{period_name}.csv"
            asset_json = output_dir / f"fii_backtest_assets_{profile}_{period_name}.json"
            asset_data.to_csv(asset_csv, index=False, encoding="utf-8-sig")
            asset_data.to_json(asset_json, orient="records", indent=2, force_ascii=False)
            written.extend((asset_csv, asset_json))
            asset_plot = output_dir / f"fii_backtest_assets_{profile}_{period_name}.png"
            _plot_assets(asset_data, asset_plot, f"FIIs otimizados - {period_name}")
            written.append(asset_plot)

        evolution_plot = output_dir / f"fii_backtest_evolution_{profile}_{period_name}.png"
        drawdown_plot = output_dir / f"fii_backtest_drawdown_{profile}_{period_name}.png"
        _plot_evolution(result, evolution_plot, f"Backtest exclusivo de FIIs - {period_name}")
        _plot_drawdown(result, drawdown_plot, f"Drawdown da carteira FII - {period_name}")
        written.extend((evolution_plot, drawdown_plot))

    metrics_path = output_dir / f"fii_backtest_metrics_{profile}.json"
    metrics_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written.append(metrics_path)
    return written


def _parse_end_date(raw: str) -> datetime:
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {raw}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest only the optimized FII consensus portfolio."
    )
    parser.add_argument("--profile", default="caio")
    parser.add_argument("--portfolio", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--end-date", type=_parse_end_date, default=None)
    parser.add_argument("--skip-ifix", action="store_true", help="skip optional IFIX benchmark")
    parser.add_argument("--no-assets", action="store_true", help="skip per-FII analysis")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if any(period <= 0 for period in args.years):
        print("ERROR: --years values must be positive", file=sys.stderr)
        return 2

    portfolio_path = args.portfolio or default_portfolio_path(args.profile)
    try:
        results = run_fii_backtest(
            portfolio_path,
            args.years,
            end_date=args.end_date,
            include_ifix=not args.skip_ifix,
            analyze_assets=not args.no_assets,
        )
        paths = write_outputs(results, args.profile, portfolio_path, args.output_dir)
    except (SnapshotError, ValueError) as exc:
        print(f"FII backtest could not run: {exc}", file=sys.stderr)
        return 2

    for period_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{period_name}: return={metrics['retorno_total_pct']:.2f}% | "
            f"annual={metrics['retorno_anualizado_pct']:.2f}% | "
            f"drawdown={metrics['drawdown_maximo_pct']:.2f}%"
        )
    print(f"Saved {len(paths)} FII backtest outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
