#!/usr/bin/env python3
"""Write a dependency-free HTML/SVG view of the allocation backtest."""

import argparse
from datetime import date
from html import escape
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import ASSET_CLASSES, ALLOCATION_DATA_DIR, ALLOCATION_OUTPUTS_DIR  # noqa: E402
from allocation_data import load_snapshot_bundle, multiply_levels, read_levels_csv  # noqa: E402
from core.allocation import simulate_portfolio  # noqa: E402


COLORS = {
    "brazilian_stocks": "var(--viz-series-1, currentColor)",
    "fiis": "var(--viz-series-2, currentColor)",
    "international_equity": "var(--viz-series-3, currentColor)",
    "fixed_income": "var(--viz-series-4, currentColor)",
    "crypto": "var(--viz-series-5, currentColor)",
}
CLASS_LABELS = {
    "brazilian_stocks": "Brazilian stocks",
    "fiis": "FIIs",
    "international_equity": "S&P 500 TR",
    "fixed_income": "Fixed income",
    "crypto": "Crypto",
}


def _scale(value: float, low: float, high: float, start: float, length: float) -> float:
    if high == low:
        return start + length / 2
    return start + (value - low) / (high - low) * length


def _sample_monthly(dates: Sequence[date], values: Sequence[float]):
    sampled = []
    seen = None
    for current_date, value in zip(dates, values):
        month = (current_date.year, current_date.month)
        if month != seen:
            sampled.append((current_date, value))
            seen = month
    return sampled


def _svg_line_chart(
    title: str,
    description: str,
    series: Mapping[str, Sequence[tuple]],
    *,
    percent: bool = False,
) -> str:
    width, height = 760, 280
    left, right, top, bottom = 58, 18, 28, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    all_points = [point for points in series.values() for point in points]
    values = [value for _, value in all_points]
    low = min(values + ([0.0] if percent else [0.0]))
    high = max(values + [0.0])
    if percent:
        low = min(low, -0.05)
        high = max(high, 0.0)
    else:
        high *= 1.08
    if high == low:
        high = low + 1.0

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    svg = [
        f'<svg class="allocation-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(description)}</desc>",
    ]
    for tick in range(5):
        value = low + (high - low) * tick / 4
        yy = y(value)
        label = f"{value * 100:.0f}%" if percent else f"{value:.1f}x"
        svg.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>')
        svg.append(f'<text class="axis-label" x="{left-8}" y="{yy+4:.1f}" text-anchor="end">{label}</text>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    svg.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')

    first = next(iter(series.values()))[0][0]
    last = next(iter(series.values()))[-1][0]
    svg.append(f'<text class="axis-label" x="{left}" y="{height-12}">{first:%Y-%m}</text>')
    svg.append(f'<text class="axis-label" x="{width-right}" y="{height-12}" text-anchor="end">{last:%Y-%m}</text>')
    for index, (label, points) in enumerate(series.items()):
        path = " ".join(
            ("M" if point_index == 0 else "L")
            + f" {_scale(point_index, 0, max(len(points)-1, 1), left, plot_width):.1f} {y(value):.1f}"
            for point_index, (_, value) in enumerate(points)
        )
        color = f"var(--viz-series-{index + 1}, currentColor)"
        svg.append(f'<path class="series" d="{path}" stroke="{color}"/>')
        last_x = left + plot_width
        last_y = y(points[-1][1])
        value = f"{points[-1][1] * 100:.1f}%" if percent else f"{points[-1][1]:.2f}x"
        svg.append(f'<text class="series-label" x="{last_x-4:.1f}" y="{last_y-6:.1f}" text-anchor="end">{escape(label)} {value}</text>')

    legend_x = left
    legend_y = height - 25
    for index, label in enumerate(series):
        x_pos = legend_x + index * 142
        svg.append(f'<line class="legend-line" x1="{x_pos}" y1="{legend_y-4}" x2="{x_pos+16}" y2="{legend_y-4}" stroke="var(--viz-series-{index+1}, currentColor)"/>')
        svg.append(f'<text class="axis-label" x="{x_pos+21}" y="{legend_y}">{escape(label)}</text>')
    svg.append("</svg>")
    return "".join(svg)


def _svg_allocation_bars(windows: Sequence[tuple]) -> str:
    width, height = 760, 260
    left, right, top, bar_height = 136, 18, 34, 32
    bar_width = width - left - right
    svg = [
        '<svg class="allocation-chart" viewBox="0 0 760 260" role="img" aria-label="Allocation weights through walk-forward windows">',
        "<title>Allocation weights through walk-forward windows</title>",
        "<desc>Each horizontal bar shows the percentage assigned to each asset class in a training window and the current target.</desc>",
    ]
    for index, (label, weights) in enumerate(windows):
        yy = top + index * 44
        svg.append(f'<text class="axis-label" x="{left-10}" y="{yy+21}" text-anchor="end">{escape(label)}</text>')
        cursor = left
        for class_name in ASSET_CLASSES:
            width_part = weights[class_name] * bar_width
            if width_part <= 0:
                continue
            svg.append(f'<rect x="{cursor:.1f}" y="{yy}" width="{width_part:.1f}" height="{bar_height}" fill="{COLORS[class_name]}"/>')
            if width_part >= 34:
                svg.append(f'<text class="bar-label" x="{cursor + width_part/2:.1f}" y="{yy+21}" text-anchor="middle">{weights[class_name]*100:.0f}%</text>')
            cursor += width_part
    legend_y = top + len(windows) * 44 + 4
    for index, class_name in enumerate(ASSET_CLASSES):
        x_pos = left + index * 122
        svg.append(f'<rect x="{x_pos}" y="{legend_y}" width="12" height="12" fill="{COLORS[class_name]}"/>')
        svg.append(f'<text class="axis-label" x="{x_pos+17}" y="{legend_y+11}">{escape(CLASS_LABELS[class_name])}</text>')
    svg.append("</svg>")
    return "".join(svg)


def _svg_risk_budget_bars(
    weights: Mapping[str, float],
    contributions: Mapping[str, float],
    cap: float,
    scenario_is_feasible: bool,
) -> str:
    width, height = 760, 220
    left, right, top, bar_height = 136, 18, 42, 32
    bar_width = width - left - right
    svg = [
        '<svg class="allocation-chart" viewBox="0 0 760 220" role="img" aria-label="Capital allocation compared with variance contribution">',
        "<title>Capital allocation compared with variance contribution</title>",
        "<desc>Two stacked bars compare the current knee's capital weights with its signed target-weight contributions to portfolio variance. The risk budget line marks the 25 percent cap.</desc>",
    ]
    bars = (("Capital weight", weights), ("Variance contribution", contributions))
    for index, (label, values) in enumerate(bars):
        yy = top + index * 64
        svg.append(f'<text class="axis-label" x="{left-10}" y="{yy+21}" text-anchor="end">{escape(label)}</text>')
        cursor = left
        for class_name in ASSET_CLASSES:
            value = max(0.0, float(values.get(class_name, 0.0)))
            width_part = value * bar_width
            if width_part <= 0:
                continue
            svg.append(f'<rect x="{cursor:.1f}" y="{yy}" width="{width_part:.1f}" height="{bar_height}" fill="{COLORS[class_name]}"/>')
            if width_part >= 34:
                svg.append(f'<text class="bar-label" x="{cursor + width_part/2:.1f}" y="{yy+21}" text-anchor="middle">{value*100:.0f}%</text>')
            cursor += width_part
    cap_x = left + cap * bar_width
    risk_y = top + 64
    svg.append(f'<line class="cap-line" x1="{cap_x:.1f}" y1="{risk_y-9}" x2="{cap_x:.1f}" y2="{risk_y+bar_height+9}"/>')
    svg.append(f'<text class="cap-label" x="{cap_x+5:.1f}" y="{risk_y-14}">{cap*100:.0f}% cap</text>')
    status = "feasible" if scenario_is_feasible else "no feasible allocation under cap"
    svg.append(f'<text class="axis-label" x="{left}" y="{height-36}">25% risk-budget scenario: {status}</text>')
    legend_y = height - 22
    for index, class_name in enumerate(ASSET_CLASSES):
        x_pos = left + index * 122
        svg.append(f'<rect x="{x_pos}" y="{legend_y-10}" width="12" height="12" fill="{COLORS[class_name]}"/>')
        svg.append(f'<text class="axis-label" x="{x_pos+17}" y="{legend_y+1}">{escape(CLASS_LABELS[class_name])}</text>')
    svg.append("</svg>")
    return "".join(svg)


def _single_snapshot_series(path: Path):
    series = read_levels_csv(path)
    if len(series) != 1:
        raise ValueError(f"expected one series in {path}; found {sorted(series)}")
    return next(iter(series.values()))


def _benchmark_series(snapshot_dir: Path, bundle, wallet_weights: Mapping[str, float]):
    """Build BRL-normalized wallet and benchmark levels on common dates."""

    wallet_path = simulate_portfolio(
        bundle.rows,
        wallet_weights,
        ASSET_CLASSES,
        annual_rebalance=True,
    )
    wallet_levels = dict(zip(wallet_path.dates, wallet_path.values))
    ptax = _single_snapshot_series(snapshot_dir / "ptax.csv")
    raw_series = {
        "Caio wallet": wallet_levels,
        "BTC (BRL)": multiply_levels(
            _single_snapshot_series(snapshot_dir / "btc_usd.csv"), ptax
        ),
        "CDI / CDB proxy": _single_snapshot_series(snapshot_dir / "di.csv"),
        "IFIX": _single_snapshot_series(snapshot_dir / "ifix.csv"),
        "Ibovespa": _single_snapshot_series(snapshot_dir / "ibovespa.csv"),
    }
    common_dates = set(next(iter(raw_series.values())))
    for levels in raw_series.values():
        common_dates.intersection_update(levels)
    dates = sorted(common_dates)
    if not dates:
        raise ValueError("wallet and benchmarks have no common dates")
    return {
        label: _sample_monthly(
            dates,
            [levels[current_date] / levels[dates[0]] for current_date in dates],
        )
        for label, levels in raw_series.items()
    }


def build_visual(
    snapshot_dir: Path,
    output_json: Path,
    portfolio_path: Path | None = None,
    benchmarks_only: bool = False,
) -> str:
    portfolio_path = portfolio_path or output_json.parent / "carteira_caio_consensus.json"
    bundle = load_snapshot_bundle(snapshot_dir, portfolio_path)
    result = json.loads(output_json.read_text(encoding="utf-8"))
    classes = tuple(ASSET_CLASSES)

    def weights_from(record):
        return {name: float(record["weights"][name]) for name in classes}

    selected = result["current_target"]["selected"]
    knee = selected["knee"]
    wallet_record = selected.get("profile_winner", knee)
    benchmark_series = _benchmark_series(snapshot_dir, bundle, weights_from(wallet_record))
    scenario = result.get("risk_budget_scenarios", {}).get(
        "max_25pct_variance_contribution", {}
    )
    scenario_is_feasible = bool(
        scenario.get("current_target", {}).get("selected")
    )
    risk_cap = float(scenario.get("risk_contribution_cap", 0.25))
    strategies = {
        "Knee": weights_from(selected["knee"]),
        "Max return": weights_from(selected["max_return"]),
        "Equal 20%": {name: 0.2 for name in classes},
        "100% stocks": {name: 1.0 if index == 0 else 0.0 for index, name in enumerate(classes)},
        "100% CDI": {name: 1.0 if name == "fixed_income" else 0.0 for name in classes},
    }
    wallet_series = {}
    drawdown_series = {}
    for label, weights in strategies.items():
        path = simulate_portfolio(bundle.rows, weights, classes, annual_rebalance=True)
        sampled = _sample_monthly(path.dates, path.values)
        wallet_series[label] = sampled
        peak = 0.0
        drawdowns = []
        for current_date, value in sampled:
            peak = max(peak, value)
            drawdowns.append((current_date, value / peak - 1.0))
        drawdown_series[label] = drawdowns

    primary = result["walk_forward"]["primary"]
    windows = [
        (f"{window['train_start'][:4]}–{window['train_end'][:4]}", window["training"]["selected"]["knee"]["weights"])
        for window in primary
    ]
    windows.append(("Current", selected["knee"]["weights"]))
    style = """<style>
#caio-allocation-evolution { font-family: system-ui, sans-serif; color: var(--foreground, currentColor); }
#caio-allocation-evolution .allocation-chart { display: block; width: 100%; height: auto; margin: 0 0 1rem; }
#caio-allocation-evolution .grid { stroke: var(--border, currentColor); stroke-width: 1; opacity: .55; }
#caio-allocation-evolution .axis { stroke: var(--muted-foreground, currentColor); stroke-width: 1; }
#caio-allocation-evolution .axis-label { fill: var(--muted-foreground, currentColor); font-size: 11px; }
#caio-allocation-evolution .series { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
#caio-allocation-evolution .series-label { fill: var(--foreground, currentColor); font-size: 11px; }
#caio-allocation-evolution .legend-line { stroke-width: 3; }
#caio-allocation-evolution .bar-label { fill: var(--background, currentColor); font-size: 11px; font-weight: 500; }
#caio-allocation-evolution .cap-line { stroke: var(--destructive, currentColor); stroke-width: 1.5; stroke-dasharray: 4 3; }
#caio-allocation-evolution .cap-label { fill: var(--destructive, currentColor); font-size: 11px; }
</style>
"""
    benchmark_chart = _svg_line_chart(
        "Personalized wallet vs benchmark indexes",
        "Growth of one Brazilian real before taxes and costs, normalized to the common valid date range. BTC is converted to BRL with PTAX; CDI is the CDB proxy; Ibovespa is a price index.",
        benchmark_series,
    )
    if benchmarks_only:
        return style + '<div id="caio-allocation-evolution">' + benchmark_chart + "</div>"
    return style + '<div id="caio-allocation-evolution">' + benchmark_chart + _svg_line_chart(
        "Historical wallet evolution",
        "One Brazilian real invested in each strategy, with annual rebalancing and no taxes or costs.",
        wallet_series,
    ) + _svg_line_chart(
        "Historical wallet drawdown",
        "Percentage decline from the previous peak for each strategy.",
        drawdown_series,
        percent=True,
    ) + _svg_allocation_bars(windows) + _svg_risk_budget_bars(
        knee["weights"],
        knee["risk_contribution"],
        risk_cap,
        scenario_is_feasible,
    ) + "</div>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate allocation evolution graphs.")
    parser.add_argument("--snapshot-dir", type=Path, default=ALLOCATION_DATA_DIR)
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=ALLOCATION_OUTPUTS_DIR / "carteira_caio_consensus.json",
    )
    parser.add_argument("--output-json", type=Path, default=ALLOCATION_OUTPUTS_DIR / "allocation_caio.json")
    parser.add_argument("--output", type=Path, default=ALLOCATION_OUTPUTS_DIR / "allocation_evolution.html")
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        help="Write only the wallet-versus-index comparison chart",
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_visual(
            args.snapshot_dir,
            args.output_json,
            args.portfolio,
            args.benchmarks_only,
        ),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
