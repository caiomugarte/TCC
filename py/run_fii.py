#!/usr/bin/env python3
"""Run the isolated FII selection flow."""

import argparse
from pathlib import Path
import sys

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from fii_selection import FII_GA_CONFIG, FiiSelectionError, run_fii_selection  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select FII assets")
    parser.add_argument("--profile", default="caio")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--processed-output", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=None, help="Custom sequential run count")
    parser.add_argument("--once", action="store_true", help="One GA execution")
    parser.add_argument("--quick", action="store_true", help="20 parallel executions")
    parser.add_argument("--production", action="store_true", help="Up to 100 adaptive parallel executions")
    parser.add_argument("--max-quality", action="store_true", help="Up to 150 adaptive sequential executions")
    parser.add_argument("--sector", default=None, help="Label for a single unlabelled segment export")
    parser.add_argument("--n-assets", type=int, default=None)
    return parser


def execution_settings(args: argparse.Namespace) -> dict:
    modes = [name for name in ("once", "quick", "production", "max_quality") if getattr(args, name)]
    if len(modes) > 1 or (modes and args.runs is not None):
        raise ValueError("choose one execution mode or --runs")
    if args.runs is not None:
        return {"mode": "custom", "n_runs": args.runs, "parallel": False, "adaptive_mode": False}
    if args.max_quality:
        return {
            "mode": "max-quality",
            "n_runs": 150,
            "parallel": False,
            "adaptive_mode": True,
            "min_runs": 40,
            "target_cv": 0.02,
            "target_jaccard": 0.75,
        }
    if args.production:
        return {
            "mode": "production",
            "n_runs": 100,
            "parallel": True,
            "adaptive_mode": True,
            "min_runs": 30,
            "target_cv": 0.03,
            "target_jaccard": 0.70,
        }
    if args.quick:
        return {"mode": "quick", "n_runs": 20, "parallel": True, "adaptive_mode": False}
    return {"mode": "once", "n_runs": 1, "parallel": False, "adaptive_mode": False}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from fetch_status_invest_fii import RAW_DATA_FILE

    settings = dict(FII_GA_CONFIG.get(args.profile, {}))
    if args.n_assets is not None:
        settings["n_assets"] = args.n_assets
    try:
        mode = execution_settings(args)
        result = run_fii_selection(
            profile=args.profile,
            input_path=args.input or RAW_DATA_FILE,
            output_path=args.output,
            processed_path=args.processed_output,
            n_runs=mode["n_runs"],
            sector_name=args.sector,
            ga_config=settings or None,
            parallel=mode["parallel"],
            adaptive_mode=mode["adaptive_mode"],
            min_runs=mode.get("min_runs", 30),
            target_cv=mode.get("target_cv", 0.03),
            target_jaccard=mode.get("target_jaccard", 0.70),
        )
    except (FiiSelectionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    portfolio = result["portfolio"]
    stability = result["stability"]
    print(f"Mode: {mode['mode']} | GA runs: {result['n_runs']}")
    print(f"Selected {len(portfolio)} FIIs from {result['n_candidates']} eligible candidates")
    print(f"Stability: CV={stability['fitness_cv']:.4f}, Jaccard={stability['jaccard_mean']:.4f}")
    print(f"Saved: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
