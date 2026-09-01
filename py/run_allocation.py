#!/usr/bin/env python3
"""Run Caio's asset-class allocation analysis from local snapshots."""

import argparse
from pathlib import Path
import sys
from typing import Optional

PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import (  # noqa: E402
    ALLOCATION_DATA_DIR,
    ALLOCATION_OUTPUTS_DIR,
    ALLOCATION_PROFILE_ANCHORS,
    ALLOCATION_PROFILE_SCORE_DEFAULTS,
)
from allocation_data import SnapshotError, load_snapshot_bundle  # noqa: E402
from allocation_profiles import build_anchor_profiles, interpolate_profile  # noqa: E402
from pipelines.asset_allocation import run_allocation, write_outputs  # noqa: E402


def resolve_suitability_score(
    profile_name: str,
    suitability_score: Optional[float],
) -> Optional[float]:
    """Use explicit score first, then a declared named-profile default."""

    if suitability_score is not None:
        return suitability_score
    return ALLOCATION_PROFILE_SCORE_DEFAULTS.get(profile_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize Caio's allocation across five benchmark sleeves."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=ALLOCATION_DATA_DIR)
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=ALLOCATION_OUTPUTS_DIR / "carteira_caio_consensus.json",
        help="fixed Caio stock portfolio JSON",
    )
    parser.add_argument(
        "--fii-portfolio",
        type=Path,
        default=ALLOCATION_OUTPUTS_DIR / "carteira_fii_caio_consensus.json",
        help="fixed FII portfolio JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=ALLOCATION_OUTPUTS_DIR)
    parser.add_argument("--output-prefix", default="allocation_caio")
    parser.add_argument(
        "--suitability-score",
        type=float,
        help="continuous suitability score (0..1); enables personalized allocation",
    )
    parser.add_argument(
        "--profile-name",
        default="caio_new",
        help=(
            "name recorded for the personalized allocation profile; "
            "caio_last defaults to the conservative anchor"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = load_snapshot_bundle(
            args.snapshot_dir,
            args.portfolio,
            fii_portfolio_path=args.fii_portfolio,
        )
        metadata = dict(bundle.metadata)
        metadata["snapshot_dir"] = str(args.snapshot_dir)
        metadata["caio_portfolio"] = str(args.portfolio)
        metadata["fii_portfolio"] = str(args.fii_portfolio)
        allocation_profile = None
        suitability_score = resolve_suitability_score(
            args.profile_name,
            args.suitability_score,
        )
        if suitability_score is not None:
            anchors = build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS)
            allocation_profile = interpolate_profile(
                suitability_score,
                anchors,
                name=args.profile_name,
                calibration_source="v2 questionnaire score interpolation",
                calibration_inputs={
                    "suitability_score": suitability_score,
                    "anchor_policies": ALLOCATION_PROFILE_ANCHORS,
                },
            )
            metadata["suitability_score"] = suitability_score
        result = run_allocation(
            bundle.rows,
            metadata=metadata,
            allocation_profile=allocation_profile,
        )
        paths = write_outputs(result, args.output_dir, args.output_prefix)
    except (SnapshotError, ValueError) as exc:
        print(f"Allocation could not run: {exc}", file=sys.stderr)
        return 2

    print(f"Allocation completed for {result['profile']}.")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
