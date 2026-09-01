from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import sys
from typing import Callable, Mapping, TypedDict

# Keep the research import seam in one adapter until the existing `py/` tree is
# packaged. Routers and services must not add their own sys.path mutations.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = PROJECT_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from allocation_config import (  # noqa: E402
    ALLOCATION_DATA_DIR,
    ALLOCATION_OUTPUTS_DIR,
    ALLOCATION_PROFILE_ANCHORS,
    ASSET_CLASSES,
)
from allocation_data import SnapshotBundle, load_snapshot_bundle  # noqa: E402
from allocation_profiles import (  # noqa: E402
    build_anchor_profiles,
    interpolate_profile,
)
from pipelines.asset_allocation import run_allocation  # noqa: E402


API_CLASS_KEYS = {
    "brazilian_stocks": "brazilian_stocks",
    "fiis": "fiis",
    "international_equity": "international",
    "fixed_income": "fixed_income",
    "crypto": "crypto",
}

CLASS_LABELS = {
    "brazilian_stocks": "Ações brasileiras",
    "fiis": "FIIs",
    "international": "Exposição internacional",
    "fixed_income": "Renda fixa",
    "crypto": "Criptoativos",
}

MODEL_VERSION = "allocation-v1"


class AllocationAdapterError(ValueError):
    """Raised when the allocation engine cannot produce a safe API result."""


class BasicAllocationClass(TypedDict):
    key: str
    label: str
    target_weight: float
    target_amount_brl: float


class BasicRecommendation(TypedDict):
    plan: str
    model_version: str
    snapshot_id: str
    snapshot_cutoff: str
    classes: list[BasicAllocationClass]
    assumptions: list[str]
    risks: list[str]


@dataclass(frozen=True)
class BasicRecommendationInput:
    generic_profile: str
    investable_capital_brl: float
    snapshot_dir: Path = ALLOCATION_DATA_DIR
    portfolio_path: Path = ALLOCATION_OUTPUTS_DIR / "carteira_caio_consensus.json"


SnapshotLoader = Callable[[Path, Path], SnapshotBundle]
AllocationRunner = Callable[..., Mapping[str, object]]


def _validate_input(request: BasicRecommendationInput) -> None:
    if request.generic_profile not in ALLOCATION_PROFILE_ANCHORS:
        raise AllocationAdapterError(
            f"unknown Basic profile: {request.generic_profile}"
        )
    if not math.isfinite(request.investable_capital_brl):
        raise AllocationAdapterError("investable capital must be finite")
    if request.investable_capital_brl <= 0:
        raise AllocationAdapterError("investable capital must be positive")


def _snapshot_id(bundle: SnapshotBundle) -> str:
    explicit_id = bundle.metadata.get("snapshot_id")
    if explicit_id:
        return str(explicit_id)
    return f"allocation:{bundle.start_date.isoformat()}:{bundle.end_date.isoformat()}"


def _selected_weights(result: Mapping[str, object]) -> Mapping[str, object]:
    current_target = result.get("current_target")
    if not isinstance(current_target, Mapping):
        raise AllocationAdapterError("allocation result has no current target")
    selected = current_target.get("selected")
    if not isinstance(selected, Mapping):
        raise AllocationAdapterError("allocation result has no selected target")
    profile_winner = selected.get("profile_winner")
    if not isinstance(profile_winner, Mapping):
        raise AllocationAdapterError("allocation result has no Basic profile winner")
    weights = profile_winner.get("weights")
    if not isinstance(weights, Mapping):
        raise AllocationAdapterError("allocation result has no class weights")
    if set(weights) != set(ASSET_CLASSES):
        raise AllocationAdapterError(
            "allocation result has unexpected class keys: "
            f"{sorted(weights)}"
        )
    return weights


def _class_targets(
    weights: Mapping[str, object],
    capital: float,
) -> list[BasicAllocationClass]:
    checked_weights: list[tuple[str, float]] = []
    for engine_key in ASSET_CLASSES:
        raw_weight = weights[engine_key]
        if not isinstance(raw_weight, (int, float)) or not math.isfinite(raw_weight):
            raise AllocationAdapterError(f"invalid weight for {engine_key}")
        weight = float(raw_weight)
        if weight < 0 or weight > 1:
            raise AllocationAdapterError(f"invalid weight for {engine_key}")
        checked_weights.append((engine_key, weight))

    weight_total = sum(weight for _, weight in checked_weights)
    if abs(weight_total - 1.0) > 1e-6:
        raise AllocationAdapterError(
            f"class weights must sum to 1; got {weight_total:.8f}"
        )

    classes: list[BasicAllocationClass] = []
    for engine_key, weight in checked_weights:
        api_key = API_CLASS_KEYS[engine_key]
        classes.append(
            {
                "key": api_key,
                "label": CLASS_LABELS[api_key],
                "target_weight": round(weight, 6),
                "target_amount_brl": round(weight * capital, 2),
            }
        )

    amount_difference = round(
        capital - sum(item["target_amount_brl"] for item in classes),
        2,
    )
    classes[-1]["target_amount_brl"] = round(
        classes[-1]["target_amount_brl"] + amount_difference,
        2,
    )
    return classes


def generate_basic_recommendation(
    request: BasicRecommendationInput,
    *,
    load_bundle: SnapshotLoader = load_snapshot_bundle,
    run_analysis: AllocationRunner = run_allocation,
) -> BasicRecommendation:
    """Run one generic Basic allocation and normalize it for the API layer."""

    _validate_input(request)
    metadata = {
        "snapshot_dir": str(request.snapshot_dir),
        "caio_portfolio": str(request.portfolio_path),
        "basic_profile": request.generic_profile,
        "investable_capital_brl": request.investable_capital_brl,
    }
    bundle = load_bundle(request.snapshot_dir, request.portfolio_path)
    metadata.update(bundle.metadata)
    anchors = build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS)
    anchor_score = float(ALLOCATION_PROFILE_ANCHORS[request.generic_profile]["score"])
    allocation_profile = interpolate_profile(
        anchor_score,
        anchors,
        name=request.generic_profile,
        calibration_source="Basic generic allocation anchor",
        calibration_inputs={"generic_profile": request.generic_profile},
    )
    result = run_analysis(
        bundle.rows,
        metadata=metadata,
        profile="caio",
        allocation_profile=allocation_profile,
    )
    weights = _selected_weights(result)
    snapshot_cutoff = str(bundle.metadata.get("cutoff_date") or bundle.end_date)

    return {
        "plan": "basic",
        "model_version": MODEL_VERSION,
        "snapshot_id": _snapshot_id(bundle),
        "snapshot_cutoff": snapshot_cutoff,
        "classes": _class_targets(weights, request.investable_capital_brl),
        "assumptions": [
            "Resultado histórico, bruto e baseado em proxies de mercado.",
            "Pesos distribuídos entre cinco classes com rebalanceamento anual no motor.",
            "Resultado não representa garantia de retorno nem ordem de execução.",
        ],
        "risks": [
            "Dados históricos não garantem resultados futuros.",
            "Custos, impostos e fluxos de caixa não fazem parte deste cálculo.",
            "A exposição de cada classe pode ter risco e liquidez diferentes.",
        ],
    }
