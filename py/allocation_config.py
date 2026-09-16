"""Configuration and validation seams for asset-class allocation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOCATION_DATA_DIR = PROJECT_ROOT / "data" / "allocation"
ALLOCATION_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ASSET_CLASSES = (
    "brazilian_stocks",
    "fiis",
    "international_equity",
    "fixed_income",
    "crypto",
)

# V2 policy anchors. These are explicit suitability-policy inputs, not
# historical guarantees: the questionnaire score interpolates between them.
ALLOCATION_PROFILE_ANCHORS = {
    "conservador": {
        "score": 0.0,
        "volatility_cap": 0.10,
        "drawdown_cap": 0.15,
        "crypto_risk_contribution_cap": 0.25,
        "hhi_penalty": 0.50,
        "risk_adjusted_weights": {
            "return": 0.30,
            "volatility": 0.35,
            "drawdown": 0.35,
        },
        "calibration_source": "v2 questionnaire risk-band policy anchor",
        "calibration_inputs": {
            "risk_band": "capital preservation",
            "basis": "lower volatility, drawdown, and crypto-risk limits",
        },
    },
    "moderado": {
        "score": 0.5,
        "volatility_cap": 0.15,
        "drawdown_cap": 0.25,
        "crypto_risk_contribution_cap": 0.40,
        "hhi_penalty": 0.25,
        "risk_adjusted_weights": {
            "return": 0.50,
            "volatility": 0.25,
            "drawdown": 0.25,
        },
        "calibration_source": "v2 questionnaire risk-band policy anchor",
        "calibration_inputs": {
            "risk_band": "balanced growth",
            "basis": "moderate risk and concentration tolerance",
        },
    },
    "arrojado": {
        "score": 1.0,
        "volatility_cap": 0.20,
        "drawdown_cap": 0.35,
        "crypto_risk_contribution_cap": 0.50,
        "hhi_penalty": 0.10,
        "risk_adjusted_weights": {
            "return": 0.70,
            "volatility": 0.15,
            "drawdown": 0.15,
        },
        "calibration_source": "v2 questionnaire risk-band policy anchor",
        "calibration_inputs": {
            "risk_band": "long-term aggressive growth",
            "basis": "higher drawdown and crypto-risk tolerance",
        },
    },
}

# Explicit named-profile aliases; never derive allocation policy from stock-GA weights.
ALLOCATION_PROFILE_SCORE_DEFAULTS = {
    "caio_last": 0.0,
}

ALLOCATION_CONFIG = {
    "caio": {
        "volatility_cap": 0.20,
        "drawdown_cap": 0.30,
        "minimum_class_weight": 0.05,
        "primary_horizon_years": 10,
        "robustness_horizon_years": 5,
        "training_years": 3,
        "test_years": 1,
        "coarse_step": 0.05,
        "refinement_step": 0.01,
        "refinement_radius": 0.02,
        "hhi_penalties": tuple(round(index * 0.05, 2) for index in range(21)),
        "rebalance_years": 1,
        "risk_budget_scenarios": {
            "max_25pct_variance_contribution": 0.25,
        },
        "crypto_weight_scenarios": (0.10, 0.15, 0.20),
    }
}


def normalize_class_constraints(
    constraints: Optional[Mapping[str, object]],
    class_names: Sequence[str] = ASSET_CLASSES,
) -> dict[str, object]:
    """Return explicit per-class bounds and an optional class-HHI cap.

    Premium policy uses this mapping to keep allocation constraints separate
    from stock/FII selector configuration.  Both the compact per-class form
    and the grouped ``minimums``/``maximums`` form are accepted.
    """

    classes = tuple(class_names)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("class_names must contain unique classes")
    raw = dict(constraints or {})
    minimums = {name: 0.0 for name in classes}
    maximums = {name: 1.0 for name in classes}

    for key in ("minimums", "minimum_weights", "min_weights", "min_weight"):
        values = raw.get(key)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"class constraint {key} must be an object")
        for name, value in values.items():
            if name not in minimums:
                raise ValueError(f"unknown allocation class: {name}")
            minimums[name] = float(value)

    for key in ("maximums", "maximum_weights", "max_weights", "max_weight"):
        values = raw.get(key)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"class constraint {key} must be an object")
        for name, value in values.items():
            if name not in maximums:
                raise ValueError(f"unknown allocation class: {name}")
            maximums[name] = float(value)

    # A direct class entry is useful for policy JSON such as
    # ``{"crypto": {"max_weight": 0}}``.
    reserved = {
        "minimums", "minimum_weights", "min_weights", "min_weight",
        "maximums", "maximum_weights", "max_weights",
        "max_weight", "hhi_max", "max_hhi", "class_hhi_max",
        "minimum_class_weight", "allow_zero",
    }
    for name in set(raw) - reserved:
        if name not in minimums:
            raise ValueError(f"unknown allocation class: {name}")
        value = raw[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"class constraint {name} must be an object")
        if "min_weight" in value or "min" in value:
            minimums[name] = float(value.get("min_weight", value.get("min")))
        if "max_weight" in value or "max" in value:
            maximums[name] = float(value.get("max_weight", value.get("max")))

    global_minimum = raw.get("minimum_class_weight")
    if global_minimum is not None:
        global_minimum = float(global_minimum)
        if raw.get("allow_zero"):
            global_minimum = 0.0
        minimums = {name: max(value, global_minimum) for name, value in minimums.items()}

    hhi_value = raw.get("hhi_max", raw.get("max_hhi", raw.get("class_hhi_max")))
    hhi_max = None if hhi_value is None else float(hhi_value)
    for name, value in minimums.items():
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"minimum weight for {name} must be between 0 and 1")
    for name, value in maximums.items():
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"maximum weight for {name} must be between 0 and 1")
        if minimums[name] > value + 1e-12:
            raise ValueError(f"minimum weight exceeds maximum for {name}")
    if hhi_max is not None and (
        not math.isfinite(hhi_max) or not 0.0 <= hhi_max <= 1.0
    ):
        raise ValueError("class HHI maximum must be between 0 and 1")
    return {
        "minimums": minimums,
        "maximums": maximums,
        "hhi_max": hhi_max,
    }


def normalize_allocation_config(
    config: Optional[Mapping[str, object]],
    *,
    base: Optional[Mapping[str, object]] = None,
    class_names: Sequence[str] = ASSET_CLASSES,
) -> dict[str, object]:
    """Merge explicit allocation policy values without resolving a profile."""

    result = dict(base or ALLOCATION_CONFIG["caio"])
    if config:
        result.update(dict(config))
    raw_constraints = config.get("class_constraints") if config else None
    if raw_constraints is None:
        raw_constraints = result.get("class_constraints")
    result["class_constraints"] = normalize_class_constraints(
        raw_constraints,
        class_names,
    )
    if config and "crypto_risk_contribution_cap" in config:
        caps = dict(result.get("risk_contribution_caps", {}))
        caps["crypto"] = float(config["crypto_risk_contribution_cap"])
        result["risk_contribution_caps"] = caps
    if "minimum_class_weight" not in result:
        result["minimum_class_weight"] = 0.0
    return result
