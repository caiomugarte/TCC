"""Configuration for the asset-class allocation analysis."""

from pathlib import Path


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
