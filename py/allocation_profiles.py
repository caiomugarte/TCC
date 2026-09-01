"""Allocation-only profile parameters and score interpolation."""

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Optional


ANCHOR_SCORES = {
    "conservador": 0.0,
    "moderado": 0.5,
    "arrojado": 1.0,
}
RISK_ADJUSTED_WEIGHT_KEYS = ("return", "volatility", "drawdown")


class AllocationProfileError(ValueError):
    """Raised when an allocation profile is incomplete or invalid."""


@dataclass(frozen=True)
class AllocationProfile:
    """Risk policy for whole-wallet allocation, independent of stock selection."""

    name: str
    score: float
    volatility_cap: float
    drawdown_cap: float
    crypto_risk_contribution_cap: Optional[float]
    hhi_penalty: float
    calibration_source: str
    risk_adjusted_weights: Mapping[str, float] = field(default_factory=dict)
    observed_metrics: Mapping[str, float] = field(default_factory=dict)
    calibration_inputs: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "score": self.score,
            "volatility_cap": self.volatility_cap,
            "drawdown_cap": self.drawdown_cap,
            "crypto_risk_contribution_cap": self.crypto_risk_contribution_cap,
            "hhi_penalty": self.hhi_penalty,
            "risk_adjusted_weights": dict(self.risk_adjusted_weights),
            "calibration_source": self.calibration_source,
            "observed_metrics": dict(self.observed_metrics),
            "calibration_inputs": dict(self.calibration_inputs),
        }


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AllocationProfileError(f"{label} must be finite")
    return result


def validate_profile(profile: AllocationProfile) -> AllocationProfile:
    score = _finite(profile.score, "profile score")
    if not 0.0 <= score <= 1.0:
        raise AllocationProfileError("profile score must be between 0 and 1")

    volatility_cap = _finite(profile.volatility_cap, "volatility cap")
    if volatility_cap <= 0.0:
        raise AllocationProfileError("volatility cap must be positive")

    drawdown_cap = _finite(profile.drawdown_cap, "drawdown cap")
    if drawdown_cap <= 0.0:
        raise AllocationProfileError("drawdown cap must be positive")

    if profile.crypto_risk_contribution_cap is not None:
        crypto_cap = _finite(
            profile.crypto_risk_contribution_cap,
            "crypto risk contribution cap",
        )
        if not 0.0 <= crypto_cap <= 1.0:
            raise AllocationProfileError(
                "crypto risk contribution cap must be between 0 and 1"
            )

    hhi_penalty = _finite(profile.hhi_penalty, "HHI penalty")
    if hhi_penalty < 0.0:
        raise AllocationProfileError("HHI penalty must be non-negative")
    preference_weights = dict(profile.risk_adjusted_weights)
    if preference_weights:
        if set(preference_weights) != set(RISK_ADJUSTED_WEIGHT_KEYS):
            raise AllocationProfileError(
                "risk-adjusted weights must contain return, volatility, and drawdown"
            )
        if any(
            _finite(preference_weights[key], f"{key} preference weight") < 0.0
            for key in RISK_ADJUSTED_WEIGHT_KEYS
        ):
            raise AllocationProfileError("risk-adjusted weights must be non-negative")
        if abs(sum(preference_weights.values()) - 1.0) > 1e-9:
            raise AllocationProfileError("risk-adjusted weights must sum to 1")
    if not profile.calibration_source.strip():
        raise AllocationProfileError("calibration source is required")
    return profile


def _interpolate(left: float, right: float, fraction: float) -> float:
    return left + (right - left) * fraction


def build_anchor_profiles(
    anchor_config: Mapping[str, Mapping[str, object]],
) -> Dict[str, AllocationProfile]:
    """Build validated allocation anchors from declarative configuration."""

    missing = sorted(set(ANCHOR_SCORES) - set(anchor_config))
    if missing:
        raise AllocationProfileError(f"missing allocation anchors: {missing}")
    return {
        name: validate_profile(
            AllocationProfile(
                name=name,
                score=float(values["score"]),
                volatility_cap=float(values["volatility_cap"]),
                drawdown_cap=float(values["drawdown_cap"]),
                crypto_risk_contribution_cap=(
                    None
                    if values.get("crypto_risk_contribution_cap") is None
                    else float(values["crypto_risk_contribution_cap"])
                ),
                hhi_penalty=float(values["hhi_penalty"]),
                calibration_source=str(values["calibration_source"]),
                risk_adjusted_weights=dict(values.get("risk_adjusted_weights", {})),
                calibration_inputs=dict(values.get("calibration_inputs", {})),
            )
        )
        for name, values in anchor_config.items()
        if name in ANCHOR_SCORES
    }


def _anchor_pair(score: float, anchors: Mapping[str, AllocationProfile]):
    if score <= 0.5:
        return anchors["conservador"], anchors["moderado"], score / 0.5
    return anchors["moderado"], anchors["arrojado"], (score - 0.5) / 0.5


def interpolate_profile(
    score: float,
    anchors: Mapping[str, AllocationProfile],
    name: str = "caio",
    calibration_source: str = "interpolated from allocation anchors",
    calibration_inputs: Optional[Mapping[str, object]] = None,
) -> AllocationProfile:
    """Interpolate allocation policy parameters at a suitability score."""

    checked_score = _finite(score, "profile score")
    if not 0.0 <= checked_score <= 1.0:
        raise AllocationProfileError("profile score must be between 0 and 1")
    missing = sorted(set(ANCHOR_SCORES) - set(anchors))
    if missing:
        raise AllocationProfileError(f"missing allocation anchors: {missing}")
    checked = {
        key: validate_profile(anchors[key])
        for key in ANCHOR_SCORES
    }
    for key, expected_score in ANCHOR_SCORES.items():
        if abs(checked[key].score - expected_score) > 1e-9:
            raise AllocationProfileError(
                f"{key} anchor must have score {expected_score}"
            )

    left, right, fraction = _anchor_pair(checked_score, checked)
    left_crypto = left.crypto_risk_contribution_cap
    right_crypto = right.crypto_risk_contribution_cap
    if (left_crypto is None) != (right_crypto is None):
        raise AllocationProfileError(
            "all adjacent anchors must either define or omit the crypto risk cap"
        )
    crypto_cap = None
    if left_crypto is not None and right_crypto is not None:
        crypto_cap = _interpolate(left_crypto, right_crypto, fraction)

    left_preferences = dict(left.risk_adjusted_weights)
    right_preferences = dict(right.risk_adjusted_weights)
    if bool(left_preferences) != bool(right_preferences):
        raise AllocationProfileError(
            "adjacent anchors must both define or both omit risk-adjusted weights"
        )
    preference_weights = {
        key: _interpolate(left_preferences[key], right_preferences[key], fraction)
        for key in RISK_ADJUSTED_WEIGHT_KEYS
    } if left_preferences else {}

    return validate_profile(
        AllocationProfile(
            name=name,
            score=checked_score,
            volatility_cap=_interpolate(
                left.volatility_cap,
                right.volatility_cap,
                fraction,
            ),
            drawdown_cap=_interpolate(
                left.drawdown_cap,
                right.drawdown_cap,
                fraction,
            ),
            crypto_risk_contribution_cap=crypto_cap,
            hhi_penalty=_interpolate(left.hhi_penalty, right.hhi_penalty, fraction),
            calibration_source=calibration_source,
            risk_adjusted_weights=preference_weights,
            calibration_inputs=dict(calibration_inputs or {}),
        )
    )
