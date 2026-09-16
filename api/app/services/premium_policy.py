from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
import math
from typing import Any, Mapping, Sequence

from app.db.models import ProfileRecord


POLICY_VERSION = "premium-policy-v1"
RULES_VERSION = POLICY_VERSION
PROFILE_SCHEMA_VERSION = 1

STOCK_FACTOR_KEYS = ("liquidez", "rent", "value", "growth", "div")
FII_FACTOR_KEYS = ("liquidity", "size_cash", "value", "growth", "dividend")
CLASS_KEYS = (
    "brazilian_stocks",
    "fiis",
    "international_equity",
    "fixed_income",
    "crypto",
)

SYSTEM_GA_CONFIG = {
    "population": 300,
    "generations": 400,
    "mutation_rate": 0.02,
    "crossover_rate": 0.8,
    "run_count": 30,
}

SELECTOR_SCORE_WEIGHTS = {
    "score": 0.40,
    "apetite": 0.15,
    "capacidade": 0.15,
    "liquidez": 0.15,
    "conhecimento": 0.15,
}

# Versioned API inputs. Runtime GA knobs stay identical across profile bands.
STOCK_RULES = {
    "conservador": {
        "n_assets": 10,
        "factor_weights": {"liquidez": 0.30, "rent": 0.25, "value": 0.15, "growth": 0.10, "div": 0.20},
        "filters": {"cap_min": 5_000_000_000, "liq_min": 2_000_000},
        "lambda_hhi": 0.50,
    },
    "moderado": {
        "n_assets": 12,
        "factor_weights": {"liquidez": 0.20, "rent": 0.25, "value": 0.25, "growth": 0.20, "div": 0.10},
        "filters": {"cap_min": 1_000_000_000, "liq_min": 500_000},
        "lambda_hhi": 0.25,
    },
    "arrojado": {
        "n_assets": 15,
        "factor_weights": {"liquidez": 0.10, "rent": 0.20, "value": 0.20, "growth": 0.40, "div": 0.10},
        "filters": {"cap_min": 200_000_000, "liq_min": 50_000},
        "lambda_hhi": 0.10,
    },
}

FII_RULES = {
    "conservador": {
        "n_assets": 10,
        "factor_weights": {"liquidity": 0.30, "size_cash": 0.25, "value": 0.15, "growth": 0.10, "dividend": 0.20},
        "filters": {"size_min": 5_000_000_000, "liq_min": 2_000_000},
        "lambda_hhi": 0.50,
    },
    "moderado": {
        "n_assets": 12,
        "factor_weights": {"liquidity": 0.20, "size_cash": 0.25, "value": 0.25, "growth": 0.20, "dividend": 0.10},
        "filters": {"size_min": 1_000_000_000, "liq_min": 500_000},
        "lambda_hhi": 0.25,
    },
    "arrojado": {
        "n_assets": 15,
        "factor_weights": {"liquidity": 0.10, "size_cash": 0.20, "value": 0.20, "growth": 0.40, "dividend": 0.10},
        "filters": {"size_min": 200_000_000, "liq_min": 50_000},
        "lambda_hhi": 0.10,
    },
}


class PremiumPolicyError(ValueError):
    """Raised when a persisted profile cannot produce a safe Premium policy."""


@dataclass(frozen=True)
class ProfilePolicy:
    schema_version: int
    profile_revision: int
    raw_score: float
    score: float
    dimensions: Mapping[str, float]
    generic_profile: str
    restrictions: tuple[str, ...]
    applied_rules: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AllocationPolicy:
    score: float
    volatility_cap: float
    drawdown_cap: float
    crypto_risk_contribution_cap: float | None
    hhi_penalty: float
    risk_adjusted_weights: Mapping[str, float]
    class_constraints: Mapping[str, object]


@dataclass(frozen=True)
class SelectorPolicy:
    selection_preset: str
    n_assets: int
    factor_weights: Mapping[str, float]
    liquidity_and_size_filters: Mapping[str, object]
    lambda_hhi: float
    system_ga_config: Mapping[str, int | float]


@dataclass(frozen=True)
class Provenance:
    policy_version: str
    model_versions: Mapping[str, str]
    source_snapshot_ids: tuple[str, ...]
    source_snapshot_hashes: Mapping[str, str]
    cutoff_date: str | None
    random_seed: int | None


@dataclass(frozen=True)
class ResolvedOptimizationPolicy:
    profile: ProfilePolicy
    allocation: AllocationPolicy
    stocks: SelectorPolicy
    fiis: SelectorPolicy
    provenance: Provenance

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": _section_dict(self.profile),
            "allocation": _section_dict(self.allocation),
            "stocks": _section_dict(self.stocks),
            "fiis": _section_dict(self.fiis),
            "provenance": _section_dict(self.provenance),
        }

    as_dict = to_dict

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _section_dict(section: object) -> dict[str, object]:
    value = asdict(section)
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _finite(value: object, label: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PremiumPolicyError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise PremiumPolicyError(f"{label} must be between {minimum} and {maximum}")
    return result


def _anchor_pair(score: float) -> tuple[str, str, float]:
    if score <= 0.5:
        return "conservador", "moderado", score / 0.5
    return "moderado", "arrojado", (score - 0.5) / 0.5


def _interpolate(left: float, right: float, fraction: float) -> float:
    return round(left + (right - left) * fraction, 6)


def _selector_policy(
    rules: Mapping[str, Mapping[str, object]],
    selector_score: float,
    restrictions: set[str],
) -> SelectorPolicy:
    left_name, right_name, fraction = _anchor_pair(selector_score)
    left = rules[left_name]
    right = rules[right_name]
    left_weights = left["factor_weights"]
    right_weights = right["factor_weights"]
    assert isinstance(left_weights, Mapping) and isinstance(right_weights, Mapping)
    weights = {
        key: _interpolate(float(left_weights[key]), float(right_weights[key]), fraction)
        for key in left_weights
    }
    total = sum(weights.values())
    weights[next(iter(weights))] = round(weights[next(iter(weights))] + 1.0 - total, 6)

    left_filters = left["filters"]
    right_filters = right["filters"]
    assert isinstance(left_filters, Mapping) and isinstance(right_filters, Mapping)
    filters = {
        key: _interpolate(float(left_filters[key]), float(right_filters[key]), fraction)
        for key in left_filters
    }
    filters["enforce_liquidity_and_size"] = "evitar_illiquidez" in restrictions
    if "limitar_concentracao" in restrictions:
        filters["hhi_max"] = 0.25

    lambda_hhi = _interpolate(float(left["lambda_hhi"]), float(right["lambda_hhi"]), fraction)
    if "limitar_concentracao" in restrictions:
        lambda_hhi = max(lambda_hhi, 0.25)

    return SelectorPolicy(
        selection_preset=left_name if fraction < 0.5 else right_name,
        n_assets=int(round(_interpolate(float(left["n_assets"]), float(right["n_assets"]), fraction))),
        factor_weights=weights,
        liquidity_and_size_filters=filters,
        lambda_hhi=lambda_hhi,
        system_ga_config=dict(SYSTEM_GA_CONFIG),
    )


def _selector_score(score: float, dimensions: Mapping[str, float]) -> float:
    return round(
        sum(
            SELECTOR_SCORE_WEIGHTS[key]
            * (score if key == "score" else dimensions[key])
            for key in SELECTOR_SCORE_WEIGHTS
        ),
        6,
    )


def _validate_restrictions(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PremiumPolicyError("profile restrictions are required")
    supported = {
        "nenhuma",
        "priorizar_renda",
        "evitar_cripto",
        "evitar_exterior",
        "limitar_concentracao",
        "evitar_illiquidez",
    }
    if any(not isinstance(item, str) or item not in supported for item in value):
        raise PremiumPolicyError("profile contains an unsupported restriction")
    order = (
        "nenhuma",
        "priorizar_renda",
        "evitar_cripto",
        "evitar_exterior",
        "limitar_concentracao",
        "evitar_illiquidez",
    )
    normalized = tuple(item for item in order if item in set(value))
    if "nenhuma" in normalized and len(normalized) > 1:
        raise PremiumPolicyError("restriction 'nenhuma' cannot coexist with another restriction")
    return normalized


def _validate_class_constraints(constraints: Mapping[str, object]) -> None:
    minimums = constraints["minimum_weights"]
    maximums = constraints["maximum_weights"]
    assert isinstance(minimums, Mapping) and isinstance(maximums, Mapping)
    if sum(float(value) for value in minimums.values()) > 1.0 + 1e-9:
        raise PremiumPolicyError("class restrictions are infeasible")
    for key, minimum in minimums.items():
        if key in maximums and float(minimum) > float(maximums[key]) + 1e-9:
            raise PremiumPolicyError("class restrictions are infeasible")
    available = [key for key in CLASS_KEYS if float(maximums.get(key, 1.0)) > 0]
    hhi_max = constraints["hhi_max"]
    if hhi_max is not None and available and 1.0 / len(available) > float(hhi_max) + 1e-9:
        raise PremiumPolicyError("class restrictions are infeasible")


def _allocation_policy(score: float, restrictions: set[str]) -> AllocationPolicy:
    try:
        from allocation_config import ALLOCATION_PROFILE_ANCHORS
        from allocation_profiles import build_anchor_profiles, interpolate_profile
    except ModuleNotFoundError:
        from app.adapters.allocation_engine import (  # type: ignore[no-redef]
            ALLOCATION_PROFILE_ANCHORS,
            build_anchor_profiles,
            interpolate_profile,
        )

    anchors = build_anchor_profiles(ALLOCATION_PROFILE_ANCHORS)
    interpolated = interpolate_profile(
        score,
        anchors,
        name="premium",
        calibration_source="Profile v1 suitability score",
    )
    constraints: dict[str, object] = {
        "minimum_class_weight": 0.0,
        "minimum_weights": {},
        "maximum_weights": {},
        "hhi_max": None,
    }
    minimums = constraints["minimum_weights"]
    maximums = constraints["maximum_weights"]
    assert isinstance(minimums, dict) and isinstance(maximums, dict)
    if "priorizar_renda" in restrictions:
        minimums["fixed_income"] = 0.40
    if "evitar_cripto" in restrictions:
        maximums["crypto"] = 0.0
    if "evitar_exterior" in restrictions:
        maximums["international_equity"] = 0.0
    if "limitar_concentracao" in restrictions:
        constraints["hhi_max"] = 0.25
    _validate_class_constraints(constraints)

    return AllocationPolicy(
        score=score,
        volatility_cap=float(interpolated.volatility_cap),
        drawdown_cap=float(interpolated.drawdown_cap),
        crypto_risk_contribution_cap=(
            None
            if interpolated.crypto_risk_contribution_cap is None
            else float(interpolated.crypto_risk_contribution_cap)
        ),
        hhi_penalty=float(interpolated.hhi_penalty),
        risk_adjusted_weights=dict(interpolated.risk_adjusted_weights),
        class_constraints=constraints,
    )


def resolve_premium_policy(
    profile_record: ProfileRecord,
    rules_version: str = POLICY_VERSION,
    *,
    model_versions: Mapping[str, str] | None = None,
    source_snapshot_ids: Sequence[str] = (),
    source_snapshot_hashes: Mapping[str, str] | None = None,
    cutoff_date: date | str | None = None,
    random_seed: int | None = None,
) -> ResolvedOptimizationPolicy:
    """Resolve one persisted Profile v1 revision into explicit engine inputs."""

    if rules_version != POLICY_VERSION:
        raise PremiumPolicyError(f"unsupported policy version: {rules_version}")
    schema_version_value = getattr(profile_record, "schema_version", None)
    schema_version = int(
        PROFILE_SCHEMA_VERSION if schema_version_value is None else schema_version_value
    )
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise PremiumPolicyError(f"unsupported profile schema version: {schema_version}")
    profile_revision = int(profile_record.version)
    if profile_revision <= 0:
        raise PremiumPolicyError("profile revision must be positive")

    score = _finite(profile_record.suitability_score, "profile score")
    raw_score_value = getattr(profile_record, "raw_score", None)
    raw_score = _finite(
        profile_record.suitability_score if raw_score_value is None else raw_score_value,
        "raw profile score",
    )
    dimensions = getattr(profile_record, "dimensions", None)
    if not isinstance(dimensions, Mapping):
        raise PremiumPolicyError("profile dimensions are required")
    checked_dimensions = {
        key: _finite(dimensions.get(key), f"profile dimension {key}")
        for key in ("apetite", "capacidade", "liquidez", "conhecimento")
    }
    generic_profile = str(profile_record.generic_profile)
    if generic_profile not in STOCK_RULES:
        raise PremiumPolicyError(f"unsupported generic profile: {generic_profile}")

    restrictions_value = getattr(profile_record, "restrictions_json", None)
    if not restrictions_value:
        answers = getattr(profile_record, "answers", {})
        restrictions_value = answers.get("restricoes") if isinstance(answers, Mapping) else None
    restrictions = _validate_restrictions(restrictions_value)
    restriction_set = set(restrictions)
    rules = getattr(profile_record, "rules_json", []) or []
    warnings = getattr(profile_record, "warnings_json", []) or []
    if not isinstance(rules, list) or not all(isinstance(item, str) for item in rules):
        raise PremiumPolicyError("profile applied rules are invalid")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise PremiumPolicyError("profile warnings are invalid")

    selector_score = _selector_score(score, checked_dimensions)
    allocation = _allocation_policy(score, restriction_set)
    stocks = _selector_policy(STOCK_RULES, selector_score, restriction_set)
    fiis = _selector_policy(FII_RULES, selector_score, restriction_set)
    cutoff = cutoff_date.isoformat() if isinstance(cutoff_date, date) else cutoff_date
    if random_seed is not None and not isinstance(random_seed, int):
        raise PremiumPolicyError("random seed must be an integer")
    provenance = Provenance(
        policy_version=rules_version,
        model_versions=dict(
            model_versions
            or {
                "allocation": "allocation-v1",
                "stocks": "stock-selection-v1",
                "fiis": "fii-selection-v1",
            }
        ),
        source_snapshot_ids=tuple(str(value) for value in source_snapshot_ids),
        source_snapshot_hashes=dict(source_snapshot_hashes or {}),
        cutoff_date=cutoff,
        random_seed=random_seed,
    )
    return ResolvedOptimizationPolicy(
        profile=ProfilePolicy(
            schema_version=schema_version,
            profile_revision=profile_revision,
            raw_score=raw_score,
            score=score,
            dimensions=checked_dimensions,
            generic_profile=generic_profile,
            restrictions=restrictions,
            applied_rules=tuple(rules),
            warnings=tuple(warnings),
        ),
        allocation=allocation,
        stocks=stocks,
        fiis=fiis,
        provenance=provenance,
    )
