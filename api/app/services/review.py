from __future__ import annotations

import math
from typing import Any

from app.db.models import PortfolioSnapshot, RecommendationRun

DRIFT_BAND = 0.05
_ACTION_ORDER = {"contribute": 0, "hold": 1, "review_sale": 2}


class ReviewUnavailableError(ValueError):
    pass


def calculate_review(
    recommendation: RecommendationRun,
    portfolio: PortfolioSnapshot,
    drift_band: float = DRIFT_BAND,
) -> dict[str, Any]:
    targets = {item["key"]: float(item["target_weight"]) for item in recommendation.classes}
    current = {key: float(value) for key, value in portfolio.normalized_weights.items()}
    values = {key: float(value) for key, value in portfolio.classes.items()}
    if set(targets) != set(current) or set(targets) != set(values):
        raise ReviewUnavailableError("recommendation and portfolio classes do not match")
    if not math.isfinite(drift_band) or drift_band <= 0:
        raise ReviewUnavailableError("drift band must be positive")

    items = []
    for key, target_weight in targets.items():
        current_weight = current[key]
        drift = round(current_weight - target_weight, 6)
        value_gap = round(target_weight * float(portfolio.total_value_brl) - values[key], 2)
        if drift > drift_band:
            status, action = "overweight", "review_sale"
        elif drift < -drift_band:
            status, action = "underweight", "contribute"
        else:
            status, action = "within_range", "hold"
        items.append(
            {
                "class_key": key,
                "current_weight": round(current_weight, 6),
                "target_weight": round(target_weight, 6),
                "drift": drift,
                "value_gap_brl": value_gap,
                "status": status,
                "suggested_action": action,
            }
        )

    items.sort(key=lambda item: (_ACTION_ORDER[item["suggested_action"]], -abs(item["value_gap_brl"])))
    return {
        "recommendation_id": recommendation.id,
        "portfolio_id": portfolio.id,
        "drift_band": drift_band,
        "items": items,
    }
