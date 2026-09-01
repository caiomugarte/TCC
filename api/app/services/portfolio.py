from __future__ import annotations

import math

from app.schemas.portfolio import PORTFOLIO_CLASS_KEYS, PortfolioInput


class PortfolioValidationError(ValueError):
    pass


def normalize_portfolio(submission: PortfolioInput) -> tuple[dict[str, float], float, dict[str, float]]:
    values = {key: round(float(submission.classes[key]), 2) for key in PORTFOLIO_CLASS_KEYS}
    total = round(sum(values.values()), 2)
    if not math.isfinite(total) or total <= 0:
        raise PortfolioValidationError("portfolio total must be greater than zero")

    weights = {key: round(value / total, 6) for key, value in values.items()}
    weights[PORTFOLIO_CLASS_KEYS[-1]] = round(
        weights[PORTFOLIO_CLASS_KEYS[-1]] + 1 - sum(weights.values()),
        6,
    )
    return values, total, weights
