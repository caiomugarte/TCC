from __future__ import annotations

from datetime import datetime
import math
from typing import Literal

from pydantic import Field, field_validator

from app.schemas.profile import ProfileSchema
from app.schemas.recommendation import AssetClassKey

PORTFOLIO_CLASS_KEYS = (
    "brazilian_stocks",
    "fiis",
    "international",
    "fixed_income",
    "crypto",
)


class PortfolioInput(ProfileSchema):
    currency: Literal["BRL"] = "BRL"
    classes: dict[AssetClassKey, float]

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: dict[AssetClassKey, float]) -> dict[AssetClassKey, float]:
        if set(value) != set(PORTFOLIO_CLASS_KEYS):
            raise ValueError("all five asset classes are required")
        for key, amount in value.items():
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"portfolio value for {key} must be finite and non-negative")
        return value


class PortfolioResponse(ProfileSchema):
    id: str
    account_id: str
    source: Literal["manual"]
    captured_at: datetime
    currency: Literal["BRL"]
    total_value_brl: float = Field(gt=0)
    classes: dict[AssetClassKey, float]
    normalized_weights: dict[AssetClassKey, float]
