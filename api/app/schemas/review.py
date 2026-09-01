from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.profile import ProfileSchema
from app.schemas.recommendation import AssetClassKey

DriftStatus = Literal["within_range", "underweight", "overweight"]
SuggestedAction = Literal["hold", "contribute", "review_sale"]


class DriftItem(ProfileSchema):
    class_key: AssetClassKey
    current_weight: float = Field(ge=0, le=1)
    target_weight: float = Field(ge=0, le=1)
    drift: float
    value_gap_brl: float
    status: DriftStatus
    suggested_action: SuggestedAction


class ReviewResponse(ProfileSchema):
    recommendation_id: str
    portfolio_id: str
    drift_band: float = Field(gt=0, lt=1)
    items: list[DriftItem]
