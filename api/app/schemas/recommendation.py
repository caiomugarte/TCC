from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.profile import ProfileSchema

AssetClassKey = Literal[
    "brazilian_stocks",
    "fiis",
    "international",
    "fixed_income",
    "crypto",
]


class RecommendationRequest(ProfileSchema):
    profile_id: str | None = None


class AllocationClass(ProfileSchema):
    key: AssetClassKey
    label: str
    target_weight: float = Field(ge=0, le=1)
    target_amount_brl: float = Field(ge=0)


class RecommendationResponse(ProfileSchema):
    id: str
    account_id: str
    profile_version: int
    plan: Literal["basic", "premium"]
    model_version: str
    snapshot_id: str
    snapshot_cutoff: str
    classes: list[AllocationClass]
    assumptions: list[str]
    risks: list[str]
    created_at: datetime
