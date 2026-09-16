from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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


class PremiumRecommendationRequest(ProfileSchema):
    profile_id: str | None = None


class AllocationClass(ProfileSchema):
    key: AssetClassKey
    label: str
    target_weight: float = Field(ge=0, le=1)
    target_amount_brl: float = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RecommendationConstituent(ProfileSchema):
    ticker: str
    sleeve_weight: float = Field(ge=0, le=1)
    portfolio_weight: float = Field(ge=0, le=1)
    target_amount_brl: float = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


RecommendationStatus = Literal["queued", "running", "completed", "failed"]


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
    status: RecommendationStatus = "completed"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    stocks: list[RecommendationConstituent] = Field(default_factory=list)
    fiis: list[RecommendationConstituent] = Field(default_factory=list)
    policy: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
