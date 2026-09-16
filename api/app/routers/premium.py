from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, Entitlement, ProfileRecord
from app.db.session import get_session
from app.entitlements.dependencies import require_premium, require_premium_pilot
from app.errors import api_error
from app.schemas.premium import PremiumAccessResponse
from app.schemas.recommendation import (
    PremiumRecommendationRequest,
    RecommendationResponse,
)
from app.services.premium_recommendation import (
    PremiumRecommendationError,
    create_premium_run,
)
from app.routers.recommendations import _response_with_profile

router = APIRouter(prefix="/v1/premium", tags=["premium"])


@router.get("", response_model=PremiumAccessResponse)
def read_premium_access(
    entitlement: Annotated[Entitlement, Depends(require_premium)],
) -> PremiumAccessResponse:
    return PremiumAccessResponse(
        feature="premium_access",
        access="granted",
        plan="premium",
        entitlement_status=entitlement.status,
    )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=202,
)
def start_premium_recommendation(
    request: PremiumRecommendationRequest,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
    _entitlement: Annotated[Entitlement, Depends(require_premium_pilot)],
) -> RecommendationResponse:
    try:
        record = create_premium_run(account, request, session)
    except PremiumRecommendationError as exc:
        raise api_error(exc.status_code, exc.code, exc.message, exc.details) from exc
    profile = session.get(ProfileRecord, record.profile_id)
    if profile is None or profile.account_id != account.id:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    return _response_with_profile(record, profile)
