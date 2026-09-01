from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.db.models import Entitlement
from app.entitlements.dependencies import require_premium
from app.schemas.premium import PremiumAccessResponse

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
