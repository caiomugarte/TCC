from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, Entitlement
from app.db.session import get_session
from app.entitlements.dependencies import get_entitlement, is_premium_active
from app.schemas.account import AccountResponse

router = APIRouter(prefix="/v1", tags=["account"])


def _response(account: Account, entitlement: Entitlement | None) -> AccountResponse:
    return AccountResponse(
        id=account.id,
        email=account.email,
        plan="premium" if is_premium_active(entitlement) else "basic",
        entitlement_status=entitlement.status if entitlement else "inactive",
    )


@router.get("/me", response_model=AccountResponse)
@router.get("/account", response_model=AccountResponse)
def read_account(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> AccountResponse:
    entitlement = get_entitlement(account, session)
    return _response(account, entitlement)
