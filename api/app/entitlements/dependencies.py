from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.config import AppSettings, get_settings
from app.db.models import Account, Entitlement
from app.db.session import get_session
from app.errors import api_error


def is_premium_active(entitlement: Entitlement | None) -> bool:
    return bool(
        entitlement
        and entitlement.plan == "premium"
        and entitlement.status in {"active", "grace_period"}
    )


def get_entitlement(account: Account, session: Session) -> Entitlement | None:
    return session.scalar(select(Entitlement).where(Entitlement.account_id == account.id))


def require_premium(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> Entitlement:
    entitlement = get_entitlement(account, session)
    if not is_premium_active(entitlement):
        raise api_error(
            403,
            "premium_required",
            "Este recurso exige um entitlement Premium ativo.",
        )
    return entitlement


def require_premium_pilot(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[AppSettings | None, Depends(get_settings)] = None,
) -> Entitlement:
    """Require Premium entitlement and an explicitly approved pilot account."""

    entitlement = require_premium(account, session)
    resolved_settings = settings or get_settings()
    if account.id not in resolved_settings.premium_pilot_account_ids:
        raise api_error(
            403,
            "premium_pilot_required",
            "Este recurso está disponível apenas para contas do piloto Premium.",
        )
    return entitlement
