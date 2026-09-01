from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

from clerk_backend_api import AuthenticateRequestOptions, authenticate_request
from fastapi import Depends, Request
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Account
from app.db.session import get_session


@dataclass(frozen=True)
class ClerkIdentity:
    subject: str
    email: str | None


def _authorized_parties() -> list[str]:
    configured = os.getenv("CLERK_AUTHORIZED_PARTIES", "http://localhost:3000")
    return [party.strip() for party in configured.split(",") if party.strip()]


def get_current_identity(request: Request) -> ClerkIdentity:
    state = authenticate_request(
        request,
        AuthenticateRequestOptions(
            secret_key=os.getenv("CLERK_SECRET_KEY"),
            jwt_key=os.getenv("CLERK_JWT_KEY"),
            audience=os.getenv("CLERK_AUDIENCE") or None,
            authorized_parties=_authorized_parties(),
            accepts_token=["session_token"],
        ),
    )
    if not state.is_signed_in or not state.payload:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthenticated", "message": "Sessão necessária."},
        )

    subject = state.payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthenticated", "message": "Sessão inválida."},
        )

    email = state.payload.get("email") or state.payload.get("email_address")
    return ClerkIdentity(subject=subject, email=email if isinstance(email, str) else None)


def get_current_account_id(
    identity: Annotated[ClerkIdentity, Depends(get_current_identity)],
) -> str:
    return identity.subject


def get_current_account(
    identity: Annotated[ClerkIdentity, Depends(get_current_identity)],
    session: Annotated[Session, Depends(get_session)],
) -> Account:
    account = session.scalar(
        select(Account).where(
            Account.auth_provider == "clerk",
            Account.auth_subject == identity.subject,
        )
    )
    if account is not None:
        return account

    account = Account(
        email=identity.email,
        auth_provider="clerk",
        auth_subject=identity.subject,
    )
    session.add(account)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        account = session.scalar(
            select(Account).where(
                Account.auth_provider == "clerk",
                Account.auth_subject == identity.subject,
            )
        )
        if account is None:
            raise HTTPException(
                status_code=409,
                detail={"code": "account_sync_failed", "message": "Não foi possível sincronizar a conta."},
            )
    else:
        session.refresh(account)
    return account
