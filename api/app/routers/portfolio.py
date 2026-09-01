from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, PortfolioSnapshot, utc_now
from app.db.session import get_session
from app.errors import api_error
from app.schemas.portfolio import PortfolioInput, PortfolioResponse
from app.services.portfolio import PortfolioValidationError, normalize_portfolio

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


def _response(record: PortfolioSnapshot) -> PortfolioResponse:
    return PortfolioResponse(
        id=record.id,
        account_id=record.account_id,
        source=record.source,
        captured_at=record.captured_at,
        currency=record.currency,
        total_value_brl=float(record.total_value_brl),
        classes=record.classes,
        normalized_weights=record.normalized_weights,
    )


def _latest(session: Session, account_id: str) -> PortfolioSnapshot | None:
    return session.scalar(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.account_id == account_id)
        .order_by(desc(PortfolioSnapshot.captured_at), desc(PortfolioSnapshot.created_at))
        .limit(1)
    )


@router.get("", response_model=PortfolioResponse | None)
def read_portfolio(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> PortfolioResponse | None:
    record = _latest(session, account.id)
    return _response(record) if record else None


@router.put("", response_model=PortfolioResponse)
def save_portfolio(
    submission: PortfolioInput,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> PortfolioResponse:
    try:
        values, total, weights = normalize_portfolio(submission)
    except PortfolioValidationError as exc:
        raise api_error(422, "portfolio_invalid", str(exc)) from exc

    record = PortfolioSnapshot(
        account_id=account.id,
        source="manual",
        captured_at=utc_now(),
        currency=submission.currency,
        total_value_brl=total,
        classes=values,
        normalized_weights=weights,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _response(record)
