from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, ProfileRecord, utc_now
from app.db.session import get_session
from app.schemas.profile import ProfileResponse, ProfileSubmission
from app.services.profile import compute_profile

router = APIRouter(prefix="/v1/profile", tags=["profile"])


def _response(record: ProfileRecord) -> ProfileResponse:
    return ProfileResponse(
        id=record.id,
        account_id=record.account_id,
        version=record.version,
        answers=record.answers,
        dimensions=record.dimensions,
        suitability_score=float(record.suitability_score),
        generic_profile=record.generic_profile,
        investable_capital_brl=float(record.investable_capital_brl),
        consented_at=record.consented_at,
        created_at=record.created_at,
    )


@router.get("", response_model=ProfileResponse | None)
def read_profile(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> ProfileResponse | None:
    record = session.scalar(
        select(ProfileRecord)
        .where(ProfileRecord.account_id == account.id)
        .order_by(desc(ProfileRecord.version))
        .limit(1)
    )
    return _response(record) if record else None


@router.put("", response_model=ProfileResponse)
def save_profile(
    submission: ProfileSubmission,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> ProfileResponse:
    computed = compute_profile(submission)
    latest = session.scalar(
        select(ProfileRecord)
        .where(ProfileRecord.account_id == account.id)
        .order_by(desc(ProfileRecord.version))
        .limit(1)
    )
    record = ProfileRecord(
        account_id=account.id,
        version=(latest.version + 1) if latest else 1,
        answers=computed.answers,
        dimensions=computed.dimensions,
        suitability_score=computed.score,
        generic_profile=computed.generic_profile,
        investable_capital_brl=computed.investable_capital_brl,
        consented_at=utc_now(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _response(record)
