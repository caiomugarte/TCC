from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.adapters.allocation_engine import (
    AllocationAdapterError,
    BasicRecommendationInput,
    generate_basic_recommendation,
)
from app.auth.dependencies import get_current_account
from app.db.models import Account, ProfileRecord, RecommendationRun
from app.db.session import get_session
from app.errors import api_error
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


def _response_with_profile(record: RecommendationRun, profile: ProfileRecord) -> RecommendationResponse:
    return RecommendationResponse(
        id=record.id,
        account_id=record.account_id,
        profile_version=profile.version,
        plan=record.plan,
        model_version=record.model_version,
        snapshot_id=record.snapshot_id,
        snapshot_cutoff=record.snapshot_cutoff,
        classes=record.classes,
        assumptions=record.assumptions,
        risks=record.risks,
        created_at=record.created_at,
    )


def _find_profile(
    session: Session,
    account_id: str,
    profile_id: str | None,
) -> ProfileRecord:
    statement = select(ProfileRecord).where(ProfileRecord.account_id == account_id)
    if profile_id:
        statement = statement.where(ProfileRecord.id == profile_id)
    else:
        statement = statement.order_by(desc(ProfileRecord.version)).limit(1)
    profile = session.scalar(statement)
    if profile is None:
        if profile_id:
            raise api_error(404, "profile_not_found", "Perfil não encontrado.")
        raise api_error(409, "profile_required", "Complete o perfil antes de gerar a recomendação.")
    return profile


@router.get("", response_model=RecommendationResponse | None)
def read_latest_recommendation(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse | None:
    record = session.scalar(
        select(RecommendationRun)
        .where(RecommendationRun.account_id == account.id)
        .order_by(desc(RecommendationRun.created_at))
        .limit(1)
    )
    if record is None:
        return None
    profile = session.get(ProfileRecord, record.profile_id)
    if profile is None or profile.account_id != account.id:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    return _response_with_profile(record, profile)


@router.post("", response_model=RecommendationResponse)
def create_recommendation(
    request: RecommendationRequest,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse:
    profile = _find_profile(session, account.id, request.profile_id)
    try:
        result = generate_basic_recommendation(
            BasicRecommendationInput(
                generic_profile=profile.generic_profile,
                investable_capital_brl=float(profile.investable_capital_brl),
            )
        )
    except (AllocationAdapterError, OSError, ValueError) as exc:
        raise api_error(
            409,
            "recommendation_unavailable",
            "A recomendação não está disponível com os dados atuais.",
            str(exc),
        ) from exc

    record = RecommendationRun(
        account_id=account.id,
        profile_id=profile.id,
        plan=result["plan"],
        model_version=result["model_version"],
        snapshot_id=result["snapshot_id"],
        snapshot_cutoff=result["snapshot_cutoff"],
        classes=result["classes"],
        assumptions=result["assumptions"],
        risks=result["risks"],
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return _response_with_profile(record, profile)


@router.get("/{recommendation_id}", response_model=RecommendationResponse)
def read_recommendation(
    recommendation_id: str,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse:
    record = session.scalar(
        select(RecommendationRun).where(
            RecommendationRun.id == recommendation_id,
            RecommendationRun.account_id == account.id,
        )
    )
    if record is None:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    profile = session.get(ProfileRecord, record.profile_id)
    if profile is None or profile.account_id != account.id:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    return _response_with_profile(record, profile)
