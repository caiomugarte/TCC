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
from app.entitlements.dependencies import get_entitlement, is_premium_active
from app.errors import api_error
from app.repositories.recommendations import RecommendationRepository
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])


def _response_with_profile(record: RecommendationRun, profile: ProfileRecord) -> RecommendationResponse:
    completed = record.status == "completed"
    result = record.result_json if completed and isinstance(record.result_json, dict) else {}
    classes = result.get("classes", record.classes) if completed else []
    assumptions = result.get("assumptions", record.assumptions) if completed else []
    risks = result.get("risks", record.risks) if completed else []
    stocks = result.get("stocks", []) if completed else []
    fiis = result.get("fiis", []) if completed else []
    policy = record.policy_json if isinstance(record.policy_json, dict) else None
    provenance = (
        result.get("provenance", record.provenance_json)
        if completed
        else record.provenance_json
    )
    return RecommendationResponse(
        id=record.id,
        account_id=record.account_id,
        profile_version=profile.version,
        plan=record.plan,
        model_version=record.model_version,
        snapshot_id=record.snapshot_id,
        snapshot_cutoff=record.snapshot_cutoff,
        classes=classes,
        assumptions=assumptions,
        risks=risks,
        created_at=record.created_at,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        failure_code=record.failure_code,
        failure_message=record.failure_message,
        stocks=stocks,
        fiis=fiis,
        policy=policy,
        provenance=provenance if isinstance(provenance, dict) else None,
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
    current_profile = session.scalar(
        select(ProfileRecord)
        .where(ProfileRecord.account_id == account.id)
        .order_by(desc(ProfileRecord.version))
        .limit(1)
    )
    if current_profile is None:
        return None
    current_plan = "premium" if is_premium_active(get_entitlement(account, session)) else "basic"
    record = session.scalar(
        select(RecommendationRun)
        .where(
            RecommendationRun.account_id == account.id,
            RecommendationRun.profile_id == current_profile.id,
            RecommendationRun.plan == current_plan,
        )
        .order_by(desc(RecommendationRun.created_at))
        .limit(1)
    )
    if record is None:
        return None
    return _response_with_profile(record, current_profile)


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
    record = RecommendationRepository(session).get_owned_run(account.id, recommendation_id)
    if record is None:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    profile = session.get(ProfileRecord, record.profile_id)
    if profile is None or profile.account_id != account.id:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
    return _response_with_profile(record, profile)
