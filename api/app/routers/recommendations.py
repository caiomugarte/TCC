from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, ProfileRecord, RecommendationRun
from app.db.session import get_session
from app.entitlements.dependencies import get_entitlement, is_premium_active
from app.errors import api_error
from app.repositories.recommendations import RecommendationRepository
from app.schemas.recommendation import (
    PremiumRecommendationRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.premium_recommendation import (
    PremiumRecommendationError,
    get_premium_recommendation_service,
)

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


def _read_latest_for_current_context(
    account: Account,
    session: Session,
    *,
    completed_only: bool,
) -> RecommendationResponse | None:
    profile = session.scalar(
        select(ProfileRecord)
        .where(ProfileRecord.account_id == account.id)
        .order_by(desc(ProfileRecord.version))
        .limit(1)
    )
    if profile is None:
        return None
    plan = "premium" if is_premium_active(get_entitlement(account, session)) else "basic"
    repository = RecommendationRepository(session)
    record = (
        repository.get_latest_completed_for_profile(account.id, profile.id, plan=plan)
        if completed_only
        else repository.get_latest_for_profile(account.id, profile.id, plan=plan)
    )
    return _response_with_profile(record, profile) if record is not None else None


@router.get("", response_model=RecommendationResponse | None)
def read_latest_recommendation(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse | None:
    return _read_latest_for_current_context(account, session, completed_only=False)


@router.get("/latest-completed", response_model=RecommendationResponse | None)
def read_latest_completed_recommendation(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse | None:
    return _read_latest_for_current_context(account, session, completed_only=True)


@router.post("", response_model=RecommendationResponse, status_code=202)
def create_recommendation(
    request: RecommendationRequest,
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> RecommendationResponse:
    try:
        record = get_premium_recommendation_service().create_run(
            account,
            PremiumRecommendationRequest(profile_id=request.profile_id),
            session,
            plan="basic",
        )
    except PremiumRecommendationError as exc:
        raise api_error(exc.status_code, exc.code, exc.message, exc.details) from exc
    profile = session.get(ProfileRecord, record.profile_id)
    if profile is None or profile.account_id != account.id:
        raise api_error(404, "recommendation_not_found", "Recomendação não encontrada.")
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
