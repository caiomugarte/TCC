from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_account
from app.db.models import Account, PortfolioSnapshot, RecommendationRun
from app.db.session import get_session
from app.errors import api_error
from app.schemas.review import ReviewResponse
from app.services.review import ReviewUnavailableError, calculate_review

router = APIRouter(prefix="/v1/review", tags=["review"])


@router.get("", response_model=ReviewResponse)
def read_review(
    account: Annotated[Account, Depends(get_current_account)],
    session: Annotated[Session, Depends(get_session)],
) -> ReviewResponse:
    recommendation = session.scalar(
        select(RecommendationRun)
        .where(RecommendationRun.account_id == account.id)
        .order_by(desc(RecommendationRun.created_at))
        .limit(1)
    )
    portfolio = session.scalar(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.account_id == account.id)
        .order_by(desc(PortfolioSnapshot.captured_at), desc(PortfolioSnapshot.created_at))
        .limit(1)
    )
    if recommendation is None or portfolio is None:
        raise api_error(
            409,
            "review_unavailable",
            "Complete a recomendação e informe sua carteira antes da revisão.",
        )
    try:
        return ReviewResponse.model_validate(calculate_review(recommendation, portfolio))
    except ReviewUnavailableError as exc:
        raise api_error(409, "review_unavailable", "A revisão não está disponível.", str(exc)) from exc
