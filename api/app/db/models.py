from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, Numeric, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, index=True, nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="clerk")
    auth_subject: Mapped[str] = mapped_column(
        String(256), unique=True, index=True, nullable=False, default=new_id
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ProfileRecord(Base):
    __tablename__ = "profiles"
    __table_args__ = (UniqueConstraint("account_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    account_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(nullable=False)
    answers: Mapped[dict] = mapped_column(JSON, nullable=False)
    dimensions: Mapped[dict] = mapped_column(JSON, nullable=False)
    suitability_score: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    raw_score: Mapped[Decimal] = mapped_column(
        Numeric(6, 5), nullable=False, default=0, server_default=text("0")
    )
    rules_json: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default=text("'[]'")
    )
    warnings_json: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default=text("'[]'")
    )
    restrictions_json: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list, server_default=text("'[]'")
    )
    schema_version: Mapped[int] = mapped_column(
        nullable=False, default=1, server_default=text("1")
    )
    generic_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    investable_capital_brl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    consented_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    account_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    profile_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    plan: Mapped[str] = mapped_column(String(16), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_cutoff: Mapped[str] = mapped_column(String(32), nullable=False)
    classes: Mapped[list] = mapped_column(JSON, nullable=False)
    assumptions: Mapped[list] = mapped_column(JSON, nullable=False)
    risks: Mapped[list] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="completed", server_default=text("'completed'")
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    policy_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provenance_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    account_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    total_value_brl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    classes: Mapped[dict] = mapped_column(JSON, nullable=False)
    normalized_weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Entitlement(Base):
    __tablename__ = "entitlements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    account_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("accounts.id", ondelete="CASCADE"), unique=True, index=True
    )
    plan: Mapped[str] = mapped_column(String(16), nullable=False, default="basic")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="inactive")
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    external_customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_subscription_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
