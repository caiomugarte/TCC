"""create initial account-owned product tables"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("auth_provider", sa.String(length=32), nullable=False),
        sa.Column("auth_subject", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("auth_subject"),
    )
    op.create_index("ix_accounts_email", "accounts", ["email"], unique=True)
    op.create_index("ix_accounts_auth_subject", "accounts", ["auth_subject"], unique=True)

    op.create_table(
        "profiles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("answers", sa.JSON(), nullable=False),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("suitability_score", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.Column("generic_profile", sa.String(length=32), nullable=False),
        sa.Column("investable_capital_brl", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("consented_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id", "version"),
    )
    op.create_index("ix_profiles_account_id", "profiles", ["account_id"], unique=False)

    op.create_table(
        "recommendation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("profile_id", sa.String(length=36), nullable=False),
        sa.Column("plan", sa.String(length=16), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("snapshot_id", sa.String(length=128), nullable=False),
        sa.Column("snapshot_cutoff", sa.String(length=32), nullable=False),
        sa.Column("classes", sa.JSON(), nullable=False),
        sa.Column("assumptions", sa.JSON(), nullable=False),
        sa.Column("risks", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recommendation_runs_account_id", "recommendation_runs", ["account_id"], unique=False)
    op.create_index("ix_recommendation_runs_profile_id", "recommendation_runs", ["profile_id"], unique=False)

    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("total_value_brl", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("classes", sa.JSON(), nullable=False),
        sa.Column("normalized_weights", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_portfolio_snapshots_account_id", "portfolio_snapshots", ["account_id"], unique=False)

    op.create_table(
        "entitlements",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("plan", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("external_customer_id", sa.String(length=128), nullable=True),
        sa.Column("external_subscription_id", sa.String(length=128), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_id"),
    )
    op.create_index("ix_entitlements_account_id", "entitlements", ["account_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_entitlements_account_id", table_name="entitlements")
    op.drop_table("entitlements")
    op.drop_index("ix_portfolio_snapshots_account_id", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")
    op.drop_index("ix_recommendation_runs_profile_id", table_name="recommendation_runs")
    op.drop_index("ix_recommendation_runs_account_id", table_name="recommendation_runs")
    op.drop_table("recommendation_runs")
    op.drop_index("ix_profiles_account_id", table_name="profiles")
    op.drop_table("profiles")
    op.drop_index("ix_accounts_auth_subject", table_name="accounts")
    op.drop_index("ix_accounts_email", table_name="accounts")
    op.drop_table("accounts")
