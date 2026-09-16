"""persist Profile v1 provenance and Premium run lifecycle data"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_premium_foundation"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _canonical_restrictions(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    order = (
        "nenhuma",
        "priorizar_renda",
        "evitar_cripto",
        "evitar_exterior",
        "limitar_concentracao",
        "evitar_illiquidez",
    )
    values = {item for item in value if isinstance(item, str)}
    return [item for item in order if item in values]


def upgrade() -> None:
    op.add_column(
        "profiles",
        sa.Column("raw_score", sa.Numeric(precision=6, scale=5), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "profiles",
        sa.Column("rules_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column(
        "profiles",
        sa.Column("warnings_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column(
        "profiles",
        sa.Column("restrictions_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.add_column(
        "profiles",
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )

    op.add_column(
        "recommendation_runs",
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'completed'")),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("failure_code", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("failure_message", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("policy_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("policy_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("result_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("provenance_json", sa.JSON(), nullable=True),
    )
    op.add_column(
        "recommendation_runs",
        sa.Column("output_hash", sa.String(length=128), nullable=True),
    )

    bind = op.get_bind()
    profiles = sa.table(
        "profiles",
        sa.column("id", sa.String(length=36)),
        sa.column("suitability_score", sa.Numeric(precision=6, scale=5)),
        sa.column("answers", sa.JSON()),
        sa.column("raw_score", sa.Numeric(precision=6, scale=5)),
        sa.column("restrictions_json", sa.JSON()),
    )
    rows = bind.execute(sa.select(profiles.c.id, profiles.c.suitability_score, profiles.c.answers))
    for row in rows:
        answers = row.answers if isinstance(row.answers, dict) else {}
        bind.execute(
            profiles.update()
            .where(profiles.c.id == row.id)
            .values(
                raw_score=row.suitability_score,
                restrictions_json=_canonical_restrictions(answers.get("restricoes")),
            )
        )

    op.execute(
        sa.text(
            "UPDATE recommendation_runs "
            "SET status = 'completed', completed_at = created_at, policy_version = 'basic-v1' "
            "WHERE plan = 'basic'"
        )
    )


def downgrade() -> None:
    op.drop_column("recommendation_runs", "output_hash")
    op.drop_column("recommendation_runs", "provenance_json")
    op.drop_column("recommendation_runs", "result_json")
    op.drop_column("recommendation_runs", "policy_json")
    op.drop_column("recommendation_runs", "policy_version")
    op.drop_column("recommendation_runs", "failure_message")
    op.drop_column("recommendation_runs", "failure_code")
    op.drop_column("recommendation_runs", "completed_at")
    op.drop_column("recommendation_runs", "started_at")
    op.drop_column("recommendation_runs", "status")
    op.drop_column("profiles", "schema_version")
    op.drop_column("profiles", "restrictions_json")
    op.drop_column("profiles", "warnings_json")
    op.drop_column("profiles", "rules_json")
    op.drop_column("profiles", "raw_score")
