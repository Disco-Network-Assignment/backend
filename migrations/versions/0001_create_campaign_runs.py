"""create campaign_runs

Revision ID: 0001
Revises:
Create Date: 2026-09-19
"""

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Databases that ran the app before migrations existed already have this table (the app used
    # to create it at startup). Adopt it as-is instead of failing on "already exists".
    # (`--sql` mode has no connection to inspect, so it always prints the CREATE TABLE.)
    if not context.is_offline_mode() and sa.inspect(op.get_bind()).has_table("campaign_runs"):
        return

    op.create_table(
        "campaign_runs",
        sa.Column("run_id", sa.String(32), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_id", sa.String(64)),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),     # done | stopped | failed
        sa.Column("input_quality", sa.String(16)),
        sa.Column("recommended", sa.Integer, nullable=False),
        sa.Column("personas", sa.Integer, nullable=False),
        sa.Column("creatives", sa.Integer, nullable=False),
        sa.Column("budget_usd", sa.Float, nullable=False),
        sa.Column("plan", postgresql.JSONB),
        sa.Column("stopped", postgresql.JSONB),
        sa.Column("error", postgresql.JSONB),
    )


def downgrade() -> None:
    op.drop_table("campaign_runs")
