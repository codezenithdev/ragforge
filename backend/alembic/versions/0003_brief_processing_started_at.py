"""brief processing_started_at (B4 / P1.2)

Adds ``briefs.processing_started_at`` so the stuck-brief sweeper can fail briefs
left in 'processing' past a deadline (worker crash / lost task).

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "briefs",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("briefs", "processing_started_at")
