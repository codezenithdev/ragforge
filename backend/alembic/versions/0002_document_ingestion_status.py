"""document ingestion status (B3 / P1.6)

Adds the ingestion lifecycle columns used by the async Celery ingestion task:
``status`` (pending/processing/ready/failed), ``num_chunks``, ``error``.
Existing rows default to ``ready`` (they were ingested synchronously before).

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DOCUMENT_STATUS = sa.Enum(
    "pending", "processing", "ready", "failed", name="document_status"
)


def upgrade() -> None:
    bind = op.get_bind()
    _DOCUMENT_STATUS.create(bind, checkfirst=True)
    op.add_column(
        "documents",
        sa.Column(
            "status",
            _DOCUMENT_STATUS,
            nullable=False,
            server_default="ready",
        ),
    )
    op.add_column(
        "documents",
        sa.Column("num_chunks", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("documents", sa.Column("error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "error")
    op.drop_column("documents", "num_chunks")
    op.drop_column("documents", "status")
    _DOCUMENT_STATUS.drop(op.get_bind(), checkfirst=True)
