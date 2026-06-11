"""drop documents.content (C3 / P2.3)

Full document text is no longer stored in Postgres — chunk text lives in Chroma,
so the column was a duplicate. Briefs store context *references* (chunk ids),
not full text.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("documents", "content")


def downgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
    )
