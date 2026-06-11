"""created_at indexes for list ordering (C5 / P2.5)

The list endpoints ORDER BY created_at DESC; index it on both tables so the
ordering/pagination queries don't scan.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-10
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_index("ix_briefs_created_at", "briefs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_briefs_created_at", table_name="briefs")
    op.drop_index("ix_documents_created_at", table_name="documents")
