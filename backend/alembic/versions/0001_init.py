"""initial schema: documents, briefs, brief_sub_queries

Revision ID: 0001
Revises:
Create Date: 2026-06-08

Vectors/chunks live in ChromaDB, so there is no document_chunks table and no
pgvector extension here — only relational tables.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column(
            "source_type",
            sa.Enum("pdf", "web", "docx", name="source_type"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "briefs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "complete", "failed", name="brief_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("faithfulness_scores", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "brief_sub_queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "brief_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("briefs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sub_query", sa.Text(), nullable=False),
        sa.Column("hyde_document", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_brief_sub_queries_brief_id", "brief_sub_queries", ["brief_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_brief_sub_queries_brief_id", table_name="brief_sub_queries")
    op.drop_table("brief_sub_queries")
    op.drop_table("briefs")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS brief_status")
    op.execute("DROP TYPE IF EXISTS source_type")
