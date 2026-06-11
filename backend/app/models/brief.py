"""Brief ORM models.

A ``Brief`` is one research-brief job: it starts ``pending``, is processed
asynchronously by the LangGraph pipeline (run via a Celery worker), and ends
``complete`` (with the structured ``result`` + per-section ``faithfulness_scores``)
or ``failed``. ``BriefSubQuery`` records the decomposed sub-questions and their
HyDE documents for traceability.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BriefStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class Brief(Base):
    __tablename__ = "briefs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[BriefStatus] = mapped_column(
        SAEnum(BriefStatus, name="brief_status"),
        nullable=False,
        default=BriefStatus.pending,
        server_default=BriefStatus.pending.value,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    faithfulness_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    # When the worker began processing — used by the stuck-brief sweeper (P1.2).
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class BriefSubQuery(Base):
    __tablename__ = "brief_sub_queries"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    brief_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("briefs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sub_query: Mapped[str] = mapped_column(Text, nullable=False)
    hyde_document: Mapped[str | None] = mapped_column(Text, nullable=True)
