"""Document ORM model.

A ``Document`` is the relational record for an ingested source (PDF, web page, or
Word doc). Its chunks/embeddings live in ChromaDB (see ``app.core.vector_db``),
keyed by this document's id in chunk metadata — there is deliberately no
``document_chunks`` table here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum as SAEnum, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SourceType(str, enum.Enum):
    """Origin of an ingested document (member name == value, so the DB enum
    stores the lowercase string)."""

    pdf = "pdf"
    web = "web"
    docx = "docx"


class DocumentStatus(str, enum.Enum):
    """Ingestion lifecycle. Upload returns immediately as ``pending``; the Celery
    ingestion task moves it ``processing`` -> ``ready`` (or ``failed``)."""

    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(SourceType, name="source_type"), nullable=False
    )
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, name="document_status"),
        nullable=False,
        default=DocumentStatus.pending,
        server_default=DocumentStatus.ready.value,  # pre-existing rows are already ingested
    )
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full document text is NOT stored here (P2.3) — chunks/text live in Chroma.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
