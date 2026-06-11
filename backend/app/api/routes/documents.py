"""Document endpoints: upload (stage + enqueue async ingestion), list, delete.

Ingestion (load -> chunk -> embed -> Chroma -> BM25) runs in a Celery worker
(P1.6/P1.3/P1.4), so the API process stays light and an upload returns as soon as
the file is validated and staged. The document's ``status`` reflects ingestion
progress (pending -> processing -> ready/failed).
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import request_id_var
from app.models import Document, DocumentStatus, SourceType
from app.rag.retrieval.bm25_index import rebuild_bm25_locked
from app.rag.retrieval.vector_store import VectorStore
from app.tasks import ingest_document_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_SUFFIX_TO_TYPE = {".pdf": SourceType.pdf, ".docx": SourceType.docx}

# Leading magic bytes per accepted type — the extension alone is not trusted
# (P0.4). DOCX is an OOXML zip, so it starts with the local-file zip signature.
_MAGIC_BYTES = {
    SourceType.pdf: b"%PDF",
    SourceType.docx: b"PK\x03\x04",
}

_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB read granularity while streaming to disk


def _upload_dir() -> Path:
    """Directory where uploads are staged for the ingestion worker (P1.6).

    Must be shared between the API and worker (a named volume in compose). Falls
    back to a per-host temp dir when unset (fine when both run on one host)."""
    base = Path(settings.upload_dir) if settings.upload_dir else Path(tempfile.gettempdir()) / "briefr-uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


async def _persist_upload(file: UploadFile, dest: Path, source_type: SourceType) -> None:
    """Stream the upload to ``dest``, enforcing size + content-type (P0.4).

    Reads in bounded chunks so an oversized body is rejected mid-stream instead
    of being buffered whole in memory, and verifies the leading magic bytes so a
    mislabelled extension is caught before any parser touches the file.
    """
    max_bytes = settings.max_upload_bytes
    magic = _MAGIC_BYTES[source_type]
    total = 0
    head = b""
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK):
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds maximum size of {max_bytes} bytes",
                    )
                if len(head) < len(magic):
                    head += chunk[: len(magic) - len(head)]
                out.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty file")
        if not head.startswith(magic):
            raise HTTPException(
                status_code=400,
                detail=f"file content does not match a valid {source_type.value.upper()}",
            )
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


@router.post("/upload", status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    source_type = _SUFFIX_TO_TYPE.get(suffix)
    if source_type is None:
        raise HTTPException(status_code=400, detail=f"unsupported file type '{suffix}' (PDF/DOCX only)")

    # Cheap fast-path reject using the declared body size before reading anything.
    declared = request.headers.get("content-length")
    if declared and settings.max_upload_bytes and int(declared) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds maximum size of {settings.max_upload_bytes} bytes",
        )

    doc_id = uuid.uuid4()
    dest = _upload_dir() / f"{doc_id}{suffix}"
    await _persist_upload(file, dest, source_type)

    document = Document(id=doc_id, name=filename, source_type=source_type, status=DocumentStatus.pending)
    db.add(document)
    await db.commit()

    ingest_document_task.delay(
        str(doc_id), str(dest), source_type.value, request_id=request_id_var.get()
    )
    logger.info("upload_document: %s staged -> %s (enqueued ingestion)", filename, doc_id)
    return {
        "document_id": str(doc_id),
        "name": filename,
        "source_type": source_type.value,
        "status": DocumentStatus.pending.value,
    }


@router.get("")
async def list_documents(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return [
        {
            "document_id": str(d.id),
            "name": d.name,
            "source_type": d.source_type.value,
            "status": d.status.value,
            "num_chunks": d.num_chunks,
            "error": d.error,
            "created_at": d.created_at.isoformat(),
        }
        for d in rows
    ]


@router.delete(
    "/{document_id}", status_code=204, response_class=Response, response_model=None
)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    await VectorStore().delete_document_chunks(str(document_id))
    await rebuild_bm25_locked()
    await db.delete(document)
    await db.commit()
    # Remove any staged upload file that ingestion didn't clean up (pending/failed docs).
    for stale in _upload_dir().glob(f"{document_id}.*"):
        stale.unlink(missing_ok=True)
    logger.info("delete_document: %s removed", document_id)
