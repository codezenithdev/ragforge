"""Document endpoints: upload (ingest -> chunk -> embed -> Chroma), list, delete."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models import Document, SourceType
from app.rag.ingestion.chunker import SemanticChunker
from app.rag.ingestion.embedder import Embedder
from app.rag.ingestion.loaders import DocxLoader, PDFLoader
from app.rag.retrieval.bm25_index import BM25Index
from app.rag.retrieval.vector_store import VectorStore

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


async def _persist_upload(file: UploadFile, suffix: str, source_type: SourceType) -> Path:
    """Stream the upload to a temp file, enforcing size + content-type (P0.4).

    Reads in bounded chunks so an oversized body is rejected mid-stream instead
    of being buffered whole in memory, and verifies the leading magic bytes so a
    mislabelled extension is caught before any parser touches the file.
    """
    max_bytes = settings.max_upload_bytes
    magic = _MAGIC_BYTES[source_type]
    total = 0
    head = b""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(_UPLOAD_CHUNK):
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds maximum size of {max_bytes} bytes",
                    )
                if len(head) < len(magic):
                    head += chunk[: len(magic) - len(head)]
                tmp.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty file")
        if not head.startswith(magic):
            raise HTTPException(
                status_code=400,
                detail=f"file content does not match a valid {source_type.value.upper()}",
            )
        return tmp_path
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


@lru_cache
def _chunker() -> SemanticChunker:
    return SemanticChunker()


@lru_cache
def _embedder() -> Embedder:
    return Embedder()


async def _rebuild_bm25(vector_store: VectorStore) -> None:
    ids, docs, metas = await vector_store.fetch_corpus()
    await BM25Index().build_index(ids, docs, metas)


@router.post("/upload", status_code=201)
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

    # Cheap fast-path reject using the declared body size before reading anything
    # (the streaming loop in _persist_upload is the authoritative cap).
    declared = request.headers.get("content-length")
    if declared and settings.max_upload_bytes and int(declared) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds maximum size of {settings.max_upload_bytes} bytes",
        )

    tmp_path = await _persist_upload(file, suffix, source_type)
    try:
        loader = PDFLoader(tmp_path) if source_type is SourceType.pdf else DocxLoader(tmp_path)
        try:
            blocks = await loader.load()
        except ValueError as exc:
            # Raised by the loaders for caps/limits (e.g. PDF page count).
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        if not blocks:
            raise HTTPException(status_code=400, detail="no extractable text in document")

        chunks = await asyncio.to_thread(_chunker().chunk_blocks, blocks)
        embeddings = await _embedder().embed_batch([c["content"] for c in chunks])

        document = Document(
            name=filename,
            source_type=source_type,
            content="\n\n".join(b["content"] for b in blocks),
        )
        db.add(document)
        await db.flush()
        doc_id = str(document.id)

        vector_store = VectorStore()
        await vector_store.upsert_chunks(
            [
                {
                    "id": f"{doc_id}::{c['metadata']['chunk_index']}",
                    "content": c["content"],
                    "embedding": embeddings[i],
                    "metadata": {**c["metadata"], "document_id": doc_id, "source": filename},
                }
                for i, c in enumerate(chunks)
            ]
        )
        await _rebuild_bm25(vector_store)
        await db.commit()
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("upload_document: %s -> %s (%d chunks)", filename, doc_id, len(chunks))
    return {"document_id": doc_id, "name": filename, "source_type": source_type.value, "num_chunks": len(chunks)}


@router.get("")
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (await db.execute(select(Document).order_by(Document.created_at.desc()))).scalars().all()
    return [
        {
            "document_id": str(d.id),
            "name": d.name,
            "source_type": d.source_type.value,
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
    vector_store = VectorStore()
    await vector_store.delete_document_chunks(str(document_id))
    await _rebuild_bm25(vector_store)
    await db.delete(document)
    await db.commit()
    logger.info("delete_document: %s removed", document_id)
