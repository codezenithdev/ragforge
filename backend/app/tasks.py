"""Celery tasks.

``generate_brief_task`` runs the full LangGraph pipeline for a brief and
persists the structured result + faithfulness scores back to Postgres.

Event-loop handling: the pipeline's cached async clients (Chroma HTTP client,
Anthropic client, asyncpg pool) are bound to the loop they were created on, so
each worker process keeps ONE persistent event loop and runs every task on it —
``asyncio.run`` per task would strand those clients on closed loops.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


# Heavy ingestion models live in the WORKER, not the API (P1.6). Imports are lazy
# so importing app.tasks (which the API does) never pulls in torch/sentence-transformers.
@lru_cache
def _chunker():
    from app.rag.ingestion.chunker import SemanticChunker

    return SemanticChunker()


@lru_cache
def _embedder():
    from app.rag.ingestion.embedder import Embedder

    return Embedder()


async def _set_status(brief_id: uuid.UUID, **fields: object) -> None:
    from app.core.database import AsyncSessionLocal
    from app.models import Brief

    async with AsyncSessionLocal() as session:
        brief = await session.get(Brief, brief_id)
        if brief is None:
            raise ValueError(f"brief {brief_id} not found")
        for key, value in fields.items():
            setattr(brief, key, value)
        await session.commit()


async def _run_pipeline(brief_id_str: str, document_ids: list[str] | None) -> None:
    from app.core.anthropic_client import UsageTotals, usage_var
    from app.core.database import AsyncSessionLocal
    from app.models import Brief, BriefStatus, BriefSubQuery
    from app.rag.pipeline.graph import briefr_graph

    brief_id = uuid.UUID(brief_id_str)

    async with AsyncSessionLocal() as session:
        brief = await session.get(Brief, brief_id)
        if brief is None:
            raise ValueError(f"brief {brief_id} not found")
        # Idempotency (P1.2): a re-delivered task (acks_late) for an already-finished
        # brief must not re-run the (billable) pipeline.
        if brief.status in (BriefStatus.complete, BriefStatus.failed):
            logger.info("generate_brief_task: brief %s already %s; skipping", brief_id, brief.status.value)
            return
        query = brief.query
        brief.status = BriefStatus.processing
        brief.processing_started_at = datetime.now(UTC)
        await session.commit()

    # Per-brief token/cost accounting. The mutable accumulator is shared across
    # the graph's concurrent nodes via the contextvar (set before ainvoke).
    usage = UsageTotals()
    usage_var.set(usage)

    try:
        # Hard timeout inside the loop (P1.2) — SIGALRM soft limits are unreliable
        # on Windows + run_until_complete, so bound the pipeline here instead.
        state = await asyncio.wait_for(
            briefr_graph.ainvoke({"query": query, "document_ids": document_ids or None}),
            timeout=settings.brief_timeout_seconds,
        )
        result = state["brief"].model_dump(mode="json")
        # Store context *references* for /eval rather than duplicating full text
        # (P2.3): document chunks live in Chroma (store id, rehydrate later); web
        # chunks aren't in Chroma, so their text is kept inline (it's the only copy).
        refs: list[dict[str, str]] = []
        for c in state.get("final_chunks", []):
            if c.metadata.get("source") == "web" or c.chunk_id.startswith("web::"):
                refs.append({"id": c.chunk_id, "text": c.content})
            else:
                refs.append({"id": c.chunk_id})
        result["context_refs"] = refs
        result["crag_action"] = state["crag_result"].action.value

        async with AsyncSessionLocal() as session:
            brief = await session.get(Brief, brief_id)
            brief.result = result
            brief.faithfulness_scores = state.get("faithfulness_scores") or {}
            brief.status = BriefStatus.complete
            brief.completed_at = datetime.now(UTC)
            for sub_query, hyde in zip(
                state.get("sub_queries", []), state.get("hyde_documents", []), strict=False
            ):
                session.add(
                    BriefSubQuery(brief_id=brief_id, sub_query=sub_query, hyde_document=hyde)
                )
            await session.commit()
        logger.info("generate_brief_task: brief %s complete", brief_id)
    except TimeoutError:
        logger.error("generate_brief_task: brief %s timed out after %ds", brief_id, settings.brief_timeout_seconds)
        await _set_status(
            brief_id,
            status=BriefStatus.failed,
            result={"error": f"generation timed out after {settings.brief_timeout_seconds}s"},
        )
    except Exception as exc:
        logger.exception("generate_brief_task: brief %s failed", brief_id)
        await _set_status(
            brief_id, status=BriefStatus.failed, result={"error": str(exc)}
        )
    finally:
        # One structured usage line per brief (emitted even on timeout/failure so
        # partial spend is visible), correlated by request id via the JSON logger.
        total_tokens = usage.input_tokens + usage.output_tokens
        logger.info(
            "generate_brief_task: brief %s usage",
            brief_id,
            extra={
                "event": "brief_usage",
                "brief_id": str(brief_id),
                "tokens_in": usage.input_tokens,
                "tokens_out": usage.output_tokens,
                "cache_read": usage.cache_read_input_tokens,
                "cache_write": usage.cache_creation_input_tokens,
                "est_cost_usd": usage.estimated_cost_usd(),
                "by_model": usage.by_model,
            },
        )
        if (
            settings.brief_token_warn_threshold
            and total_tokens > settings.brief_token_warn_threshold
        ):
            logger.warning(
                "generate_brief_task: brief %s used %d tokens (> warn threshold %d)",
                brief_id,
                total_tokens,
                settings.brief_token_warn_threshold,
            )


@celery_app.task(
    name="app.tasks.generate_brief_task",
    # Pipeline-level failures are caught in _run_pipeline (the brief is marked
    # failed). Only infra errors that escape it (e.g. DB unreachable during the
    # acquire/idempotency phase) propagate here and get a bounded, backed-off retry.
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=settings.brief_max_retries,
)
def generate_brief_task(
    brief_id: str, document_ids: list[str] | None = None, request_id: str = "-"
) -> None:
    request_id_var.set(request_id)  # correlate worker logs with the originating request
    _get_loop().run_until_complete(_run_pipeline(brief_id, document_ids))


# --------------------------------------------------------------------------- #
# Document ingestion (P1.6/P1.3/P1.4) — moved off the API request path.
# --------------------------------------------------------------------------- #
async def _set_doc_status(doc_id: uuid.UUID, **fields: object) -> None:
    from app.core.database import AsyncSessionLocal
    from app.models import Document

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return
        for key, value in fields.items():
            setattr(doc, key, value)
        await session.commit()


async def _run_ingest(doc_id_str: str, file_path_str: str, source_type_value: str) -> None:
    from pathlib import Path

    from app.core.database import AsyncSessionLocal
    from app.models import Document, DocumentStatus, SourceType
    from app.rag.ingestion.loaders import DocxLoader, PDFLoader
    from app.rag.retrieval.bm25_index import rebuild_bm25_locked
    from app.rag.retrieval.vector_store import VectorStore

    doc_id = uuid.UUID(doc_id_str)
    source_type = SourceType(source_type_value)
    file_path = Path(file_path_str)
    vector_store = VectorStore()

    try:
        await _set_doc_status(doc_id, status=DocumentStatus.processing)

        async def _ingest_core() -> int:
            loader = (
                PDFLoader(file_path) if source_type is SourceType.pdf else DocxLoader(file_path)
            )
            blocks = await loader.load()
            if not blocks:
                raise ValueError("no extractable text in document")
            chunks = await asyncio.to_thread(_chunker().chunk_blocks, blocks)
            embeddings = await _embedder().embed_batch([c["content"] for c in chunks])

            await vector_store.upsert_chunks(
                [
                    {
                        "id": f"{doc_id_str}::{c['metadata']['chunk_index']}",
                        "content": c["content"],
                        "embedding": embeddings[i],
                        "metadata": {**c["metadata"], "document_id": doc_id_str},
                    }
                    for i, c in enumerate(chunks)
                ]
            )
            await rebuild_bm25_locked()

            async with AsyncSessionLocal() as session:
                doc = await session.get(Document, doc_id)
                if doc is None:
                    # Row deleted mid-ingest — drop the vectors we just wrote.
                    await vector_store.delete_document_chunks(doc_id_str)
                    await rebuild_bm25_locked()
                    return 0
                # Full text is NOT stored in PG (P2.3) — chunks live in Chroma.
                doc.num_chunks = len(chunks)
                doc.status = DocumentStatus.ready
                doc.error = None
                await session.commit()
            return len(chunks)

        n = await asyncio.wait_for(_ingest_core(), timeout=settings.ingest_timeout_seconds)
        logger.info("ingest_document_task: %s ready (%d chunks)", doc_id, n)
    except Exception as exc:  # noqa: BLE001 -- record + compensate, never crash the worker
        logger.exception("ingest_document_task: %s failed", doc_id)
        # Compensate: remove any partial vectors so PG (the source of truth) and
        # Chroma stay reconcilable (P1.4), then flag the row failed.
        try:
            await vector_store.delete_document_chunks(doc_id_str)
            await rebuild_bm25_locked()
        except Exception:  # noqa: BLE001
            logger.exception("ingest_document_task: cleanup failed for %s", doc_id)
        from app.models import DocumentStatus

        await _set_doc_status(doc_id, status=DocumentStatus.failed, error=str(exc)[:1000])
    finally:
        file_path.unlink(missing_ok=True)


@celery_app.task(name="app.tasks.ingest_document_task")
def ingest_document_task(
    doc_id: str, file_path: str, source_type: str, request_id: str = "-"
) -> None:
    request_id_var.set(request_id)
    _get_loop().run_until_complete(_run_ingest(doc_id, file_path, source_type))


async def _run_reconcile() -> None:
    """Purge orphaned Chroma vectors and fail documents stuck mid-ingest (P1.4)."""
    from sqlalchemy import select

    from app.core.database import AsyncSessionLocal
    from app.models import Document, DocumentStatus
    from app.rag.retrieval.bm25_index import rebuild_bm25_locked
    from app.rag.retrieval.vector_store import VectorStore

    vector_store = VectorStore()
    _, _, metas = await vector_store.fetch_corpus()
    chroma_doc_ids = {m.get("document_id") for m in metas if m.get("document_id")}

    deadline = datetime.now(UTC) - timedelta(seconds=settings.ingest_timeout_seconds * 2)
    async with AsyncSessionLocal() as session:
        pg_ids = {
            str(r) for r in (await session.execute(select(Document.id))).scalars().all()
        }
        stuck = (
            await session.execute(
                select(Document).where(
                    Document.status.in_((DocumentStatus.pending, DocumentStatus.processing)),
                    Document.created_at < deadline,
                )
            )
        ).scalars().all()
        for doc in stuck:
            doc.status = DocumentStatus.failed
            doc.error = "ingestion did not complete (worker lost or timed out)"
        await session.commit()

    orphans = [did for did in chroma_doc_ids if did not in pg_ids]
    for did in orphans:
        await vector_store.delete_document_chunks(did)
    if orphans:
        await rebuild_bm25_locked()
    logger.info(
        "reconcile_storage_task: %d orphan doc(s) purged, %d stuck doc(s) failed",
        len(orphans),
        len(stuck),
    )


@celery_app.task(name="app.tasks.reconcile_storage_task")
def reconcile_storage_task() -> None:
    _get_loop().run_until_complete(_run_reconcile())


async def _run_prune_briefs() -> None:
    """Delete briefs older than the retention window (P2.3). No-op when disabled."""
    if settings.brief_retention_days <= 0:
        return
    from sqlalchemy import delete

    from app.core.database import AsyncSessionLocal
    from app.models import Brief

    cutoff = datetime.now(UTC) - timedelta(days=settings.brief_retention_days)
    async with AsyncSessionLocal() as session:
        result = await session.execute(delete(Brief).where(Brief.created_at < cutoff))
        await session.commit()
    logger.info(
        "prune_old_briefs_task: deleted %s brief(s) older than %d days",
        result.rowcount,
        settings.brief_retention_days,
    )


@celery_app.task(name="app.tasks.prune_old_briefs_task")
def prune_old_briefs_task() -> None:
    _get_loop().run_until_complete(_run_prune_briefs())


async def _run_sweep_stuck_briefs() -> None:
    """Fail briefs stranded mid-generation so the frontend stops polling (P1.2)."""
    from sqlalchemy import or_, select

    from app.core.database import AsyncSessionLocal
    from app.models import Brief, BriefStatus

    now = datetime.now(UTC)
    processing_deadline = now - timedelta(seconds=settings.brief_stuck_seconds)
    pending_deadline = now - timedelta(seconds=settings.brief_stuck_seconds)

    async with AsyncSessionLocal() as session:
        stuck = (
            await session.execute(
                select(Brief).where(
                    or_(
                        # Picked up but never finished (worker crash / lost task).
                        (Brief.status == BriefStatus.processing)
                        & (Brief.processing_started_at < processing_deadline),
                        # Enqueued but never picked up.
                        (Brief.status == BriefStatus.pending)
                        & (Brief.created_at < pending_deadline),
                    )
                )
            )
        ).scalars().all()
        for brief in stuck:
            brief.status = BriefStatus.failed
            brief.result = {"error": "generation stalled (worker lost); failed by sweeper"}
            brief.completed_at = now
        await session.commit()
    if stuck:
        logger.warning("sweep_stuck_briefs_task: failed %d stuck brief(s)", len(stuck))


@celery_app.task(name="app.tasks.sweep_stuck_briefs_task")
def sweep_stuck_briefs_task() -> None:
    _get_loop().run_until_complete(_run_sweep_stuck_briefs())
