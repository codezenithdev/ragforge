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
from datetime import datetime, timezone

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


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
    from app.core.database import AsyncSessionLocal
    from app.models import Brief, BriefStatus, BriefSubQuery
    from app.rag.pipeline.graph import briefr_graph

    brief_id = uuid.UUID(brief_id_str)

    async with AsyncSessionLocal() as session:
        brief = await session.get(Brief, brief_id)
        if brief is None:
            raise ValueError(f"brief {brief_id} not found")
        query = brief.query
        brief.status = BriefStatus.processing
        await session.commit()

    try:
        state = await briefr_graph.ainvoke(
            {"query": query, "document_ids": document_ids or None}
        )
        result = state["brief"].model_dump(mode="json")
        # Retain the generation contexts so /eval can run RAGAS later.
        result["contexts"] = [c.content for c in state.get("final_chunks", [])]
        result["crag_action"] = state["crag_result"].action.value

        async with AsyncSessionLocal() as session:
            brief = await session.get(Brief, brief_id)
            brief.result = result
            brief.faithfulness_scores = state.get("faithfulness_scores") or {}
            brief.status = BriefStatus.complete
            brief.completed_at = datetime.now(timezone.utc)
            for sub_query, hyde in zip(
                state.get("sub_queries", []), state.get("hyde_documents", [])
            ):
                session.add(
                    BriefSubQuery(brief_id=brief_id, sub_query=sub_query, hyde_document=hyde)
                )
            await session.commit()
        logger.info("generate_brief_task: brief %s complete", brief_id)
    except Exception as exc:
        logger.exception("generate_brief_task: brief %s failed", brief_id)
        await _set_status(
            brief_id, status=BriefStatus.failed, result={"error": str(exc)}
        )


@celery_app.task(name="app.tasks.generate_brief_task")
def generate_brief_task(brief_id: str, document_ids: list[str] | None = None) -> None:
    _get_loop().run_until_complete(_run_pipeline(brief_id, document_ids))
