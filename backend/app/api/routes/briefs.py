"""Brief endpoints: create (enqueue async pipeline), get status/result, list."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models import Brief, BriefStatus
from app.tasks import generate_brief_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefs", tags=["briefs"])

_ACTIVE_STATUSES = (BriefStatus.pending, BriefStatus.processing)


async def _enforce_brief_quota(db: AsyncSession) -> None:
    """Reject brief creation that would breach the spend controls (P0.2).

    Two caps, both evaluated against Postgres (the source of truth for brief
    state): a global in-flight ceiling and a per-UTC-day creation ceiling. Each
    brief is ~15 LLM calls, so these bound runaway cost. A cap of 0 disables it.
    """
    if settings.max_concurrent_briefs:
        in_flight = await db.scalar(
            select(func.count())
            .select_from(Brief)
            .where(Brief.status.in_(_ACTIVE_STATUSES))
        )
        if (in_flight or 0) >= settings.max_concurrent_briefs:
            raise HTTPException(
                status_code=429,
                detail=f"too many briefs in flight (max {settings.max_concurrent_briefs}); retry shortly",
            )

    if settings.daily_brief_limit:
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today = await db.scalar(
            select(func.count()).select_from(Brief).where(Brief.created_at >= day_start)
        )
        if (today or 0) >= settings.daily_brief_limit:
            raise HTTPException(
                status_code=429,
                detail=f"daily brief limit reached (max {settings.daily_brief_limit})",
            )


class BriefCreateRequest(BaseModel):
    query: str = Field(..., min_length=3)
    document_ids: list[str] = Field(default_factory=list)


def _serialize(brief: Brief) -> dict[str, Any]:
    return {
        "brief_id": str(brief.id),
        "query": brief.query,
        "status": brief.status.value,
        "result": brief.result,
        "faithfulness_scores": brief.faithfulness_scores,
        "created_at": brief.created_at.isoformat(),
        "completed_at": brief.completed_at.isoformat() if brief.completed_at else None,
    }


@router.post("", status_code=202)
async def create_brief(
    request: BriefCreateRequest, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    # Per-identity request rate limiting is applied globally by SlowAPIMiddleware;
    # these caps bound brief-specific LLM spend (P0.2).
    await _enforce_brief_quota(db)

    brief = Brief(query=request.query)
    db.add(brief)
    await db.commit()

    generate_brief_task.delay(str(brief.id), request.document_ids or None)
    logger.info("create_brief: %s enqueued (%d scoped docs)", brief.id, len(request.document_ids))
    return {"brief_id": str(brief.id), "status": brief.status.value}


@router.get("/{brief_id}")
async def get_brief(brief_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    brief = await db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="brief not found")
    return _serialize(brief)


@router.get("")
async def list_briefs(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (
        (await db.execute(select(Brief).order_by(Brief.created_at.desc()).limit(20)))
        .scalars()
        .all()
    )
    return [
        {
            "brief_id": str(b.id),
            "query": b.query,
            "status": b.status.value,
            "created_at": b.created_at.isoformat(),
            "completed_at": b.completed_at.isoformat() if b.completed_at else None,
        }
        for b in rows
    ]
