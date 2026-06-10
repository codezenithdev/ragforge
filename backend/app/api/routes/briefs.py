"""Brief endpoints: create (enqueue async pipeline), get status/result, list."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import Brief
from app.tasks import generate_brief_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefs", tags=["briefs"])


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
