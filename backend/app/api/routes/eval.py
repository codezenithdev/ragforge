"""Evaluation endpoints: run RAGAS on a completed brief, fetch stored scores.

RAGAS scores are stored on the brief's ``result`` JSONB under ``ragas_eval``
(the generation contexts needed for evaluation are persisted there by the
pipeline task).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import Brief, BriefStatus
from app.rag.evaluation.ragas_runner import RAGASRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRunRequest(BaseModel):
    brief_id: str
    ground_truth: str | None = None


async def _rehydrate_contexts(result: dict[str, Any]) -> list[str]:
    """Reconstruct the generation contexts for evaluation (P2.3).

    New briefs store ``context_refs`` (document-chunk ids + inline web text);
    document text is fetched from Chroma by id. Falls back to the legacy
    ``contexts`` field (full text) for briefs created before the redesign.
    """
    refs = result.get("context_refs")
    if not refs:
        return result.get("contexts") or []

    from app.rag.retrieval.vector_store import VectorStore

    need_ids = [r["id"] for r in refs if "text" not in r]
    texts_by_id = await VectorStore().fetch_texts_by_ids(need_ids) if need_ids else {}
    contexts: list[str] = []
    for ref in refs:
        text = ref.get("text") if "text" in ref else texts_by_id.get(ref["id"], "")
        if text:
            contexts.append(text)
    return contexts


@router.post("/run")
async def run_eval(request: EvalRunRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    brief = await db.get(Brief, uuid.UUID(request.brief_id))
    if brief is None:
        raise HTTPException(status_code=404, detail="brief not found")
    if brief.status is not BriefStatus.complete or not brief.result:
        raise HTTPException(status_code=409, detail=f"brief is '{brief.status.value}', not complete")

    answer = (brief.result.get("executive_summary") or {}).get("content", "")
    contexts = await _rehydrate_contexts(brief.result)
    if not answer or not contexts:
        raise HTTPException(status_code=409, detail="brief result lacks answer/contexts for evaluation")

    ragas = await RAGASRunner().evaluate(
        question=brief.query,
        answer=answer,
        contexts=contexts,
        ground_truth=request.ground_truth,
    )
    eval_payload = {
        "faithfulness": ragas.faithfulness,
        "answer_relevancy": ragas.answer_relevancy,
        "context_precision": ragas.context_precision,
        "context_recall": ragas.context_recall,
        "overall": ragas.overall,
        "raw": ragas.raw,
    }
    # Reassign (not mutate) so SQLAlchemy detects the JSONB change.
    brief.result = {**brief.result, "ragas_eval": eval_payload}
    await db.commit()
    logger.info("run_eval: brief %s overall=%.3f", brief.id, ragas.overall)
    return {"brief_id": str(brief.id), **eval_payload}


@router.get("/{brief_id}")
async def get_eval(brief_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    brief = await db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="brief not found")
    eval_payload = (brief.result or {}).get("ragas_eval")
    if not eval_payload:
        raise HTTPException(status_code=404, detail="brief has not been evaluated yet")
    return {"brief_id": str(brief.id), **eval_payload}
