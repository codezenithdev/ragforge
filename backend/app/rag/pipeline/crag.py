"""Corrective RAG (CRAG) evaluation.

Before generating, judge whether retrieval was good enough. We score the
retrieved chunks with the cross-encoder; if even the best chunk is not
confidently relevant — its relevance probability (sigmoid of the cross-encoder
logit) falls below a threshold — we return ``CORRECTIVE_SEARCH`` with a few
focused web-search queries so the pipeline can pull in fresh external context.
Otherwise we ``PROCEED``.
"""

from __future__ import annotations

import asyncio
import logging
import math

import anthropic
import instructor
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.types import CRAGAction, CRAGResult, ScoredChunk

logger = logging.getLogger(__name__)

_WEBQ_SYSTEM = (
    "You generate focused web-search queries to fill gaps in retrieved context."
)


class _WebQueries(BaseModel):
    queries: list[str] = Field(..., description="3 focused web-search queries")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CRAGEvaluator:
    def __init__(
        self,
        reranker: CrossEncoderReranker,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.reranker = reranker
        client = anthropic_client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._instructor = instructor.from_anthropic(client)
        self.model = settings.subtask_model
        self.threshold = settings.crag_confidence_threshold

    async def evaluate_chunks(
        self, query: str, chunks: list[ScoredChunk]
    ) -> CRAGResult:
        if not chunks:
            logger.info("CRAG: no chunks retrieved -> corrective search")
            return CRAGResult(
                action=CRAGAction.CORRECTIVE_SEARCH,
                top_score=0.0,
                suggested_web_queries=await self._web_queries(query),
            )

        # Cross-encoder score the candidates; the top score gauges confidence.
        scored = await asyncio.to_thread(self.reranker.rerank, query, list(chunks), len(chunks))
        top_prob = _sigmoid(scored[0].rerank_score or 0.0)

        if top_prob < self.threshold:
            logger.info(
                "CRAG: LOW_CONFIDENCE (top_prob=%.3f < %.2f) -> corrective search",
                top_prob,
                self.threshold,
            )
            return CRAGResult(
                action=CRAGAction.CORRECTIVE_SEARCH,
                top_score=top_prob,
                suggested_web_queries=await self._web_queries(query),
            )

        logger.info("CRAG: PROCEED (top_prob=%.3f >= %.2f)", top_prob, self.threshold)
        return CRAGResult(action=CRAGAction.PROCEED, top_score=top_prob)

    async def _web_queries(self, query: str) -> list[str]:
        result = await self._instructor.messages.create(
            model=self.model,
            max_tokens=512,
            system=_WEBQ_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original question: {query}\n\n"
                        "Generate 3 focused web-search queries that would retrieve "
                        "information to help answer it."
                    ),
                }
            ],
            response_model=_WebQueries,
        )
        return [q.strip() for q in result.queries if q.strip()][:3]
