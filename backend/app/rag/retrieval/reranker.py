"""Cross-encoder reranker.

A bi-encoder (vector search) scores query and document independently; a
cross-encoder scores the (query, document) *pair* jointly, which is far more
accurate for final ranking. We rerank the fused candidate set down to the top-k
that actually go into generation. ``predict`` is CPU-bound and synchronous —
async callers should wrap ``rerank`` in ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import time

from app.core.config import settings
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_rerank
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        start = time.perf_counter()
        scores = self.model.predict(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
            chunk.score = float(score)

        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]
        logger.info(
            "CrossEncoderReranker: reranked %d chunks in %.1f ms -> top %d",
            len(chunks),
            elapsed_ms,
            len(ranked),
        )
        return ranked
