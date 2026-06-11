"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Runs dense (vector) and sparse (BM25) retrieval concurrently and fuses them with
RRF: ``score(d) = Σ 1 / (k + rank(d))`` over each result list. RRF blends rankings
with very different score scales without any tuning — which is exactly why it
beats naive score normalization here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from app.core.config import settings
from app.rag.retrieval.bm25_index import BM25Index
from app.rag.retrieval.vector_store import VectorStore
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k or settings.rrf_k

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_doc_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_retrieval
        # Both sides scope to the same doc set; BM25 now filters before ranking
        # (P1.7) rather than post-filtering a globally-truncated top_k.
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_store.similarity_search(
                query_embedding, top_k=top_k, filter_doc_ids=filter_doc_ids
            ),
            self.bm25_index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids),
        )
        fused = self._rrf_fuse(vector_hits, bm25_hits)
        logger.info(
            "HybridRetriever: vector=%d bm25=%d fused=%d (rrf_k=%d)",
            len(vector_hits),
            len(bm25_hits),
            len(fused),
            self.rrf_k,
        )
        return fused[:top_k]

    def _rrf_fuse(
        self, vector_hits: list[ScoredChunk], bm25_hits: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        rrf_scores: dict[str, float] = {}
        chunks: dict[str, ScoredChunk] = {}

        for result_list in (vector_hits, bm25_hits):
            for rank, chunk in enumerate(result_list):
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank + 1
                )
                existing = chunks.get(chunk.chunk_id)
                if existing is None:
                    chunks[chunk.chunk_id] = chunk
                else:
                    # Merge component scores so both are retained on the kept object.
                    if chunk.vector_score is not None:
                        existing.vector_score = chunk.vector_score
                    if chunk.bm25_score is not None:
                        existing.bm25_score = chunk.bm25_score

        merged: list[ScoredChunk] = []
        for chunk_id, rrf in rrf_scores.items():
            chunk = chunks[chunk_id]
            chunk.rrf_score = rrf
            chunk.score = rrf
            merged.append(chunk)

        merged.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)
        return merged
