"""Sparse BM25 retrieval.

rank_bm25 is in-memory, so the corpus (ids/texts/metadata) is cached in Redis and
the ``BM25Okapi`` index is rebuilt from it on first use per process. Rebuilding on
update is acceptable for the MVP.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import redis.asyncio as aioredis
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)


class BM25Index:
    REDIS_KEY = "briefr:bm25:corpus"

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = aioredis.from_url(redis_url or settings.redis_url)
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    async def build_index(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in ids]
        self._ids, self._texts, self._metadatas = ids, texts, metadatas
        self._bm25 = BM25Okapi([self._tokenize(t) for t in texts]) if texts else None
        await self._redis.set(
            self.REDIS_KEY,
            pickle.dumps({"ids": ids, "texts": texts, "metadatas": metadatas}),
        )
        logger.info("BM25Index: built index over %d chunks", len(ids))

    async def update_index(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        await self.build_index(ids, texts, metadatas)

    async def _ensure_loaded(self) -> None:
        if self._bm25 is not None:
            return
        payload = await self._redis.get(self.REDIS_KEY)
        if not payload:
            return
        data = pickle.loads(payload)
        self._ids = data["ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        self._bm25 = BM25Okapi([self._tokenize(t) for t in self._texts]) if self._texts else None

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_retrieval
        await self._ensure_loaded()
        if not self._bm25 or not self._ids:
            logger.info("BM25Index: empty index, no results")
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [
            ScoredChunk(
                chunk_id=self._ids[i],
                content=self._texts[i],
                metadata=self._metadatas[i],
                score=float(scores[i]),
                bm25_score=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]
        logger.info("BM25Index: query -> %d hits (top_k=%d)", len(results), top_k)
        return results
