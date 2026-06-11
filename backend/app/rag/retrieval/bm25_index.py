"""Sparse BM25 retrieval.

rank_bm25 is in-memory, so the corpus (ids/texts/metadata) is cached in Redis and
the ``BM25Okapi`` index is rebuilt from it on first use per process. Rebuilding on
update is acceptable for the MVP.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from typing import Any, Iterable

import redis.asyncio as aioredis
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.redis_client import get_redis
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)

_REBUILD_LOCK_KEY = "briefr:bm25:rebuild:lock"


class BM25Index:
    REDIS_KEY = "briefr:bm25:corpus"
    # Monotonic counter bumped on every corpus write. A long-lived instance (e.g.
    # the worker's PipelineComponents.bm25 singleton) compares this against the
    # version it last loaded and reloads when it advances, so a doc uploaded by
    # another process becomes visible to sparse retrieval without a restart (P1.1).
    VERSION_KEY = "briefr:bm25:version"

    def __init__(self, redis_url: str | None = None) -> None:
        # Reuse the shared client by default (no per-instance connection churn,
        # P3.4); a custom redis_url still gets its own client when needed.
        self._redis = aioredis.from_url(redis_url) if redis_url else get_redis()
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        # -1 (never loaded) forces the first _ensure_loaded to load.
        self._version: int = -1

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
        # Bump the corpus version and adopt it locally so the writing instance
        # doesn't needlessly reload what it just built.
        self._version = int(await self._redis.incr(self.VERSION_KEY))
        logger.info("BM25Index: built index over %d chunks (version=%d)", len(ids), self._version)

    async def update_index(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        await self.build_index(ids, texts, metadatas)

    async def _ensure_loaded(self) -> None:
        # No Redis (pre-loaded/test instances): trust the in-memory index as-is.
        if self._redis is None:
            return
        # Reload when never loaded OR when the shared corpus version has advanced
        # since this instance last loaded (another process rebuilt the index).
        current = int(await self._redis.get(self.VERSION_KEY) or 0)
        if self._bm25 is not None and current == self._version:
            return
        payload = await self._redis.get(self.REDIS_KEY)
        if not payload:
            return
        data = pickle.loads(payload)
        self._ids = data["ids"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        self._bm25 = BM25Okapi([self._tokenize(t) for t in self._texts]) if self._texts else None
        self._version = current
        logger.info("BM25Index: (re)loaded corpus of %d chunks (version=%d)", len(self._ids), current)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_doc_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_retrieval
        await self._ensure_loaded()
        if not self._bm25 or not self._ids:
            logger.info("BM25Index: empty index, no results")
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        # Restrict ranking to the requested doc set *before* taking top_k (P1.7),
        # so a scoped doc isn't lost just because it fell outside the global top_k.
        if filter_doc_ids is not None:
            allowed = set(filter_doc_ids)
            candidates = [
                i for i in range(len(scores)) if self._metadatas[i].get("document_id") in allowed
            ]
        else:
            candidates = list(range(len(scores)))
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
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


async def rebuild_bm25_locked(*, wait_timeout: float = 30.0) -> None:
    """Rebuild the BM25 index from the full Chroma corpus under a Redis lock (P1.3/P1.4).

    Serializes concurrent rebuilds (upload ingestion + deletes) so they can't
    clobber each other. Each rebuild reads the current corpus from Chroma — the
    source of truth — so even a queued rebuild converges to the latest state, and
    the version counter (P1.1) makes long-lived readers reload.
    """
    from app.rag.retrieval.vector_store import VectorStore

    redis = get_redis()
    acquired = False
    start = time.monotonic()
    while True:
        acquired = bool(await redis.set(_REBUILD_LOCK_KEY, "1", nx=True, ex=120))
        if acquired:
            break
        if time.monotonic() - start > wait_timeout:
            logger.warning("rebuild_bm25_locked: lock wait exceeded %.0fs; proceeding", wait_timeout)
            break
        await asyncio.sleep(0.25)
    try:
        ids, docs, metas = await VectorStore().fetch_corpus()
        await BM25Index().build_index(ids, docs, metas)
    finally:
        if acquired:
            await redis.delete(_REBUILD_LOCK_KEY)
