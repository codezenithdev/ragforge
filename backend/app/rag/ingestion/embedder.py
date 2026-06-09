"""OpenAI embedder.

Wraps the OpenAI embeddings API with request batching, exponential-backoff retry
(tenacity), and **unit-normalization** of every vector so that ChromaDB cosine
distance behaves consistently.
"""

from __future__ import annotations

import logging

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class Embedder:
    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr.tolist()
        return (arr / norm).tolist()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _embed_request(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            raw = await self._embed_request(batch)
            vectors.extend(self._normalize(v) for v in raw)
        logger.info(
            "Embedder: embedded %d texts (model=%s, dim=%d)",
            len(texts),
            self.model,
            len(vectors[0]) if vectors else 0,
        )
        return vectors

    async def embed_single(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]
