"""Shared async Redis client.

A single process-wide ``redis.asyncio`` client, reused everywhere instead of a
new connection pool per object (P3.4: ``BM25Index`` previously created — and
never closed — one ``aioredis.from_url`` per instance, churning connections on
every upload/delete/search). Call ``get_redis()`` to obtain it and ``close_redis()``
on shutdown.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the process-wide shared async Redis client (created on first use)."""
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis_url)
        logger.info("Redis client initialized (%s)", settings.redis_url)
    return _client


async def close_redis() -> None:
    """Close the shared client (call on app/worker shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
