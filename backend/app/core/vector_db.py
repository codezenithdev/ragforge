"""ChromaDB vector store (server mode).

The FastAPI backend and the Celery worker are separate processes that both
connect to the shared Chroma HTTP server. Chroma owns the chunk corpus — ids,
embeddings, chunk text, and metadata (document_id, source, page, section,
chunk_index) — while Postgres (see ``app.core.database``) holds documents and
briefs.

Collections use cosine distance to match the unit-normalized OpenAI embeddings.
An async client is used so the backend stays async end-to-end.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.async_api import AsyncClientAPI
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Cosine space matches the normalized embeddings produced by the embedder.
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}

def _build_chroma_settings() -> ChromaSettings:
    """Chroma client settings, with token auth wired in when configured (P0.6)."""
    kwargs: dict[str, object] = {"anonymized_telemetry": False}
    if settings.chroma_auth_token:
        kwargs.update(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=settings.chroma_auth_token,
        )
    return ChromaSettings(**kwargs)


_CHROMA_SETTINGS = _build_chroma_settings()


async def get_async_chroma_client() -> AsyncClientAPI:
    """Return an async Chroma HTTP client connected to the server."""
    return await chromadb.AsyncHttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=_CHROMA_SETTINGS,
    )


def get_chroma_client() -> ClientAPI:
    """Return a sync Chroma HTTP client (for non-async contexts, e.g. scripts)."""
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=_CHROMA_SETTINGS,
    )


async def get_or_create_collection(client: AsyncClientAPI | None = None):
    """Return the Briefr chunk collection, creating it (cosine) if needed."""
    client = client or await get_async_chroma_client()
    return await client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata=COLLECTION_METADATA,
    )


async def chroma_heartbeat() -> int:
    """Ping the Chroma server. Called on app startup to fail fast if it is down."""
    client = await get_async_chroma_client()
    nanoseconds = await client.heartbeat()
    logger.info(
        "Chroma server reachable (heartbeat=%s ns, collection=%s)",
        nanoseconds,
        settings.chroma_collection,
    )
    return nanoseconds
