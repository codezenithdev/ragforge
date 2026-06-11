"""ChromaDB-backed vector store.

Wraps the async Chroma collection (from ``app.core.vector_db``) with the
retrieval operations Briefr needs: bulk upsert, cosine similarity search (with an
optional document-id filter), corpus fetch (for the BM25 index), and
delete-by-document. Cosine *distance* from Chroma is converted to a similarity
``score = 1 - distance``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from app.core.config import settings
from app.core.vector_db import get_or_create_collection
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)

_SCALAR = (str, int, float, bool)


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values; drop None/non-scalars."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if isinstance(v, _SCALAR)}


class VectorStore:
    def __init__(self, collection: Any | None = None) -> None:
        self._collection = collection

    async def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = await get_or_create_collection()
        return self._collection

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert chunks. Each item: {id, content, embedding, metadata}."""
        if not chunks:
            return
        collection = await self._get_collection()
        await collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["content"] for c in chunks],
            metadatas=[_clean_metadata(c.get("metadata")) or {"_": ""} for c in chunks],
        )
        logger.info("VectorStore: upserted %d chunks", len(chunks))

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_doc_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        top_k = top_k or settings.top_k_retrieval
        collection = await self._get_collection()
        where = None
        if filter_doc_ids:
            where = {"document_id": {"$in": list(filter_doc_ids)}}

        result = await collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[ScoredChunk] = []
        for cid, doc, md, dist in zip(ids, documents, metadatas, distances, strict=False):
            similarity = 1.0 - float(dist)
            chunks.append(
                ScoredChunk(
                    chunk_id=cid,
                    content=doc or "",
                    metadata=md or {},
                    score=similarity,
                    vector_score=similarity,
                )
            )
        logger.info("VectorStore: similarity_search -> %d hits (top_k=%d)", len(chunks), top_k)
        return chunks

    async def fetch_corpus(
        self, page_size: int = 1000
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Return (ids, documents, metadatas) for the whole collection — used to
        (re)build the BM25 index. Paginated (P1.3) so an unbounded ``get()`` can't
        pull the entire corpus into memory in one allocation."""
        collection = await self._get_collection()
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = await collection.get(
                include=["documents", "metadatas"], limit=page_size, offset=offset
            )
            page_ids = page.get("ids") or []
            if not page_ids:
                break
            ids.extend(page_ids)
            documents.extend(page.get("documents") or [])
            metadatas.extend(page.get("metadatas") or [{} for _ in page_ids])
            if len(page_ids) < page_size:
                break
            offset += page_size
        logger.info("VectorStore: fetched corpus of %d chunks", len(ids))
        return ids, documents, metadatas

    async def fetch_texts_by_ids(self, ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: text} for the given ids (P2.3 context rehydration)."""
        if not ids:
            return {}
        collection = await self._get_collection()
        result = await collection.get(ids=ids, include=["documents"])
        got_ids = result.get("ids") or []
        documents = result.get("documents") or []
        return {cid: doc or "" for cid, doc in zip(got_ids, documents, strict=False)}

    async def delete_document_chunks(self, document_id: str) -> None:
        collection = await self._get_collection()
        await collection.delete(where={"document_id": document_id})
        logger.info("VectorStore: deleted chunks for document_id=%s", document_id)
