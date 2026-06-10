"""Unit tests for the retrieval layer: hybrid RRF fusion, BM25, reranker.

External services (Chroma, Redis) and ML models are stubbed — these tests
exercise the fusion/ranking logic itself.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Iterable

from app.rag.retrieval.bm25_index import BM25Index
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.types import ScoredChunk
from tests.conftest import make_chunk

QUERY_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


class StubVectorStore:
    def __init__(self, hits_factory) -> None:
        self._hits_factory = hits_factory
        self.calls: list[dict[str, Any]] = []

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filter_doc_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        self.calls.append({"top_k": top_k, "filter_doc_ids": filter_doc_ids})
        return self._hits_factory()


class StubBM25:
    def __init__(self, hits_factory) -> None:
        self._hits_factory = hits_factory

    async def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        return self._hits_factory()


def _vector_hits() -> list[ScoredChunk]:
    # Dense ranking: A > B > C
    return [
        make_chunk("A", vector_score=0.9, document_id="d1"),
        make_chunk("B", vector_score=0.7, document_id="d1"),
        make_chunk("C", vector_score=0.5, document_id="d2"),
    ]


def _bm25_hits() -> list[ScoredChunk]:
    # Sparse ranking disagrees: C > B, and never saw A.
    return [
        make_chunk("C", bm25_score=4.2, document_id="d2"),
        make_chunk("B", bm25_score=1.1, document_id="d1"),
    ]


def _retriever() -> HybridRetriever:
    return HybridRetriever(StubVectorStore(_vector_hits), StubBM25(_bm25_hits))


async def test_hybrid_returns_results_from_both_retrievers() -> None:
    fused = await _retriever().retrieve("query", QUERY_EMBEDDING)
    ids = {c.chunk_id for c in fused}
    # A is dense-only, C is in both -> fusion must retain results of both kinds.
    assert {"A", "B", "C"} == ids
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["A"].vector_score is not None and by_id["A"].bm25_score is None
    assert all(c.rrf_score is not None for c in fused)


async def test_rrf_fusion_produces_different_ordering_than_either_alone() -> None:
    fused = await _retriever().retrieve("query", QUERY_EMBEDDING)
    fused_ids = [c.chunk_id for c in fused]
    vector_ids = [c.chunk_id for c in _vector_hits()]
    bm25_ids = [c.chunk_id for c in _bm25_hits()]

    # RRF(k=60): A=1/61; B=1/62+1/62; C=1/63+1/61 -> C, B, A.
    assert fused_ids != vector_ids
    assert fused_ids != bm25_ids
    assert fused_ids[0] == "C"
    # Sorted descending by fused score.
    rrf_scores = [c.rrf_score for c in fused]
    assert rrf_scores == sorted(rrf_scores, reverse=True)


async def test_rrf_merges_duplicates_keeping_both_component_scores() -> None:
    fused = await _retriever().retrieve("query", QUERY_EMBEDDING)
    c = next(chunk for chunk in fused if chunk.chunk_id == "C")
    assert [ch.chunk_id for ch in fused].count("C") == 1
    assert c.vector_score == 0.5 and c.bm25_score == 4.2
    # Sum of both reciprocal-rank contributions (ranks: vector #3, bm25 #1).
    assert c.rrf_score == (1 / 63) + (1 / 61)


async def test_hybrid_scopes_bm25_hits_to_filtered_documents() -> None:
    retriever = _retriever()
    fused = await retriever.retrieve(
        "query", QUERY_EMBEDDING, filter_doc_ids=["d1"]
    )
    bm25_scored = [c for c in fused if c.bm25_score is not None]
    # C (document d2) must be dropped from the sparse side by the post-filter.
    assert all(c.metadata["document_id"] == "d1" for c in bm25_scored)
    # And the doc filter was forwarded to the vector store.
    assert retriever.vector_store.calls[0]["filter_doc_ids"] == ["d1"]


async def test_cross_encoder_reranker_changes_ordering(monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            # Longer documents score higher -> reverses the input ordering below.
            return [float(len(doc)) for _, doc in pairs]

    # Avoid importing the real (heavy) sentence_transformers package entirely.
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    from app.rag.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="fake-model")
    chunks = [
        make_chunk("short", "aa"),
        make_chunk("medium", "aaaa"),
        make_chunk("long", "aaaaaaaa"),
    ]
    ranked = reranker.rerank("query", list(chunks), top_k=2)

    assert [c.chunk_id for c in ranked] == ["long", "medium"]  # reordered + top_k cut
    assert all(c.rerank_score is not None for c in ranked)
    assert ranked[0].score == ranked[0].rerank_score


async def test_bm25_index_search_ranks_matching_documents() -> None:
    from rank_bm25 import BM25Okapi

    # Build the index without touching Redis (__init__ bypassed on purpose).
    index = BM25Index.__new__(BM25Index)
    index._redis = None
    index._ids = ["c1", "c2", "c3"]
    index._texts = [
        "anthropic builds the claude language models",
        "photosynthesis happens in plants",
        "claude is a large language model by anthropic",
    ]
    index._metadatas = [{"document_id": "d1"}, {"document_id": "d2"}, {"document_id": "d3"}]
    index._bm25 = BM25Okapi([index._tokenize(t) for t in index._texts])

    hits = await index.search("anthropic claude", top_k=2)

    assert len(hits) == 2
    assert {h.chunk_id for h in hits} <= {"c1", "c3"}  # the biology chunk loses
    assert all(h.bm25_score is not None and h.bm25_score > 0 for h in hits)
