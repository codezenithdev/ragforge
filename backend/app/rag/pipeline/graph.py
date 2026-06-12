"""LangGraph pipeline: decompose -> retrieve -> CRAG -> (corrective) -> rerank.

Wires the retrieval + agentic components into a StateGraph. Generation and
faithfulness scoring are appended in Phase 6; this graph currently ends at
rerank, exposing the reranked ``final_chunks``.

Heavy components (embedder, cross-encoder, Chroma client, Anthropic client) are
built once, lazily, on first invocation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.rag.evaluation.faithfulness_scorer import FaithfulnessScorer
from app.rag.generation.brief_generator import BriefGenerator
from app.rag.generation.schemas import BriefOutput
from app.rag.ingestion.embedder import Embedder
from app.rag.pipeline.crag import CRAGEvaluator
from app.rag.pipeline.query_decomposer import QueryDecomposer
from app.rag.retrieval.bm25_index import BM25Index
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.retrieval.vector_store import VectorStore
from app.rag.types import CRAGAction, CRAGResult, ScoredChunk

logger = logging.getLogger(__name__)


class BriefState(TypedDict, total=False):
    query: str
    document_ids: list[str] | None
    sub_queries: list[str]
    hyde_documents: list[str]
    retrieved_chunks: list[ScoredChunk]
    crag_result: CRAGResult
    corrective_chunks: list[ScoredChunk]
    final_chunks: list[ScoredChunk]
    brief: BriefOutput
    faithfulness_scores: dict[str, float]
    error: str | None


class PipelineComponents:
    """Heavy, reusable singletons (embedder, cross-encoder, clients)."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self.hybrid = HybridRetriever(self.vector_store, self.bm25)
        self.reranker = CrossEncoderReranker()
        self.decomposer = QueryDecomposer()
        self.crag = CRAGEvaluator(self.reranker)
        self.brief_generator = BriefGenerator()
        self.faithfulness_scorer = FaithfulnessScorer()


_components: PipelineComponents | None = None


def get_components() -> PipelineComponents:
    global _components
    if _components is None:
        logger.info("Initializing pipeline components (loading models)...")
        _components = PipelineComponents()
    return _components


def _dedup(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    seen: dict[str, ScoredChunk] = {}
    for chunk in chunks:
        seen.setdefault(chunk.chunk_id, chunk)
    return list(seen.values())


# --- Nodes (each async, each logs entry/exit) ---


async def decompose_query(state: BriefState) -> dict[str, Any]:
    components = get_components()
    query = state["query"]

    # Degraded path: if decomposition fails entirely, fall back to the bare query
    # rather than failing the whole brief.
    try:
        sub_queries = await components.decomposer.decompose(query)
    except Exception:  # noqa: BLE001 - transient LLM failure shouldn't abort the brief
        logger.warning("node decompose_query: decomposition failed; using bare query", exc_info=True)
        sub_queries = []
    if not sub_queries:
        sub_queries = [query]

    # HyDE per sub-query, fault-tolerant: one failed generation falls back to
    # embedding the sub-query text itself (keeps coverage) instead of aborting.
    hyde_results = await asyncio.gather(
        *(components.decomposer.generate_hyde_document(sq) for sq in sub_queries),
        return_exceptions=True,
    )
    hyde_documents: list[str] = []
    for sq, res in zip(sub_queries, hyde_results, strict=False):
        if isinstance(res, BaseException) or not res:
            logger.warning("node decompose_query: HyDE failed for %r; using sub-query text", sq[:60])
            hyde_documents.append(sq)
        else:
            hyde_documents.append(res)
    logger.info("node decompose_query: %d sub-queries", len(sub_queries))
    return {"sub_queries": sub_queries, "hyde_documents": hyde_documents}


async def retrieve(state: BriefState) -> dict[str, Any]:
    components = get_components()
    sub_queries = state["sub_queries"]
    hyde_documents = state["hyde_documents"]
    doc_ids = state.get("document_ids")

    # HyDE: embed the hypothetical-answer paragraphs and retrieve with those vectors.
    embeddings = await components.embedder.embed_batch(hyde_documents)
    # Fault-tolerant fan-out: one failed sub-query retrieval contributes no chunks
    # instead of aborting the brief.
    results = await asyncio.gather(
        *(
            components.hybrid.retrieve(sub_queries[i], embeddings[i], filter_doc_ids=doc_ids)
            for i in range(len(sub_queries))
        ),
        return_exceptions=True,
    )

    # Merge across sub-queries, dedup by chunk_id, keep the best RRF score.
    merged: dict[str, ScoredChunk] = {}
    failures = 0
    for hits in results:
        if isinstance(hits, BaseException):
            failures += 1
            continue
        for chunk in hits:
            existing = merged.get(chunk.chunk_id)
            if existing is None or (chunk.rrf_score or 0.0) > (existing.rrf_score or 0.0):
                merged[chunk.chunk_id] = chunk
    chunks = list(merged.values())
    if failures:
        logger.warning(
            "node retrieve: %d/%d sub-query retrievals failed", failures, len(sub_queries)
        )
    logger.info(
        "node retrieve: %d sub-queries -> %d unique chunks", len(sub_queries), len(chunks)
    )
    return {"retrieved_chunks": chunks}


async def evaluate_retrieval(state: BriefState) -> dict[str, Any]:
    components = get_components()
    result = await components.crag.evaluate_chunks(
        state["query"], state.get("retrieved_chunks", [])
    )
    logger.info(
        "node evaluate_retrieval: action=%s top=%.3f", result.action.value, result.top_score
    )
    return {"crag_result": result}


async def corrective_search(state: BriefState) -> dict[str, Any]:
    queries = state["crag_result"].suggested_web_queries
    chunks = await _tavily_search(queries)
    logger.info("node corrective_search: %d queries -> %d web chunks", len(queries), len(chunks))
    return {"corrective_chunks": chunks}


async def rerank(state: BriefState) -> dict[str, Any]:
    components = get_components()
    candidates = _dedup(
        list(state.get("retrieved_chunks", [])) + list(state.get("corrective_chunks", []))
    )
    final = await asyncio.to_thread(
        components.reranker.rerank, state["query"], candidates, settings.top_k_rerank
    )
    logger.info("node rerank: %d candidates -> %d final chunks", len(candidates), len(final))
    return {"final_chunks": final}


async def generate_brief(state: BriefState) -> dict[str, Any]:
    components = get_components()
    brief = await components.brief_generator.generate(
        state["query"], state.get("final_chunks", [])
    )
    logger.info("node generate_brief: title=%r, %d key_facts", brief.title, len(brief.key_facts))
    return {"brief": brief}


async def score_faithfulness(state: BriefState) -> dict[str, Any]:
    components = get_components()
    brief = state["brief"]
    scores = await components.faithfulness_scorer.score_brief(
        brief, state.get("final_chunks", [])
    )
    logger.info("node score_faithfulness: %d section scores", len(scores))
    # ``brief`` is returned too because score_brief filled each section's confidence in place.
    return {"brief": brief, "faithfulness_scores": scores}


async def _tavily_search(queries: list[str]) -> list[ScoredChunk]:
    if not settings.tavily_api_key:
        logger.warning("corrective_search: no TAVILY_API_KEY set; skipping web search")
        return []

    from tavily import TavilyClient

    def _search(query: str) -> dict[str, Any]:
        return TavilyClient(api_key=settings.tavily_api_key).search(
            query, max_results=settings.tavily_max_results
        )

    responses = await asyncio.gather(*(asyncio.to_thread(_search, q) for q in queries))
    chunks: list[ScoredChunk] = []
    for resp in responses:
        for item in (resp or {}).get("results", []):
            content = (item.get("content") or "").strip()
            if not content:
                continue
            chunks.append(
                ScoredChunk(
                    chunk_id=f"web::{uuid.uuid4().hex[:12]}",
                    content=content,
                    metadata={
                        "source": "web",
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                    },
                )
            )
    return chunks


def _route_after_crag(state: BriefState) -> str:
    if state["crag_result"].action == CRAGAction.CORRECTIVE_SEARCH:
        return "corrective_search"
    return "rerank"


def build_graph():
    graph = StateGraph(BriefState)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate_retrieval", evaluate_retrieval)
    graph.add_node("corrective_search", corrective_search)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_brief", generate_brief)
    graph.add_node("score_faithfulness", score_faithfulness)

    graph.set_entry_point("decompose_query")
    graph.add_edge("decompose_query", "retrieve")
    graph.add_edge("retrieve", "evaluate_retrieval")
    graph.add_conditional_edges(
        "evaluate_retrieval",
        _route_after_crag,
        {"corrective_search": "corrective_search", "rerank": "rerank"},
    )
    graph.add_edge("corrective_search", "rerank")
    graph.add_edge("rerank", "generate_brief")
    graph.add_edge("generate_brief", "score_faithfulness")
    graph.add_edge("score_faithfulness", END)
    return graph.compile()


briefr_graph = build_graph()
