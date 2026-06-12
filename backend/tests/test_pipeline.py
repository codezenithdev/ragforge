"""Integration tests for the LangGraph pipeline (all components faked).

Verifies the graph wiring itself: node transition order, state propagation,
chunk deduplication, and the CRAG conditional edge (PROCEED vs corrective web
search). Also unit-tests the CRAGEvaluator routing thresholds.
"""

from __future__ import annotations

import anthropic
import pytest

import app.rag.pipeline.graph as graph_module
from app.rag.generation.schemas import BriefOutput, BriefSection
from app.rag.pipeline.crag import CRAGEvaluator
from app.rag.types import CRAGAction, CRAGResult, ScoredChunk
from tests.conftest import make_chunk


def _fake_brief() -> BriefOutput:
    section = lambda: BriefSection(content="text", sources=["1"])  # noqa: E731
    return BriefOutput(
        title="Test Brief",
        executive_summary=section(),
        key_facts=[section(), section(), section()],
        risks_and_limitations=section(),
        opportunities=section(),
        open_questions=["q?"],
        sources=[],
    )


class FakeComponents:
    """Stands in for PipelineComponents; every method logs its stage."""

    def __init__(self, crag_result: CRAGResult) -> None:
        self.log: list[str] = []
        self._crag_result = crag_result
        log = self.log

        class Decomposer:
            async def decompose(self, query: str) -> list[str]:
                log.append("decompose")
                return ["sub one", "sub two"]

            async def generate_hyde_document(self, sub_query: str) -> str:
                log.append("hyde")
                return f"hyde for {sub_query}"

        class Embedder:
            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                log.append("embed")
                return [[0.1, 0.2] for _ in texts]

        class Hybrid:
            async def retrieve(self, query, query_embedding, top_k=None, filter_doc_ids=None):
                log.append("hybrid")
                dup = make_chunk("dup", document_id="d1")
                dup.rrf_score = 0.03
                unique = make_chunk("unique", document_id="d1")
                unique.rrf_score = 0.02
                return [dup, unique]

        class Crag:
            async def evaluate_chunks(self_inner, query, chunks) -> CRAGResult:
                log.append("crag")
                return crag_result

        class Reranker:
            def rerank(self, query, chunks: list[ScoredChunk], top_k=None):
                log.append("rerank")
                for i, chunk in enumerate(chunks):
                    chunk.rerank_score = chunk.score = float(len(chunks) - i)
                return chunks[: (top_k or len(chunks))]

        class Generator:
            async def generate(self, query, chunks) -> BriefOutput:
                log.append("generate")
                return _fake_brief()

        class Scorer:
            async def score_brief(self, brief: BriefOutput, chunks) -> dict[str, float]:
                log.append("score")
                brief.executive_summary.confidence = 0.9
                return {"executive_summary": 0.9}

        self.decomposer = Decomposer()
        self.embedder = Embedder()
        self.hybrid = Hybrid()
        self.crag = Crag()
        self.reranker = Reranker()
        self.brief_generator = Generator()
        self.faithfulness_scorer = Scorer()


def _first_occurrence(log: list[str]) -> list[str]:
    seen: list[str] = []
    for entry in log:
        if entry not in seen:
            seen.append(entry)
    return seen


@pytest.fixture
def proceed_components(monkeypatch) -> FakeComponents:
    fake = FakeComponents(CRAGResult(action=CRAGAction.PROCEED, top_score=0.95))
    monkeypatch.setattr(graph_module, "_components", fake)
    return fake


@pytest.fixture
def corrective_components(monkeypatch) -> FakeComponents:
    fake = FakeComponents(
        CRAGResult(
            action=CRAGAction.CORRECTIVE_SEARCH,
            top_score=0.05,
            suggested_web_queries=["w1", "w2", "w3"],
        )
    )
    monkeypatch.setattr(graph_module, "_components", fake)

    async def fake_tavily(queries: list[str]) -> list[ScoredChunk]:
        fake.log.append("tavily")
        return [make_chunk("web::1", source="web", url="https://example.com")]

    monkeypatch.setattr(graph_module, "_tavily_search", fake_tavily)
    return fake


async def test_graph_transitions_happy_path(proceed_components: FakeComponents) -> None:
    state = await graph_module.briefr_graph.ainvoke({"query": "test question"})

    assert _first_occurrence(proceed_components.log) == [
        "decompose",
        "hyde",
        "embed",
        "hybrid",
        "crag",
        "rerank",
        "generate",
        "score",
    ]
    assert state["sub_queries"] == ["sub one", "sub two"]
    assert len(state["hyde_documents"]) == len(state["sub_queries"])
    assert state["crag_result"].action is CRAGAction.PROCEED
    assert "corrective_chunks" not in state  # corrective node must not run
    assert state["final_chunks"]
    assert isinstance(state["brief"], BriefOutput)
    assert state["faithfulness_scores"] == {"executive_summary": 0.9}
    assert state["brief"].executive_summary.confidence == 0.9


async def test_retrieve_deduplicates_chunks_across_sub_queries(
    proceed_components: FakeComponents,
) -> None:
    state = await graph_module.briefr_graph.ainvoke({"query": "test question"})
    # Both sub-queries returned the same two chunk ids -> exactly 2 after dedup.
    ids = [c.chunk_id for c in state["retrieved_chunks"]]
    assert sorted(ids) == ["dup", "unique"]
    assert proceed_components.log.count("hybrid") == 2  # one retrieval per sub-query


async def test_crag_corrective_path_triggers_web_search(
    corrective_components: FakeComponents,
) -> None:
    state = await graph_module.briefr_graph.ainvoke({"query": "off-topic question"})

    order = _first_occurrence(corrective_components.log)
    assert order.index("crag") < order.index("tavily") < order.index("rerank")
    assert [c.chunk_id for c in state["corrective_chunks"]] == ["web::1"]
    # The web chunk flows into the final reranked context.
    assert "web::1" in {c.chunk_id for c in state["final_chunks"]}
    assert state["brief"].title == "Test Brief"


# --- Intra-pipeline resilience (#4) ---


async def test_decompose_tolerates_hyde_failure(proceed_components: FakeComponents) -> None:
    fake = proceed_components

    async def flaky_hyde(sub_query: str) -> str:
        if sub_query == "sub one":
            raise RuntimeError("hyde 529")
        return f"hyde for {sub_query}"

    fake.decomposer.generate_hyde_document = flaky_hyde  # type: ignore[method-assign]

    state = await graph_module.briefr_graph.ainvoke({"query": "q"})

    # Both sub-queries are retained; the failed HyDE falls back to embedding the
    # sub-query text, and the brief still completes.
    assert state["sub_queries"] == ["sub one", "sub two"]
    assert state["hyde_documents"][0] == "sub one"
    assert state["hyde_documents"][1] == "hyde for sub two"
    assert isinstance(state["brief"], BriefOutput)


async def test_decompose_falls_back_to_bare_query(proceed_components: FakeComponents) -> None:
    fake = proceed_components

    async def boom(query: str) -> list[str]:
        raise RuntimeError("decompose down")

    fake.decomposer.decompose = boom  # type: ignore[method-assign]

    state = await graph_module.briefr_graph.ainvoke({"query": "the bare question"})

    assert state["sub_queries"] == ["the bare question"]
    assert len(state["hyde_documents"]) == 1
    assert isinstance(state["brief"], BriefOutput)


async def test_retrieve_tolerates_partial_subquery_failure(
    proceed_components: FakeComponents,
) -> None:
    fake = proceed_components
    calls = {"n": 0}

    async def flaky_retrieve(query, query_embedding, top_k=None, filter_doc_ids=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("vector store timeout")
        chunk = make_chunk("survivor", document_id="d1")
        chunk.rrf_score = 0.02
        return [chunk]

    fake.hybrid.retrieve = flaky_retrieve  # type: ignore[method-assign]

    state = await graph_module.briefr_graph.ainvoke({"query": "q"})

    # One sub-query retrieval failed; the brief still completes on the survivor.
    assert {c.chunk_id for c in state["retrieved_chunks"]} == {"survivor"}
    assert isinstance(state["brief"], BriefOutput)


# --- CRAGEvaluator routing thresholds (real evaluator, fake reranker) ---


class _FixedReranker:
    def __init__(self, top_logit: float) -> None:
        self._top_logit = top_logit

    def rerank(self, query, chunks, top_k=None):
        for i, chunk in enumerate(chunks):
            chunk.rerank_score = self._top_logit - i
        return chunks


def _evaluator(top_logit: float, monkeypatch) -> CRAGEvaluator:
    evaluator = CRAGEvaluator(
        _FixedReranker(top_logit),
        anthropic_client=anthropic.AsyncAnthropic(api_key="test-key"),
    )

    async def fake_web_queries(query: str) -> list[str]:
        return ["q1", "q2", "q3"]

    monkeypatch.setattr(evaluator, "_web_queries", fake_web_queries)
    return evaluator


async def test_crag_proceeds_when_top_chunk_is_confident(monkeypatch) -> None:
    evaluator = _evaluator(top_logit=5.0, monkeypatch=monkeypatch)  # sigmoid ~0.99
    result = await evaluator.evaluate_chunks("q", [make_chunk("a"), make_chunk("b")])
    assert result.action is CRAGAction.PROCEED
    assert result.top_score > 0.9
    assert result.suggested_web_queries == []


async def test_crag_requests_corrective_search_on_low_confidence(monkeypatch) -> None:
    evaluator = _evaluator(top_logit=-5.0, monkeypatch=monkeypatch)  # sigmoid ~0.007
    result = await evaluator.evaluate_chunks("q", [make_chunk("a")])
    assert result.action is CRAGAction.CORRECTIVE_SEARCH
    assert result.top_score < 0.3
    assert result.suggested_web_queries == ["q1", "q2", "q3"]


async def test_crag_requests_corrective_search_when_nothing_retrieved(monkeypatch) -> None:
    evaluator = _evaluator(top_logit=5.0, monkeypatch=monkeypatch)
    result = await evaluator.evaluate_chunks("q", [])
    assert result.action is CRAGAction.CORRECTIVE_SEARCH
    assert result.suggested_web_queries == ["q1", "q2", "q3"]
