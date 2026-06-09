"""Shared RAG data types.

Kept in one module so retrieval, pipeline, and generation can all import them
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ScoredChunk:
    """A retrieved chunk with whatever scores the current stage has attached.

    ``score`` is the *active* score for the current stage (cosine similarity after
    vector search, RRF after fusion, cross-encoder score after reranking); the
    component scores are retained for logging/debugging.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class CRAGAction(str, Enum):
    PROCEED = "proceed"
    CORRECTIVE_SEARCH = "corrective_search"


@dataclass
class CRAGResult:
    action: CRAGAction
    top_score: float
    suggested_web_queries: list[str] = field(default_factory=list)
