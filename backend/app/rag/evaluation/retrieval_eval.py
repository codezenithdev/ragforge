"""Deterministic, offline retrieval-quality eval (the CI regression gate).

Scores recall@k and MRR for the hybrid *ranking logic* — BM25 lexical retrieval
fused with RRF (``app.rag.retrieval.hybrid.rrf_fuse``) — over a committed golden
set. No embedding API, vector DB, or Redis: the dense signal is a deterministic
token-overlap stand-in, so the same inputs always produce the same ranking. This
gates regressions in our fusion/ranking code. End-to-end *answer* quality
(faithfulness, RAGAS) is graded separately by ``scripts/run_eval.py``, which
needs API keys and is run on demand, not in CI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.retrieval.hybrid import rrf_fuse
from app.rag.types import ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")

# Default golden set location (committed alongside the tests).
GOLDEN_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "golden" / "golden.json"


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class GoldenItem:
    query: str
    expected_ids: list[str]


@dataclass
class EvalSummary:
    recall_at_k: float
    mrr: float
    k: int
    n: int
    per_query: list[dict] = field(default_factory=list)


def load_golden(path: str | Path = GOLDEN_PATH) -> tuple[list[tuple[str, str]], list[GoldenItem]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus = [(c["id"], c["text"]) for c in data["corpus"]]
    items = [GoldenItem(q["query"], list(q["expected_ids"])) for q in data["queries"]]
    return corpus, items


def recall_at_k(ranked_ids: list[str], expected_ids: list[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return sum(1 for e in expected_ids if e in topk) / len(expected_ids)


def reciprocal_rank(ranked_ids: list[str], expected_ids: list[str]) -> float:
    expected = set(expected_ids)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def _bm25_ranked(query: str, corpus: list[tuple[str, str]]) -> list[ScoredChunk]:
    bm = BM25Okapi([_tok(t) for _, t in corpus])
    scores = bm.get_scores(_tok(query))
    order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [
        ScoredChunk(chunk_id=corpus[i][0], content=corpus[i][1], bm25_score=float(scores[i]))
        for i in order
    ]


def _dense_ranked(query: str, corpus: list[tuple[str, str]]) -> list[ScoredChunk]:
    """Deterministic dense stand-in: Jaccard token overlap (no embedding model)."""
    q = set(_tok(query))

    def sim(text: str) -> float:
        toks = set(_tok(text))
        union = q | toks
        return len(q & toks) / len(union) if union else 0.0

    order = sorted(range(len(corpus)), key=lambda i: sim(corpus[i][1]), reverse=True)
    return [
        ScoredChunk(chunk_id=corpus[i][0], content=corpus[i][1], vector_score=sim(corpus[i][1]))
        for i in order
    ]


def rank(query: str, corpus: list[tuple[str, str]], top_k: int | None = None) -> list[str]:
    """Return chunk ids best-first via BM25 + deterministic-dense, fused with RRF."""
    top_k = top_k or settings.top_k_retrieval
    fused = rrf_fuse(_dense_ranked(query, corpus), _bm25_ranked(query, corpus), settings.rrf_k)
    return [c.chunk_id for c in fused[:top_k]]


def evaluate(
    corpus: list[tuple[str, str]], items: list[GoldenItem], k: int = 5
) -> EvalSummary:
    recalls: list[float] = []
    rrs: list[float] = []
    per_query: list[dict] = []
    for it in items:
        ranked = rank(it.query, corpus)
        r = recall_at_k(ranked, it.expected_ids, k)
        rr = reciprocal_rank(ranked, it.expected_ids)
        recalls.append(r)
        rrs.append(rr)
        per_query.append(
            {"query": it.query, "recall@k": r, "rr": rr, "top": ranked[:k]}
        )
    n = len(items) or 1
    return EvalSummary(
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        k=k,
        n=len(items),
        per_query=per_query,
    )
