"""Offline retrieval regression gate (#1, Layer 1).

Runs the deterministic BM25 + RRF ranking over the committed golden set and
asserts recall@k / MRR stay above a floor. Fully offline (no keys/services), so
it runs in the normal CI pytest job. Lowering a retrieval knob that breaks
ranking makes this fail — that's the point.
"""

from __future__ import annotations

from app.rag.evaluation.retrieval_eval import (
    evaluate,
    load_golden,
    rank,
    recall_at_k,
    reciprocal_rank,
)

# Conservative floors: the golden queries are strongly lexical so healthy
# ranking scores ~1.0; these floors bite if fusion/ranking regresses.
RECALL_AT_5_FLOOR = 0.8
MRR_FLOOR = 0.6


def test_retrieval_eval_meets_floor() -> None:
    corpus, items = load_golden()
    assert len(items) >= 10, "golden set should have a meaningful number of queries"

    summary = evaluate(corpus, items, k=5)

    assert summary.recall_at_k >= RECALL_AT_5_FLOOR, (
        f"recall@5 {summary.recall_at_k:.3f} below floor {RECALL_AT_5_FLOOR}: "
        f"{[q for q in summary.per_query if q['recall@k'] < 1.0]}"
    )
    assert summary.mrr >= MRR_FLOOR, f"MRR {summary.mrr:.3f} below floor {MRR_FLOOR}"


def test_metric_helpers() -> None:
    # recall@k counts how many expected ids land in the top-k.
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5
    assert recall_at_k(["x"], ["a"], k=1) == 0.0
    # reciprocal rank is 1/(rank of first hit).
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_rank_is_deterministic() -> None:
    corpus, items = load_golden()
    q = items[0].query
    assert rank(q, corpus) == rank(q, corpus)
