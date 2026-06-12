"""On-demand RAG quality eval (#1, Layer 2) — run before a release.

Runs the FULL pipeline (briefr_graph) over the golden queries, then aggregates
the per-section faithfulness scores (computed inline by the pipeline) and RAGAS
metrics into a summary table + eval-out/report.json. Also reports estimated
token cost per query via the usage accumulator.

Unlike the offline retrieval gate (tests/test_eval_retrieval.py), this needs API
keys (ANTHROPIC_API_KEY, OPENAI_API_KEY), a running ChromaDB, and the golden
corpus actually ingested — so it is NOT wired into CI. Use it to tune knobs
(crag_confidence_threshold, num_sub_queries, …) against a measured baseline.

Usage:
    cd backend && python scripts/run_eval.py [--golden PATH] [--limit N] [--out DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make `app` importable when run as a script from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def _run_one(query: str) -> dict:
    from app.core.anthropic_client import UsageTotals, usage_var
    from app.rag.evaluation.ragas_runner import RAGASRunner
    from app.rag.pipeline.graph import briefr_graph

    usage = UsageTotals()
    usage_var.set(usage)

    state = await briefr_graph.ainvoke({"query": query})
    brief = state["brief"]
    contexts = [c.content for c in state.get("final_chunks", [])]
    answer = (brief.executive_summary.content or "").strip()

    faith = state.get("faithfulness_scores", {}) or {}
    mean_faith = sum(faith.values()) / len(faith) if faith else 0.0

    ragas_overall = None
    ragas_raw: dict = {}
    if answer and contexts:
        ragas = await RAGASRunner().evaluate(question=query, answer=answer, contexts=contexts)
        ragas_overall = ragas.overall
        ragas_raw = ragas.raw

    return {
        "query": query,
        "crag_action": state.get("crag_result").action.value if state.get("crag_result") else None,
        "num_chunks": len(contexts),
        "mean_faithfulness": round(mean_faith, 4),
        "faithfulness_by_section": {k: round(v, 4) for k, v in faith.items()},
        "ragas_overall": round(ragas_overall, 4) if ragas_overall is not None else None,
        "ragas_raw": ragas_raw,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "est_cost_usd": usage.estimated_cost_usd(),
    }


async def _main(golden_path: str, limit: int | None, out_dir: str) -> int:
    from app.rag.evaluation.retrieval_eval import load_golden

    _, items = load_golden(golden_path)
    if limit:
        items = items[:limit]

    rows: list[dict] = []
    for it in items:
        print(f"running: {it.query!r} ...", flush=True)
        try:
            rows.append(await _run_one(it.query))
        except Exception as exc:  # noqa: BLE001 - report per-query failures, keep going
            print(f"  FAILED: {exc}", flush=True)
            rows.append({"query": it.query, "error": str(exc)})

    ok = [r for r in rows if "error" not in r]
    agg = {
        "n_queries": len(rows),
        "n_ok": len(ok),
        "mean_faithfulness": round(sum(r["mean_faithfulness"] for r in ok) / len(ok), 4) if ok else 0.0,
        "mean_ragas_overall": round(
            sum(r["ragas_overall"] for r in ok if r["ragas_overall"] is not None)
            / max(1, sum(1 for r in ok if r["ragas_overall"] is not None)),
            4,
        ),
        "total_est_cost_usd": round(sum(r.get("est_cost_usd", 0.0) for r in ok), 4),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report = {"aggregate": agg, "per_query": rows}
    (out_path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RAG eval summary ===")
    print(f"{'query':45.45}  faith  ragas  cost$")
    for r in rows:
        if "error" in r:
            print(f"{r['query']:45.45}  ERROR  ({r['error'][:30]})")
            continue
        ragas = f"{r['ragas_overall']:.2f}" if r["ragas_overall"] is not None else "  - "
        print(f"{r['query']:45.45}  {r['mean_faithfulness']:.2f}   {ragas}   {r['est_cost_usd']:.4f}")
    print("\naggregate:", json.dumps(agg))
    print(f"report written to {out_path / 'report.json'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the on-demand RAG quality eval.")
    parser.add_argument("--golden", default=None, help="Path to golden.json (default: committed set)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries")
    parser.add_argument("--out", default="eval-out", help="Output directory for report.json")
    args = parser.parse_args()

    from app.rag.evaluation.retrieval_eval import GOLDEN_PATH

    golden = args.golden or str(GOLDEN_PATH)
    raise SystemExit(asyncio.run(_main(golden, args.limit, args.out)))


if __name__ == "__main__":
    main()
