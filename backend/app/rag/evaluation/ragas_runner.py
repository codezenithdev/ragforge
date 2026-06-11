"""Batch RAG evaluation via RAGAS.

Wraps RAGAS (0.2) metrics — faithfulness, answer relevancy, context precision,
and (when a ground truth is supplied) context recall — wired to Claude as the
judge LLM and OpenAI for embeddings. Used by the ``/eval`` route to score a
completed brief. RAGAS' ``evaluate`` is synchronous, so it runs in a thread.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    overall: float = 0.0
    raw: dict[str, float] = field(default_factory=dict)


class RAGASRunner:
    def __init__(self) -> None:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        self._llm = LangchainLLMWrapper(
            ChatAnthropic(
                model=settings.subtask_model,
                anthropic_api_key=settings.anthropic_api_key,
                max_tokens=1024,
                timeout=60,
            )
        )
        self._embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        )

    def _evaluate_sync(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> dict[str, float]:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            LLMContextPrecisionWithReference,
            answer_relevancy,
            context_recall,
            faithfulness,
        )

        # faithfulness + answer_relevancy need no reference. Context precision has
        # two variants: with-reference (needs ground truth) and without-reference
        # (uses the response). Context recall needs a reference.
        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics += [LLMContextPrecisionWithReference(), context_recall]
        else:
            metrics.append(LLMContextPrecisionWithoutReference())

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
        )
        row = result.to_pandas().iloc[0].to_dict()
        # Keep only numeric metric columns.
        return {
            k: float(v)
            for k, v in row.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASResult:
        scores = await asyncio.to_thread(
            self._evaluate_sync, question, answer, contexts, ground_truth
        )
        numeric = [v for v in scores.values() if v == v]  # drop NaN
        overall = sum(numeric) / len(numeric) if numeric else 0.0

        def pick(*needles: str) -> float | None:
            # Metric column names vary by variant (e.g. llm_context_precision_without_reference).
            for key, value in scores.items():
                if any(n in key for n in needles):
                    return value
            return None

        logger.info("RAGASRunner: scores=%s overall=%.3f", scores, overall)
        return RAGASResult(
            faithfulness=pick("faithfulness"),
            answer_relevancy=pick("answer_relevancy", "relevancy"),
            context_precision=pick("precision"),
            context_recall=pick("recall"),
            overall=overall,
            raw=scores,
        )
