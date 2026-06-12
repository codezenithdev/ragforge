"""Per-section faithfulness scoring (LLM-as-judge).

For each section of a generated brief, asks the cheaper model to estimate what
fraction of the section's factual claims are actually supported by its cited
source passages — a real grounding signal, not a guess. Sections are scored in
parallel; each section's ``confidence`` is filled in place and a
``{section_key: score}`` map is returned for storage on the brief.
"""

from __future__ import annotations

import asyncio
import logging

import instructor
from pydantic import BaseModel, Field

from app.core.anthropic_client import get_anthropic_client, record_anthropic_usage
from app.core.config import settings
from app.rag.generation.context_format import UNTRUSTED_DATA_NOTE, wrap_untrusted
from app.rag.generation.schemas import BriefOutput, BriefSection
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You evaluate faithfulness. Given a SECTION of a research brief and its SOURCE "
    "passages, estimate the fraction (0.0-1.0) of the factual claims in SECTION that "
    "are directly supported by SOURCES. 1.0 = every claim is grounded; 0.0 = the "
    "claims are unsupported or contradicted. Judge only against the sources provided. "
    + UNTRUSTED_DATA_NOTE
)


class _FaithScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of supported claims")
    justification: str = Field(default="", description="One-sentence rationale")


class FaithfulnessScorer:
    def __init__(self) -> None:
        self._anthropic = get_anthropic_client()
        self._instructor = instructor.from_anthropic(self._anthropic)
        self.model = settings.subtask_model
        self.warn_threshold = settings.faithfulness_warn_threshold

    async def score_section(self, section_content: str, source_chunks: list[str]) -> float:
        if not section_content.strip() or not source_chunks:
            return 0.0
        # Wrap each source as an untrusted, delimited block (P3.2) so injected
        # directives in a web/document chunk can't manipulate the judge's score.
        sources = "\n\n".join(wrap_untrusted(t, i + 1) for i, t in enumerate(source_chunks))
        result, completion = await self._instructor.messages.create_with_completion(
            model=self.model,
            max_tokens=256,
            system=_JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"SECTION:\n{section_content}\n\nSOURCES:\n{sources}\n\n"
                        "What fraction of the factual claims in SECTION are directly "
                        "supported by SOURCES?"
                    ),
                }
            ],
            response_model=_FaithScore,
        )
        record_anthropic_usage(getattr(completion, "usage", None), self.model)
        return max(0.0, min(1.0, float(result.score)))

    async def score_brief(
        self, brief: BriefOutput, chunks: list[ScoredChunk]
    ) -> dict[str, float]:
        # Citation number (1-based) -> chunk content, matching the generator's numbering.
        idx_to_content = {str(i + 1): c.content for i, c in enumerate(chunks)}

        def sources_for(section: BriefSection) -> list[str]:
            # No fallback to the full context: a section whose citations don't
            # resolve (uncited or hallucinated ids) is judged against an EMPTY
            # evidence set -> score 0.0. The old `or all_contents` fallback gave
            # the least-grounded sections the most generous evidence, inflating
            # the very signal that's meant to flag them.
            return [idx_to_content[s] for s in section.sources if s in idx_to_content]

        targets: list[tuple[str, BriefSection]] = [
            ("executive_summary", brief.executive_summary),
            ("risks_and_limitations", brief.risks_and_limitations),
            ("opportunities", brief.opportunities),
        ]
        targets += [(f"key_fact_{i + 1}", kf) for i, kf in enumerate(brief.key_facts)]

        section_sources = [sources_for(sec) for _, sec in targets]
        scores = await asyncio.gather(
            *(
                self.score_section(sec.content, src)
                for (_, sec), src in zip(targets, section_sources, strict=False)
            )
        )

        result: dict[str, float] = {}
        low: list[str] = []
        uncited: list[str] = []
        for (key, section), src, score in zip(targets, section_sources, scores, strict=False):
            section.confidence = score
            result[key] = score
            if not src:
                uncited.append(key)  # citations didn't resolve -> forced 0.0, surfaced
            if score < self.warn_threshold:
                low.append(key)
        logger.info(
            "FaithfulnessScorer: scored %d sections, %d below warn threshold %.2f%s; "
            "%d uncited%s",
            len(targets),
            len(low),
            self.warn_threshold,
            f" ({', '.join(low)})" if low else "",
            len(uncited),
            f" ({', '.join(uncited)})" if uncited else "",
        )
        return result
