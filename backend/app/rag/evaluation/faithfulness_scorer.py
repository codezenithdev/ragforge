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

import anthropic
import instructor
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.generation.schemas import BriefOutput, BriefSection
from app.rag.types import ScoredChunk

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You evaluate faithfulness. Given a SECTION of a research brief and its SOURCE "
    "passages, estimate the fraction (0.0-1.0) of the factual claims in SECTION that "
    "are directly supported by SOURCES. 1.0 = every claim is grounded; 0.0 = the "
    "claims are unsupported or contradicted. Judge only against the sources provided."
)


class _FaithScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of supported claims")
    justification: str = Field(default="", description="One-sentence rationale")


class FaithfulnessScorer:
    def __init__(self) -> None:
        self._anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._instructor = instructor.from_anthropic(self._anthropic)
        self.model = settings.subtask_model
        self.warn_threshold = settings.faithfulness_warn_threshold

    async def score_section(self, section_content: str, source_chunks: list[str]) -> float:
        if not section_content.strip() or not source_chunks:
            return 0.0
        sources = "\n\n".join(f"[Source {i + 1}] {t}" for i, t in enumerate(source_chunks))
        result: _FaithScore = await self._instructor.messages.create(
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
        return max(0.0, min(1.0, float(result.score)))

    async def score_brief(
        self, brief: BriefOutput, chunks: list[ScoredChunk]
    ) -> dict[str, float]:
        # Citation number (1-based) -> chunk content, matching the generator's numbering.
        idx_to_content = {str(i + 1): c.content for i, c in enumerate(chunks)}
        all_contents = [c.content for c in chunks]

        def sources_for(section: BriefSection) -> list[str]:
            texts = [idx_to_content[s] for s in section.sources if s in idx_to_content]
            return texts or all_contents  # fall back to full context if citations don't resolve

        targets: list[tuple[str, BriefSection]] = [
            ("executive_summary", brief.executive_summary),
            ("risks_and_limitations", brief.risks_and_limitations),
            ("opportunities", brief.opportunities),
        ]
        targets += [(f"key_fact_{i + 1}", kf) for i, kf in enumerate(brief.key_facts)]

        scores = await asyncio.gather(
            *(self.score_section(sec.content, sources_for(sec)) for _, sec in targets)
        )

        result: dict[str, float] = {}
        low = []
        for (key, section), score in zip(targets, scores):
            section.confidence = score
            result[key] = score
            if score < self.warn_threshold:
                low.append(key)
        logger.info(
            "FaithfulnessScorer: scored %d sections, %d below warn threshold %.2f%s",
            len(targets),
            len(low),
            self.warn_threshold,
            f" ({', '.join(low)})" if low else "",
        )
        return result
