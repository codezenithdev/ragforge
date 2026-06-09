"""Pydantic v2 schemas for the generated brief.

``BriefOutput`` is the public contract returned by the generator and stored on
the ``briefs`` row. Each ``BriefSection`` carries the source numbers it cites
(resolvable against the top-level ``sources`` list) and a ``confidence`` score
that the faithfulness scorer fills in after generation.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    id: str = Field(..., description="Citation number used inline by sections (e.g. '1')")
    source_type: str = Field(..., description="'document' or 'web'")
    title: str | None = None
    url: str | None = None


class BriefSection(BaseModel):
    content: str
    sources: list[str] = Field(
        default_factory=list, description="Citation numbers referenced in this section"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Per-section faithfulness score, filled after generation",
    )


class BriefOutput(BaseModel):
    title: str
    executive_summary: BriefSection
    key_facts: list[BriefSection] = Field(default_factory=list)
    risks_and_limitations: BriefSection
    opportunities: BriefSection
    open_questions: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    generated_at: datetime | None = None
