"""Query decomposition + HyDE.

Splits the user's question into focused, self-contained sub-questions (so
retrieval covers the topic from several angles), and for each writes a
Hypothetical Document (HyDE) — a short, made-up "ideal answer" paragraph. The
HyDE paragraph is embedded and used for retrieval instead of the bare question,
which surfaces semantically richer matches for abstract or technical queries.

Uses the cheaper sub-task model. Sub-question generation uses ``instructor`` for
structured output (a validated list); HyDE uses a plain text completion.
"""

from __future__ import annotations

import logging

import instructor
from pydantic import BaseModel, Field

from app.core.anthropic_client import get_anthropic_client, record_anthropic_usage
from app.core.config import settings

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "You are a research assistant that breaks a question into focused, "
    "self-contained sub-questions for document retrieval."
)

_HYDE_SYSTEM = (
    "You write a single concise, factual-sounding paragraph (3-5 sentences) that "
    "would be an ideal answer to a question. Write authoritatively even if unsure; "
    "this hypothetical passage is used only to improve document retrieval and is "
    "never shown to users. Output only the paragraph."
)


class _SubQuestions(BaseModel):
    questions: list[str] = Field(..., description="The list of specific sub-questions")


class QueryDecomposer:
    def __init__(self) -> None:
        self._anthropic = get_anthropic_client()
        self._instructor = instructor.from_anthropic(self._anthropic)
        self.model = settings.subtask_model
        self.n = settings.num_sub_queries

    async def decompose(self, query: str) -> list[str]:
        # System prompt is short (< cache minimum) — not cached intentionally.
        result, completion = await self._instructor.messages.create_with_completion(
            model=self.model,
            max_tokens=1024,
            system=_DECOMPOSE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Break the following research question into {self.n} specific, "
                        "self-contained sub-questions that together cover the topic "
                        "comprehensively. Each sub-question must be answerable on its own "
                        "(no pronouns referring to the others).\n\n"
                        f"Question: {query}"
                    ),
                }
            ],
            response_model=_SubQuestions,
        )
        record_anthropic_usage(getattr(completion, "usage", None), self.model)
        sub_queries = [q.strip() for q in result.questions if q.strip()][: self.n]
        logger.info("QueryDecomposer: '%s' -> %d sub-queries", query[:60], len(sub_queries))
        return sub_queries

    async def generate_hyde_document(self, sub_query: str) -> str:
        resp = await self._anthropic.messages.create(
            model=self.model,
            max_tokens=400,
            system=_HYDE_SYSTEM,
            messages=[
                {"role": "user", "content": f"Question: {sub_query}\n\nWrite the paragraph:"}
            ],
        )
        record_anthropic_usage(getattr(resp, "usage", None), self.model)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        logger.info("QueryDecomposer: HyDE for '%s' (%d chars)", sub_query[:50], len(text))
        return text
