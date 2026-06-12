"""Shared async Anthropic client + per-brief usage accounting.

One process-wide ``anthropic.AsyncAnthropic`` (reused everywhere instead of a
new client per RAG component, mirroring ``redis_client``), plus a token-usage
accumulator threaded through a brief via a ``contextvars`` variable. Each LLM
call records its ``usage`` into the active accumulator; ``tasks.py`` resets it at
the start of a brief and logs the totals once at the end (P-telemetry).

The accumulator is a *mutable* object stored in the contextvar: ``asyncio.gather``
children copy the context (the reference), so increments from concurrent nodes
(HyDE fan-out, parallel faithfulness scoring) all land on the same object.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


def get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return the process-wide shared async Anthropic client (created on first use)."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        logger.info("Anthropic client initialized")
    return _client


@dataclass
class UsageTotals:
    """Aggregate token usage for one brief, with a per-model breakdown for cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def _bucket(self, model: str) -> dict[str, int]:
        return self.by_model.setdefault(
            model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        )

    def add_anthropic(self, usage: Any, model: str) -> None:
        if usage is None:
            return
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        c_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        c_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cache_read_input_tokens += c_read
        self.cache_creation_input_tokens += c_write
        b = self._bucket(model)
        b["input"] += in_tok
        b["output"] += out_tok
        b["cache_read"] += c_read
        b["cache_write"] += c_write

    def add_openai_embedding(self, usage: Any, model: str) -> None:
        """OpenAI embeddings report ``prompt_tokens`` (input only)."""
        if usage is None:
            return
        in_tok = getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0
        self.input_tokens += in_tok
        self._bucket(model)["input"] += in_tok

    def estimated_cost_usd(self) -> float:
        """Estimate $ from the per-model breakdown and ``settings.model_prices``.

        Cache reads cost ~0.1x input price; cache writes ~1.25x. Prices are
        $/1M tokens. Unknown models contribute 0 (logged-only estimate).
        """
        total = 0.0
        for model, b in self.by_model.items():
            price_in, price_out = settings.model_prices.get(model, (0.0, 0.0))
            total += (
                b["input"] * price_in
                + b["output"] * price_out
                + b["cache_read"] * price_in * 0.1
                + b["cache_write"] * price_in * 1.25
            ) / 1_000_000
        return round(total, 6)


# Active per-brief accumulator. ``None`` outside a brief (tests, ad-hoc calls) →
# recording is a no-op, so call sites can record unconditionally.
usage_var: contextvars.ContextVar[UsageTotals | None] = contextvars.ContextVar(
    "usage_totals", default=None
)


def record_anthropic_usage(usage: Any, model: str) -> None:
    totals = usage_var.get()
    if totals is not None:
        totals.add_anthropic(usage, model)


def record_openai_embedding_usage(usage: Any, model: str) -> None:
    totals = usage_var.get()
    if totals is not None:
        totals.add_openai_embedding(usage, model)
