"""Shared test setup.

Ensures the backend root is importable regardless of how pytest is invoked, and
provides a chunk factory used across test modules. All tests are fully mocked —
no API keys, Docker services, or network access are required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# backend/ (parent of tests/) must be on sys.path for `import app`.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.types import ScoredChunk  # noqa: E402


def make_chunk(
    chunk_id: str,
    content: str | None = None,
    *,
    vector_score: float | None = None,
    bm25_score: float | None = None,
    **metadata: Any,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content if content is not None else f"content of {chunk_id}",
        metadata=metadata,
        score=vector_score or bm25_score or 0.0,
        vector_score=vector_score,
        bm25_score=bm25_score,
    )
