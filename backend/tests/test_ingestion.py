"""Ingestion tests: embedder input guards (P3.4) and the deterministic chunker.

The OpenAI client is never called — ``_embed_request`` is stubbed — so no key or
network is needed.
"""

from __future__ import annotations

from app.core.config import settings
from app.rag.ingestion.chunker import RecursiveChunker
from app.rag.ingestion.embedder import Embedder


async def test_embedder_sanitizes_empty_and_overlong_inputs() -> None:
    embedder = Embedder()
    captured: dict[str, list[str]] = {}

    async def _fake_request(texts: list[str]) -> list[list[float]]:
        captured["texts"] = texts
        return [[1.0, 0.0, 0.0] for _ in texts]

    embedder._embed_request = _fake_request  # type: ignore[method-assign]

    long_text = "a" * (settings.embedding_max_chars + 5000)
    out = await embedder.embed_batch(["hello world", "   ", "", long_text])

    sent = captured["texts"]
    assert sent[0] == "hello world"
    assert sent[1] == " " and sent[2] == " "  # empty/whitespace -> single space
    assert len(sent[3]) == settings.embedding_max_chars  # truncated to the cap
    # Output stays aligned 1:1 with the input list.
    assert len(out) == 4


async def test_embedder_returns_empty_for_no_input() -> None:
    assert await Embedder().embed_batch([]) == []


def test_recursive_chunker_windows_with_overlap_and_global_index() -> None:
    chunker = RecursiveChunker(chunk_size=5, chunk_overlap=2)
    chunks = chunker.chunk("one two three four five six seven eight nine ten")

    assert len(chunks) >= 2
    assert all(c["content"] for c in chunks)
    # chunk_index is assigned 0..n-1.
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
