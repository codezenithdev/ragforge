"""Chunkers.

``SemanticChunker`` (primary): split text into sentences (nltk), embed adjacent
sentences with a *local* sentence-transformer, and start a new chunk wherever the
adjacent cosine similarity drops below a threshold — yielding semantically
coherent chunks. ``RecursiveChunker`` (fallback): deterministic word-window
chunking with overlap.

Both expose ``chunk(text, metadata) -> list[dict]`` and ``chunk_blocks(blocks)``
(which re-indexes ``chunk_index`` globally across a document's blocks). Chunking
is CPU-bound and synchronous; async callers should wrap calls in
``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def _ensure_nltk() -> None:
    """Make sure the nltk sentence tokenizer data is available."""
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:  # noqa: BLE001 -- punkt_tab may not exist on older nltk
                pass


class _BaseChunker:
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def chunk_blocks(
        self, blocks: list[dict[str, Any]], content_key: str = "content"
    ) -> list[dict[str, Any]]:
        """Chunk every loader block, carrying block metadata onto each chunk and
        assigning a document-global ``chunk_index``."""
        out: list[dict[str, Any]] = []
        running = 0
        for block in blocks:
            block_meta = {k: v for k, v in block.items() if k != content_key}
            for ch in self.chunk(block.get(content_key, ""), block_meta):
                ch["metadata"]["chunk_index"] = running
                running += 1
                out.append(ch)
        return out

    @staticmethod
    def _make(content: str, metadata: dict[str, Any], idx: int) -> dict[str, Any]:
        md = dict(metadata)
        md["chunk_index"] = idx
        return {"content": content, "metadata": md}


class SemanticChunker(_BaseChunker):
    def __init__(
        self, breakpoint_percentile: float | None = None, model_name: str | None = None
    ) -> None:
        from sentence_transformers import SentenceTransformer

        _ensure_nltk()
        self.breakpoint_percentile = (
            settings.semantic_chunk_breakpoint_percentile
            if breakpoint_percentile is None
            else breakpoint_percentile
        )
        self.model = SentenceTransformer(model_name or settings.chunker_embedding_model)
        # Soft cap so a long low-variance region cannot grow into one giant chunk.
        self._max_chars = settings.chunk_size * 4

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        import nltk

        metadata = dict(metadata or {})
        text = (text or "").strip()
        if not text:
            return []

        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        if len(sentences) <= 1:
            return [self._make(text, metadata, 0)]

        embeddings = self.model.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )

        # Distance (1 - cosine) between each adjacent sentence pair. Split at the
        # boundaries whose distance exceeds the per-document percentile breakpoint
        # — this groups coherent runs and cuts at topic shifts, adapting to the
        # document instead of relying on a model-specific absolute threshold.
        distances = [
            1.0 - float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(sentences) - 1)
        ]
        breakpoint = float(np.percentile(distances, self.breakpoint_percentile))

        chunk_texts: list[str] = []
        current = [sentences[0]]
        for i, dist in enumerate(distances):
            current_len = sum(len(s) for s in current)
            if dist > breakpoint or current_len >= self._max_chars:
                chunk_texts.append(" ".join(current))
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        if current:
            chunk_texts.append(" ".join(current))

        result = [self._make(t, metadata, idx) for idx, t in enumerate(chunk_texts)]
        logger.info(
            "SemanticChunker: %d sentences -> %d chunks (p%g breakpoint=%.3f)",
            len(sentences),
            len(result),
            self.breakpoint_percentile,
            breakpoint,
        )
        return result


class RecursiveChunker(_BaseChunker):
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        metadata = dict(metadata or {})
        words = (text or "").split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[dict[str, Any]] = []
        idx = 0
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(self._make(" ".join(window), metadata, idx))
            idx += 1
            if start + self.chunk_size >= len(words):
                break
        logger.info(
            "RecursiveChunker: %d words -> %d chunks (size=%d, overlap=%d)",
            len(words),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
