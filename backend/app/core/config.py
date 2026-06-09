"""Application configuration.

All secrets, model identifiers, and tunable thresholds are defined here and
sourced from environment variables / the ``.env`` file. Nothing in the rest of
the codebase should hardcode an API key, model name, or threshold (project
rule #5). Later phases add their own stage-specific knobs (RRF k, CRAG and
faithfulness cutoffs, semantic-chunk threshold) to this same Settings object.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets / external services ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""

    # --- Infrastructure ---
    database_url: str = "postgresql+asyncpg://briefr:briefr@localhost:5432/briefr"
    redis_url: str = "redis://localhost:6379"
    environment: str = "development"

    # --- Vector store (ChromaDB, server mode) ---
    # Local-dev default targets the host-published port (8001 -> container 8000).
    # In compose the backend overrides chroma_host=chroma, chroma_port=8000.
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "briefr_chunks"

    # --- Embeddings (text-embedding-3-small => 1536 dims) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100

    # --- Cross-encoder re-ranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Claude models (latest IDs; generation uses the stronger model) ---
    generation_model: str = "claude-sonnet-4-6"
    subtask_model: str = "claude-haiku-4-5"

    # --- Retrieval / chunking knobs ---
    top_k_retrieval: int = 50
    top_k_rerank: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Semantic chunking (local model for sentence-adjacency similarity) ---
    # Split at adjacent-sentence distance outliers above this percentile. This
    # adapts per-document and is robust across embedding models; a fixed cosine
    # threshold over-splits badly with MiniLM.
    chunker_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_breakpoint_percentile: float = 90.0


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` singleton."""
    return Settings()


# Convenience module-level singleton for direct imports.
settings: Settings = get_settings()
