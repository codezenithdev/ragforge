"""Application configuration.

All secrets, model identifiers, and tunable thresholds are defined here and
sourced from environment variables / the ``.env`` file. Nothing in the rest of
the codebase should hardcode an API key, model name, or threshold (project
rule #5). Later phases add their own stage-specific knobs (RRF k, CRAG and
faithfulness cutoffs, semantic-chunk threshold) to this same Settings object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The .env lives at the repo root. Resolve it from this file's location so the
# settings load regardless of the process CWD (uvicorn/celery run from
# backend/); real environment variables (e.g. injected by docker compose)
# always take precedence over the file. Missing candidates are skipped.
_REPO_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    """Typed settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT_ENV, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets / external services ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""

    # --- Infrastructure ---
    # Local default uses host port 5433 (the container's 5432 is remapped there
    # because a native PostgreSQL 17 owns host :5432). Compose overrides this to
    # postgres:5432 for in-network services.
    database_url: str = "postgresql+asyncpg://briefr:briefr@localhost:5433/briefr"
    redis_url: str = "redis://localhost:6379"
    environment: str = "development"
    cors_origins: str = "*"

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
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Semantic chunking (local model for sentence-adjacency similarity) ---
    # Split at adjacent-sentence distance outliers above this percentile. This
    # adapts per-document and is robust across embedding models; a fixed cosine
    # threshold over-splits badly with MiniLM.
    chunker_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_chunk_breakpoint_percentile: float = 90.0

    # --- Agentic pipeline (decomposition, HyDE, CRAG, corrective web search) ---
    num_sub_queries: int = 6
    # Top-chunk relevance probability (sigmoid of cross-encoder score) below which
    # CRAG triggers a corrective web search.
    crag_confidence_threshold: float = 0.3
    tavily_max_results: int = 5

    # --- Generation / evaluation ---
    # Sections scoring below this faithfulness value get a low-confidence flag.
    faithfulness_warn_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` singleton."""
    return Settings()


# Convenience module-level singleton for direct imports.
settings: Settings = get_settings()
