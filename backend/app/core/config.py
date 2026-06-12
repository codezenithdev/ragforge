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

    # --- API authentication (P0.1) ---
    # Shared secret required in the ``X-API-Key`` header on every API endpoint.
    # In development an empty value disables the check (local/tests/frontend run
    # without friction); in any other environment an empty value is a fatal
    # boot error (see ``_validate_production``).
    api_key: str = ""

    # --- Rate limiting & spend controls (P0.2) ---
    # ``rate_limit`` is the per-identity request ceiling (slowapi string form)
    # applied to every endpoint by the middleware. The brief caps bound LLM
    # spend: ``max_concurrent_briefs`` limits in-flight (pending/processing)
    # briefs globally; ``daily_brief_limit`` limits briefs created per UTC day.
    # 0 disables a given cap.
    rate_limit: str = "60/minute"
    max_concurrent_briefs: int = 5
    daily_brief_limit: int = 200
    # How long an Idempotency-Key maps to its brief (P2.6); repeats within this
    # window return the same brief instead of creating (and billing) a new one.
    idempotency_ttl_seconds: int = 86400

    # --- Upload hardening (P0.4) ---
    # Reject uploads larger than this before reading them into memory, and cap
    # the page count parsed from a PDF.
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB
    max_pdf_pages: int = 200

    # --- Async ingestion (P1.6/P1.3) ---
    # Directory where the API stages uploaded files for the Celery ingestion
    # worker to read. Must be shared between the API and worker (a named volume
    # in compose). Empty => a per-host temp dir (fine when both run on one host).
    upload_dir: str = ""
    # Hard ceiling on a single ingestion run, enforced inside the task.
    ingest_timeout_seconds: int = 300
    # How often the reconcile sweeper purges orphaned vectors / fails stuck docs.
    reconcile_interval_seconds: int = 600
    # Delete briefs older than this many days (P2.3 retention). 0 disables pruning.
    brief_retention_days: int = 0

    # --- Celery task safety (P1.2) ---
    # Hard ceiling on a single brief pipeline, enforced inside the task via
    # asyncio.wait_for (SIGALRM soft limits are unreliable on Windows). Briefs
    # left in 'processing' past brief_stuck_seconds are failed by a sweeper.
    brief_timeout_seconds: int = 600
    brief_stuck_seconds: int = 1200
    brief_max_retries: int = 2
    # How often the stuck-brief sweeper runs.
    brief_sweep_interval_seconds: int = 300

    # --- Vector store (ChromaDB, server mode) ---
    # Local-dev default targets the host-published port (8001 -> container 8000).
    # In compose the backend overrides chroma_host=chroma, chroma_port=8000.
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "briefr_chunks"
    # Optional token (P0.6). When set, the client authenticates with the Chroma
    # server's token authn provider; empty disables it (local dev runs open).
    chroma_auth_token: str = ""

    # --- Embeddings (text-embedding-3-small => 1536 dims) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100
    # Truncate any single text longer than this before embedding (P3.4) — keeps it
    # under the model's ~8191-token limit (≈4 chars/token).
    embedding_max_chars: int = 30000

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

    # --- Cost telemetry ---
    # Per-model price ($/1M tokens) as (input, output). Used only to annotate the
    # per-brief usage log line with an estimate; not billing-accurate. Update when
    # model pricing changes. Embedding models price output at 0.
    model_prices: dict[str, tuple[float, float]] = {
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "text-embedding-3-small": (0.02, 0.0),
    }
    # Log a warning when a single brief's total tokens exceed this. 0 disables.
    brief_token_warn_threshold: int = 200_000

    # ------------------------------------------------------------------ #
    # Derived helpers & production validation
    # ------------------------------------------------------------------ #

    @property
    def is_production(self) -> bool:
        """True for any environment other than local development."""
        return self.environment.strip().lower() not in {"development", "dev", "local", "test"}

    @property
    def cors_origin_list(self) -> list[str]:
        """``cors_origins`` parsed into a clean list of origins."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def model_post_init(self, _context: object) -> None:
        """Fail fast on insecure production configuration (P0.3 / P0.5).

        In development everything is permitted so local runs, tests, and the
        frontend work without keys. In any other environment a missing API key,
        a wildcard CORS policy, or a missing required service key aborts boot
        instead of silently exposing the service / failing per-request.
        """
        if not self.is_production:
            return

        problems: list[str] = []
        if not self.api_key:
            problems.append("api_key is required (set API_KEY)")
        if "*" in self.cors_origin_list or not self.cors_origin_list:
            problems.append("cors_origins must be an explicit allowlist, not '*'")
        for name in ("anthropic_api_key", "openai_api_key", "tavily_api_key"):
            if not getattr(self, name):
                problems.append(f"{name} is required")
        if problems:
            raise RuntimeError(
                "Insecure production configuration (environment="
                f"{self.environment!r}): " + "; ".join(problems)
            )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` singleton."""
    return Settings()


# Convenience module-level singleton for direct imports.
settings: Settings = get_settings()
