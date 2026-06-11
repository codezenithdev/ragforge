"""Briefr API entrypoint.

FastAPI app with environment-aware CORS, per-identity rate limiting, API-key
authentication on every API route, an async lifespan that prepares Postgres
tables and verifies the ChromaDB server is reachable, and the v1 routers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import app.models  # noqa: F401  -- register ORM models on Base.metadata
from app.api.routes import briefs, documents, eval as eval_routes
from app.core.config import settings
from app.core.database import init_models
from app.core.ratelimit import limiter
from app.core.redis_client import close_redis
from app.core.security import require_api_key
from app.core.vector_db import chroma_heartbeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await init_models()
    await chroma_heartbeat()
    logger.info("Briefr API ready (environment=%s)", settings.environment)
    yield
    await close_redis()


# Gate the interactive docs / OpenAPI schema in production (P0.1): they expose
# the full API surface, so don't serve them on an internet-facing deployment.
_docs_kwargs: dict[str, Any] = (
    {"docs_url": None, "redoc_url": None, "openapi_url": None} if settings.is_production else {}
)

app = FastAPI(title="Briefr", version="0.1.0", lifespan=lifespan, **_docs_kwargs)

# --- Rate limiting (P0.2) ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- CORS (P0.3) ---
# config.model_post_init refuses to boot in production with a '*' allowlist, so
# by the time we get here the list is safe to apply verbatim.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication (P0.1) ---
# Applied as a router-level dependency so every API endpoint is guarded without
# each route opting in. /health stays open for load-balancer probes.
_auth = [Depends(require_api_key)]
app.include_router(documents.router, prefix="/api/v1", dependencies=_auth)
app.include_router(briefs.router, prefix="/api/v1", dependencies=_auth)
app.include_router(eval_routes.router, prefix="/api/v1", dependencies=_auth)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "environment": settings.environment}
