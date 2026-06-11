"""Briefr API entrypoint.

FastAPI app with environment-aware CORS, per-identity rate limiting, API-key
authentication on every API route, an async lifespan that prepares Postgres
tables and verifies the ChromaDB server is reachable, and the v1 routers.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

import app.models  # noqa: F401  -- register ORM models on Base.metadata
from app.api.routes import briefs, documents, eval as eval_routes
from app.core.config import settings
from app.core.database import engine
from app.core.logging import configure_logging, request_id_var
from app.core.ratelimit import limiter
from app.core.redis_client import close_redis, get_redis
from app.core.security import require_api_key
from app.core.vector_db import chroma_heartbeat

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Schema is owned by Alembic (P2.1): migrations run as a one-shot `migrate`
    # step before the app starts (see docker-compose). The app no longer calls
    # create_all — it must not diverge from the migration history.
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


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation id to every request (P2.2): read an inbound
    ``X-Request-ID`` or mint one, expose it in logs + the response header, and
    log a structured access line with latency."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        # Log inside the contextvar scope so the access line carries request_id.
        response.headers["X-Request-ID"] = rid
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Deep readiness probe (P2.2): verify Postgres, Redis, and Chroma are
    reachable. Returns 503 if any dependency is down."""
    checks: dict[str, str] = {}

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {type(exc).__name__}"

    try:
        await get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {type(exc).__name__}"

    try:
        await chroma_heartbeat()
        checks["chroma"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["chroma"] = f"error: {type(exc).__name__}"

    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "degraded", "checks": checks},
        status_code=200 if ready else 503,
    )
