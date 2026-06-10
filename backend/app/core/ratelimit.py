"""Per-identity request rate limiting (P0.2).

A Redis-backed slowapi ``Limiter`` keyed by the caller's API key (falling back
to client IP for the development-open path). Shared across all uvicorn/gunicorn
workers via Redis storage, so the limit is global rather than per-process.

The per-endpoint request limit lives here; the brief-specific spend controls
(global in-flight cap + daily ceiling) are enforced in ``briefs.create_brief``
against Postgres, which is the source of truth for brief state.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.core.config import settings
from app.core.security import API_KEY_HEADER


def identity_key(request: Request) -> str:
    """Rate-limit bucket: the API key if present, else the client IP."""
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    return f"ip:{get_remote_address(request)}"


# storage_uri points slowapi at Redis so the window is shared across workers.
# in-memory fallback (storage_uri=None) is used only if Redis is unreachable.
limiter = Limiter(
    key_func=identity_key,
    default_limits=[settings.rate_limit],
    storage_uri=settings.redis_url,
    headers_enabled=True,
    # Fail open: a Redis hiccup must not take the API down (and keeps the test
    # suite hermetic when Redis is absent).
    swallow_errors=True,
)
