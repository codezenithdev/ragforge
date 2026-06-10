"""API authentication (P0.1).

A single shared secret supplied in the ``X-API-Key`` request header guards every
API endpoint. The check is wired in as a router-level FastAPI dependency (see
``app.main``), so it applies uniformly without each route opting in.

Development convenience: when ``settings.api_key`` is empty *and* the environment
is development, the check is disabled so local runs, the test suite, and the
frontend work without a key. Production refuses to boot without a key at all
(``Settings.model_post_init``), so the open path can only ever be reached in
development.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Security
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# auto_error=False: we raise our own 401 (with the right header advertised) and
# tolerate a missing header in the development-open path below.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

if not settings.api_key and not settings.is_production:
    logger.warning(
        "API_KEY is unset in development: API authentication is DISABLED. "
        "Set API_KEY to require the %s header.",
        API_KEY_HEADER,
    )


async def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency: authenticate the caller via the ``X-API-Key`` header.

    Raises 401 when authentication fails. No-ops when authentication is disabled
    (development with no configured key).
    """
    if not settings.api_key:
        # Disabled — only reachable in development (production boot would have
        # aborted on a missing key).
        return

    # Constant-time comparison to avoid leaking the key via timing.
    if provided is None or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
