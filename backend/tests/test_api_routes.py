"""API route security & hardening tests (P0.1, P0.2, P0.4).

Exercised through the real ASGI app via httpx, with the Postgres session
dependency replaced by an in-memory fake so no services are required. Auth is
enabled by setting ``settings.api_key`` for the duration of each test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.database import get_db
from app.main import app

API_KEY = "test-secret-key"
AUTH = {"X-API-Key": API_KEY}


# --------------------------------------------------------------------------- #
# Fakes & fixtures
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, items: list) -> None:
        self._items = items

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._items


class _FakeSession:
    """Minimal async-session stand-in covering the calls the routes make."""

    def __init__(self, *, scalar: int = 0, rows: list | None = None) -> None:
        self._scalar = scalar
        self._rows = rows or []

    async def execute(self, *_a, **_k) -> _FakeResult:
        return _FakeResult(self._rows)

    async def scalar(self, *_a, **_k) -> int:
        return self._scalar

    async def get(self, *_a, **_k):
        return None

    def add(self, *_a, **_k) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def delete(self, *_a, **_k) -> None:
        pass


def _use_session(session: _FakeSession) -> None:
    async def _override() -> AsyncIterator[_FakeSession]:
        yield session

    app.dependency_overrides[get_db] = _override


@pytest.fixture(autouse=True)
def _auth_enabled():
    """Enable API-key auth for the test, restoring prior config afterwards."""
    prev_key = settings.api_key
    settings.api_key = API_KEY
    yield
    settings.api_key = prev_key
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --------------------------------------------------------------------------- #
# P0.1 — authentication
# --------------------------------------------------------------------------- #
async def test_health_is_open(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_protected_route_rejects_missing_key(client: AsyncClient) -> None:
    _use_session(_FakeSession(rows=[]))
    resp = await client.get("/api/v1/documents")
    assert resp.status_code == 401


async def test_protected_route_rejects_wrong_key(client: AsyncClient) -> None:
    _use_session(_FakeSession(rows=[]))
    resp = await client.get("/api/v1/documents", headers={"X-API-Key": "nope"})
    assert resp.status_code == 401


async def test_protected_route_accepts_valid_key(client: AsyncClient) -> None:
    _use_session(_FakeSession(rows=[]))
    resp = await client.get("/api/v1/documents", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == []


# --------------------------------------------------------------------------- #
# P0.2 — brief spend controls
# --------------------------------------------------------------------------- #
async def test_create_brief_rejected_when_concurrency_cap_hit(client: AsyncClient) -> None:
    # scalar() returns the in-flight count; force it to the cap.
    _use_session(_FakeSession(scalar=settings.max_concurrent_briefs))
    resp = await client.post(
        "/api/v1/briefs", json={"query": "a real query"}, headers=AUTH
    )
    assert resp.status_code == 429
    assert "in flight" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# P0.4 — upload hardening
# --------------------------------------------------------------------------- #
async def test_upload_rejects_mismatched_magic_bytes(client: AsyncClient) -> None:
    _use_session(_FakeSession())
    files = {"file": ("evil.pdf", b"this is not really a pdf", "application/pdf")}
    resp = await client.post("/api/v1/documents/upload", files=files, headers=AUTH)
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]


async def test_upload_rejects_oversized_file(client: AsyncClient) -> None:
    _use_session(_FakeSession())
    prev = settings.max_upload_bytes
    settings.max_upload_bytes = 8  # tiny cap
    try:
        files = {"file": ("big.pdf", b"%PDF-1.4 and then some more bytes", "application/pdf")}
        resp = await client.post("/api/v1/documents/upload", files=files, headers=AUTH)
        assert resp.status_code == 413
    finally:
        settings.max_upload_bytes = prev


async def test_upload_rejects_unsupported_extension(client: AsyncClient) -> None:
    _use_session(_FakeSession())
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    resp = await client.post("/api/v1/documents/upload", files=files, headers=AUTH)
    assert resp.status_code == 400
    assert "unsupported file type" in resp.json()["detail"]
