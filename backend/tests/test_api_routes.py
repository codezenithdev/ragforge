"""API route security & hardening tests (P0.1, P0.2, P0.4).

Exercised through the real ASGI app via httpx, with the Postgres session
dependency replaced by an in-memory fake so no services are required. Auth is
enabled by setting ``settings.api_key`` for the duration of each test.
"""

from __future__ import annotations

import uuid
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

    def add(self, obj: object = None, *_a, **_k) -> None:
        # Simulate SQLAlchemy applying Python-side column defaults at flush time
        # (PK uuid, status enum, num_chunks, ...) so post-commit reads work.
        if obj is None:
            return
        try:
            from sqlalchemy import inspect as sa_inspect

            for col in sa_inspect(type(obj)).mapper.columns:
                if getattr(obj, col.key, None) is not None or col.default is None:
                    continue
                value = col.default.arg
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = value(None)
                setattr(obj, col.key, value)
        except Exception:
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


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


async def test_create_brief_is_idempotent_per_key(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P2.6: same Idempotency-Key -> same brief, enqueued only once.
    _use_session(_FakeSession(scalar=0))
    fake_redis = _FakeRedis()
    monkeypatch.setattr("app.api.routes.briefs.get_redis", lambda: fake_redis)
    enqueues: list = []

    class _StubTask:
        def delay(self, *args, **kwargs) -> None:
            enqueues.append(args)

    monkeypatch.setattr("app.api.routes.briefs.generate_brief_task", _StubTask())

    headers = {**AUTH, "Idempotency-Key": "abc-123"}
    first = await client.post("/api/v1/briefs", json={"query": "same question"}, headers=headers)
    second = await client.post("/api/v1/briefs", json={"query": "same question"}, headers=headers)

    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["brief_id"] == second.json()["brief_id"]
    assert len(enqueues) == 1  # the replay did not enqueue a second pipeline


# --------------------------------------------------------------------------- #
# P0.4 — upload hardening
# --------------------------------------------------------------------------- #
async def test_upload_accepts_valid_pdf_and_enqueues(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slim async upload (B3): validate + stage + enqueue, returns 202 pending.
    _use_session(_FakeSession())
    calls: list[tuple] = []

    class _StubTask:
        def delay(self, *args, **kwargs) -> None:
            calls.append(args)

    monkeypatch.setattr("app.api.routes.documents.ingest_document_task", _StubTask())

    files = {"file": ("doc.pdf", b"%PDF-1.4 minimal pdf content here", "application/pdf")}
    resp = await client.post("/api/v1/documents/upload", files=files, headers=AUTH)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    # Ingestion was enqueued with (doc_id, staged_path, source_type).
    assert len(calls) == 1
    assert calls[0][0] == body["document_id"]
    assert calls[0][2] == "pdf"


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
