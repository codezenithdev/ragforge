"""Structured logging / correlation tests (P2.2)."""

from __future__ import annotations

import json
import logging

from app.core.logging import JsonFormatter, request_id_var


def test_json_formatter_emits_request_id_and_extra_fields() -> None:
    token = request_id_var.set("rid-abc123")
    try:
        record = logging.LogRecord(
            "app.test", logging.INFO, __file__, 10, "http_request", None, None
        )
        record.method = "POST"
        record.path = "/api/v1/briefs"
        record.status = 202
        record.duration_ms = 12.3

        out = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert out["request_id"] == "rid-abc123"
    assert out["level"] == "INFO"
    assert out["msg"] == "http_request"
    # Fields passed via `extra=` are surfaced as structured keys.
    assert out["method"] == "POST"
    assert out["path"] == "/api/v1/briefs"
    assert out["status"] == 202
    assert out["duration_ms"] == 12.3


def test_request_id_defaults_to_dash_outside_a_request() -> None:
    # Fresh contextvar default (no request in scope).
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "boot", None, None)
    out = json.loads(JsonFormatter().format(record))
    assert out["request_id"] == "-"
