"""Structured logging / correlation tests (P2.2)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from app.core.anthropic_client import (
    UsageTotals,
    record_anthropic_usage,
    usage_var,
)
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


def _fake_usage(inp: int, out: int, c_read: int = 0, c_write: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=c_read,
        cache_creation_input_tokens=c_write,
    )


def test_usage_totals_accumulate_and_estimate_cost() -> None:
    totals = UsageTotals()
    totals.add_anthropic(_fake_usage(1000, 500), "claude-sonnet-4-6")
    totals.add_anthropic(_fake_usage(200, 100), "claude-haiku-4-5")
    totals.add_openai_embedding(SimpleNamespace(prompt_tokens=50, total_tokens=50), "text-embedding-3-small")

    assert totals.input_tokens == 1250  # 1000 + 200 + 50
    assert totals.output_tokens == 600
    # Per-model cost: sonnet (1000*3 + 500*15) + haiku (200*1 + 100*5) + embed (50*0.02), /1e6
    expected = (1000 * 3 + 500 * 15 + 200 * 1 + 100 * 5 + 50 * 0.02) / 1_000_000
    assert totals.estimated_cost_usd() == round(expected, 6)
    assert set(totals.by_model) == {"claude-sonnet-4-6", "claude-haiku-4-5", "text-embedding-3-small"}


def test_record_usage_is_noop_without_active_accumulator() -> None:
    # Outside a brief the contextvar is None — recording must be a safe no-op.
    assert usage_var.get() is None
    record_anthropic_usage(_fake_usage(10, 10), "claude-sonnet-4-6")  # must not raise


def test_record_usage_lands_on_active_accumulator() -> None:
    totals = UsageTotals()
    token = usage_var.set(totals)
    try:
        record_anthropic_usage(_fake_usage(7, 3, c_read=2), "claude-haiku-4-5")
    finally:
        usage_var.reset(token)
    assert totals.input_tokens == 7
    assert totals.output_tokens == 3
    assert totals.cache_read_input_tokens == 2


def test_brief_usage_log_fields_serialize() -> None:
    record = logging.LogRecord("app.tasks", logging.INFO, __file__, 1, "brief usage", None, None)
    record.event = "brief_usage"
    record.tokens_in = 1250
    record.tokens_out = 600
    record.est_cost_usd = 0.012
    record.by_model = {"claude-sonnet-4-6": {"input": 1000, "output": 500}}

    out = json.loads(JsonFormatter().format(record))
    assert out["event"] == "brief_usage"
    assert out["tokens_in"] == 1250
    assert out["est_cost_usd"] == 0.012
    assert out["by_model"]["claude-sonnet-4-6"]["input"] == 1000
