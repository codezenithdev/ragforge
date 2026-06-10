"""Production configuration fail-fast tests (P0.3 / P0.5).

``Settings.model_post_init`` must refuse to construct an insecure production
configuration: a wildcard CORS policy, a missing API key, or a missing required
service key. Development stays permissive.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings

_VALID_PROD = dict(
    environment="production",
    cors_origins="https://app.example.com",
    api_key="a-real-key",
    anthropic_api_key="a",
    openai_api_key="o",
    tavily_api_key="t",
)


def test_valid_production_config_constructs() -> None:
    settings = Settings(**_VALID_PROD)
    assert settings.is_production
    assert settings.cors_origin_list == ["https://app.example.com"]


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(RuntimeError, match="cors_origins"):
        Settings(**{**_VALID_PROD, "cors_origins": "*"})


def test_production_rejects_missing_api_key() -> None:
    with pytest.raises(RuntimeError, match="api_key"):
        Settings(**{**_VALID_PROD, "api_key": ""})


def test_production_rejects_missing_service_key() -> None:
    with pytest.raises(RuntimeError, match="anthropic_api_key"):
        Settings(**{**_VALID_PROD, "anthropic_api_key": ""})


def test_development_is_permissive() -> None:
    # No keys, wildcard CORS — all fine in development.
    settings = Settings(
        environment="development",
        cors_origins="*",
        api_key="",
        anthropic_api_key="",
        openai_api_key="",
        tavily_api_key="",
    )
    assert not settings.is_production
