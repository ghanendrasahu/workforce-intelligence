"""Security tests: API-key gate + CORS allowlist."""
from __future__ import annotations

import pytest


@pytest.fixture()
def _keys_enabled(monkeypatch: pytest.MonkeyPatch):
    from core.config import get_settings

    monkeypatch.setenv("API_KEYS", '["ci-key-123"]')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def _no_keys(monkeypatch: pytest.MonkeyPatch):
    from core.config import get_settings

    monkeypatch.delenv("API_KEYS", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_health_always_public(_keys_enabled, api_client) -> None:
    assert api_client.get("/health").status_code == 200


def test_api_401_without_key(_keys_enabled, api_client) -> None:
    resp = api_client.get("/api/v1/employees/search", params={"q": "", "limit": 1})
    assert resp.status_code == 401
    assert "API key" in resp.json()["detail"]


def test_api_401_with_wrong_key(_keys_enabled, api_client) -> None:
    resp = api_client.get(
        "/api/v1/employees/search", params={"q": "", "limit": 1},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_api_200_with_valid_key(_keys_enabled, api_client) -> None:
    resp = api_client.get(
        "/api/v1/employees/search", params={"q": "", "limit": 1},
        headers={"X-API-Key": "ci-key-123"},
    )
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_api_open_when_no_keys(_no_keys, api_client) -> None:
    resp = api_client.get("/api/v1/employees/search", params={"q": "", "limit": 1})
    assert resp.status_code == 200


def test_cors_default_allows_dashboard_origin(api_client) -> None:
    resp = api_client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_rejects_unknown_origin(api_client) -> None:
    resp = api_client.get("/health", headers={"Origin": "http://evil.example"})
    assert resp.headers.get("access-control-allow-origin") is None


def test_env_list_parsing() -> None:
    from core.config import Settings

    parsed = Settings(
        _env_file=None,  # type: ignore[call-arg]
        cors_origins="http://a.example, http://b.example",  # type: ignore[arg-type]
        api_keys="",  # type: ignore[arg-type]
    )
    assert parsed.cors_origins == ["http://a.example", "http://b.example"]
    assert parsed.api_keys == []
