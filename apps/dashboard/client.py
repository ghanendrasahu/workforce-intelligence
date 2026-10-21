"""Thin httpx client wrapper around the Workforce Intelligence API."""
from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

API_KEY = os.environ.get("API_KEY", "")
DEFAULT_HEADERS: dict[str, str] = {"X-API-Key": API_KEY} if API_KEY else {}


class ApiError(RuntimeError):
    pass


def get(path: str, params: dict | None = None,
        base_url: str = DEFAULT_BASE_URL) -> Any:
    try:
        resp = httpx.get(f"{base_url}{path}", params=params, timeout=60,
                         headers=DEFAULT_HEADERS)
    except httpx.ConnectError as exc:
        raise ApiError(f"API not reachable at {base_url}. "
                       f"Start it with: uvicorn apps.api.app.main:app") from exc
    if resp.status_code >= 400:
        raise ApiError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def post(path: str, payload: dict, base_url: str = DEFAULT_BASE_URL) -> Any:
    try:
        resp = httpx.post(f"{base_url}{path}", json=payload, timeout=120,
                          headers=DEFAULT_HEADERS)
    except httpx.ConnectError as exc:
        raise ApiError(f"API not reachable at {base_url}. "
                       f"Start it with: uvicorn apps.api.app.main:app") from exc
    if resp.status_code >= 400:
        raise ApiError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()
