"""Unit tests for the news fetcher port (no network; httpx is monkeypatched)."""

from __future__ import annotations

import httpx
import pytest

from notamusicbot.content.news import NewsFetcher


def _fake_response(data: dict) -> httpx.Response:
    return httpx.Response(200, json=data)


def test_missing_api_key_returns_none_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("network must not be touched without an API key")

    monkeypatch.setattr(httpx, "get", boom)
    import asyncio

    assert asyncio.run(NewsFetcher(None).get_random_article()) is None


def test_successful_fetch_returns_random_article(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "status": "ok",
        "articles": [
            {"title": "T1", "source": {"name": "S1"}, "description": "D1", "url": "u1"},
            {"title": "T2", "source": {"name": "S2"}, "description": "D2", "url": "u2"},
        ],
    }
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _fake_response(data)

    monkeypatch.setattr(httpx, "get", fake_get)
    import asyncio

    article = asyncio.run(NewsFetcher("secret").get_random_article(country="gb", category="tech"))
    assert article is not None and article.title in {"T1", "T2"}
    assert captured["params"]["apiKey"] == "secret"
    assert captured["params"]["country"] == "gb"


def test_failed_request_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, params=None, timeout=None):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", fake_get)
    import asyncio

    assert asyncio.run(NewsFetcher("secret").get_random_article()) is None


def test_error_status_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda url, params=None, timeout=None: _fake_response({"status": "error"})
    )
    import asyncio

    assert asyncio.run(NewsFetcher("secret").get_random_article()) is None
