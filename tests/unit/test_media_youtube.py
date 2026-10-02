"""Unit tests for yt-dlp option construction and YouTubeResolver (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from notamusicbot.media.youtube import (
    DESKTOP_UA,
    YouTubeResolutionError,
    YouTubeResolver,
    build_yt_opts,
    discover_js_runtimes,
)


class FakeYdl:
    def __init__(self, results: dict[str, object], fail_clients: set[str] | None = None) -> None:
        self.results = results
        self.fail_clients = fail_clients or set()
        self.seen_clients: list[str] = []
        self.current: dict | None = None

    def __call__(self, opts: dict) -> FakeYdl:
        client = opts["extractor_args"]["youtube"]["player_client"][0]
        self.seen_clients.append(client)
        self.current = opts
        return self

    def __enter__(self) -> FakeYdl:
        return self

    def __exit__(self, *exc: object) -> None:
        if (
            self.current
            and self.current["extractor_args"]["youtube"]["player_client"][0] in self.fail_clients
        ):
            raise RuntimeError("client failed")

    def extract_info(self, url: str, download: bool = False) -> object:
        if (
            self.current
            and self.current["extractor_args"]["youtube"]["player_client"][0] in self.fail_clients
        ):
            raise RuntimeError("extraction failed")
        result = self.results.get(url)
        if result is None:
            raise RuntimeError("no result")
        return result


def test_build_yt_opts_sets_client_headers_and_js_runtimes() -> None:
    opts = build_yt_opts("web", js_runtimes={"deno": {"path": "/tmp/deno"}})
    assert opts["extractor_args"]["youtube"]["player_client"] == ["web"]
    assert opts["http_headers"]["User-Agent"] == DESKTOP_UA
    assert opts["js_runtimes"] == {"deno": {"path": "/tmp/deno"}}
    assert opts["quiet"] is True and opts["noplaylist"] is True


def test_build_yt_opts_attaches_cookies_only_when_file_exists(tmp_path: Path) -> None:
    cookies = tmp_path / "youtube_cookies.txt"
    # missing file -> anonymous extraction, graceful fallback
    assert "cookiefile" not in build_yt_opts("default", cookies_file=cookies)
    cookies.write_text("# Netscape HTTP Cookie File\n")
    assert build_yt_opts("default", cookies_file=cookies)["cookiefile"] == str(cookies)


def test_build_yt_opts_without_js_runtimes_discovers(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
    opts = build_yt_opts("default")
    assert opts["js_runtimes"] == {"node": {"path": "/usr/bin/node"}}


def test_discover_js_runtimes_prefers_explicit_deno(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    deno = tmp_path / "deno"
    deno.write_text("#!/bin/sh\n")
    deno.chmod(0o755)
    runtimes = discover_js_runtimes({"deno": deno})
    assert runtimes["deno"]["path"] == str(deno)


def test_discover_js_runtimes_ignores_missing_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    runtimes = discover_js_runtimes({"deno": tmp_path / "nope"})
    assert runtimes == {}


def test_resolve_returns_stream_media_with_headers() -> None:
    info = {
        "title": "Song",
        "webpage_url": "https://youtube.com/watch?v=1",
        "url": "https://stream",
        "http_headers": {"User-Agent": "x"},
    }
    ydl = FakeYdl({"https://youtube.com/watch?v=1": info})
    resolver = YouTubeResolver(ydl_factory=ydl)
    media = resolver_sync(resolver, "https://youtube.com/watch?v=1")
    assert media.kind.value == "stream"
    assert media.title == "Song"
    assert media.stream_url == "https://stream"
    assert media.stream_headers == {"User-Agent": "x"}


def resolver_sync(resolver: YouTubeResolver, url: str):
    import asyncio

    return asyncio.run(resolver.resolve(url))


def test_resolve_multi_client_fallback() -> None:
    info = {"title": "t", "webpage_url": "u", "url": "s"}
    ydl = FakeYdl({"u": info}, fail_clients={"default", "web"})
    resolver = YouTubeResolver(clients=("default", "web", "tv_embedded"), ydl_factory=ydl)
    media = resolver_sync(resolver, "u")
    assert media.stream_url == "s"
    assert ydl.seen_clients == ["default", "web", "tv_embedded"]


def test_resolve_ytsearch_takes_first_entry() -> None:
    info = {"entries": [None, {"title": "Hit", "webpage_url": "u", "url": "s"}]}
    ydl = FakeYdl({"ytsearch1:artist song": info})
    resolver = YouTubeResolver(ydl_factory=ydl)
    media = resolver_sync(resolver, "ytsearch1:artist song")
    assert media.title == "Hit"


def test_resolve_raises_when_no_stream_url() -> None:
    ydl = FakeYdl({"u": {"title": "t", "webpage_url": "u", "url": ""}})
    resolver = YouTubeResolver(ydl_factory=ydl)
    with pytest.raises(YouTubeResolutionError):
        resolver_sync(resolver, "u")


def test_resolve_raises_when_all_clients_fail() -> None:
    ydl = FakeYdl({}, fail_clients={"default", "web", "tv_embedded"})
    resolver = YouTubeResolver(clients=("default", "web", "tv_embedded"), ydl_factory=ydl)
    with pytest.raises(RuntimeError):
        resolver_sync(resolver, "u")
