"""Unit tests for the praw-based RedditPostFetcher (fakes only, no network)."""

from __future__ import annotations

import asyncio
import random
import sys
import types

import pytest

from notamusicbot.content.reddit import RedditPostFetcher


class _FakeAuthor:
    def __init__(self, name: str | None) -> None:
        self.name = name


def _fake_praw(posts: list, *, reddit_error: Exception | None = None) -> types.ModuleType:
    """Build a fake ``praw`` module whose Reddit records construction args."""

    class _FakeSubreddit:
        def __init__(self, fetcher: _FakeReddit, name: str) -> None:
            self._fetcher = fetcher
            self._name = name
            if reddit_error is not None and name == "boom":
                raise reddit_error

        def hot(self, limit: int = 100):
            return iter(posts)

    class _FakeReddit:
        instances: list[_FakeReddit] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.read_only = False
            self.front = _FakeSubreddit(fetcher := _FakeRedditHolder(self), "front")
            self._sub_cache: dict[str, _FakeSubreddit] = {}
            _FakeReddit.instances.append(self)
            self._holder = fetcher

        def subreddit(self, name: str) -> _FakeSubreddit:
            if name not in self._sub_cache:
                self._sub_cache[name] = _FakeSubreddit(self._holder, name)
            return self._sub_cache[name]

    class _FakeRedditHolder:
        def __init__(self, reddit: _FakeReddit) -> None:
            self.reddit = reddit

    module = types.ModuleType("praw")
    module.Reddit = _FakeReddit  # type: ignore[attr-defined]
    return module


def _post(
    pid: str, *, is_self: bool = True, author: str | None = "someone"
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id=pid,
        title=f"Title {pid}",
        url=f"https://reddit.com/{pid}",
        author=_FakeAuthor(author),
        selftext=f"Body {pid}" if is_self else "",
        is_self=is_self,
    )


def test_no_praw_installed_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "praw", None)
    fetcher = RedditPostFetcher("id", "secret")
    assert asyncio.run(fetcher.get_random_post()) is None


def test_missing_credentials_returns_none_without_praw_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    module = _fake_praw([_post("a")])
    original = module.Reddit

    class _Spy(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)
            super().__init__(**kwargs)

    module.Reddit = _Spy
    monkeypatch.setitem(sys.modules, "praw", module)
    fetcher = RedditPostFetcher("", "secret")
    assert asyncio.run(fetcher.get_random_post()) is None
    assert calls == []


def test_fetches_and_filters_self_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = [_post("a"), _post("b", is_self=False), _post("c", author=None)]
    module = _fake_praw(posts)
    monkeypatch.setitem(sys.modules, "praw", module)
    fetcher = RedditPostFetcher(
        "id", "secret", subreddits=["AskReddit"], post_limit=10, rng=random.Random(1)
    )
    first = asyncio.run(fetcher.get_random_post())
    assert first is not None
    # Only self posts survive; deleted authors are mapped to "unknown".
    seen_ids = {"a", "c"}
    assert first.post_id in seen_ids
    assert fetcher._pending  # a batch was cached, not just one post
    reddit_instance = module.Reddit.instances[0]  # type: ignore[attr-defined]
    assert reddit_instance.kwargs["client_id"] == "id"
    assert reddit_instance.kwargs["check_for_async"] is False
    assert reddit_instance.read_only is True


def test_praw_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _fake_praw([], reddit_error=RuntimeError("oauth down"))
    monkeypatch.setitem(sys.modules, "praw", module)
    fetcher = RedditPostFetcher("id", "secret", subreddits=["boom"])
    assert asyncio.run(fetcher.get_random_post()) is None


def test_batch_is_shuffled_and_exhausted_then_refilled(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = [_post(p) for p in ("a", "b", "c", "d")]
    module = _fake_praw(posts)
    monkeypatch.setitem(sys.modules, "praw", module)
    fetcher = RedditPostFetcher("id", "secret", rng=random.Random(7))
    drawn = [asyncio.run(fetcher.get_random_post()).post_id for _ in range(4)]
    assert sorted(drawn) == ["a", "b", "c", "d"]  # all served exactly once
    fifth = asyncio.run(fetcher.get_random_post())
    assert fifth is not None  # batch refilled (second fetch)


def test_from_settings_maps_fields() -> None:
    class _Settings:
        subreddits = ["r/askreddit", " tifu "]
        post_limit = 5
        user_agent = "ua"
        request_timeout_s = 3.0

    fetcher = RedditPostFetcher.from_settings("id", "secret", _Settings())
    assert fetcher._subreddits == ["askreddit", "tifu"]
    assert fetcher._post_limit == 5
    assert fetcher._user_agent == "ua"
    assert fetcher._timeout_s == 3.0
