"""Unit tests for Spotify metadata resolution with a fake spotipy client."""

from __future__ import annotations

import asyncio

from notamusicbot.media.spotify import SpotifyResolver


class FakeSpotifyClient:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.queries: list[str] = []

    def search(self, q: str, limit: int = 1) -> dict:
        self.queries.append(q)
        return {"tracks": {"items": self.items[:limit]}}


def make_resolver(items: list[dict]) -> tuple[SpotifyResolver, FakeSpotifyClient]:
    client = FakeSpotifyClient(items)
    resolver = SpotifyResolver(
        client_id="id",
        client_secret="secret",
        client_factory=lambda cid, csecret: client,
    )
    return resolver, client


TRACK_ITEM = {
    "name": "Total Eclipse of the Heart",
    "artists": [{"name": "Bonnie Tyler"}],
    "album": {"name": " Faster Than the Speed of Night"},
    "uri": "spotify:track:abc",
}


def test_search_track_maps_metadata() -> None:
    resolver, client = make_resolver([TRACK_ITEM])
    track = asyncio.run(resolver.search_track("spotify:track:abc"))
    assert track is not None
    assert track.name == "Total Eclipse of the Heart"
    assert track.artists == ["Bonnie Tyler"]
    assert track.uri == "spotify:track:abc"
    assert client.queries == ["spotify:track:abc"]


def test_search_track_returns_none_when_no_results() -> None:
    resolver, _ = make_resolver([])
    assert asyncio.run(resolver.search_track("nothing")) is None


def test_search_query_joins_artists_and_name() -> None:
    resolver, _ = make_resolver([TRACK_ITEM])
    track = asyncio.run(resolver.search_track("q"))
    assert track is not None
    assert track.search_query == "Bonnie Tyler Total Eclipse of the Heart"


def test_client_is_created_lazily_once() -> None:
    calls: list[tuple] = []

    def factory(cid: str | None, csecret: str | None) -> FakeSpotifyClient:
        calls.append((cid, csecret))
        return FakeSpotifyClient([TRACK_ITEM])

    resolver = SpotifyResolver(client_id="a", client_secret="b", client_factory=factory)
    asyncio.run(resolver.search_track("x"))
    asyncio.run(resolver.search_track("y"))
    assert len(calls) == 1
