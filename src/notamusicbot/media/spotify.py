"""Spotify metadata resolution (spotipy, client-credentials flow).

Decision (Phase E, documented deviation): the legacy ``!gplay_spotify`` path
never actually produced audio — it fed a ``spotify:track:...`` URI straight
into ``FFmpegPCMAudio`` (which cannot play a URI), generated article commentary
into an unassigned local variable, and called
``generate_song_comment_audio_file`` with an argument mismatch. The rework
keeps only the part that works (metadata search via spotipy, off the event
loop) and routes the resolved track name/artist through the same YouTube
resolver (``ytsearch:``) and guild queue as every other request. Commentary is
generated through the shared speech pipeline like other media. This is simpler
than porting the legacy path and makes the command actually audible.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SpotifyClientFactory(Protocol):
    def __call__(self, client_id: str | None, client_secret: str | None) -> Any: ...


def _default_client_factory(client_id: str | None, client_secret: str | None) -> Any:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    )


@dataclass(frozen=True, slots=True)
class SpotifyTrack:
    """Minimal track metadata from a Spotify search."""

    name: str
    artists: list[str]
    album: str
    uri: str

    @property
    def search_query(self) -> str:
        """A query usable with a YouTube search to find playable audio."""
        return " ".join([*self.artists, self.name])


class SpotifyResolver:
    """Search Spotify tracks without blocking the event loop."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        client_factory: SpotifyClientFactory | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._client_factory = client_factory or _default_client_factory
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self._client_id, self._client_secret)
        return self._client

    async def search_track(self, query: str) -> SpotifyTrack | None:
        """Return the first matching track, or ``None`` when nothing matches."""
        client = await asyncio.to_thread(self._get_client)

        def _search() -> Any:
            return client.search(q=query, limit=1)

        results = await asyncio.to_thread(_search)
        items = (results or {}).get("tracks", {}).get("items", [])
        if not items:
            return None
        track_info = items[0]
        return SpotifyTrack(
            name=track_info["name"],
            artists=[artist["name"] for artist in track_info["artists"]],
            album=track_info["album"]["name"],
            uri=track_info["uri"],
        )
