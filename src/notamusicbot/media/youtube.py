"""yt-dlp YouTube extraction: option construction, JS runtime discovery, resolve.

All blocking yt-dlp work is executed through :func:`asyncio.to_thread` (see
``YouTubeResolver``); this module's pure helpers (option construction, JS
runtime discovery) are synchronous and unit-testable without network access.

Logging must never include cookie contents or authentication headers; only
client names and non-sensitive failure messages are emitted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from notamusicbot.domain.media import MediaKind, PlayableMedia

logger = logging.getLogger(__name__)

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMPEG_OPTIONS = "-vn -filter:a volume=0.5"

# yt-dlp needs a player client that can derive the signed stream URL. Avoid the
# android client (requires a PO token) and try a small set of desktop-friendly
# clients instead.
DEFAULT_YT_CLIENTS = ("default", "web", "tv_embedded")

# Use a stable desktop UA to reduce PO-token prompts.
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BASE_OPTS: dict[str, Any] = {
    "quiet": True,
    "format": "bestaudio/best",
    "noplaylist": True,
}


def discover_js_runtimes(
    extra: Mapping[str, Path | str] | None = None,
) -> dict[str, dict[str, str]]:
    """Find available JS runtimes (node/deno/qjs/quickjs) for yt-dlp.

    ``extra`` maps runtime names to candidate binary paths (e.g. a deno install
    at ``~/.deno/bin/deno``); existing, executable entries win over PATH hits.
    """
    runtimes: dict[str, dict[str, str]] = {}
    for name in ("node", "deno", "qjs", "quickjs"):
        found = shutil.which(name)
        if found:
            runtimes[name] = {"path": found}

    for name, raw_path in (extra or {}).items():
        path = Path(raw_path).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            runtimes[name] = {"path": str(path)}

    if not runtimes:
        logger.warning(
            "No JavaScript runtime found for yt-dlp. Install Node or Deno "
            "(yt_dlp_ejs needs one for YouTube JS challenge solving)."
        )
    else:
        logger.info(
            "JS runtimes detected for yt-dlp: %s", {k: v["path"] for k, v in runtimes.items()}
        )
    return runtimes


def build_yt_opts(
    client: str,
    *,
    cookies_file: Path | str | None = None,
    js_runtimes: Mapping[str, Mapping[str, str]] | None = None,
    base: Mapping[str, Any] | None = None,
    user_agent: str = DESKTOP_UA,
) -> dict[str, Any]:
    """Build yt-dlp options for one player client.

    Cookies are attached only when the file exists; a missing cookies file is
    handled gracefully (extraction continues anonymously).
    """
    opts: dict[str, Any] = {
        **(base or BASE_OPTS),
        "extractor_args": {"youtube": {"player_client": [client]}},
        "http_headers": {"User-Agent": user_agent},
        "js_runtimes": dict(js_runtimes) if js_runtimes else discover_js_runtimes(),
    }
    if cookies_file is not None and Path(cookies_file).is_file():
        opts["cookiefile"] = str(cookies_file)
    return opts


class YouTubeResolutionError(RuntimeError):
    """Raised when yt-dlp cannot produce a usable result for a URL."""


class _YoutubeDLFactory(Protocol):
    """Injection point so unit tests never import or invoke real yt-dlp."""

    def __call__(self, opts: Mapping[str, Any]) -> Any: ...


def default_ydl_factory(opts: Mapping[str, Any]) -> Any:
    import yt_dlp

    try:
        import yt_dlp_ejs  # noqa: F401  (registers JS runtimes with yt-dlp)
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("yt_dlp_ejs not installed; JS challenge solving may degrade")

    return yt_dlp.YoutubeDL(dict(opts))


@dataclass(frozen=True, slots=True)
class ResolvedInfo:
    """Metadata plus stream access for one resolved media item."""

    title: str
    webpage_url: str
    duration_s: int | None
    uploader: str
    stream_url: str
    http_headers: Mapping[str, str]


class YouTubeResolver:
    """Resolves YouTube URLs (and ``ytsearch:`` queries) to playable media.

    Extraction runs on a worker thread with a multi-client fallback chain
    (default -> web -> tv_embedded by default). A failed/absent stream URL is
    handled later by the download fallback in the player layer.
    """

    def __init__(
        self,
        *,
        clients: tuple[str, ...] = DEFAULT_YT_CLIENTS,
        cookies_file: Path | str | None = None,
        js_runtimes: Mapping[str, Mapping[str, str]] | None = None,
        user_agent: str = DESKTOP_UA,
        ydl_factory: _YoutubeDLFactory | None = None,
    ) -> None:
        self._clients = tuple(clients) or DEFAULT_YT_CLIENTS
        self._cookies_file = cookies_file
        self._js_runtimes = js_runtimes
        self._user_agent = user_agent
        self._ydl_factory = ydl_factory or default_ydl_factory

    @classmethod
    def from_settings(cls, settings: Any) -> YouTubeResolver:
        """Build from :class:`notamusicbot.config.MediaSettings`."""
        return cls(
            clients=tuple(settings.yt_clients),
            cookies_file=settings.cookies_file,
            js_runtimes=discover_js_runtimes(settings.js_runtimes),
            user_agent=settings.user_agent,
        )

    def _opts(self, client: str) -> dict[str, Any]:
        return build_yt_opts(
            client,
            cookies_file=self._cookies_file,
            js_runtimes=self._js_runtimes,
            user_agent=self._user_agent,
        )

    def opts_for(self, client: str) -> dict[str, Any]:
        """Public option builder reused by the download fallback path."""
        return self._opts(client)

    def _extract_sync(self, url: str) -> ResolvedInfo:
        last_error: Exception | None = None
        for client in self._clients:
            try:
                with self._ydl_factory(self._opts(client)) as ydl:
                    info = ydl.extract_info(url, download=False)
                if info and "entries" in info:  # ytsearch<ns>:query results
                    info = next((entry for entry in info["entries"] if entry), None)
                if not info:
                    raise YouTubeResolutionError("no result")
                return ResolvedInfo(
                    title=info.get("title", ""),
                    webpage_url=info.get("webpage_url", url),
                    duration_s=info.get("duration"),
                    uploader=info.get("uploader", "Unknown"),
                    stream_url=info.get("url", "") or "",
                    http_headers=info.get("http_headers", {}) or {},
                )
            except Exception as exc:  # yt-dlp raises generic exceptions here
                logger.warning("yt-dlp client '%s' failed for %s: %s", client, url, exc)
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError(f"yt-dlp could not resolve {url}")

    async def resolve_info(self, url: str) -> ResolvedInfo:
        """Extract metadata and stream URL off the event loop."""
        return await asyncio.to_thread(self._extract_sync, url)

    async def resolve(self, url: str, *, media_id: str | None = None) -> PlayableMedia:
        """Resolve a YouTube URL or ``ytsearch:`` query into a stream item.

        The stream URL is best-effort: playback falls back to a local download
        when it is empty or fails mid-stream (see the player layer).
        """
        info = await self.resolve_info(url)
        return PlayableMedia(
            media_id=media_id or uuid.uuid4().hex[:12],
            kind=MediaKind.STREAM,
            title=info.title,
            # The direct stream URL is best-effort; when it is absent the item
            # still resolves with the canonical webpage URL and the player's
            # download fallback fetches audio via that URL.
            stream_url=info.stream_url or None,
            source_url=info.webpage_url,
            stream_headers=info.http_headers,
        )
