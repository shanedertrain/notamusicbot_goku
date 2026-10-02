"""Download cache for resilient playback and explicit temp-file cleanup.

Downloads go to per-run files under ``var/cache/youtube`` (configured via
``MediaSettings.cache_dir``); every downloaded file is deleted explicitly after
its playback finishes. Nothing is destructively cleared at startup.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from notamusicbot.domain.media import PlayableMedia
from notamusicbot.media.youtube import default_ydl_factory
from notamusicbot.speech.artifacts import discard_temp_speech_artifact, is_temp_speech_artifact

logger = logging.getLogger(__name__)

DOWNLOAD_SUFFIX = ".m4a"


def download_with_fallback(
    url: str,
    *,
    build_opts: Callable[[str], dict[str, Any]],
    clients: Iterable[str],
    cache_dir: Path,
    ydl_factory: Any = None,
) -> Path:
    """Blocking worker-thread body: download ``url`` trying each player client.

    Returns the downloaded file path, or raises the last client error.
    """
    factory = ydl_factory or default_ydl_factory
    last_error: Exception | None = None
    for client in clients:
        out_path = cache_dir / f"{uuid.uuid4().hex}{DOWNLOAD_SUFFIX}"
        opts = {**build_opts(client), "outtmpl": str(out_path)}
        try:
            with factory(opts) as ydl:
                ydl.download([url])
            return out_path
        except Exception as exc:
            logger.warning("yt-dlp download with client '%s' failed for %s: %s", client, url, exc)
            last_error = exc
            out_path.unlink(missing_ok=True)
    if last_error:
        raise last_error
    raise RuntimeError(f"no yt-dlp clients configured for download of {url}")


async def download_audio(
    url: str,
    *,
    build_opts: Callable[[str], dict[str, Any]],
    clients: Iterable[str],
    cache_dir: Path,
    ydl_factory: Any = None,
) -> Path:
    """Download audio into the cache dir, off the event loop."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(
        download_with_fallback,
        url,
        build_opts=build_opts,
        clients=tuple(clients),
        cache_dir=cache_dir,
        ydl_factory=ydl_factory,
    )


async def download_media_audio(
    media: PlayableMedia,
    *,
    build_opts: Callable[[str], dict[str, Any]],
    clients: Iterable[str],
    cache_dir: Path,
    ydl_factory: Any = None,
) -> Path:
    """Download a resolved media item via its canonical URL, off the event loop.

    The canonical ``source_url`` (the original webpage) is preferred over the
    direct ``stream_url``: yt-dlp re-extracts from the page, so an expired or
    missing direct stream URL never breaks the fallback.
    """
    url = media.source_url or media.stream_url
    if not url:
        raise ValueError(f"media {media.media_id} has no downloadable URL")
    return await download_audio(
        url,
        build_opts=build_opts,
        clients=clients,
        cache_dir=cache_dir,
        ydl_factory=ydl_factory,
    )


def cleanup_file(path: Path | None) -> None:
    """Explicitly delete one temp artifact, tolerating a missing file."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete temporary media file %s", path, exc_info=True)


def cleanup_media(media: PlayableMedia) -> None:
    """Delete the backing file of a temp media item after playback.

    Two sources of temp files are honored: items explicitly marked
    ``temp_file=True`` (download-cache entries), and generated speech artifacts
    the speech pipeline registered as play-once queue media (see
    :mod:`notamusicbot.speech.artifacts`).
    """
    if media.source_path is None:
        return
    if media.temp_file:
        cleanup_file(media.source_path)
        return
    if is_temp_speech_artifact(media.source_path):
        discard_temp_speech_artifact(media.source_path)
        cleanup_file(media.source_path)
