"""Event-driven guild player.

One player task per guild consumes an ``asyncio.Queue[PlayableMedia]``. There
is no list polling and no ``sleep(1)`` loop: the task awaits the queue, then
awaits a playback-completion event set by the voice client's ``after``
callback. Skip and stop are driven by ``voice_client.stop()`` (which fires the
same callback) plus a stop event; the queue is drained explicitly.

Any playback failure is logged and the item skipped — the queue survives
(behavioral requirement from handoff section 20).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.media.downloader import cleanup_media
from notamusicbot.media.youtube import FFMPEG_BEFORE_OPTIONS, FFMPEG_OPTIONS

logger = logging.getLogger(__name__)

AudioSourceFactory = Callable[[PlayableMedia], Any]
StreamDownloader = Callable[[PlayableMedia], Awaitable[Path | None]]
NowPlayingHook = Callable[[PlayableMedia], Awaitable[None] | None]


def ffmpeg_source_factory(media: PlayableMedia) -> Any:
    """Build a discord.py FFmpeg source for local files and stream URLs."""
    import discord

    if media.kind is MediaKind.LOCAL:
        return discord.FFmpegPCMAudio(str(media.source_path))
    header_opts = ""
    if media.stream_headers:
        header_blob = "".join(
            f"{key}: {value}\\r\\n" for key, value in media.stream_headers.items()
        )
        header_opts = f' -headers "{header_blob}"'
    return discord.FFmpegPCMAudio(
        media.stream_url,
        before_options=f"{FFMPEG_BEFORE_OPTIONS}{header_opts}",
        options=FFMPEG_OPTIONS,
    )


class GuildPlayer:
    """Plays the guild queue sequentially through the guild's voice client."""

    def __init__(
        self,
        state: GuildState,
        *,
        get_voice_client: Callable[[], Any],
        source_factory: AudioSourceFactory = ffmpeg_source_factory,
        stream_downloader: StreamDownloader | None = None,
        on_now_playing: NowPlayingHook | None = None,
    ) -> None:
        self._state = state
        self._get_voice_client = get_voice_client
        self._source_factory = source_factory
        self._stream_downloader = stream_downloader
        self._on_now_playing = on_now_playing
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def state(self) -> GuildState:
        return self._state

    def start(self) -> asyncio.Task[None]:
        """Start the playback task if it is not already running."""
        if self._state.player_task is None or self._state.player_task.done():
            self._state.reset()
            self._loop = asyncio.get_running_loop()
            self._state.player_task = self._loop.create_task(
                self.run(), name=f"guild-player-{self._state.guild_id}"
            )
        assert self._state.player_task is not None
        return self._state.player_task

    async def run(self) -> None:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        state = self._state
        while True:
            get_task: asyncio.Task[PlayableMedia] = asyncio.create_task(state.queue.get())
            stop_task: asyncio.Task[bool] = asyncio.create_task(state.stop_event.wait())
            try:
                await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                stop_task.cancel()
            if state.stop_event.is_set():
                get_task.cancel()
                break
            media = await get_task
            play_media = media
            try:
                play_media = await self._play_one(media)
            except Exception:
                logger.exception(
                    "Playback failed for media %s (%r) in guild %s; skipping it",
                    media.media_id,
                    media.title,
                    state.guild_id,
                )
            finally:
                # Clean up the item that was actually played: when a stream
                # fell back to a downloaded copy, that LOCAL copy (temp_file)
                # is the file to delete, not the original STREAM item.
                cleanup_media(play_media)
        logger.info("Playback task exiting for guild %s", state.guild_id)

    async def _play_one(self, media: PlayableMedia) -> PlayableMedia:
        """Play one item; return the media item whose file was actually played.

        When a STREAM item is downloaded first, the returned copy is the LOCAL
        temp file backing playback so the caller cleans that file up.
        """
        voice_client = self._get_voice_client()
        if voice_client is None or not voice_client.is_connected():
            logger.info(
                "No connected voice client in guild %s; dropping media %s",
                self._state.guild_id,
                media.media_id,
            )
            return media

        play_media = media
        source: Any = None
        if media.kind is MediaKind.STREAM and self._stream_downloader is not None:
            # Legacy behavior preserved: prefer a downloaded copy over the raw
            # stream URL to avoid mid-stream signature/403 failures.
            try:
                path = await self._stream_downloader(media)
                if path is not None and path.exists():
                    play_media = media.as_local(path)
                    source = self._source_factory(play_media)
            except Exception:
                logger.warning(
                    "Download fallback failed for media %s in guild %s; trying the stream URL",
                    media.media_id,
                    self._state.guild_id,
                    exc_info=True,
                )
        if source is None:
            source = self._source_factory(play_media)

        done = self._state.playback_done
        done.clear()
        voice_client.play(source, after=self._make_after_callback(media))
        if self._on_now_playing is not None:
            result = self._on_now_playing(media)
            if inspect.isawaitable(result):
                await result
        await done.wait()
        return play_media

    def _make_after_callback(self, media: PlayableMedia) -> Callable[[Exception | None], None]:
        def _after(error: Exception | None) -> None:
            if error is not None:
                logger.error(
                    "Audio playback error for media %s in guild %s: %s",
                    media.media_id,
                    self._state.guild_id,
                    error,
                )
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._state.playback_done.set)

        return _after

    def skip(self) -> bool:
        """Stop the current item; the queue loop advances to the next one."""
        voice_client = self._get_voice_client()
        if voice_client is not None and voice_client.is_playing():
            voice_client.stop()
            return True
        return False

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop playback, clear the queue, and end the playback task."""
        state = self._state
        state.request_stop()
        voice_client = self._get_voice_client()
        if voice_client is not None:
            voice_client.stop()  # fires the after-callback; unblocks _play_one
        for media in state.drain_queue():
            cleanup_media(media)
        task = state.player_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
                logger.warning("Playback task for guild %s cancelled after timeout", state.guild_id)
        state.player_task = None
