"""Event-driven guild player.

One player task per guild consumes an ``asyncio.Queue[PlayableMedia]``. There
is no list polling and no ``sleep(1)`` loop: the task awaits the queue, then
awaits a playback-completion event set by the voice client's ``after``
callback. Skip and stop are driven by ``voice_client.stop()`` (which fires the
same callback) plus a stop event; the queue is drained explicitly.

Any playback failure is logged and the item skipped — the queue survives
(behavioral requirement from handoff section 20).

Playback outcomes (review Round-3 P0): ``_play_one()`` returns
``(played_media, PlaybackOutcome)``. ``skip()``/``stop()`` record the
interruption reason in ``state.interrupt_reason`` BEFORE calling
``voice_client.stop()``, so the after-callback/``_play_one`` can report
SKIPPED/STOPPED; an after-callback playback error with no recorded
interruption yields FAILED; a dequeued item with no connected voice client is
DROPPED.

Completion-hook contract (review Round-3 P0): the player invokes
``on_playback_complete(media, outcome)`` ONLY when ``media.kind == STREAM``
AND ``outcome == NATURAL`` — a song that was skipped, stopped, failed, or
dropped (and any generated/LOCAL media, which can never recurse into
post-song content) never triggers post-song generation. The outcome is still
passed to the hook so the bot-side caller can assert/log the completion kind.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.domain.media import MediaKind, PlayableMedia, PlaybackOutcome
from notamusicbot.media.downloader import cleanup_media
from notamusicbot.media.youtube import FFMPEG_BEFORE_OPTIONS, FFMPEG_OPTIONS
from notamusicbot.speech.artifacts import cleanup_temp_artifact, is_temp_speech_artifact

logger = logging.getLogger(__name__)

AudioSourceFactory = Callable[[PlayableMedia], Any]
StreamDownloader = Callable[[PlayableMedia], Awaitable[Path | None]]
NowPlayingHook = Callable[[PlayableMedia], Awaitable[None] | None]
PlaybackCompleteHook = Callable[[PlayableMedia, PlaybackOutcome], Awaitable[None] | None]


def cleanup_media_item(media: PlayableMedia) -> None:
    """Full play-once cleanup for a played/dropped queue item.

    Generated speech artifacts go through the full temporary-artifact helper
    (deletes the final file AND the emptied per-job directory — review Round-3
    P1); every other temp item uses the plain media cleanup.
    """
    path = media.source_path
    if path is not None and is_temp_speech_artifact(path):
        cleanup_temp_artifact(path)
        return
    cleanup_media(media)


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
        on_playback_complete: PlaybackCompleteHook | None = None,
    ) -> None:
        self._state = state
        self._get_voice_client = get_voice_client
        self._source_factory = source_factory
        self._stream_downloader = stream_downloader
        self._on_now_playing = on_now_playing
        self._on_playback_complete = on_playback_complete
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
                # Stop/get race: if the queue-get task already completed, the
                # item has been removed from the queue and would otherwise be
                # orphaned — recover it, clean it up, then exit.
                if get_task.done() and not get_task.cancelled():
                    recovered = get_task.result()
                    logger.info(
                        "Recovered dequeued media %s during stop in guild %s",
                        recovered.media_id,
                        state.guild_id,
                    )
                    cleanup_media_item(recovered)
                else:
                    get_task.cancel()
                break
            media = await get_task
            play_media = media
            outcome = PlaybackOutcome.FAILED
            try:
                play_media, outcome = await self._play_one(media)
            except asyncio.CancelledError:
                cleanup_media_item(play_media)
                raise
            except Exception:
                logger.exception(
                    "Playback failed for media %s (%r) in guild %s; skipping it",
                    media.media_id,
                    media.title,
                    state.guild_id,
                )
                outcome = PlaybackOutcome.FAILED
            else:
                # Clean up the item that was actually played: when a stream
                # fell back to a downloaded copy, that LOCAL copy (temp_file)
                # is the file to delete, not the original STREAM item.
                cleanup_media_item(play_media)
            # Never let a stale interruption reason leak into the next item.
            state.interrupt_reason = None
            # Post-song content hook (review Round-2 P0, Round-3 P0): awaited
            # to completion BEFORE the next dequeue. Fired only for STREAM
            # items that ended NATURAL — skip/stop/failure/drop (and any
            # generated/LOCAL item) never trigger post-song generation.
            if (
                self._on_playback_complete is not None
                and media.kind is MediaKind.STREAM
                and outcome is PlaybackOutcome.NATURAL
            ):
                try:
                    result = self._on_playback_complete(media, outcome)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception(
                        "Playback-complete hook failed for media %s in guild %s; continuing",
                        media.media_id,
                        state.guild_id,
                    )
        logger.info("Playback task exiting for guild %s", state.guild_id)

    async def _play_one(self, media: PlayableMedia) -> tuple[PlayableMedia, PlaybackOutcome]:
        """Play one item; return the media whose file played plus the outcome.

        When a STREAM item is downloaded first, the returned copy is the LOCAL
        temp file backing playback so the caller cleans that file up.
        Outcomes (review Round-3 P0): DROPPED when there is no connected voice
        client; SKIPPED/STOPPED when skip()/stop() recorded an interruption
        before ``voice_client.stop()``; FAILED when the after-callback reports
        a playback error with no recorded interruption; NATURAL otherwise.
        """
        voice_client = self._get_voice_client()
        if voice_client is None or not voice_client.is_connected():
            logger.info(
                "No connected voice client in guild %s; dropping media %s",
                self._state.guild_id,
                media.media_id,
            )
            return media, PlaybackOutcome.DROPPED

        play_media = media
        source: Any = None
        if media.kind is MediaKind.STREAM and self._stream_downloader is not None:
            # Legacy behavior preserved: prefer a downloaded copy over the raw
            # stream URL to avoid mid-stream signature/403 failures.
            try:
                path = await self._stream_downloader(media)
            except Exception:
                logger.warning(
                    "Download fallback failed for media %s in guild %s; trying the stream URL",
                    media.media_id,
                    self._state.guild_id,
                    exc_info=True,
                )
                path = None
            if path is not None and path.exists():
                play_media = media.as_local(path)
                try:
                    source = self._source_factory(play_media)
                except Exception:
                    # P1: we own the downloaded temp file until playback
                    # starts — remove it before falling back to the stream
                    # URL so a source-creation failure cannot leak it.
                    logger.warning(
                        "Source creation failed for downloaded copy of media %s in guild %s; "
                        "removing it and trying the stream URL",
                        media.media_id,
                        self._state.guild_id,
                        exc_info=True,
                    )
                    cleanup_media(play_media)
                    play_media = media
                    source = None
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
        state = self._state
        if state.interrupt_reason is not None:
            outcome, state.interrupt_reason = state.interrupt_reason, None
        else:
            outcome = PlaybackOutcome.NATURAL
        return play_media, outcome

    def _make_after_callback(self, media: PlayableMedia) -> Callable[[Exception | None], None]:
        def _after(error: Exception | None) -> None:
            if error is not None:
                logger.error(
                    "Audio playback error for media %s in guild %s: %s",
                    media.media_id,
                    self._state.guild_id,
                    error,
                )
                # Playback error (review Round-3 P0): report FAILED unless an
                # interruption (skip/stop) was already recorded — the recorded
                # reason wins because it was set BEFORE voice_client.stop().
                if self._state.interrupt_reason is None:
                    self._state.interrupt_reason = PlaybackOutcome.FAILED
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._state.playback_done.set)

        return _after

    def skip(self) -> bool:
        """Stop the current item; the queue loop advances to the next one."""
        voice_client = self._get_voice_client()
        if voice_client is not None and voice_client.is_playing():
            # Record the interruption reason BEFORE stopping playback so the
            # after-callback reports SKIPPED, not natural completion.
            self._state.interrupt_reason = PlaybackOutcome.SKIPPED
            voice_client.stop()
            return True
        return False

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop playback, clear the queue, and end the playback task."""
        state = self._state
        state.request_stop()
        voice_client = self._get_voice_client()
        if voice_client is not None:
            # Record the interruption reason BEFORE stopping playback so the
            # after-callback reports STOPPED, not natural completion.
            state.interrupt_reason = PlaybackOutcome.STOPPED
            voice_client.stop()  # fires the after-callback; unblocks _play_one
        for media in state.drain_queue():
            cleanup_media_item(media)
        task = state.player_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
                logger.warning("Playback task for guild %s cancelled after timeout", state.guild_id)
        state.player_task = None
