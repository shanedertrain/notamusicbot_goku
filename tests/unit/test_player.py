"""Unit tests for the event-driven GuildPlayer (fakes, no Discord I/O)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.discord_bot.player import GuildPlayer
from notamusicbot.domain.media import MediaKind, PlayableMedia, PlaybackOutcome


class FakeVoiceClient:
    """Records played sources; ``stop()`` fires the discord.py after-callback."""

    def __init__(self) -> None:
        self.played: list[object] = []
        self.after_callback = None
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def is_playing(self) -> bool:
        return self.after_callback is not None and self._playing

    _playing = False

    def play(self, source: object, after=None) -> None:  # noqa: ANN001
        self.played.append(source)
        self.after_callback = after
        self._playing = True

    def stop(self) -> None:
        self._playing = False
        if self.after_callback is not None:
            cb = self.after_callback
            self.after_callback = None
            cb(None)

    def finish(self) -> None:
        """Simulate natural end of playback."""
        self._playing = False
        if self.after_callback is not None:
            cb = self.after_callback
            self.after_callback = None
            cb(None)

    def crash(self, error: Exception) -> None:
        """Simulate a Discord/FFmpeg playback error ending playback."""
        self._playing = False
        if self.after_callback is not None:
            cb = self.after_callback
            self.after_callback = None
            cb(error)


def make_media(name: str, tmp_path: Path, *, temp: bool = False) -> PlayableMedia:
    path = tmp_path / f"{name}.wav"
    path.write_bytes(b"RIFFfake")
    return PlayableMedia(
        media_id=name,
        kind=MediaKind.LOCAL,
        title=name,
        source_path=path,
        temp_file=temp,
    )


def make_player(
    state: GuildState,
    voice_client: FakeVoiceClient,
    *,
    source_factory=None,
    stream_downloader=None,
    on_now_playing=None,
    on_playback_complete=None,
) -> GuildPlayer:
    return GuildPlayer(
        state,
        get_voice_client=lambda: voice_client,
        source_factory=source_factory or (lambda media: object()),
        stream_downloader=stream_downloader,
        on_now_playing=on_now_playing,
        on_playback_complete=on_playback_complete,
    )


async def wait_until(predicate, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


def run_and_drain(player: GuildPlayer):
    task = asyncio.create_task(player.run())
    return task


@contextlib.asynccontextmanager
async def running_player(player: GuildPlayer):
    task = run_and_drain(player)
    try:
        yield task
    finally:
        player.state.request_stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_queue_order_is_preserved(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    played_ids: list[str] = []

    def factory(media: PlayableMedia) -> object:
        played_ids.append(media.media_id)
        return object()

    player = make_player(state, vc, source_factory=factory)
    for name in ("a", "b", "c"):
        state.queue.put_nowait(make_media(name, tmp_path))

    async with running_player(player):
        for i in range(3):
            await wait_until(lambda bound=i: len(played_ids) >= bound + 1)
            vc.finish()
    assert played_ids == ["a", "b", "c"]


async def test_skip_advances_to_next_item(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    player = make_player(state, vc)
    state.queue.put_nowait(make_media("first", tmp_path))
    state.queue.put_nowait(make_media("second", tmp_path))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        assert player.skip() is True  # stops the voice client -> after fires
        await wait_until(lambda: len(vc.played) == 2)
        vc.finish()
    assert len(vc.played) == 2


async def test_skip_without_playing_returns_false(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    player = make_player(GuildState(1), vc)
    assert player.skip() is False


async def test_stop_ends_task_and_clears_queue(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    player = make_player(state, vc)
    state.queue.put_nowait(make_media("current", tmp_path))
    state.queue.put_nowait(make_media("pending1", tmp_path))
    state.queue.put_nowait(make_media("pending2", tmp_path))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        await player.stop()  # also disconnects nothing; just stops the task
    assert state.player_task is None
    assert state.queue.empty()
    assert len(vc.played) == 1  # pending items never played
    assert state.stop_event.is_set()


async def test_queue_clearing_cleans_temp_files(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    player = make_player(state, vc)
    current = make_media("current", tmp_path)
    pending = make_media("pending", tmp_path, temp=True)
    state.queue.put_nowait(current)
    state.queue.put_nowait(pending)

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        await player.stop()
    assert state.queue.empty()
    assert not pending.source_path.exists()


async def test_temp_file_deleted_after_playback(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    player = make_player(state, vc)
    temp_media = make_media("temp", tmp_path, temp=True)
    state.queue.put_nowait(temp_media)

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.finish()
        await wait_until(lambda: not temp_media.source_path.exists())
    assert not temp_media.source_path.exists()


async def test_playback_failure_skips_item_and_continues(tmp_path: Path) -> None:
    vc = FakeVoiceClient()

    def failing_for_first(media: PlayableMedia) -> object:
        if media.media_id == "bad":
            raise RuntimeError("boom")
        return object()

    state = GuildState(1)
    player = make_player(state, vc, source_factory=failing_for_first)
    state.queue.put_nowait(make_media("bad", tmp_path))
    state.queue.put_nowait(make_media("good", tmp_path))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        assert state.queue.empty()  # bad item consumed and skipped
        vc.finish()
    assert len(vc.played) == 1  # only the good item produced a source


async def test_stream_media_prefers_downloaded_copy_and_falls_back(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    seen_kinds: list[MediaKind] = []

    def factory(media: PlayableMedia) -> object:
        seen_kinds.append(media.kind)
        return object()

    downloaded = tmp_path / "dl.m4a"
    downloaded.write_bytes(b"audio")

    async def downloader(media: PlayableMedia) -> Path | None:
        if media.media_id == "ok":
            return downloaded
        raise RuntimeError("download failed")

    player = make_player(state, vc, source_factory=factory, stream_downloader=downloader)
    stream_ok = PlayableMedia(
        media_id="ok", kind=MediaKind.STREAM, title="ok", stream_url="https://example/stream"
    )
    stream_fail = PlayableMedia(
        media_id="fail", kind=MediaKind.STREAM, title="fail", stream_url="https://example/stream2"
    )
    state.queue.put_nowait(stream_ok)
    state.queue.put_nowait(stream_fail)

    async with running_player(player):
        for i in (1, 2):
            await wait_until(lambda count=i: len(vc.played) >= count)
            vc.finish()
    assert seen_kinds == [MediaKind.LOCAL, MediaKind.STREAM]  # download first, then stream URL


async def test_no_voice_client_drops_item_without_crash(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    vc.connected = False
    state = GuildState(1)
    player = make_player(state, vc)
    state.queue.put_nowait(make_media("orphan", tmp_path))

    async with running_player(player):
        await wait_until(lambda: state.queue.empty())
    assert vc.played == []


async def test_downloaded_fallback_file_deleted_after_playback(tmp_path: Path) -> None:
    """P1-6: the LOCAL copy created for a stream fallback must be cleaned up,
    not the original STREAM item."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    downloaded = tmp_path / "fallback.m4a"
    downloaded.write_bytes(b"audio")
    seen: list[PlayableMedia] = []

    def factory(media: PlayableMedia) -> object:
        seen.append(media)
        return object()

    async def downloader(media: PlayableMedia) -> Path:
        return downloaded

    player = make_player(state, vc, source_factory=factory, stream_downloader=downloader)
    stream_media = PlayableMedia(
        media_id="yt",
        kind=MediaKind.STREAM,
        title="yt",
        stream_url="https://example/stream",
        source_url="https://youtube.com/watch?v=yt",
    )
    state.queue.put_nowait(stream_media)

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.finish()
        await wait_until(lambda: not downloaded.exists())

    # The local copy (temp) was played, and the downloaded file is gone.
    assert seen[0].kind is MediaKind.LOCAL
    assert seen[0].source_path == downloaded
    assert not downloaded.exists()
    # The original STREAM item has no backing file to delete.
    assert stream_media.source_path is None


async def test_download_failure_cleans_only_played_media(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)

    async def downloader(media: PlayableMedia) -> Path:
        raise RuntimeError("yt-dlp down")

    player = make_player(state, vc, stream_downloader=downloader)
    temp_media = make_media("local-temp", tmp_path, temp=True)
    state.queue.put_nowait(temp_media)

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.finish()
        await wait_until(lambda: not temp_media.source_path.exists())
    assert not temp_media.source_path.exists()


# --- Round-3: post-song lifecycle, stop race, download ownership ------------


async def test_completion_hook_fires_once_per_song_not_for_generated(
    tmp_path: Path,
) -> None:
    """Round-3 P0: song A -> exactly one generated segment -> song B, and the
    generated LOCAL segment never triggers another completion hook (no
    recursion)."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    played_ids: list[str] = []
    hook_calls: list[str] = []
    segment_no = 0

    def factory(media: PlayableMedia) -> object:
        played_ids.append(media.media_id)
        return object()

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        nonlocal segment_no
        hook_calls.append(media.media_id)
        segment_no += 1
        state.put_front(make_media(f"seg-{segment_no}", tmp_path))

    player = make_player(state, vc, source_factory=factory, on_playback_complete=hook)
    state.queue.put_nowait(
        PlayableMedia(media_id="song-a", kind=MediaKind.STREAM, stream_url="https://example/a")
    )
    state.queue.put_nowait(
        PlayableMedia(media_id="song-b", kind=MediaKind.STREAM, stream_url="https://example/b")
    )

    async with running_player(player):
        for count in range(1, 5):
            await wait_until(lambda c=count: len(played_ids) >= c)
            vc.finish()
        await wait_until(lambda: state.queue.empty() and len(hook_calls) == 2)

    # Interstitial segments are enqueued ahead of the next song, and each song
    # produced exactly one segment (generated items fire nothing).
    assert played_ids == ["song-a", "seg-1", "song-b", "seg-2"]
    assert hook_calls == ["song-a", "song-b"]

    # No leftover tasks after the queue drains (review section 9).
    current = asyncio.current_task()
    await asyncio.sleep(0)
    leftover = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert leftover == []


async def test_completion_hook_not_fired_for_local_media(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    hook_calls: list[str] = []

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        hook_calls.append(media.media_id)

    player = make_player(state, vc, on_playback_complete=hook)
    state.queue.put_nowait(make_media("generated-speech", tmp_path))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.finish()
        await wait_until(lambda: state.queue.empty())
    assert hook_calls == []


async def test_completion_hook_awaited_before_next_dequeue(tmp_path: Path) -> None:
    """The hook must complete (including its enqueue) before song B starts."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    events: list[str] = []

    def factory(media: PlayableMedia) -> object:
        events.append(f"play:{media.media_id}")
        return object()

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        events.append("hook:start")
        await asyncio.sleep(0.02)  # simulated LLM/TTS latency
        state.put_front(make_media("interstitial", tmp_path))
        events.append("hook:end")

    player = make_player(state, vc, source_factory=factory, on_playback_complete=hook)
    state.queue.put_nowait(
        PlayableMedia(media_id="a", kind=MediaKind.STREAM, stream_url="https://example/a")
    )
    state.queue.put_nowait(
        PlayableMedia(media_id="b", kind=MediaKind.STREAM, stream_url="https://example/b")
    )

    async with running_player(player):
        await wait_until(lambda: "play:a" in events)
        vc.finish()  # song A ends
        await wait_until(lambda: "hook:end" in events)
        assert "play:b" not in events  # B did not start before the hook finished
        await wait_until(lambda: "play:interstitial" in events)
        vc.finish()
        await wait_until(lambda: "play:b" in events)
        vc.finish()

    assert events.index("hook:end") < events.index("play:b")
    assert events.index("play:interstitial") < events.index("play:b")
    assert events[:3] == ["play:a", "hook:start", "hook:end"]


async def test_stop_recovers_completed_queue_get_item(tmp_path: Path) -> None:
    """Round-3 P1 stop race: queue.get resolved while stop won — the removed
    item is recovered and its temp file cleaned up, deterministically."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    player = make_player(state, vc)
    orphan = make_media("orphan", tmp_path, temp=True)
    state.queue.put_nowait(orphan)
    # Stop is requested before the loop even waits, so the queue-get task and
    # the stop wait complete in the same event-loop step.
    state.request_stop()

    task = asyncio.create_task(player.run())
    await asyncio.wait_for(task, timeout=2.0)

    assert state.queue.empty()
    assert not orphan.source_path.exists()
    assert vc.played == []


async def test_source_factory_failure_after_download_removes_temp_file(
    tmp_path: Path,
) -> None:
    """Round-3 P1 download leak: download succeeds, source_factory raises —
    the downloaded temp file must not survive."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    downloaded = tmp_path / "fallback.m4a"
    downloaded.write_bytes(b"audio")

    def factory(media: PlayableMedia) -> object:
        raise RuntimeError("source creation boom")

    async def downloader(media: PlayableMedia) -> Path:
        return downloaded

    player = make_player(state, vc, source_factory=factory, stream_downloader=downloader)
    state.queue.put_nowait(
        PlayableMedia(media_id="yt", kind=MediaKind.STREAM, stream_url="https://example/stream")
    )

    async with running_player(player):
        await wait_until(lambda: state.queue.empty())
        await asyncio.sleep(0.05)  # allow the skip path to settle

    assert not downloaded.exists()
    assert vc.played == []  # item skipped, not played


# ------------------------------------------------------- playback outcomes (R3)


def stream_media(media_id: str) -> PlayableMedia:
    return PlayableMedia(
        media_id=media_id, kind=MediaKind.STREAM, stream_url=f"https://example/{media_id}"
    )


async def test_outcome_natural_fires_hook_exactly_once(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("song"))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.finish()
        await wait_until(lambda: not vc.is_playing())
        await asyncio.sleep(0.05)  # let the run loop finish the item
    assert calls == [("song", PlaybackOutcome.NATURAL)]


async def test_outcome_skipped_no_hook_next_song_proceeds(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []
    played: list[str] = []

    def factory(media: PlayableMedia) -> object:
        played.append(media.media_id)
        return object()

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, source_factory=factory, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("a"))
    state.queue.put_nowait(stream_media("b"))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        # skip() must record the interruption reason BEFORE voice_client.stop()
        assert player.skip() is True
        assert state.interrupt_reason is PlaybackOutcome.SKIPPED
        await wait_until(lambda: len(played) == 2)  # next song proceeds directly
        vc.finish()
        await wait_until(lambda: not vc.is_playing())
        await asyncio.sleep(0.05)
    assert calls == [("b", PlaybackOutcome.NATURAL)]  # skipped song fired nothing
    assert played == ["a", "b"]


async def test_outcome_stopped_no_hook_player_exits_promptly(tmp_path: Path) -> None:
    vc = FakeVoiceClient()
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("a"))
    state.queue.put_nowait(stream_media("b"))

    task = asyncio.create_task(player.run())
    await wait_until(lambda: len(vc.played) == 1)
    await asyncio.wait_for(player.stop(timeout=2.0), timeout=2.0)
    await asyncio.wait_for(task, timeout=2.0)  # exits promptly
    assert calls == []  # no post-song generation on stop
    assert state.queue.empty()


async def test_outcome_playback_error_yields_failed_no_hook(tmp_path: Path) -> None:
    """An after-callback playback error reports FAILED — no post-song hook."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []
    played: list[str] = []

    def factory(media: PlayableMedia) -> object:
        played.append(media.media_id)
        return object()

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, source_factory=factory, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("a"))
    state.queue.put_nowait(stream_media("b"))

    async with running_player(player):
        await wait_until(lambda: len(vc.played) == 1)
        vc.crash(RuntimeError("ffmpeg exploded"))
        await wait_until(lambda: len(played) == 2)  # queue survives the failure
        vc.finish()
        await wait_until(lambda: not vc.is_playing())
        await asyncio.sleep(0.05)
    assert calls == [("b", PlaybackOutcome.NATURAL)]


async def test_outcome_source_error_yields_failed_no_hook(tmp_path: Path) -> None:
    """A source-creation failure (unplayable dequeued item) fires no hook."""
    vc = FakeVoiceClient()
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []
    attempts: list[str] = []

    def factory(media: PlayableMedia) -> object:
        attempts.append(media.media_id)
        if media.media_id == "broken":
            raise RuntimeError("bad source")
        return object()

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, source_factory=factory, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("broken"))
    state.queue.put_nowait(stream_media("good"))

    async with running_player(player):
        await wait_until(lambda: "good" in attempts)
        vc.finish()
        await wait_until(lambda: not vc.is_playing())
        await asyncio.sleep(0.05)  # let the run loop finish the item
    assert calls == [("good", PlaybackOutcome.NATURAL)]


async def test_outcome_dropped_when_no_voice_client(tmp_path: Path) -> None:
    """A dequeued item with no connected voice client is DROPPED, no hook."""
    vc = FakeVoiceClient()
    vc.connected = False
    state = GuildState(1)
    calls: list[tuple[str, PlaybackOutcome]] = []

    async def hook(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
        calls.append((media.media_id, outcome))

    player = make_player(state, vc, on_playback_complete=hook)
    state.queue.put_nowait(stream_media("orphan"))

    async with running_player(player):
        await wait_until(lambda: state.queue.empty())
    assert calls == []
    assert vc.played == []


async def test_cleanup_of_speech_artifact_removes_job_dir(tmp_path: Path) -> None:
    """Round-3 P1: playing a generated speech artifact removes the final file
    AND the now-empty per-job directory (full helper, no dir buildup)."""
    from notamusicbot.discord_bot.player import cleanup_media_item
    from notamusicbot.speech.artifacts import (
        clear_registry,
        mark_temp_artifact,
        mark_temporary_job,
    )

    clear_registry()
    try:
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        final = job_dir / "final.wav"
        final.write_bytes(b"RIFFfake")
        mark_temp_artifact(final)
        mark_temporary_job(job_dir, status="generated")

        media = PlayableMedia(media_id="commentary", kind=MediaKind.LOCAL, source_path=final)
        cleanup_media_item(media)
        assert not final.exists()
        assert not job_dir.exists()  # marker + file gone, dir removed
    finally:
        clear_registry()
