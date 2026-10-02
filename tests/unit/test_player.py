"""Unit tests for the event-driven GuildPlayer (fakes, no Discord I/O)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.discord_bot.player import GuildPlayer
from notamusicbot.domain.media import MediaKind, PlayableMedia


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
) -> GuildPlayer:
    return GuildPlayer(
        state,
        get_voice_client=lambda: voice_client,
        source_factory=source_factory or (lambda media: object()),
        stream_downloader=stream_downloader,
        on_now_playing=on_now_playing,
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
