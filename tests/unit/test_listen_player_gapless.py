"""The reply player is one continuous stream across chunks (no per-chunk restart)."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from notamusicbot.discord_bot import listen_glue
from notamusicbot.discord_bot.listen_glue import (
    FRAME_BYTES,
    ChunkedSource,
    VoicePlayer,
    frame_align,
)

A = b"\x01" * FRAME_BYTES
B = b"\x02" * FRAME_BYTES
SILENCE = b"\x00" * FRAME_BYTES


def test_frame_align_drops_a_partial_tail_and_pads_tiny_clips() -> None:
    assert frame_align(A * 2 + b"\x09" * 100) == A * 2
    assert frame_align(A) == A
    assert frame_align(b"\x05" * 10) == b"\x05" * 10 + b"\x00" * (FRAME_BYTES - 10)
    assert frame_align(b"") == SILENCE


async def test_next_chunk_follows_the_previous_with_no_silence_between() -> None:
    src = ChunkedSource(asyncio.get_running_loop(), grace_frames=3)
    first = src.feed(A * 2)
    second = src.feed(B)
    assert [src.read() for _ in range(3)] == [A, A, B]
    await asyncio.sleep(0)
    assert first.done() and second.done()


async def test_a_chunk_resolves_only_when_its_last_frame_is_read() -> None:
    src = ChunkedSource(asyncio.get_running_loop(), grace_frames=3)
    first = src.feed(A * 2)
    second = src.feed(B)
    src.read()
    await asyncio.sleep(0)
    assert not first.done()
    src.read()
    await asyncio.sleep(0)
    assert first.done() and not second.done()


async def test_silence_bridges_a_late_chunk_then_the_source_ends() -> None:
    src = ChunkedSource(asyncio.get_running_loop(), grace_frames=2)
    src.feed(A)
    assert src.read() == A
    assert src.read() == SILENCE
    src.feed(B)  # arrives within the grace: still the same source
    assert src.read() == B
    assert [src.read() for _ in range(3)] == [SILENCE, SILENCE, b""]
    with pytest.raises(listen_glue._SourceClosed):
        src.feed(A)


async def test_finish_releases_waiters_and_propagates_an_error() -> None:
    src = ChunkedSource(asyncio.get_running_loop(), grace_frames=3)
    fut = src.feed(A * 4)
    src.finish(RuntimeError("player died"))
    with pytest.raises(RuntimeError):
        await fut


class ThreadedVoiceClient:
    """Plays a source on a real thread at (sped-up) frame cadence, like discord's player."""

    def __init__(self) -> None:
        self.play_calls = 0
        self.frames: list[bytes] = []
        self._thread: threading.Thread | None = None

    def play(self, source: Any, *, after: Any) -> None:
        self.play_calls += 1

        def run() -> None:
            while frame := source.read():
                self.frames.append(frame)
                time.sleep(0.001)
            after(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop_playing(self) -> None: ...

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


async def test_consecutive_plays_share_one_player_and_one_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(listen_glue, "SOURCE_GRACE_FRAMES", 200)
    vc = ThreadedVoiceClient()
    player = VoicePlayer(vc)
    await player.play(A * 5)
    await player.play(B * 5)
    await player.play(A * 5)
    await asyncio.sleep(0.05)  # the player thread records a frame just after it is read
    # One vc.play for the whole reply, and no silence frame between the chunks.
    assert vc.play_calls == 1
    audible = [f for f in vc.frames if f != SILENCE]
    assert audible == [A] * 5 + [B] * 5 + [A] * 5
    seam = vc.frames.index(B)
    assert vc.frames[seam - 1] == A
    assert vc.frames[: seam + 5].count(SILENCE) == 0


async def test_a_new_reply_after_the_source_ended_starts_a_new_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(listen_glue, "SOURCE_GRACE_FRAMES", 2)
    vc = ThreadedVoiceClient()
    player = VoicePlayer(vc)
    await player.play(A * 2)
    await asyncio.sleep(0.1)  # longer than the grace: the first source has ended
    await player.play(B * 2)
    assert vc.play_calls == 2
