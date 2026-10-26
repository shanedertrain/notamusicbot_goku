"""Streaming reply path: ReplySynthesizer.stream_sentence, TurnManager, VoicePlayer.

Fakes only: no GPU, no Discord, no models.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _listen_turn_fakes import (
    FakePlayer,
    FakeReplySynth,
    StreamPlayer,
    StreamSynth,
    TimelineClock,
    make_rig,
    speech,
    spin,
    utterance,
)

from conftest import FakeContentProvider
from notamusicbot.discord_bot import listen_glue
from notamusicbot.discord_bot.listen_glue import (
    FRAME_BYTES,
    PREBUFFER_FRAMES,
    VoicePlayer,
)
from notamusicbot.domain.voice_profiles import VoiceProfile
from notamusicbot.listen.reply import (
    REPLY_SAMPLE_RATE_HZ,
    STREAM_GAIN_DB,
    ReplySynthesisError,
    ReplySynthesizer,
    _Upsampler2x,
)


def _profile(backend: str = "passthrough", speed: float = 1.0) -> VoiceProfile:
    return VoiceProfile.model_validate(
        {
            "id": "goku_raw",
            "persona": {"display_name": "Goku"},
            "tts": {"speaker": "Ryan"},
            "voice_conversion": {"backend": backend},
            "audio": {"speed": speed},
        }
    )


class SineTts:
    """Fake streaming TTS: a 440 Hz sine in uneven 24 kHz chunks (float32 mono)."""

    def __init__(
        self,
        seconds: float = 1.0,
        sizes: tuple[int, ...] = (1000, 3001, 7, 2500),
        amp: float = 0.3,
        supports_streaming: bool = True,
    ) -> None:
        self.supports_streaming = supports_streaming
        n = int(24000 * seconds)
        self.signal = (amp * np.sin(2 * np.pi * 440 * np.arange(n) / 24000)).astype(np.float32)
        self.sizes = sizes
        self.requests: list[Any] = []
        self.closed = False
        self.pulled = 0

    async def stream_pcm(self, request: Any):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        try:
            pos, i = 0, 0
            while pos < self.signal.size:
                size = self.sizes[i % len(self.sizes)]
                i += 1
                self.pulled += 1
                yield self.signal[pos : pos + size]
                pos += size
                await asyncio.sleep(0)
        finally:
            self.closed = True


def _synth(tts: Any, **kw: Any) -> ReplySynthesizer:
    pipeline = SimpleNamespace(_synthesizer=tts)
    return ReplySynthesizer(pipeline, _profile(**kw))  # type: ignore[arg-type]


async def _collect(synth: ReplySynthesizer, text: str) -> list[bytes]:
    return [c async for c in synth.stream_sentence(text)]


def _frames(chunks: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(chunks), dtype="<i2").reshape(-1, 2)


# ------------------------------------------------------------- can_stream gate


def test_can_stream_only_for_passthrough_and_streaming_tts() -> None:
    assert _synth(SineTts()).can_stream
    assert not _synth(SineTts(supports_streaming=False)).can_stream
    assert not _synth(SineTts(), backend="rvc").can_stream
    assert not _synth(SineTts(), speed=1.2).can_stream
    assert not _synth(SimpleNamespace()).can_stream  # a TTS with no streaming API
    assert not _synth(None).can_stream


async def test_stream_sentence_refuses_when_not_streamable() -> None:
    with pytest.raises(ReplySynthesisError):
        await _collect(_synth(SineTts(), backend="rvc"), "hello there")


# ------------------------------------------------------------ resample / gain


def test_upsampler_is_exactly_2x_and_chunking_invariant() -> None:
    x = np.sin(np.linspace(0, 50, 9973)).astype(np.float32)
    whole = _Upsampler2x()
    ref = np.concatenate([whole.push(x), whole.flush()])
    assert ref.size == 2 * x.size
    up = _Upsampler2x()
    parts = []
    pos = 0
    for size in (1, 5000, 17, 1, 4954):
        parts.append(up.push(x[pos : pos + size]))
        pos += size
    parts.append(up.flush())
    assert np.array_equal(np.concatenate(parts), ref)


async def test_stream_is_2x_stereo_without_boundary_jumps() -> None:
    tts = SineTts(seconds=1.0)
    chunks = await _collect(_synth(tts), "one two three four five six")
    pcm = _frames(chunks)
    assert pcm.shape == (2 * tts.signal.size, 2)  # exactly 2x, stereo
    assert np.array_equal(pcm[:, 0], pcm[:, 1])
    gain = 10 ** (STREAM_GAIN_DB / 20)
    # 440 Hz at 48 kHz, amp 0.3 * gain: max sample-to-sample step of a clean sine.
    max_step = 2 * np.pi * 440 / REPLY_SAMPLE_RATE_HZ * 0.3 * gain * 32767
    assert np.abs(np.diff(pcm[:, 0].astype(np.int32))).max() <= 1.1 * max_step
    assert np.abs(pcm).max() <= 0.3 * gain * 32767 * 1.01
    assert tts.closed


async def test_stream_hard_clips_loud_input() -> None:
    tts = SineTts(seconds=0.2, amp=0.99)
    pcm = _frames(await _collect(_synth(tts), "loud words here please now"))
    assert pcm.max() <= 32767
    assert pcm.min() >= -32767


async def test_stream_request_carries_the_cap() -> None:
    tts = SineTts(seconds=0.1)
    await _collect(_synth(tts), "one two three")
    req = tts.requests[0]
    assert req.max_audio_seconds == 3.0
    assert req.speaker == "Ryan"


# ------------------------------------------------------------------------ cap


async def test_cap_truncates_with_fade_and_closes_tts(caplog: pytest.LogCaptureFixture) -> None:
    tts = SineTts(seconds=6.0, amp=0.5)
    # two words -> 2.0 s floor cap; the TTS would produce 6 s.
    with caplog.at_level(logging.WARNING):
        chunks = await _collect(_synth(tts), "Hi there")
    pcm = _frames(chunks)
    assert pcm.shape[0] == 2 * REPLY_SAMPLE_RATE_HZ
    fade = int(0.08 * REPLY_SAMPLE_RATE_HZ)
    tail = np.abs(pcm[-fade:, 0]).astype(float)
    head_peak = np.abs(pcm[:REPLY_SAMPLE_RATE_HZ, 0]).max()
    assert np.abs(pcm[-1, 0]) < 0.02 * head_peak  # fades to ~0
    assert tail[: fade // 4].max() > 0.5 * head_peak  # but not before the fade window
    assert tts.closed
    assert tts.pulled < 100  # stopped pulling once the cap was reached
    assert sum("over-generated" in r.message for r in caplog.records) == 1


async def test_audio_under_the_cap_is_not_faded_or_cut(caplog: pytest.LogCaptureFixture) -> None:
    tts = SineTts(seconds=1.0)
    with caplog.at_level(logging.WARNING):
        pcm = _frames(await _collect(_synth(tts), "Hi there"))
    assert pcm.shape[0] == 2 * tts.signal.size
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_tts_error_becomes_reply_synthesis_error() -> None:
    class Boom(SineTts):
        async def stream_pcm(self, request: Any):  # type: ignore[no-untyped-def]
            yield self.signal[:100]
            raise RuntimeError("cuda oom")

    with pytest.raises(ReplySynthesisError):
        await _collect(_synth(Boom()), "hello there")


async def test_closing_stream_sentence_closes_the_tts() -> None:
    tts = SineTts(seconds=5.0)
    gen = _synth(tts).stream_sentence("a b c d e f g h i j")
    await anext(gen)
    await gen.aclose()
    assert tts.closed


# ----------------------------------------------------------------- TurnManager

TWO_SENTENCES = ["First one here. Second one there."]


def _stream_rig(
    *,
    n_chunks: int = 3,
    chunk_s: float = 0.2,
    chunk_play_s: float = 0.5,
    texts: list[str] | None = None,
    **synth_kw: Any,
):  # type: ignore[no-untyped-def]
    clock = TimelineClock(start=100.0)
    log: list[tuple[str, float, str]] = []
    synth = StreamSynth(clock, log, n_chunks=n_chunks, chunk_s=chunk_s, **synth_kw)
    player = StreamPlayer(clock, log, chunk_play_s=chunk_play_s)
    rig = make_rig(
        clock=clock,
        synth=synth,  # type: ignore[arg-type]
        player=player,  # type: ignore[arg-type]
        provider=FakeContentProvider(texts or TWO_SENTENCES),
    )
    return rig, clock, log, synth, player


async def _run_turn(rig: Any, clock: TimelineClock) -> None:
    await rig.manager.on_utterance(utterance(1))
    await clock.run()
    await rig.manager.wait_idle()


def _times(log: list[tuple[str, float, str]], kind: str, text: str | None = None) -> list[float]:
    return [t for k, t, x in log if k == kind and (text is None or x == text or x.startswith(text))]


async def test_streaming_path_used_and_audio_arrives_in_order() -> None:
    rig, clock, _log, synth, player = _stream_rig()
    await _run_turn(rig, clock)
    assert synth.calls == ["First one here.", "Second one there."]
    assert player.plays == []  # never the whole-sentence player.play
    assert player.chunks == [
        *(f"First one here.|{i}".encode() for i in range(3)),
        *(f"Second one there.|{i}".encode() for i in range(3)),
    ]
    assert [t.render() for t in rig.conversation.history()][-1] == (
        "Goku: First one here. Second one there."
    )
    assert len(rig.metrics) == 1


async def test_first_pcm_reaches_the_player_before_tts_finished() -> None:
    rig, clock, log, _synth, _player = _stream_rig(n_chunks=5, chunk_s=0.4)
    await _run_turn(rig, clock)
    first_play = _times(log, "play_chunk", "First one here.|0")[0]
    gen_end = _times(log, "gen_end", "First one here.")[0]
    assert first_play < gen_end
    # about one TTS chunk after generation started, not after the whole sentence
    gen_start = _times(log, "gen_start", "First one here.")[0]
    assert first_play - gen_start == pytest.approx(0.4)


async def test_next_sentence_generates_after_previous_generation_before_playback_ends() -> None:
    rig, clock, log, _synth, _player = _stream_rig(n_chunks=3, chunk_s=0.2, chunk_play_s=1.0)
    await _run_turn(rig, clock)
    gen_end_1 = _times(log, "gen_end", "First one here.")[0]
    gen_start_2 = _times(log, "gen_start", "Second one there.")[0]
    last_play_1 = _times(log, "play_chunk", "First one here.|2")[0] + 1.0
    assert gen_start_2 == pytest.approx(gen_end_1)  # right when k's generation ended
    assert gen_start_2 < last_play_1  # and while k is still playing
    # never two generations at once
    assert _times(log, "gen_end", "First one here.")[0] <= gen_start_2
    # no gap: sentence 2's first chunk plays right after sentence 1's last one
    start_2 = _times(log, "play_chunk", "Second one there.|0")[0]
    assert start_2 == pytest.approx(last_play_1)


async def test_metrics_measure_first_pcm_not_whole_sentence() -> None:
    rig, clock, _log, _synth, _player = _stream_rig(n_chunks=5, chunk_s=0.4)
    await _run_turn(rig, clock)
    from notamusicbot.listen.metrics import parse_turn_line

    m = parse_turn_line(rig.metrics[0])
    assert m["llm->tts1"] == pytest.approx(0.4, abs=0.01)
    assert m["tts1->audio"] is None  # ~0 prints as n/a


async def test_barge_in_stops_player_and_closes_generator_in_one_tick() -> None:
    rig, clock, log, synth, player = _stream_rig(n_chunks=50, chunk_s=0.2, chunk_play_s=0.2)
    await rig.manager.on_utterance(utterance(1))
    # advance until audio is playing
    await clock.run(until=clock() + 1.0)
    assert player.chunks, "audio never started"
    assert synth.closed == 0
    rig.manager._barge_in(1)  # noqa: SLF001 (the public path is timer driven)
    await asyncio.sleep(0)  # one loop tick
    assert player.stop_calls >= 1
    assert synth.closed == 1
    assert any(k == "gen_closed" for k, _, _ in log)
    await clock.run()
    await rig.manager.wait_idle()
    assert synth.calls == ["First one here."]  # nothing generated after the barge-in
    assert rig.conversation.history()[-1].interrupted_after == 1


async def test_barge_in_via_speech_event(caplog: pytest.LogCaptureFixture) -> None:
    rig, clock, _log, synth, player = _stream_rig(n_chunks=50, chunk_s=0.2, chunk_play_s=0.2)
    await rig.manager.on_utterance(utterance(1))
    await clock.run(until=clock() + 1.0)
    await rig.manager.on_speech_started(speech(1, clock()))
    await clock.run(until=clock() + 2.0)
    await spin()
    assert player.stop_calls == 1
    assert synth.closed == 1


async def test_failure_before_audio_drops_the_reply(caplog: pytest.LogCaptureFixture) -> None:
    rig, clock, _log, synth, player = _stream_rig(fail_on_call=1, fail_after_chunks=0)
    with caplog.at_level(logging.WARNING):
        await _run_turn(rig, clock)
    assert player.chunks == []
    assert rig.metrics == []
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    assert synth.calls == ["First one here."]


async def test_failure_mid_stream_drops_the_rest_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig, clock, _log, synth, player = _stream_rig(fail_on_call=2, fail_after_chunks=1)
    with caplog.at_level(logging.WARNING):
        await _run_turn(rig, clock)
    assert player.chunks[:3] == [f"First one here.|{i}".encode() for i in range(3)]
    assert player.chunks[3:] == [b"Second one there.|0"]  # audio up to the failure only
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "mid-reply" in warnings[0].getMessage()
    assert rig.manager.state.name == "IDLE"


async def test_non_streaming_synth_keeps_the_whole_sentence_path() -> None:
    synth = FakeReplySynth()
    assert not getattr(synth, "can_stream", False)
    rig = make_rig(synth=synth, player=FakePlayer())
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert rig.player.plays == [b"Hey, that sounds awesome!"]
    assert rig.player.stream_plays == []


async def test_streaming_plays_one_continuous_stream_per_reply() -> None:
    rig, clock, _log, _synth, player = _stream_rig(texts=["One here. Two there. Three gone."])
    calls = 0
    inner = player.play_stream

    async def counting(chunks):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        await inner(chunks)

    player.play_stream = counting  # type: ignore[method-assign]
    await _run_turn(rig, clock)
    assert calls == 1  # one play_stream per reply, so no gap between sentences
    assert player.stop_calls == 0
    assert len(player.chunks) == 6  # default max_sentences is 2


# ------------------------------------------------------------------ VoicePlayer


class StreamVc:
    """Voice client whose ``play`` just records the source; the test plays the thread."""

    def __init__(self) -> None:
        self.source: Any = None
        self.after: Any = None
        self.stop_calls = 0
        self.stop_playing_calls = 0
        self.play_calls = 0

    def play(self, source: Any, *, after: Any) -> None:
        self.play_calls += 1
        self.source, self.after = source, after

    def stop(self) -> None:
        self.stop_calls += 1

    def stop_playing(self) -> None:
        self.stop_playing_calls += 1
        if self.after is not None:
            after, self.after = self.after, None
            after(None)

    def is_playing(self) -> bool:
        return self.after is not None


def _frame(n: int) -> bytes:
    return bytes([n % 250 + 1]) * FRAME_BYTES


async def _chunks(*items: bytes, gate: asyncio.Event | None = None):  # type: ignore[no-untyped-def]
    for item in items:
        yield item
        await asyncio.sleep(0)
    if gate is not None:
        await gate.wait()


async def test_voice_player_prebuffers_then_starts() -> None:
    vc = StreamVc()
    gate = asyncio.Event()
    task = asyncio.ensure_future(
        VoicePlayer(vc).play_stream(
            _chunks(*(_frame(i) for i in range(PREBUFFER_FRAMES - 1)), gate=gate)
        )
    )
    await spin()
    assert vc.play_calls == 0  # 5 frames: still pre-buffering
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_voice_player_starts_at_prebuffer_and_streams_frames_in_order() -> None:
    vc = StreamVc()
    gate = asyncio.Event()
    frames = [_frame(i) for i in range(PREBUFFER_FRAMES + 2)]
    task = asyncio.ensure_future(VoicePlayer(vc).play_stream(_chunks(*frames, gate=gate)))
    await spin()
    assert vc.play_calls == 1
    got = [vc.source.read() for _ in range(len(frames))]
    assert got == frames
    assert not vc.source.is_opus()
    # more is coming (gate not set): an empty buffer pads silence, never ends
    assert vc.source.read() == b"\x00" * FRAME_BYTES
    assert vc.source.underruns == 1
    assert not task.done()
    gate.set()
    await spin()
    assert vc.source.read() == b""  # iterator ended and queue drained -> end of source
    vc.after(None)
    await asyncio.wait_for(task, 1.0)
    assert vc.stop_calls == 0


async def test_voice_player_splits_odd_chunks_into_exact_frames_and_pads_the_tail() -> None:
    vc = StreamVc()
    blob = b"".join(_frame(i) for i in range(PREBUFFER_FRAMES)) + b"\x07" * 100
    cut = [blob[:1000], blob[1000:5000], blob[5000:]]
    task = asyncio.ensure_future(VoicePlayer(vc).play_stream(_chunks(*cut)))
    await spin()
    out = [vc.source.read() for _ in range(PREBUFFER_FRAMES)]
    assert all(len(f) == FRAME_BYTES for f in out)
    tail = vc.source.read()
    assert tail == b"\x07" * 100 + b"\x00" * (FRAME_BYTES - 100)
    assert vc.source.read() == b""
    vc.after(None)
    await asyncio.wait_for(task, 1.0)


async def test_voice_player_short_reply_still_plays() -> None:
    vc = StreamVc()
    task = asyncio.ensure_future(VoicePlayer(vc).play_stream(_chunks(_frame(1), _frame(2))))
    await spin()
    assert vc.play_calls == 1  # ended below the pre-buffer: play what there is
    assert vc.source.read() == _frame(1)
    assert vc.source.read() == _frame(2)
    assert vc.source.read() == b""
    vc.after(None)
    await asyncio.wait_for(task, 1.0)


async def test_voice_player_empty_stream_returns_without_playing() -> None:
    vc = StreamVc()
    await asyncio.wait_for(VoicePlayer(vc).play_stream(_chunks()), 1.0)
    assert vc.play_calls == 0


async def test_voice_player_stop_cancels_producer_and_uses_stop_playing() -> None:
    vc = StreamVc()
    closed = False

    async def endless():  # type: ignore[no-untyped-def]
        nonlocal closed
        try:
            while True:
                yield _frame(1)
                await asyncio.sleep(0)
        finally:
            closed = True

    player = VoicePlayer(vc)
    task = asyncio.ensure_future(player.play_stream(endless()))
    await spin()
    assert vc.play_calls == 1
    player.stop()
    await asyncio.wait_for(task, 1.0)
    assert vc.stop_playing_calls == 1
    assert vc.stop_calls == 0  # VoiceRecvClient.stop() would end reception
    assert closed  # the chunk iterator was closed


async def test_voice_player_stop_before_playback_started_returns() -> None:
    vc = StreamVc()
    gate = asyncio.Event()
    player = VoicePlayer(vc)
    task = asyncio.ensure_future(player.play_stream(_chunks(_frame(1), gate=gate)))
    await spin()
    assert vc.play_calls == 0
    player.stop()
    await asyncio.wait_for(task, 1.0)
    assert vc.play_calls == 0
    assert vc.stop_calls == 0


async def test_voice_player_cancel_stops_playback_and_closes_iterator() -> None:
    vc = StreamVc()
    closed = False

    async def endless():  # type: ignore[no-untyped-def]
        nonlocal closed
        try:
            while True:
                yield _frame(1)
                await asyncio.sleep(0)
        finally:
            closed = True

    task = asyncio.ensure_future(VoicePlayer(vc).play_stream(endless()))
    await spin()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert vc.stop_playing_calls == 1
    assert closed


async def test_voice_player_backpressure_bounds_the_buffer() -> None:
    vc = StreamVc()
    produced = 0

    async def endless():  # type: ignore[no-untyped-def]
        nonlocal produced
        while True:
            produced += 1
            yield _frame(1)
            await asyncio.sleep(0)

    player = VoicePlayer(vc)
    task = asyncio.ensure_future(player.play_stream(endless()))
    await asyncio.sleep(0.1)
    assert produced <= listen_glue.MAX_BUFFER_FRAMES + 2
    player.stop()
    await asyncio.wait_for(task, 1.0)


async def test_voice_player_propagates_an_iterator_error() -> None:
    vc = StreamVc()

    async def bad():  # type: ignore[no-untyped-def]
        for i in range(PREBUFFER_FRAMES):
            yield _frame(i)
        raise RuntimeError("tts died")

    task = asyncio.ensure_future(VoicePlayer(vc).play_stream(bad()))
    await spin()
    assert vc.source.read() == b""  # aborted: buffered audio dropped
    vc.after(None)
    with pytest.raises(RuntimeError, match="tts died"):
        await asyncio.wait_for(task, 1.0)
