"""Streaming TTS: model pieces -> lease-held pipeline -> 48 kHz PCM -> player."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from _listen_turn_fakes import FakePlayer, make_rig, utterance

from notamusicbot.discord_bot.listen_glue import FRAME_BYTES, VoicePlayer
from notamusicbot.domain.speech import AudioProcessingProfile, TtsRequest
from notamusicbot.domain.voice_profiles import AudioProfile, Persona, TtsProfile, VoiceProfile
from notamusicbot.listen.reply import (
    REPLY_SAMPLE_RATE_HZ,
    ReplySynthesisError,
    ReplySynthesizer,
    StreamedReply,
)
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.postprocess import PeakTracker, StreamingPostprocess
from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

PROFILE = AudioProcessingProfile(target_sample_rate_hz=48000, channels=2, target_peak_dbfs=-1.0)


def _pcm_array(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)


# ------------------------------------------------------------- postprocess


def test_streaming_postprocess_resamples_to_stereo_without_losing_samples() -> None:
    post = StreamingPostprocess(PROFILE, PeakTracker())
    tone = (0.4 * np.sin(np.linspace(0, 200, 24000))).astype(np.float32)
    out = b"".join(post.push(tone[i : i + 6000], 24000) for i in range(0, 24000, 6000))
    out += post.finish()
    frames = _pcm_array(out)
    assert abs(frames.shape[0] - 48000) <= 64  # 1 s in, 1 s out, whatever the chunking
    assert (frames[:, 0] == frames[:, 1]).all()


def test_streaming_gain_never_exceeds_the_target_peak_and_is_shared_across_clips() -> None:
    tracker = PeakTracker()
    target = 10 ** (-1.0 / 20) * 32767
    loud = (0.9 * np.sin(np.linspace(0, 100, 12000))).astype(np.float32)
    post = StreamingPostprocess(PROFILE, tracker)
    out = _pcm_array(post.push(loud, 24000) + post.finish())
    assert np.abs(out).max() <= target + 2
    assert tracker.ema == pytest.approx(0.9, abs=0.01)
    # The next clip starts from the shared reference, so a quiet clip stays quiet.
    quiet = (0.1 * np.sin(np.linspace(0, 100, 12000))).astype(np.float32)
    second = StreamingPostprocess(PROFILE, tracker)
    out2 = _pcm_array(second.push(quiet, 24000) + second.finish())
    assert np.abs(out2).max() < 0.2 * 32767


# ------------------------------------------------------------- model wrapper


class StreamingModel:
    def __init__(self, pieces: int = 3, delay: float = 0.0) -> None:
        self.pieces, self.delay = pieces, delay
        self.produced = 0
        self.closed = threading.Event()
        self.kwargs: dict[str, Any] = {}

    def create_voice_clone_prompt(self, **_kw: Any) -> list[object]:
        return [object()]

    def generate_voice_clone_streaming(self, **kwargs: Any):  # noqa: ANN201
        self.kwargs = kwargs
        try:
            for _ in range(self.pieces):
                time.sleep(self.delay)
                self.produced += 1
                yield np.full(2400, 0.3, dtype=np.float32), 24000, {}
        finally:
            self.closed.set()


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "ref.wav"
    sf.write(path, 0.3 * np.ones(4800, dtype=np.float32), 24000, subtype="PCM_16")
    return path


def _request(tmp_path: Path, **kw: Any) -> TtsRequest:
    return TtsRequest(text="Hi there.", reference_audio=str(_reference(tmp_path)), **kw)


async def test_clone_stream_yields_pieces_and_forwards_sampling(tmp_path: Path) -> None:
    model = StreamingModel()
    synth = QwenCloneSynthesizer(model, fast=True)
    assert synth.supports_chunk_streaming
    got = [
        p
        async for p in synth.stream(
            _request(tmp_path, temperature=0.7, top_k=20), threading.Event()
        )
    ]
    assert len(got) == 3 and got[0][1] == 24000
    assert model.kwargs["temperature"] == 0.7 and model.kwargs["top_k"] == 20
    assert model.kwargs["chunk_size"] == 12


async def test_clone_stream_stops_the_model_when_closed_early(tmp_path: Path) -> None:
    model = StreamingModel(pieces=50, delay=0.01)
    synth = QwenCloneSynthesizer(model, fast=True)
    stream = synth.stream(_request(tmp_path), threading.Event())
    await anext(stream)
    await stream.aclose()
    assert model.closed.is_set()  # the worker finished before aclose returned
    assert model.produced < 50


def test_a_stock_model_cannot_stream() -> None:
    assert not QwenCloneSynthesizer(StreamingModel(), fast=False).supports_chunk_streaming


# ------------------------------------------------------------- pipeline lease


class ChunkSynth:
    supports_chunk_streaming = True

    def __init__(self, pieces: int = 40) -> None:
        self.pieces = pieces
        self.produced = 0

    async def stream(self, request: TtsRequest, stop: threading.Event) -> AsyncIterator[Any]:
        for i in range(self.pieces):
            if stop.is_set():
                return
            await asyncio.sleep(0.005)
            self.produced = i + 1
            yield np.full(2400, 0.3, dtype=np.float32), 24000


async def test_pipeline_stream_holds_the_lease_until_the_worker_stops() -> None:
    coordinator = InferenceCoordinator()
    synth = ChunkSynth()
    pipeline = SpeechPipeline(synth, coordinator=coordinator)  # type: ignore[arg-type]
    assert pipeline.can_stream
    request = TtsRequest(text="x", reference_audio="r.wav")
    first = pipeline.stream(request)
    await anext(first)
    await first.aclose()
    # A second stream can only start once the first released the lease.
    synth2 = ChunkSynth(pieces=2)
    pipeline2 = SpeechPipeline(synth2, coordinator=coordinator)  # type: ignore[arg-type]
    assert len([p async for p in pipeline2.stream(request)]) == 2
    assert synth.produced < 40  # stopped early, not run to the end


async def test_pipeline_stream_surfaces_a_synthesizer_failure() -> None:
    class Boom(ChunkSynth):
        async def stream(self, request: TtsRequest, stop: threading.Event):  # noqa: ANN201
            yield np.zeros(10, dtype=np.float32), 24000
            raise RuntimeError("cuda boom")

    pipeline = SpeechPipeline(Boom(), coordinator=InferenceCoordinator())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cuda boom"):
        _ = [p async for p in pipeline.stream(TtsRequest(text="x", reference_audio="r.wav"))]


# ------------------------------------------------------------- ReplySynthesizer


class PieceSource:
    can_stream = True

    def __init__(self, pieces: list[np.ndarray], fail: bool = False) -> None:
        self.pieces, self.fail = pieces, fail
        self.closed = False

    async def stream(self, request: TtsRequest) -> AsyncIterator[Any]:
        try:
            for piece in self.pieces:
                yield piece, 24000
            if self.fail:
                raise RuntimeError("boom")
        finally:
            self.closed = True


def _profile() -> VoiceProfile:
    return VoiceProfile(
        id="t",
        persona=Persona(display_name="T", description="a test voice"),
        tts=TtsProfile(reference_audio=Path("tests/fixtures/sine_2s.wav")),
        audio=AudioProfile(),
    )


async def test_stream_sentence_yields_stereo_pcm_in_pieces() -> None:
    src = PieceSource([np.full(12000, 0.3, dtype=np.float32)] * 2)
    synth = ReplySynthesizer(src, _profile())  # type: ignore[arg-type]
    assert synth.can_stream
    pieces = [p async for p in synth.stream_sentence("hello there friend")]
    total = sum(len(p) for p in pieces) // 4
    assert len(pieces) >= 2 and abs(total - 48000) <= 64
    assert src.closed


async def test_stream_sentence_cuts_at_the_length_cap_with_a_fade() -> None:
    src = PieceSource([np.full(24000, 0.3, dtype=np.float32)] * 5)  # 5 s of audio
    synth = ReplySynthesizer(
        src,  # type: ignore[arg-type]
        _profile(),
        tts_cap_floor_seconds=2.0,
        tts_max_seconds_per_word=0.1,
    )
    pieces = [p async for p in synth.stream_sentence("two words")]
    frames = _pcm_array(b"".join(pieces))
    assert frames.shape[0] == 2 * REPLY_SAMPLE_RATE_HZ
    assert abs(int(frames[-1, 0])) < 50  # faded out
    assert src.closed


async def test_stream_sentence_failure_becomes_a_synthesis_error() -> None:
    src = PieceSource([np.full(2400, 0.3, dtype=np.float32)], fail=True)
    synth = ReplySynthesizer(src, _profile())  # type: ignore[arg-type]
    with pytest.raises(ReplySynthesisError):
        _ = [p async for p in synth.stream_sentence("hello there")]


async def test_speed_other_than_one_disables_streaming() -> None:
    synth = ReplySynthesizer(PieceSource([]), _profile())  # type: ignore[arg-type]
    synth._processing = AudioProcessingProfile(speed=1.1)
    assert not synth.can_stream


# ------------------------------------------------------------- turn manager


class StreamingReplySynth:
    """Streams each chunk as two pieces and records what it was asked for."""

    can_stream = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def stream_sentence(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        half = len(text) // 2
        yield text[:half].encode()
        await asyncio.sleep(0)
        yield text[half:].encode()


class StreamPlayer(FakePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.streams: list[list[bytes]] = []

    async def play_stream(self, pieces: AsyncIterator[bytes]) -> None:
        self.streams.append([p async for p in pieces])


async def test_streamed_chunks_reach_the_player_piece_by_piece() -> None:
    synth = StreamingReplySynth()
    player = StreamPlayer()
    rig = make_rig(synth=synth, player=player)  # type: ignore[arg-type]
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert synth.calls == ["Hey, that sounds awesome!"]
    assert player.plays == []  # nothing went through the whole-clip path
    assert [b"".join(s) for s in player.streams] == [b"Hey, that sounds awesome!"]
    assert len(player.streams[0]) == 2


async def test_a_streaming_failure_before_any_audio_drops_the_reply() -> None:
    class Failing(StreamingReplySynth):
        async def stream_sentence(self, text: str) -> AsyncIterator[bytes]:
            raise ReplySynthesisError("tts unavailable")
            yield b""  # pragma: no cover

    player = StreamPlayer()
    rig = make_rig(synth=Failing(), player=player)  # type: ignore[arg-type]
    await rig.manager.on_utterance(utterance(1))
    await rig.manager.wait_idle()
    assert player.streams == []


async def test_streamed_reply_ends_its_iterator() -> None:
    reply = StreamedReply()
    reply.push(b"\x00" * 3840)
    reply.end()
    assert [p async for p in reply.pieces()] == [b"\x00" * 3840]
    assert reply.duration_s == pytest.approx(0.02)


# ------------------------------------------------------------- player


class _Vc:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.plays = 0
        self._t: threading.Thread | None = None

    def play(self, source: Any, *, after: Any) -> None:
        self.plays += 1

        def run() -> None:
            while frame := source.read():
                self.frames.append(frame)
                time.sleep(0.001)
            after(None)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop_playing(self) -> None: ...

    def is_playing(self) -> bool:
        return bool(self._t and self._t.is_alive())


async def test_play_stream_keeps_every_byte_across_odd_sized_pieces() -> None:
    vc = _Vc()
    player = VoicePlayer(vc)
    data = bytes(range(256)) * (FRAME_BYTES * 6 // 256)  # six frames
    cuts = [0, 1000, 1001, 5000, 9000, len(data)]

    async def pieces() -> AsyncIterator[bytes]:
        for a, b in zip(cuts, cuts[1:], strict=False):
            yield data[a:b]
            await asyncio.sleep(0.002)

    await player.play_stream(pieces())
    await asyncio.sleep(0.05)  # the player thread records a frame just after it is read
    audible = b"".join(f for f in vc.frames if f != b"\x00" * FRAME_BYTES)
    assert audible == data  # bridging silence is allowed between pieces, lost bytes are not
    assert vc.plays == 1
