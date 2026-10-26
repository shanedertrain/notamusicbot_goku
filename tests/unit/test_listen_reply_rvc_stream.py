"""ReplySynthesizer streaming through a (fake) streaming RVC converter.

Fakes only: no GPU, no worker, no models. The converter contract is duck-typed
(supports_streaming / convert_stream / warm_stream).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from notamusicbot.domain.voice_profiles import VoiceProfile
from notamusicbot.listen.reply import (
    REPLY_SAMPLE_RATE_HZ,
    RVC_STREAM_GAIN_DB,
    ReplySynthesisError,
    ReplySynthesizer,
    warm_voice_converter,
)


def _profile(backend: str = "rvc", speed: float = 1.0) -> VoiceProfile:
    return VoiceProfile.model_validate(
        {
            "id": "goku",
            "persona": {"display_name": "Goku"},
            "tts": {"speaker": "Ryan"},
            "voice_conversion": {"backend": backend},
            "audio": {"speed": speed},
        }
    )


class FakeTts:
    def __init__(
        self, chunks: int = 4, size: int = 6000, amp: float = 0.2, fail_at: int | None = None
    ) -> None:
        self.supports_streaming = True
        self.n_chunks = chunks
        self.size = size
        self.amp = amp
        self.fail_at = fail_at
        self.closed = False
        self.events: list[str] = []

    async def stream_pcm(self, request: Any):  # type: ignore[no-untyped-def]
        try:
            for i in range(self.n_chunks):
                if i == self.fail_at:
                    raise RuntimeError("tts boom")
                yield np.full(self.size, self.amp, dtype=np.float32)
                await asyncio.sleep(0)
        finally:
            self.closed = True
            self.events.append("tts_closed")


class FakeConverter:
    def __init__(self, tts: FakeTts, supports: bool = True, fail_after: int | None = None) -> None:
        self.delay = 0.001
        self.tts = tts
        self.supports = supports
        self.fail_after = fail_after
        self.warmed: list[Any] = []
        self.warm_error: Exception | None = None
        self.closed = False
        self.order: list[str] = []

    def supports_streaming(self, profile: Any) -> bool:
        return self.supports

    async def convert_stream(self, chunks, profile):  # type: ignore[no-untyped-def]
        n = 0
        try:
            async for chunk in chunks:
                await asyncio.sleep(self.delay)
                if self.fail_after is not None and n >= self.fail_after:
                    raise RuntimeError("conv boom")
                n += 1
                yield np.repeat(chunk, 2)
        finally:
            self.closed = True
            self.order.append("converter_closed")
            self.tts.events.append("converter_closed")

    async def warm_stream(self, profile: Any) -> bool:
        if self.warm_error:
            raise self.warm_error
        self.warmed.append(profile)
        return True


def _synth(tts: Any, converter: Any, *, via_pipeline: bool = False, **kw: Any) -> ReplySynthesizer:
    pipeline = SimpleNamespace(_synthesizer=tts)
    if via_pipeline:
        pipeline._converter = converter
        return ReplySynthesizer(pipeline, _profile(**kw))  # type: ignore[arg-type]
    return ReplySynthesizer(
        pipeline,  # type: ignore[arg-type]
        _profile(**kw),
        stream_converter=converter,
    )


async def _collect(synth: ReplySynthesizer, text: str = "one two three") -> bytes:
    return b"".join([c async for c in synth.stream_sentence(text)])


def _pcm(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").reshape(-1, 2)


# ------------------------------------------------------------------ can_stream


def test_can_stream_true_for_supported_rvc() -> None:
    tts = FakeTts()
    assert _synth(tts, FakeConverter(tts)).can_stream
    assert _synth(tts, FakeConverter(tts), via_pipeline=True).can_stream


def test_can_stream_false_cases() -> None:
    tts = FakeTts()
    assert not _synth(tts, FakeConverter(tts, supports=False)).can_stream
    assert not _synth(tts, None).can_stream
    assert not _synth(tts, SimpleNamespace(supports_streaming=lambda p: True)).can_stream
    assert not _synth(tts, FakeConverter(tts), speed=1.2).can_stream
    no_stream_tts = FakeTts()
    no_stream_tts.supports_streaming = False
    assert not _synth(no_stream_tts, FakeConverter(no_stream_tts)).can_stream


def test_can_stream_false_when_supports_streaming_raises() -> None:
    tts = FakeTts()
    conv = FakeConverter(tts)

    def boom(profile: Any) -> bool:
        raise RuntimeError("x")

    conv.supports_streaming = boom  # type: ignore[method-assign]
    assert not _synth(tts, conv).can_stream


# --------------------------------------------------------------- stream output


async def test_stream_length_stereo_gain_and_close_order() -> None:
    tts = FakeTts(chunks=4, size=6000, amp=0.2)
    conv = FakeConverter(tts)
    data = await _collect(_synth(tts, conv))
    assert len(data) == 2 * 4 * 6000 * 4  # 2x frames, 2 channels, 2 bytes
    pcm = _pcm(data)
    assert np.array_equal(pcm[:, 0], pcm[:, 1])
    gain = 10 ** (RVC_STREAM_GAIN_DB / 20)
    expected = 0.2 * gain
    assert pytest.approx(expected * 32767, rel=0.01) == np.abs(pcm[:, 0]).max()
    # Natural end: the TTS is exhausted (and finalised) before the converter returns.
    assert sorted(tts.events) == ["converter_closed", "tts_closed"]


# ------------------------------------------------------------------- cap / fade


async def test_cap_truncates_fades_and_closes_both() -> None:
    # "one" -> 2.0 s floor cap; fake stream is 10 s long.
    tts = FakeTts(chunks=40, size=6000)
    conv = FakeConverter(tts)
    data = await _collect(_synth(tts, conv), "one")
    pcm = _pcm(data)
    assert pcm.shape[0] == 2 * REPLY_SAMPLE_RATE_HZ  # 2.0 s cap at 48 kHz
    fade = int(round(0.08 * REPLY_SAMPLE_RATE_HZ))
    assert abs(int(pcm[-1, 0])) < abs(int(pcm[0, 0])) * 0.05
    assert abs(int(pcm[-fade - 1000, 0])) > 0.9 * abs(int(pcm[0, 0]))
    assert tts.events == ["converter_closed", "tts_closed"]


# ---------------------------------------------------------------- cancel / error


async def test_early_aclose_closes_both() -> None:
    tts = FakeTts(chunks=40)
    conv = FakeConverter(tts)
    gen = _synth(tts, conv).stream_sentence("one two three")
    await gen.__anext__()
    await gen.aclose()
    assert tts.events == ["converter_closed", "tts_closed"]


async def test_cancellation_closes_both() -> None:
    tts = FakeTts(chunks=1000)
    conv = FakeConverter(tts)
    conv.delay = 0.05
    synth = _synth(tts, conv)

    async def consume() -> None:
        async for _ in synth.stream_sentence("one two three"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tts.events == ["converter_closed", "tts_closed"]


async def test_tts_error_is_reply_synthesis_error() -> None:
    tts = FakeTts(chunks=4, fail_at=2)
    with pytest.raises(ReplySynthesisError, match="reply streaming failed"):
        await _collect(_synth(tts, FakeConverter(tts)))
    assert tts.closed


async def test_converter_error_is_reply_synthesis_error() -> None:
    tts = FakeTts(chunks=6)
    conv = FakeConverter(tts, fail_after=1)
    with pytest.raises(ReplySynthesisError, match="reply streaming failed"):
        await _collect(_synth(tts, conv))
    assert tts.events == ["converter_closed", "tts_closed"]


async def test_refuses_when_not_streamable() -> None:
    tts = FakeTts()
    with pytest.raises(ReplySynthesisError):
        await _collect(_synth(tts, FakeConverter(tts, supports=False)))


# --------------------------------------------------------------------- warm-up


class WarmConverter(FakeConverter):
    async def convert(self, source: Any, profile: Any, out_path: Path) -> None:
        out_path.write_bytes(b"x")


async def test_warm_calls_warm_stream_when_supported() -> None:
    conv = WarmConverter(FakeTts())
    profile = _profile()
    assert await warm_voice_converter(SimpleNamespace(_converter=conv), profile) is True
    assert len(conv.warmed) == 1


async def test_warm_skips_warm_stream_when_unsupported() -> None:
    conv = WarmConverter(FakeTts(), supports=False)
    assert await warm_voice_converter(SimpleNamespace(_converter=conv), _profile()) is True
    assert conv.warmed == []


async def test_warm_survives_warm_stream_exception() -> None:
    conv = WarmConverter(FakeTts())
    conv.warm_error = RuntimeError("worker gone")
    assert await warm_voice_converter(SimpleNamespace(_converter=conv), _profile()) is True
