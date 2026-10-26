"""Shared pytest fixtures: fakes at the TTS/VC and listen-mode boundaries
(no GPU, no weights, no Discord)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from notamusicbot.content.base import ContentRequest
from notamusicbot.domain.listen import AudioChunk, Transcript
from notamusicbot.domain.speech import AudioArtifact, TtsRequest, VoiceConversionProfile


class FakeSynthesizer:
    """Writes a deterministic sine-wave WAV instead of calling a TTS model."""

    def __init__(self, sample_rate: int = 24000, seconds: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.calls: list[TtsRequest] = []

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        import asyncio

        self.calls.append(request)
        n = int(self.sample_rate * self.seconds)
        wave = 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / self.sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            sf.write(output_path, wave, self.sample_rate, subtype="PCM_16")

        await asyncio.to_thread(_write)
        return AudioArtifact(
            path=output_path,
            sample_rate_hz=self.sample_rate,
            duration_s=n / self.sample_rate,
        )


class FakeConverter:
    """Copies the source artifact, recording calls, to emulate a VC backend."""

    def __init__(self) -> None:
        self.calls: list[tuple[AudioArtifact, VoiceConversionProfile]] = []

    async def convert(
        self, source: AudioArtifact, profile: VoiceConversionProfile, output_path: Path
    ) -> AudioArtifact:
        import asyncio
        import shutil

        self.calls.append((source, profile))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source.path, output_path)
        return AudioArtifact(
            path=output_path,
            sample_rate_hz=source.sample_rate_hz,
            duration_s=source.duration_s,
        )


class FailingSynthesizer(FakeSynthesizer):
    """Synthesizer that raises to exercise failure fallback behavior."""

    def __init__(self, message: str = "boom") -> None:
        super().__init__()
        self.message = message

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        raise RuntimeError(self.message)


# ----------------------------------------------------------------- listen mode


class FakeClock:
    """Manually advanced monotonic clock; ``sleep`` advances instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        import asyncio

        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # still yield to the loop


class FakeTranscriber:
    """Returns scripted transcripts in order (the last one repeats)."""

    def __init__(self, texts: list[str] | None = None, elapsed_s: float = 0.0) -> None:
        self.texts = list(texts) if texts else ["hello there goku"]
        self.elapsed_s = elapsed_s
        self.calls: list[AudioChunk] = []
        self.shutdowns = 0

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        index = min(len(self.calls), len(self.texts) - 1)
        self.calls.append(audio)
        return Transcript(text=self.texts[index], language="en", elapsed_s=self.elapsed_s)

    async def shutdown(self) -> None:
        self.shutdowns += 1


class FakeVad:
    """Returns scripted speech probabilities per frame, then ``default``."""

    def __init__(self, probabilities: list[float] | None = None, default: float = 0.0) -> None:
        self.probabilities = list(probabilities or [])
        self.default = default
        self.frames = 0
        self.resets = 0

    def process(self, frame: np.ndarray) -> float:
        index = self.frames
        self.frames += 1
        if index < len(self.probabilities):
            return self.probabilities[index]
        return self.default

    def reset(self) -> None:
        self.resets += 1


class FakePacketSource:
    """Delivers scripted ``(user_id, opus_payload)`` packets to a receive callback."""

    def __init__(self, packets: list[tuple[int | None, bytes]] | None = None) -> None:
        self.packets = list(packets or [])

    @classmethod
    def interleaved(cls, user_ids: list[int | None], count: int) -> FakePacketSource:
        """``count`` packets per user, round-robin; payload encodes user and index."""
        packets = [
            (user_id, f"{user_id}:{i}".encode()) for i in range(count) for user_id in user_ids
        ]
        return cls(packets)

    def deliver(self, callback: Callable[[int | None, bytes], None]) -> int:
        for user_id, payload in self.packets:
            callback(user_id, payload)
        return len(self.packets)


class FakeContentProvider:
    """Scripted ContentProvider: replies in order (last repeats); an Exception
    item is raised instead of returned. ``delay_s`` awaits before replying."""

    def __init__(self, replies: list[str | Exception] | None = None, delay_s: float = 0.0) -> None:
        self.replies = list(replies) if replies else ["Hey, that sounds awesome!"]
        self.delay_s = delay_s
        self.calls: list[ContentRequest] = []

    async def generate(self, request: ContentRequest) -> str:
        import asyncio

        index = min(len(self.calls), len(self.replies) - 1)
        self.calls.append(request)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        reply = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture()
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def fake_synthesizer() -> FakeSynthesizer:
    return FakeSynthesizer()


@pytest.fixture()
def fake_converter() -> FakeConverter:
    return FakeConverter()


@pytest.fixture()
def tmp_wav(tmp_path: Path) -> Callable[[int, float], Path]:
    def _make(sample_rate: int = 24000, seconds: float = 0.5) -> Path:
        n = int(sample_rate * seconds)
        wave = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate)
        path = tmp_path / f"src_{sample_rate}_{seconds}.wav"
        sf.write(path, wave, sample_rate, subtype="PCM_16")
        return path

    return _make


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"
