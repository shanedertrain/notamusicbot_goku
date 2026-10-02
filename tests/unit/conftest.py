"""Shared pytest fixtures: fakes at the TTS/VC boundaries (no GPU, no weights)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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
