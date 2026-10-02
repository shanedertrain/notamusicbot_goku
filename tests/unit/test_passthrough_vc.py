"""Unit tests for the passthrough voice converter."""

from __future__ import annotations

import asyncio
from pathlib import Path

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter


def test_passthrough_copies_audio_identically(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.5)
    data, rate = sf.read(src, dtype="int16")
    artifact = AudioArtifact(path=src, sample_rate_hz=rate, duration_s=0.5)
    out = tmp_path / "conv" / "out.wav"
    converter = PassthroughVoiceConverter()
    result = asyncio.run(
        converter.convert(artifact, VoiceConversionProfile(backend="passthrough"), out)
    )
    out_data, out_rate = sf.read(out, dtype="int16")
    assert out_rate == rate
    assert out_data.shape == data.shape
    assert (out_data == data).all()
    assert result.path == out
    assert result.sample_rate_hz == rate
