"""Unit tests for post-processing contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, AudioProcessingProfile
from notamusicbot.speech.postprocess import peak_dbfs, postprocess


def _artifact(path: Path, rate: int) -> AudioArtifact:
    data, _ = sf.read(path, dtype="float64")
    return AudioArtifact(path=path, sample_rate_hz=rate, duration_s=len(data) / rate)


def test_resample_to_target_rate(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.5)
    out = tmp_path / "out.wav"
    result = asyncio.run(
        postprocess(_artifact(src, 24000), out, AudioProcessingProfile(target_sample_rate_hz=48000))
    )
    assert result.sample_rate_hz == 48000
    assert sf.info(out).samplerate == 48000
    assert abs(result.duration_s - 0.5) < 0.05


def test_mono_to_stereo(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.25)
    out = tmp_path / "stereo.wav"
    asyncio.run(postprocess(_artifact(src, 24000), out, AudioProcessingProfile(channels=2)))
    info = sf.info(out)
    assert info.channels == 2


def test_speed_changes_duration(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 1.0)
    out = tmp_path / "fast.wav"
    result = asyncio.run(postprocess(_artifact(src, 24000), out, AudioProcessingProfile(speed=2.0)))
    assert result.duration_s == pytest.approx(0.5, abs=0.02)


def test_peak_protection_targets_dbfs(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.5)
    out = tmp_path / "peak.wav"
    asyncio.run(
        postprocess(
            _artifact(src, 24000),
            out,
            AudioProcessingProfile(target_peak_dbfs=-3.0),
        )
    )
    assert abs(peak_dbfs(out) - (-3.0)) < 0.2


def test_peak_never_above_zero_dbfs(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.5)
    out = tmp_path / "peak0.wav"
    asyncio.run(
        postprocess(_artifact(src, 24000), out, AudioProcessingProfile(target_peak_dbfs=6.0))
    )
    assert peak_dbfs(out) <= 0.0 + 0.01


def test_speed_one_identity(tmp_wav, tmp_path: Path) -> None:
    src = tmp_wav(24000, 0.5)
    out = tmp_path / "same.wav"
    data_before, _ = sf.read(src, dtype="float64")
    asyncio.run(
        postprocess(
            _artifact(src, 24000),
            out,
            AudioProcessingProfile(speed=1.0, target_sample_rate_hz=24000, channels=1),
        )
    )
    data_after, _ = sf.read(out, dtype="float64")
    data_after = data_after.reshape(data_before.shape)
    # Peak protection normalizes to target dBFS, so compare normalized shapes.
    before_norm = data_before / np.abs(data_before).max()
    after_norm = data_after / np.abs(data_after).max()
    assert np.allclose(before_norm, after_norm, atol=1e-3)
    assert len(data_before) == len(data_after)
