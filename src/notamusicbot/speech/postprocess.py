"""Audio post-processing: resample/channels -> speed -> peak protection -> WAV.

Intermediate audio between inference stages is always uncompressed PCM/WAV.
Speed change is explicit here, never an opaque RVC side effect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, AudioProcessingProfile

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def _load(path: Path) -> tuple[npt.NDArray[np.floating], int]:
    data, rate = sf.read(path, dtype="float64", always_2d=True)
    return data, int(rate)


def _to_channels(data: npt.NDArray[np.floating], channels: int) -> npt.NDArray[np.floating]:
    if data.shape[1] == channels:
        return data
    if channels == 1:
        return data.mean(axis=1, keepdims=True)
    if data.shape[1] == 1:
        return data.repeat(channels, axis=1)
    return data[:, :channels]


def _resample(
    data: npt.NDArray[np.floating], src_rate: int, dst_rate: int
) -> npt.NDArray[np.floating]:
    if src_rate == dst_rate:
        return data
    try:
        import soxr

        return soxr.resample(data, src_rate, dst_rate)
    except ImportError:
        # Linear resampling fallback keeps the pipeline functional without soxr.
        import numpy as np

        n_src = data.shape[0]
        n_dst = int(round(n_src * dst_rate / src_rate))
        if n_dst <= 0:
            msg = f"Resampled length is non-positive ({n_dst})"
            raise ValueError(msg) from None
        idx = np.linspace(0, n_src - 1, num=n_dst)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, n_src - 1)
        frac = (idx - i0)[:, None]
        return data[i0] * (1.0 - frac) + data[i1] * frac


def _change_speed(data: npt.NDArray[np.floating], speed: float) -> npt.NDArray[np.floating]:
    """Time-domain speed change by index interpolation (duration /= speed)."""
    import numpy as np

    if speed == 1.0:
        return data
    n = data.shape[0]
    n_out = max(1, int(round(n / speed)))
    idx = np.linspace(0, n - 1, num=n_out)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (idx - i0)[:, None]
    return data[i0] * (1.0 - frac) + data[i1] * frac


def _apply_peak_protection(
    data: npt.NDArray[np.floating], target_peak_dbfs: float
) -> npt.NDArray[np.floating]:
    """Scale the waveform so its peak equals ``target_peak_dbfs`` (never above 0 dBFS)."""
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data
    target_peak = 10.0 ** (min(target_peak_dbfs, 0.0) / 20.0)
    return data * (target_peak / peak)


def _write(
    path: Path, data: npt.NDArray[np.floating], sample_rate: int, subtype: str = "PCM_16"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate, subtype=subtype)


async def postprocess(
    source: AudioArtifact,
    output_path: Path,
    profile: AudioProcessingProfile,
) -> AudioArtifact:
    """Apply the standard post-processing chain and write the final WAV."""
    start = time.perf_counter()
    data, src_rate = await asyncio.to_thread(_load, source.path)
    data = _to_channels(data, profile.channels)
    data = _resample(data, src_rate, profile.target_sample_rate_hz)
    data = _change_speed(data, profile.speed)
    data = _apply_peak_protection(data, profile.target_peak_dbfs)
    await asyncio.to_thread(_write, output_path, data, profile.target_sample_rate_hz)
    duration = float(data.shape[0] / profile.target_sample_rate_hz)
    logger.info(
        "Postprocess done rate=%d channels=%d speed=%.2f duration=%.2fs latency=%.3fs",
        profile.target_sample_rate_hz,
        profile.channels,
        profile.speed,
        duration,
        time.perf_counter() - start,
    )
    return AudioArtifact(
        path=output_path,
        sample_rate_hz=profile.target_sample_rate_hz,
        duration_s=duration,
    )


def peak_dbfs(path: Path) -> float:
    """Read a WAV file and return its peak level in dBFS (for tests/tools)."""
    data, _ = _load(path)
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(peak))
