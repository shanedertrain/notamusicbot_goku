"""Audio post-processing: resample/channels -> speed -> peak protection -> WAV.

Intermediate audio between inference stages is always uncompressed PCM/WAV.
Speed change is explicit here, never an opaque conversion side effect, and
uses the
FFmpeg ``atempo`` filter (chained for ratios outside [0.5, 2.0]) so duration
scales without shifting pitch.
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


def _atempo_chain(speed: float) -> list[str]:
    """Split ``speed`` into FFmpeg ``atempo`` factors (each within [0.5, 2.0])."""
    factors: list[str] = []
    remaining = speed
    while remaining > 2.0:
        factors.append("2.0")
        remaining /= 2.0
    while remaining < 0.5:
        factors.append("0.500000")
        remaining /= 0.5
    factors.append(f"{remaining:.6f}")
    return factors


def _run_atempo(input_path: Path, output_path: Path, speed: float) -> None:
    """Pitch-preserving time-stretch via the FFmpeg CLI (subprocess, blocking).

    FFmpeg is a documented prerequisite of this project; a missing binary is a
    clear configuration error, not a silent no-op.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = (
            "ffmpeg is required for speed != 1.0 (pitch-preserving atempo "
            "time-stretch) but was not found on PATH"
        )
        raise RuntimeError(msg)
    filter_chain = ",".join(f"atempo={factor}" for factor in _atempo_chain(speed))
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-filter:a",
        filter_chain,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = f"ffmpeg atempo failed (exit {result.returncode}): {result.stderr[-500:]}"
        raise RuntimeError(msg)


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
    data = _apply_peak_protection(data, profile.target_peak_dbfs)
    if profile.speed == 1.0:
        await asyncio.to_thread(_write, output_path, data, profile.target_sample_rate_hz)
        duration = float(data.shape[0] / profile.target_sample_rate_hz)
    else:
        # Pitch-preserving time-stretch: run FFmpeg atempo on temp WAV files
        # instead of resampling-style interpolation (which shifts pitch).
        tmp_in = output_path.with_name(output_path.stem + ".pre_speed.wav")
        tmp_out = output_path.with_name(output_path.stem + ".post_speed.wav")
        try:
            await asyncio.to_thread(_write, tmp_in, data, profile.target_sample_rate_hz)
            await asyncio.to_thread(_run_atempo, tmp_in, tmp_out, profile.speed)
            stretched, out_rate = await asyncio.to_thread(_load, tmp_out)
            duration = float(stretched.shape[0] / out_rate)
            await asyncio.to_thread(_write, output_path, stretched, profile.target_sample_rate_hz)
        finally:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
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


# Quietest reference peak the streaming gain will boost from (about -26 dBFS), so a
# near-silent opening piece is never amplified into noise.
_MIN_REF_PEAK = 0.05
# Weight of the newest clip when the cross-clip peak reference is updated.
_PEAK_EMA_ALPHA = 0.3


class PeakTracker:
    """Typical clip peak, shared by every clip a voice renders.

    A streamed clip cannot be normalized to its own peak (the peak is unknown
    until it ends), so it is scaled by the running reference instead, which keeps
    level steady from one chunk of a reply to the next.
    """

    def __init__(self) -> None:
        self.ema: float | None = None

    def update(self, clip_peak: float) -> None:
        if clip_peak <= 0.0:
            return
        if self.ema is None:
            self.ema = clip_peak
        else:
            self.ema += _PEAK_EMA_ALPHA * (clip_peak - self.ema)


class StreamingPostprocess:
    """Chunk-at-a-time form of :func:`postprocess` for speed 1.0: float mono in, s16le out.

    Resampling is stateful (no seam between chunks) and the gain comes from the
    reference peak so far, never above the target, so a chunk can never clip.
    One instance per clip; ``push`` per piece, then ``finish``.
    """

    def __init__(self, profile: AudioProcessingProfile, tracker: PeakTracker) -> None:
        self._profile = profile
        self._tracker = tracker
        self._ref = tracker.ema or 0.0
        self._clip_peak = 0.0
        self._resampler: object | None = None
        self._rate: int | None = None

    def push(self, audio: npt.NDArray[np.floating], sample_rate: int) -> bytes:
        return self._emit(self._resample(audio, sample_rate, last=False))

    def finish(self) -> bytes:
        import numpy as np

        tail = self._resample(np.zeros(0, dtype=np.float32), self._rate or 0, last=True)
        out = self._emit(tail)
        self._tracker.update(self._clip_peak)
        return out

    def _resample(
        self, audio: npt.NDArray[np.floating], sample_rate: int, *, last: bool
    ) -> npt.NDArray[np.float32]:
        import numpy as np

        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=-1)
        mono = mono.reshape(-1)
        target = self._profile.target_sample_rate_hz
        if self._rate is None:
            self._rate = sample_rate
            if sample_rate != target:
                import soxr

                self._resampler = soxr.ResampleStream(sample_rate, target, 1, dtype="float32")
        if self._resampler is None:
            return mono
        return np.asarray(
            self._resampler.resample_chunk(mono, last=last),  # type: ignore[attr-defined]
            dtype=np.float32,
        ).reshape(-1)

    def _emit(self, mono: npt.NDArray[np.float32]) -> bytes:
        import numpy as np

        if mono.size == 0:
            return b""
        peak = float(np.max(np.abs(mono)))
        self._ref = max(self._ref, peak)
        self._clip_peak = max(self._clip_peak, peak)
        target = 10.0 ** (min(self._profile.target_peak_dbfs, 0.0) / 20.0)
        scaled = np.clip(mono * (target / max(self._ref, _MIN_REF_PEAK)), -1.0, 1.0)
        pcm = np.rint(scaled * 32767.0).astype("<i2")
        if self._profile.channels > 1:
            pcm = np.repeat(pcm[:, None], self._profile.channels, axis=1)
        return pcm.tobytes()


def peak_dbfs(path: Path) -> float:
    """Read a WAV file and return its peak level in dBFS (for tests/tools)."""
    data, _ = _load(path)
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(peak))
