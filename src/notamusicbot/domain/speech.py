"""Speech domain models (dataclasses and enums shared across the pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class TtsRequest:
    """A single text-to-speech request.

    The only backend is Qwen3-TTS voice cloning: ``reference_audio`` is the
    absolute path of the target voice's reference clip, ``reference_text`` its
    transcript (required for ``clone_mode="icl"``) and ``clone_mode`` selects
    x-vector-only (``xvec``) or in-context (``icl``) cloning. ``instruction``
    is kept in the interface for future backends; the clone backend ignores it.
    """

    text: str
    language: str = "English"
    speaker: str | None = None
    instruction: str | None = None
    # Upper bound on generated audio, in seconds. ``None`` = unbounded (the
    # model's own limit). Backends turn it into a generation-length bound.
    max_audio_seconds: float | None = None
    # Absolute path to the voice-clone reference clip (mono, any rate).
    reference_audio: str | None = None
    # Transcript of the reference clip; required for ``clone_mode="icl"``.
    reference_text: str | None = None
    # xvec = speaker-embedding-only clone; icl = in-context clone.
    clone_mode: Literal["xvec", "icl"] = "xvec"


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """A produced audio file plus its measured properties."""

    path: Path
    sample_rate_hz: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class AudioProcessingProfile:
    """Post-processing parameters applied after speech synthesis."""

    target_sample_rate_hz: int = 48000
    channels: int = 2
    speed: float = 1.0
    target_peak_dbfs: float = -1.0


@dataclass(frozen=True, slots=True)
class SpeechJobMetrics:
    """Timing metrics recorded for one speech job."""

    job_id: str
    queue_wait_s: float = 0.0
    tts_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0
    output_duration_s: float = 0.0
    tts_rtf: float = 0.0
    total_rtf: float = 0.0
