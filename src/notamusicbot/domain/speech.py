"""Speech domain models (dataclasses and enums shared across the pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TtsRequest:
    """A single text-to-speech request.

    ``instruction`` is kept in the interface for future backends (e.g. Qwen 1.7B
    CustomVoice); the current 0.6B CustomVoice backend must not depend on it.
    """

    text: str
    language: str
    speaker: str
    instruction: str | None = None
    # Upper bound on generated audio, in seconds. ``None`` = unbounded (the
    # model's own limit). Backends turn it into a generation-length bound.
    max_audio_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """A produced audio file plus its measured properties."""

    path: Path
    sample_rate_hz: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class VoiceConversionProfile:
    """Parameters for the voice-conversion stage of a job."""

    backend: str
    model: str | None = None
    # YAML config variant inside the RVC model directory (obama vs
    # obama_trans share the Obama weights). None = directory default.
    variant: str | None = None
    pitch_shift: int = 0
    # Optional per-request override of the RVC f0 extraction method carried by
    # the model YAML (pm | harvest | dio | crepe | crepe-tiny). ``None`` keeps
    # the yaml value (harvest in the shipped profiles). Applied by the speech
    # pipeline from ``speech.voice_conversion.f0_method_override``.
    f0_method: str | None = None


@dataclass(frozen=True, slots=True)
class AudioProcessingProfile:
    """Post-processing parameters applied after voice conversion."""

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
    voice_conversion_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0
    output_duration_s: float = 0.0
    tts_rtf: float = 0.0
    voice_conversion_rtf: float = 0.0
    total_rtf: float = 0.0
