"""Typed voice-profile model loaded from ``config/voices/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class Persona(BaseModel):
    display_name: str
    description: str = ""


class TtsProfile(BaseModel):
    backend: str = "qwen_custom_voice"
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    language: str = "English"
    speaker: str


class VoiceConversionProfileModel(BaseModel):
    backend: str = "passthrough"
    model: str | None = None
    pitch_shift: int = 0


class AudioProfile(BaseModel):
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    target_peak_dbfs: float = Field(default=-1.0, le=0.0)


class VoiceProfile(BaseModel):
    """A full voice/persona profile. Persona data is intentionally separate
    from any RVC model configuration."""

    id: str
    persona: Persona
    tts: TtsProfile
    voice_conversion: VoiceConversionProfileModel
    audio: AudioProfile


class VoiceProfileError(ValueError):
    """Raised when a voice profile file is missing or invalid."""


def load_voice_profile(path: Path) -> VoiceProfile:
    """Load and validate one voice profile YAML file."""
    if not path.is_file():
        msg = f"Voice profile file not found: {path}"
        raise VoiceProfileError(msg)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return VoiceProfile.model_validate(data)
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        msg = f"Invalid voice profile {path}: {exc}"
        raise VoiceProfileError(msg) from exc


def load_voice_profiles(dir_path: Path) -> dict[str, VoiceProfile]:
    """Load all ``*.yaml`` voice profiles in a directory, keyed by profile id."""
    profiles: dict[str, VoiceProfile] = {}
    if not dir_path.is_dir():
        return profiles
    for path in sorted(dir_path.glob("*.yaml")):
        profile = load_voice_profile(path)
        profiles[profile.id] = profile
    return profiles


def to_conversion_profile(profile: VoiceProfile) -> Any:
    """Map a voice profile's VC section to the domain VoiceConversionProfile."""
    from notamusicbot.domain.speech import VoiceConversionProfile

    return VoiceConversionProfile(
        backend=profile.voice_conversion.backend,
        model=profile.voice_conversion.model,
        pitch_shift=profile.voice_conversion.pitch_shift,
    )


def to_processing_profile(profile: VoiceProfile, *, sample_rate_hz: int, channels: int) -> Any:
    """Map a voice profile's audio section to the domain AudioProcessingProfile."""
    from notamusicbot.domain.speech import AudioProcessingProfile

    return AudioProcessingProfile(
        target_sample_rate_hz=sample_rate_hz,
        channels=channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )
