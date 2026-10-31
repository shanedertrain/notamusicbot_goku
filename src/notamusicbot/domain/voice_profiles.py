"""Typed voice-profile model loaded from ``config/voices/*.yaml``.

Clone-only (owner decision 2026-10-30): a voice profile carries the reference
clip and transcript that Qwen3-TTS Base zero-shot cloning renders. There is no
trained model per voice and no voice-conversion stage — adding a voice is
dropping a WAV plus a transcript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from notamusicbot.domain.speech import AudioProcessingProfile, TtsRequest


class Persona(BaseModel):
    display_name: str
    description: str = ""
    # Optional listen-mode system prompt; empty = the default Goku chat prompt.
    chat_prompt: str = ""


class TtsProfile(BaseModel):
    """Voice-clone settings for the shared Qwen3-TTS Base model."""

    language: str = "English"
    # Reference clip path (absolute, or relative to the repo root / profile dir).
    reference_audio: Path
    # Transcript of the reference clip; required for icl cloning.
    reference_text: str | None = None
    # xvec = speaker embedding only (no transcript needed); icl = in-context.
    clone_mode: Literal["xvec", "icl"] = "xvec"
    # Sampling for the TTS decoder; ``None`` keeps the library default (0.9 / 50).
    # Lower values give steadier delivery from one sentence to the next.
    temperature: float | None = Field(default=None, gt=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("reference_audio", mode="before")
    @classmethod
    def _reference_audio_is_set(cls, value: Any) -> Any:
        if value is None or not str(value).strip():
            msg = "tts.reference_audio is required (path to the voice reference clip)"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _icl_requires_reference_text(self) -> TtsProfile:
        if self.clone_mode == "icl" and not (self.reference_text or "").strip():
            msg = "tts.reference_text is required when tts.clone_mode is 'icl'"
            raise ValueError(msg)
        return self


class AudioProfile(BaseModel):
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    target_peak_dbfs: float = Field(default=-1.0, le=0.0)


class VoiceProfile(BaseModel):
    """A full voice/persona profile: persona text plus its clone reference."""

    id: str
    persona: Persona
    tts: TtsProfile
    audio: AudioProfile


class VoiceProfileError(ValueError):
    """Raised when a voice profile file is missing or invalid."""


def resolve_reference_audio(reference_audio: Path | str, profile_path: Path | None = None) -> Path:
    """Existing absolute path for a profile's reference clip.

    Relative paths are resolved against the repo working directory first and
    then against the profile file's directory, so both ``config/voices/refs``
    and a profile-local ``refs/`` layout work. Raises ``VoiceProfileError``
    when no candidate exists.
    """
    path = Path(reference_audio)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        if profile_path is not None:
            candidates.insert(0, profile_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    msg = f"Voice reference audio not found: {path}"
    raise VoiceProfileError(msg)


def load_voice_profile(path: Path) -> VoiceProfile:
    """Load and validate one voice profile YAML file.

    The reference clip must exist; its absolute path replaces the configured
    value so every later request points at a real file.
    """
    if not path.is_file():
        msg = f"Voice profile file not found: {path}"
        raise VoiceProfileError(msg)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = VoiceProfile.model_validate(data)
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        msg = f"Invalid voice profile {path}: {exc}"
        raise VoiceProfileError(msg) from exc
    profile.tts.reference_audio = resolve_reference_audio(profile.tts.reference_audio, path)
    return profile


def load_voice_profiles(dir_path: Path) -> dict[str, VoiceProfile]:
    """Load all ``*.yaml`` voice profiles in a directory, keyed by profile id."""
    profiles: dict[str, VoiceProfile] = {}
    if not dir_path.is_dir():
        return profiles
    for path in sorted(dir_path.glob("*.yaml")):
        profile = load_voice_profile(path)
        profiles[profile.id] = profile
    return profiles


def tts_request_for(
    profile: VoiceProfile,
    text: str,
    *,
    max_audio_seconds: float | None = None,
) -> TtsRequest:
    """Build the clone TTS request for ``profile`` (absolute reference path)."""
    reference = Path(profile.tts.reference_audio)
    if not reference.is_absolute():
        reference = (Path.cwd() / reference).resolve()
    return TtsRequest(
        text=text,
        language=profile.tts.language,
        max_audio_seconds=max_audio_seconds,
        reference_audio=str(reference),
        reference_text=profile.tts.reference_text,
        clone_mode=profile.tts.clone_mode,
        temperature=profile.tts.temperature,
        top_k=profile.tts.top_k,
    )


def to_processing_profile(
    profile: VoiceProfile, *, sample_rate_hz: int, channels: int
) -> AudioProcessingProfile:
    """Map a voice profile's audio section to the domain AudioProcessingProfile."""
    return AudioProcessingProfile(
        target_sample_rate_hz=sample_rate_hz,
        channels=channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )
