"""Unit tests for clone voice-profile loading and validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from notamusicbot.domain.voice_profiles import (
    VoiceProfileError,
    load_voice_profile,
    load_voice_profiles,
    resolve_reference_audio,
    tts_request_for,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_load_goku_profile() -> None:
    profile = load_voice_profile(FIXTURES / "voice_goku.yaml")
    assert profile.id == "goku"
    assert profile.tts.language == "English"
    assert profile.tts.clone_mode == "xvec"
    assert profile.tts.reference_text == "Hey, it's me, Goku!"
    assert profile.audio.speed == 1.0
    assert profile.audio.target_peak_dbfs == -1.0
    # The loader resolves the reference to a real absolute path.
    assert profile.tts.reference_audio.is_absolute()
    assert profile.tts.reference_audio.is_file()


def test_persona_is_separate_from_tts_settings() -> None:
    profile = load_voice_profile(FIXTURES / "voice_goku.yaml")
    assert profile.persona.display_name == "Goku"
    assert "Hey, it's me" not in profile.persona.description


def test_load_all_profiles_keyed_by_id() -> None:
    profiles = load_voice_profiles(FIXTURES)
    assert set(profiles) == {"goku", "vegeta"}
    assert profiles["vegeta"].tts.clone_mode == "icl"


def test_missing_file_raises_voice_profile_error() -> None:
    with pytest.raises(VoiceProfileError):
        load_voice_profile(FIXTURES / "nope.yaml")


def test_invalid_profile_raises_voice_profile_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: [unclosed\n")
    with pytest.raises(VoiceProfileError):
        load_voice_profile(bad)


def _write_profile(tmp_path: Path, reference: str, **tts: object) -> Path:
    payload: dict = {
        "id": "test",
        "persona": {"display_name": "Test"},
        "tts": {"language": "English", "reference_audio": reference, **tts},
        "audio": {},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_missing_reference_audio_is_rejected(tmp_path: Path) -> None:
    """A profile whose reference clip does not exist must not load."""
    path = _write_profile(tmp_path, "refs/missing.wav")
    with pytest.raises(VoiceProfileError, match="reference audio not found"):
        load_voice_profile(path)


def test_empty_reference_audio_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(tmp_path, "")
    with pytest.raises(VoiceProfileError, match="reference_audio is required"):
        load_voice_profile(path)


def test_reference_found_next_to_profile(tmp_path: Path) -> None:
    """Relative paths resolve against the profile directory as well."""
    (tmp_path / "refs").mkdir()
    shutil.copyfile(FIXTURES / "sine_2s.wav", tmp_path / "refs" / "ref.wav")
    profile = load_voice_profile(_write_profile(tmp_path, "refs/ref.wav"))
    assert profile.tts.reference_audio == (tmp_path / "refs" / "ref.wav").resolve()


def test_icl_requires_a_transcript(tmp_path: Path) -> None:
    (tmp_path / "ref.wav").write_bytes((FIXTURES / "sine_2s.wav").read_bytes())
    path = _write_profile(tmp_path, "ref.wav", clone_mode="icl")
    with pytest.raises(VoiceProfileError, match="reference_text is required"):
        load_voice_profile(path)


def test_icl_with_transcript_loads(tmp_path: Path) -> None:
    (tmp_path / "ref.wav").write_bytes((FIXTURES / "sine_2s.wav").read_bytes())
    path = _write_profile(tmp_path, "ref.wav", clone_mode="icl", reference_text="A reference line.")
    profile = load_voice_profile(path)
    assert profile.tts.clone_mode == "icl"
    assert profile.tts.reference_text == "A reference line."


def test_unknown_clone_mode_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "ref.wav").write_bytes((FIXTURES / "sine_2s.wav").read_bytes())
    path = _write_profile(tmp_path, "ref.wav", clone_mode="nope")
    with pytest.raises(VoiceProfileError):
        load_voice_profile(path)


def test_tts_request_for_carries_absolute_reference_and_mode() -> None:
    profile = load_voice_profile(FIXTURES / "voice_goku.yaml")
    request = tts_request_for(profile, "hello", max_audio_seconds=4.0)
    assert request.text == "hello"
    assert request.language == "English"
    assert request.max_audio_seconds == 4.0
    assert request.clone_mode == "xvec"
    assert Path(request.reference_audio or "").is_absolute()
    assert Path(request.reference_audio or "").is_file()
    assert request.reference_text == "Hey, it's me, Goku!"


def test_resolve_reference_audio_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(VoiceProfileError):
        resolve_reference_audio("nope.wav", tmp_path / "profile.yaml")
