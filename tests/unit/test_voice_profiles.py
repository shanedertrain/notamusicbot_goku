"""Unit tests for voice-profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from notamusicbot.domain.voice_profiles import (
    VoiceProfileError,
    load_voice_profile,
    load_voice_profiles,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_load_goku_profile() -> None:
    profile = load_voice_profile(FIXTURES / "voice_goku.yaml")
    assert profile.id == "goku"
    assert profile.tts.speaker == "Ryan"
    assert profile.tts.language == "English"
    assert profile.voice_conversion.backend == "rvc"
    assert profile.voice_conversion.model == "GOKU"
    assert profile.audio.speed == 1.0
    assert profile.audio.target_peak_dbfs == -1.0


def test_persona_is_separate_from_vc_model() -> None:
    profile = load_voice_profile(FIXTURES / "voice_goku.yaml")
    assert profile.persona.display_name == "Goku"
    assert "GOKU" not in profile.persona.description


def test_load_all_profiles_keyed_by_id() -> None:
    profiles = load_voice_profiles(FIXTURES)
    assert set(profiles) == {"goku", "vegeta"}
    assert profiles["vegeta"].tts.speaker == "Aiden"


def test_missing_file_raises_voice_profile_error() -> None:
    with pytest.raises(VoiceProfileError):
        load_voice_profile(FIXTURES / "nope.yaml")


def test_invalid_profile_raises_voice_profile_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: [unclosed\n")
    with pytest.raises(VoiceProfileError):
        load_voice_profile(bad)
