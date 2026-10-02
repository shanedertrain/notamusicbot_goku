"""Unit tests for the typed settings layer."""

from __future__ import annotations

from notamusicbot.config import AppSettings, SpeechSettings, load_app_settings


def test_default_settings_are_valid() -> None:
    settings = AppSettings()
    assert settings.speech.max_concurrent_inference == 1
    assert settings.speech.queue_size >= 1
    assert settings.speech.tts_model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def test_settings_reject_bad_concurrency() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SpeechSettings(max_concurrent_inference=0)


def test_load_app_settings_missing_file_returns_defaults(tmp_path) -> None:
    settings = load_app_settings(tmp_path / "does-not-exist.yaml")
    assert settings == AppSettings()


def test_load_app_settings_reads_yaml(tmp_path) -> None:
    import textwrap

    path = tmp_path / "app.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            speech:
              max_concurrent_inference: 2
              sample_rate_hz: 44100
            """
        )
    )
    settings = load_app_settings(path)
    assert settings.speech.max_concurrent_inference == 2
    assert settings.speech.sample_rate_hz == 44100
