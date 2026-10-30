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


# ------------------------------------------------------------------ listen mode


def test_listen_defaults_pinned() -> None:
    listen = AppSettings().listen
    assert listen.enabled is False
    assert listen.allowed_channel_ids == []
    assert listen.operator_user_ids == []
    assert listen.voice_profile == "goku"
    assert listen.announce_on_join is True
    assert listen.announcement_text == (
        "Heads up, I'm listening now. Everyone in this channel is transcribed, "
        "and nothing is saved."
    )
    assert listen.announcement_text_stored == (
        "Heads up, I'm listening now. Everyone in this channel is transcribed. "
        "The text of what you say and what I reply is saved for {days} days (no audio)."
    )
    assert listen.respond_to == "all"
    assert listen.wake_word == "goku"
    assert listen.follow_up_window_s == 8.0

    vad = listen.vad
    assert vad.backend == "silero"
    assert vad.threshold == 0.5
    assert vad.min_speech_ms == 250
    assert vad.end_silence_ms == 600
    assert vad.max_utterance_s == 15.0
    assert vad.pre_roll_ms == 300
    assert vad.ring_buffer_s == 30.0

    asr = listen.asr
    assert asr.backend == "parakeet_onnx"
    assert asr.model_id == "istupakov/parakeet-tdt-0.6b-v2-onnx"
    assert asr.whisper_model_id == "openai/whisper-small.en"
    assert asr.device == "cpu"
    assert asr.threads == 4
    assert asr.quantization == "int8"
    assert asr.language == "en"
    assert asr.timeout_s == 10.0

    reply = listen.reply
    assert reply.max_sentences == 2
    assert reply.max_new_tokens == 60
    assert reply.backchannel_enabled is True
    assert reply.backchannel_texts == ["Hm?", "Oh yeah?", "Let me think."]
    assert reply.barge_in_min_speech_ms == 400
    assert reply.per_user_cooldown_s == 2.0
    assert reply.max_replies_per_minute == 12
    assert reply.history_turns == 8
    assert reply.llm_timeout_s == 20.0
    assert reply.first_chunk_max_words == 8
    assert reply.later_chunk_max_words == 12
    assert reply.tts_max_seconds_per_word == 1.0
    assert reply.tts_cap_floor_seconds == 2.0

    llm = listen.llm
    assert llm.provider == "spark"
    assert llm.base_url == "http://127.0.0.1:8000/v1"
    assert llm.model == "auto"
    assert llm.timeout_s == 20.0
    assert llm.first_token_timeout_s == 8.0
    assert llm.fallback == "local"
    assert llm.health_interval_s == 30.0

    assert listen.privacy.log_transcripts is False
    # The per-session opt-in system is gone, so its persistence flag is too.
    assert "persist_opt_ins" not in type(listen.privacy).model_fields


def test_listen_validators_reject_bad_values() -> None:
    import pytest
    from pydantic import ValidationError

    from notamusicbot.config import (
        ListenAsrSettings,
        ListenReplySettings,
        ListenSettings,
        ListenVadSettings,
    )

    with pytest.raises(ValidationError, match="end_silence_ms"):
        ListenVadSettings(min_speech_ms=700, end_silence_ms=600)
    with pytest.raises(ValidationError):
        ListenReplySettings(max_sentences=0)
    with pytest.raises(ValidationError):
        ListenSettings(respond_to="everyone")
    with pytest.raises(ValidationError):
        ListenAsrSettings(backend="qwen_worker")
    with pytest.raises(ValidationError):
        ListenAsrSettings(threads=0)
    with pytest.raises(ValidationError):
        ListenAsrSettings(device="tpu")
    with pytest.raises(ValidationError):
        ListenAsrSettings(quantization="int4")


def test_listen_validators_accept_boundaries() -> None:
    from notamusicbot.config import ListenAsrSettings, ListenSettings, ListenVadSettings

    assert ListenVadSettings(min_speech_ms=600, end_silence_ms=600).end_silence_ms == 600
    assert ListenSettings(respond_to="wake_word").respond_to == "wake_word"
    assert ListenAsrSettings(backend="whisper_hf").backend == "whisper_hf"
    assert ListenAsrSettings(threads=1).threads == 1


def test_listen_enabled_with_empty_channels_still_loads(tmp_path) -> None:
    # Section 6: such a config loads; the session start refuses it.
    path = tmp_path / "app.yaml"
    path.write_text("listen:\n  enabled: true\n  allowed_channel_ids: []\n")
    settings = load_app_settings(path)
    assert settings.listen.enabled is True
    assert settings.listen.allowed_channel_ids == []


def test_example_yaml_listen_block_matches_defaults() -> None:
    from pathlib import Path

    settings = load_app_settings(Path("config/app.example.yaml"))
    assert settings.listen.enabled is False
    assert settings.listen == AppSettings().listen


def test_listen_llm_rejects_non_loopback_hosts() -> None:
    import pytest
    from pydantic import ValidationError

    from notamusicbot.config import ListenLlmSettings

    with pytest.raises(ValidationError, match=r"host '10\.0\.0\.5' is not loopback"):
        ListenLlmSettings(base_url="http://10.0.0.5:8000/v1")
    with pytest.raises(ValidationError, match=r"host 'example\.com' is not loopback"):
        ListenLlmSettings(base_url="http://example.com/v1")
    with pytest.raises(ValidationError, match="not loopback"):
        ListenLlmSettings(base_url="http://127.0.0.1.evil.com/v1")
    for ok in ("http://127.0.0.1:9/v1", "http://localhost:8000/v1", "http://[::1]:8000/v1"):
        assert ListenLlmSettings(base_url=ok).base_url == ok


def test_listen_llm_and_reply_validators() -> None:
    import pytest
    from pydantic import ValidationError

    from notamusicbot.config import ListenLlmSettings, ListenReplySettings

    with pytest.raises(ValidationError):
        ListenLlmSettings(provider="openai")
    with pytest.raises(ValidationError):
        ListenLlmSettings(fallback="remote")
    with pytest.raises(ValidationError):
        ListenLlmSettings(timeout_s=0)
    with pytest.raises(ValidationError):
        ListenReplySettings(first_chunk_max_words=2)
    with pytest.raises(ValidationError):
        ListenReplySettings(later_chunk_max_words=2)
    assert ListenReplySettings(first_chunk_max_words=3).first_chunk_max_words == 3
    assert ListenLlmSettings(provider="local", fallback="none").provider == "local"
