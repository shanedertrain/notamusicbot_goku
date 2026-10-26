"""Config tests for the opt-in streaming RVC settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from notamusicbot.config import VoiceConversionSettings, load_app_settings
from notamusicbot.speech.voice_conversion.base import VoiceStreamParams


def test_defaults_off_and_numeric_defaults():
    s = VoiceConversionSettings()
    assert s.streaming is False
    assert s.f0_method_override is None
    assert (
        s.stream_first_block_ms,
        s.stream_block_ms,
        s.stream_context_ms,
        s.stream_lookahead_ms,
        s.stream_crossfade_ms,
        s.stream_search_ms,
    ) == (100, 320, 1000, 100, 40, 20)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_first_block_ms", 130),
        ("stream_block_ms", 330),
        ("stream_context_ms", 1010),
        ("stream_lookahead_ms", 110),
        ("stream_crossfade_ms", 50),
        ("stream_search_ms", 30),
    ],
)
def test_non_multiple_of_20_rejected(field, value):
    with pytest.raises(ValidationError, match="multiple of 20"):
        VoiceConversionSettings(**{field: value})


def test_30_ms_rejected():
    with pytest.raises(ValidationError):
        VoiceConversionSettings(stream_crossfade_ms=30)


def test_search_zero_allowed():
    assert VoiceConversionSettings(stream_search_ms=0).stream_search_ms == 0


def test_range_violations_rejected():
    with pytest.raises(ValidationError):
        VoiceConversionSettings(stream_block_ms=80)
    with pytest.raises(ValidationError):
        VoiceConversionSettings(stream_lookahead_ms=20)
    with pytest.raises(ValidationError):
        VoiceConversionSettings(stream_context_ms=4020)


def test_crossfade_not_above_block():
    with pytest.raises(ValidationError, match="crossfade"):
        VoiceConversionSettings(stream_block_ms=100, stream_crossfade_ms=120)
    VoiceConversionSettings(stream_block_ms=100, stream_crossfade_ms=100)


def test_yaml_round_trip(tmp_path: Path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text(
        "speech:\n"
        "  voice_conversion:\n"
        "    f0_method_override: pm\n"
        "    streaming: true\n"
        "    stream_block_ms: 400\n"
        "    stream_search_ms: 0\n",
        encoding="utf-8",
    )
    vc = load_app_settings(cfg).speech.voice_conversion
    assert vc.streaming is True and vc.f0_method_override == "pm"
    assert vc.stream_block_ms == 400 and vc.stream_search_ms == 0
    assert vc.stream_context_ms == 1000


def test_yaml_invalid_value_rejected(tmp_path: Path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text("speech:\n  voice_conversion:\n    stream_block_ms: 330\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_app_settings(cfg)


def test_example_yaml_still_loads():
    example = Path(__file__).resolve().parents[2] / "config" / "app.example.yaml"
    assert load_app_settings(example).speech.voice_conversion.streaming is False


def test_from_settings_mapping():
    s = VoiceConversionSettings(
        streaming=True,
        stream_first_block_ms=60,
        stream_block_ms=400,
        stream_context_ms=1200,
        stream_lookahead_ms=80,
        stream_crossfade_ms=60,
        stream_search_ms=40,
    )
    assert VoiceStreamParams.from_settings(s) == VoiceStreamParams(
        enabled=True,
        first_block_ms=60,
        block_ms=400,
        context_ms=1200,
        lookahead_ms=80,
        crossfade_ms=60,
        search_ms=40,
    )
    assert VoiceStreamParams.from_settings(VoiceConversionSettings()).enabled is False
    assert VoiceStreamParams() == VoiceStreamParams.from_settings(VoiceConversionSettings())
