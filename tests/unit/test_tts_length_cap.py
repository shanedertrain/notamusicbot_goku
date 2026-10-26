"""TTS length cap: generation bound and post-generation trim (listen mode)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import FakeConverter, FakeSynthesizer
from notamusicbot.domain.speech import TtsRequest
from notamusicbot.domain.voice_profiles import VoiceProfile
from notamusicbot.listen.reply import ReplyAudio, ReplySynthesizer, tts_cap_seconds
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.tts.qwen_custom_voice import (
    QwenCustomVoiceSynthesizer,
    max_new_tokens_for,
)

_RATE = 48000


def _profile() -> VoiceProfile:
    return VoiceProfile.model_validate(
        {
            "id": "goku",
            "persona": {"display_name": "Goku"},
            "tts": {"speaker": "Ryan"},
            "voice_conversion": {"backend": "passthrough"},
            "audio": {},
        }
    )


def _synth(tmp_path: Path, fake: FakeSynthesizer) -> ReplySynthesizer:
    pipeline = SpeechPipeline(fake, FakeConverter(), output_dir=tmp_path / "speech")
    return ReplySynthesizer(pipeline, _profile(), artifact_root=tmp_path / "speech")


def _frames(audio: ReplyAudio) -> np.ndarray:
    return np.frombuffer(audio.pcm, dtype="<i2").reshape(-1, 2)


def test_cap_values() -> None:
    assert tts_cap_seconds("one two three four five") == 5.0
    assert tts_cap_seconds("Hm?") == 2.0
    assert tts_cap_seconds(" ".join(["w"] * 16)) == 16.0
    assert tts_cap_seconds("a b c", seconds_per_word=2.0, floor_seconds=1.0) == 6.0


def test_max_new_tokens_bound() -> None:
    assert max_new_tokens_for(None) is None
    assert max_new_tokens_for(5.0) == 76  # ceil(5 s * 12.5 tok/s) + 13 slack
    assert max_new_tokens_for(2.0) == 38  # 25 + 13
    assert max_new_tokens_for(24.0) == 313  # bound stays above the cap: 313 / 12.5 = 25.04 s


async def test_over_generation_is_trimmed_and_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeSynthesizer(seconds=12.4)
    text = "one two three four five six seven eight"  # 8 words -> 8.0 s cap
    with caplog.at_level(logging.WARNING, logger="notamusicbot.listen.reply"):
        audio = await _synth(tmp_path, fake).synthesize_sentence(text)
    assert fake.calls[0].max_audio_seconds == 8.0
    assert audio.truncated is True
    assert audio.duration_s == pytest.approx(8.0, abs=1 / _RATE)
    frames = _frames(audio)
    assert abs(frames.shape[0] - 8 * _RATE) <= 1
    fade = int(0.08 * _RATE)
    assert np.abs(frames[-fade - 200 : -fade]).max() > 10000  # body is loud
    assert np.abs(frames[-8:]).max() < 100  # fade-out ends near zero
    assert np.abs(frames[-fade:]).max() < np.abs(frames[:fade]).max()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "8 words" in warnings[0].getMessage()
    assert "12.40s" in warnings[0].getMessage()
    assert "8.00s" in warnings[0].getMessage()


async def test_within_cap_untouched(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeSynthesizer(seconds=3.0)
    with caplog.at_level(logging.WARNING, logger="notamusicbot.listen.reply"):
        audio = await _synth(tmp_path, fake).synthesize_sentence("one two three four five")
    assert fake.calls[0].max_audio_seconds == 5.0
    assert audio.truncated is False
    assert audio.duration_s == pytest.approx(3.0, abs=1 / _RATE)
    assert len(audio.pcm) == 3 * _RATE * 4
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_no_cap_leaves_long_audio_untouched(tmp_path: Path) -> None:
    fake = FakeSynthesizer(seconds=12.4)
    pipeline = SpeechPipeline(fake, FakeConverter(), output_dir=tmp_path / "speech")
    request = TtsRequest(text="hi", language="English", speaker="Ryan")
    assert request.max_audio_seconds is None
    artifact, _ = await pipeline.run(request, _synth(tmp_path, fake)._conversion)
    assert fake.calls[0].max_audio_seconds is None
    assert artifact.duration_s == pytest.approx(12.4, abs=0.01)


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate_custom_voice(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.kwargs = kwargs
        return [np.zeros(2400, dtype=np.float32)], 24000


@pytest.mark.parametrize(("cap", "expected"), [(5.0, 76), (None, None)])
def test_qwen_passes_max_new_tokens(
    tmp_path: Path, cap: float | None, expected: int | None
) -> None:
    model = _FakeModel()
    synth = QwenCustomVoiceSynthesizer(model)
    request = TtsRequest(text="Hi.", language="English", speaker="Ryan", max_audio_seconds=cap)
    asyncio.run(synth.synthesize(request, tmp_path / "o.wav"))
    if expected is None:
        assert "max_new_tokens" not in model.kwargs
    else:
        assert model.kwargs["max_new_tokens"] == expected
    assert model.kwargs["text"] == "Hi."
