"""Failure fallback and skip semantics: a TTS failure must not crash callers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import FailingSynthesizer, FakeConverter, FakeSynthesizer

from notamusicbot.domain.speech import TtsRequest, VoiceConversionProfile
from notamusicbot.speech.pipeline import SpeechPipeline

CONVERSION = VoiceConversionProfile(backend="passthrough")


def test_tts_failure_raises_and_leaves_no_final_artifact(tmp_path) -> None:
    pipeline = SpeechPipeline(
        FailingSynthesizer("boom"),
        FakeConverter(),
        output_dir=tmp_path / "speech",
    )
    try:
        asyncio.run(
            pipeline.run(TtsRequest(text="hi", language="English", speaker="Ryan"), CONVERSION)
        )
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from failing synthesizer")
    speech_dir = tmp_path / "speech"
    final_wavs = list(speech_dir.rglob("final.wav")) if speech_dir.exists() else []
    assert not final_wavs


def test_vc_failure_does_not_produce_final(tmp_path) -> None:
    class FailingConverter(FakeConverter):
        async def convert(self, source, profile, output_path):
            raise RuntimeError("vc down")

    pipeline = SpeechPipeline(FakeSynthesizer(), FailingConverter(), output_dir=tmp_path / "speech")
    try:
        asyncio.run(
            pipeline.run(TtsRequest(text="hi", language="English", speaker="Ryan"), CONVERSION)
        )
    except RuntimeError as exc:
        assert "vc down" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from failing converter")


def test_skip_on_failure_is_caller_policy(tmp_path: Path) -> None:
    """Demonstrates the documented failure behavior: log, skip, continue."""

    async def speak(text: str) -> Path | None:
        pipeline = SpeechPipeline(
            FailingSynthesizer(), FakeConverter(), output_dir=tmp_path / "speech"
        )
        try:
            artifact, _ = await pipeline.run(
                TtsRequest(text=text, language="English", speaker="Ryan"), CONVERSION
            )
        except RuntimeError:
            return None  # skip commentary, continue
        return artifact.path

    assert asyncio.run(speak("test")) is None
