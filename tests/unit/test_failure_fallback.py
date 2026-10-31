"""Failure fallback and skip semantics: a speech failure must not crash callers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import FailingSynthesizer, FakeSynthesizer
from notamusicbot.domain.speech import TtsRequest
from notamusicbot.speech.pipeline import SpeechPipeline


def test_tts_failure_raises_and_leaves_no_final_artifact(tmp_path) -> None:
    pipeline = SpeechPipeline(
        FailingSynthesizer("boom"),
        output_dir=tmp_path / "speech",
    )
    try:
        asyncio.run(pipeline.run(TtsRequest(text="hi")))
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from failing synthesizer")
    speech_dir = tmp_path / "speech"
    final_wavs = list(speech_dir.rglob("final.wav")) if speech_dir.exists() else []
    assert not final_wavs


def test_postprocess_failure_does_not_produce_final(tmp_path, monkeypatch) -> None:
    from notamusicbot.speech import pipeline as pipeline_mod

    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("processing down")

    monkeypatch.setattr(pipeline_mod, "postprocess", boom)
    pipeline = SpeechPipeline(FakeSynthesizer(), output_dir=tmp_path / "speech")
    with pytest.raises(RuntimeError, match="processing down"):
        asyncio.run(pipeline.run(TtsRequest(text="hi")))
    final_wavs = list((tmp_path / "speech").rglob("final.wav"))
    assert not final_wavs


def test_skip_on_failure_is_caller_policy(tmp_path: Path) -> None:
    """Demonstrates the documented failure behavior: log, skip, continue."""

    async def speak(text: str) -> Path | None:
        pipeline = SpeechPipeline(FailingSynthesizer(), output_dir=tmp_path / "speech")
        try:
            artifact, _ = await pipeline.run(TtsRequest(text=text))
        except RuntimeError:
            return None  # skip commentary, continue
        return artifact.path

    assert asyncio.run(speak("test")) is None
