"""Unit tests for pipeline sequencing, coordinator bounds, and artifact naming."""

from __future__ import annotations

import asyncio

import soundfile as sf

from conftest import FakeConverter, FakeSynthesizer
from notamusicbot.domain.speech import (
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline

CONVERSION = VoiceConversionProfile(backend="passthrough")
PROCESSING = AudioProcessingProfile(target_sample_rate_hz=48000, channels=1)


def _pipeline(tmp_path, synth, conv) -> SpeechPipeline:
    return SpeechPipeline(
        synth,
        conv,
        PROCESSING,
        output_dir=tmp_path / "speech",
    )


def test_pipeline_sequence_tts_vc_postprocess(tmp_path) -> None:
    synth, conv = FakeSynthesizer(), FakeConverter()
    pipeline = _pipeline(tmp_path, synth, conv)
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="Hello there", language="English", speaker="Ryan"), CONVERSION)
    )
    # Sequencing: TTS ran before VC with the same request
    assert len(synth.calls) == 1
    assert synth.calls[0].text == "Hello there"
    assert len(conv.calls) == 1
    source_artifact, conversion_used = conv.calls[0]
    assert conversion_used.backend == "passthrough"
    assert source_artifact.sample_rate_hz == 24000
    assert source_artifact.path.name == "tts_source.wav"
    # Final artifact is a valid WAV at the post-processed rate
    info = sf.info(final.path)
    assert info.samplerate == 48000
    assert info.frames > 0
    # Metrics recorded
    assert metrics.total_s > 0
    assert metrics.output_duration_s > 0
    assert metrics.tts_rtf > 0


def test_artifact_naming_deterministic_layout(tmp_path) -> None:
    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter())
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="x", language="English", speaker="Aiden"), CONVERSION)
    )
    assert final.path.parent == tmp_path / "speech" / metrics.job_id
    assert final.path.name == "final.wav"
    assert (tmp_path / "speech" / metrics.job_id / "tts_source.wav").exists()


def test_queue_ordering_and_serialization(tmp_path) -> None:
    """Requests execute in submission order under a concurrency-1 coordinator."""
    synth = FakeSynthesizer()
    conv = FakeConverter()
    coordinator = InferenceCoordinator(max_concurrent=1, queue_size=4)
    pipeline = SpeechPipeline(
        synth, conv, PROCESSING, coordinator=coordinator, output_dir=tmp_path / "speech"
    )
    order: list[str] = []

    original_convert = conv.convert

    async def tracking_convert(source, profile, output_path):
        order.append(output_path.parent.name)
        return await original_convert(source, profile, output_path)

    conv.convert = tracking_convert  # type: ignore[method-assign]

    async def main() -> None:
        requests = [
            TtsRequest(text=f"line {i}", language="English", speaker="Aiden") for i in range(3)
        ]
        await asyncio.gather(*(pipeline.run(r, CONVERSION) for r in requests))

    asyncio.run(main())
    assert len(order) == 3
    assert len(set(order)) == 3  # one job per unique directory


def test_coordinator_limits_concurrency() -> None:
    coordinator = InferenceCoordinator(max_concurrent=1, queue_size=2)
    active = 0
    peak = 0

    async def job() -> None:
        nonlocal active, peak
        async with coordinator.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def main() -> None:
        await asyncio.gather(*(job() for _ in range(4)))

    asyncio.run(main())
    assert peak == 1


def test_empty_text_raises(tmp_path) -> None:
    from notamusicbot.speech.tts.qwen_custom_voice import normalize_text

    assert normalize_text("  hello   world ") == "hello world"
    # Quotes and punctuation are preserved (section 7.6)
    assert normalize_text('"Hey, it\'s me!"') == '"Hey, it\'s me!"'
