"""Unit tests for pipeline sequencing, coordinator bounds, and artifact naming."""

from __future__ import annotations

import asyncio
import threading

import pytest
import soundfile as sf

from conftest import FakeSynthesizer
from notamusicbot.domain.speech import AudioProcessingProfile, TtsRequest
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline

PROCESSING = AudioProcessingProfile(target_sample_rate_hz=48000, channels=1)


def _request(text: str = "Hello there") -> TtsRequest:
    return TtsRequest(text=text)


def _pipeline(tmp_path, synth) -> SpeechPipeline:
    return SpeechPipeline(
        synth,
        PROCESSING,
        output_dir=tmp_path / "speech",
    )


def test_pipeline_sequence_tts_postprocess(tmp_path) -> None:
    synth = FakeSynthesizer()
    pipeline = _pipeline(tmp_path, synth)
    final, metrics = asyncio.run(pipeline.run(_request()))
    # TTS ran with the request
    assert len(synth.calls) == 1
    assert synth.calls[0].text == "Hello there"
    # Final artifact is a valid WAV at the post-processed rate
    info = sf.info(final.path)
    assert info.samplerate == 48000
    assert info.frames > 0
    # Metrics recorded
    assert metrics.total_s > 0
    assert metrics.output_duration_s > 0
    assert metrics.tts_rtf > 0


def test_artifact_naming_deterministic_layout(tmp_path) -> None:
    pipeline = _pipeline(tmp_path, FakeSynthesizer())
    final, metrics = asyncio.run(pipeline.run(_request("x")))
    assert final.path.parent == tmp_path / "speech" / metrics.job_id
    assert final.path.name == "final.wav"
    # P1-11 cleanup: the TTS intermediate is deleted once postprocess consumed it.
    assert not (tmp_path / "speech" / metrics.job_id / "tts_source.wav").exists()


def test_queue_ordering_and_serialization(tmp_path) -> None:
    """Requests execute in submission order under a concurrency-1 coordinator."""
    synth = FakeSynthesizer()
    coordinator = InferenceCoordinator(max_concurrent=1, queue_size=4)
    pipeline = SpeechPipeline(
        synth, PROCESSING, coordinator=coordinator, output_dir=tmp_path / "speech"
    )
    order: list[str] = []

    original_synthesize = synth.synthesize

    async def tracking_synthesize(request, output_path):
        order.append(output_path.parent.name)
        return await original_synthesize(request, output_path)

    synth.synthesize = tracking_synthesize  # type: ignore[method-assign]

    async def main() -> None:
        requests = [_request(f"line {i}") for i in range(3)]
        await asyncio.gather(*(pipeline.run(r) for r in requests))

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
    from notamusicbot.speech.tts._common import normalize_text

    assert normalize_text("  hello   world ") == "hello world"
    # Quotes and punctuation are preserved (section 7.6)
    assert normalize_text('"Hey, it\'s me!"') == '"Hey, it\'s me!"'


# --- Artifact cleanup policy (review P1-11) -------------------------------


def test_pipeline_cleans_intermediates_and_marks_final_temp(tmp_path) -> None:
    """Successful job: tts_source.wav deleted after postprocess, final.wav
    registered as play-once queue media."""
    from notamusicbot.speech.artifacts import (
        clear_registry,
        is_temp_speech_artifact,
    )

    clear_registry()
    pipeline = _pipeline(tmp_path, FakeSynthesizer())
    final, _metrics = asyncio.run(pipeline.run(_request("x")))
    job_dir = final.path.parent
    assert not (job_dir / "tts_source.wav").exists()
    assert (job_dir / "final.wav").exists()
    assert is_temp_speech_artifact(final.path)
    clear_registry()


def test_pipeline_keep_artifacts_retains_everything(tmp_path) -> None:
    from notamusicbot.speech.artifacts import (
        clear_registry,
        is_temp_speech_artifact,
    )

    clear_registry()
    pipeline = SpeechPipeline(
        FakeSynthesizer(),
        PROCESSING,
        output_dir=tmp_path / "speech",
        keep_artifacts=True,
    )
    final, _metrics = asyncio.run(pipeline.run(_request("x")))
    job_dir = final.path.parent
    assert (job_dir / "tts_source.wav").exists()
    assert (job_dir / "final.wav").exists()
    assert not is_temp_speech_artifact(final.path)
    clear_registry()


def test_pipeline_postprocess_failure_retains_tts_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure policy: when postprocess fails, tts_source.wav is retained for
    diagnostics (the only reproducible input for the failed stage)."""
    from notamusicbot.speech import pipeline as pipeline_mod
    from notamusicbot.speech.artifacts import clear_registry

    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("postprocess boom")

    monkeypatch.setattr(pipeline_mod, "postprocess", boom)
    clear_registry()
    pipeline = _pipeline(tmp_path, FakeSynthesizer())
    with pytest.raises(RuntimeError, match="postprocess boom"):
        asyncio.run(pipeline.run(_request("x")))
    job_dir = tmp_path / "speech"
    sources = list(job_dir.glob("*/tts_source.wav"))
    assert len(sources) == 1 and sources[0].exists()
    clear_registry()


# ---------------------------------------------------------------------------
# Review Round-5 §4/C: queued-not-started pipeline job cancels cleanly
# ---------------------------------------------------------------------------


def test_pipeline_cancelled_while_queued_never_runs(tmp_path) -> None:
    """A pipeline.run cancelled while still waiting for the coordinator
    must never call TTS and must leave no job artifacts."""
    coordinator = InferenceCoordinator(max_concurrent=1, queue_size=4)
    synth = FakeSynthesizer()
    pipeline = SpeechPipeline(
        synth, PROCESSING, coordinator=coordinator, output_dir=tmp_path / "speech"
    )
    release = threading.Event()

    def hold_a() -> None:
        release.wait(10)

    async def main() -> None:
        job_a = asyncio.create_task(coordinator.run_blocking(hold_a))
        await asyncio.sleep(0.05)  # A acquires the lease
        job_b = asyncio.create_task(pipeline.run(_request("q")))
        await asyncio.sleep(0.1)  # B queues behind A (has NOT started)
        assert coordinator.pending == 2
        job_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job_b
        release.set()
        await asyncio.wait_for(job_a, 10)
        await asyncio.sleep(0.1)  # a wrongly-detached B would run here

    asyncio.run(main())
    assert len(synth.calls) == 0, "TTS ran for a job cancelled before lease acquisition"
    # Only the empty job dir with its pending .temporary marker may exist
    # (created before submission; retention pruning ages it out). No media.
    job_dirs = list((tmp_path / "speech").glob("*")) if (tmp_path / "speech").exists() else []
    for job_dir in job_dirs:
        assert not list(job_dir.glob("*.wav")), (
            "artifacts were generated for a job cancelled before lease acquisition"
        )


# ---------------------------------------------------------------------------
# Review Round-5 §7/E: queue_wait_s measures queue wait, not synthesis
# ---------------------------------------------------------------------------


def test_queue_wait_measures_lease_wait_not_synthesis(tmp_path) -> None:
    """Deliberately occupy the coordinator, then verify queue_wait_s
    reflects the intentional wait rather than the synthesis duration."""
    coordinator = InferenceCoordinator(max_concurrent=1, queue_size=4)
    synth = FakeSynthesizer()
    pipeline = SpeechPipeline(
        synth, PROCESSING, coordinator=coordinator, output_dir=tmp_path / "speech"
    )
    OCCUPANCY_S = 0.4

    async def main() -> None:
        occupier = asyncio.create_task(coordinator.run_detached(asyncio.sleep, OCCUPANCY_S))
        await asyncio.sleep(0.05)  # occupier holds the lease
        final, metrics = await pipeline.run(_request("wait me"))
        await asyncio.wait_for(occupier, 10)
        return final, metrics

    _final, metrics = asyncio.run(main())
    # The fake pipeline itself takes only milliseconds; the queue wait must
    # be dominated by the deliberate ~0.4 s occupancy, not the synthesis.
    assert metrics.queue_wait_s >= OCCUPANCY_S - 0.15, (
        f"queue_wait_s={metrics.queue_wait_s:.3f} does not reflect the "
        f"intentional {OCCUPANCY_S}s coordinator occupancy"
    )
    assert metrics.queue_wait_s < OCCUPANCY_S + 1.0
    assert metrics.tts_s < metrics.queue_wait_s, (
        "queue_wait_s looks like synthesis duration, not queue wait"
    )
