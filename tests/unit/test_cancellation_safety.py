"""Review Round-4 P0: cancellation-safe GPU lease (Option B design).

Proves the Round-4 review's required invariant:

    cancelling the awaiter must NOT make the GPU available while the
    underlying blocking inference is still running.

Covers:

- ``InferenceCoordinator.run_blocking`` core lease semantics (lease held until
  the thread future really completes, even after caller cancellation).
- Double-acquire guard (no nested run_blocking from a slot holder).
- A-LLM: cancelled ``LocalQwenLlmProvider.generate`` keeps the shared lease
  until the blocking ``_generate_sync`` finished.
- A-TTS / A-RVC: cancelled ``SpeechPipeline.run`` keeps the lease until the
  fake blocking synthesizer / RVC converter work finished (the whole
  TTS -> VC -> postprocess chain runs inside one ``run_blocking`` lease; the
  RVC adapter's worker-reply wait thread lives inside that chain).
- RVC orphan replies: a late reply for a cancelled request id is discarded.
- B: post-song generation in flight, player ``stop()`` cancels it, a second
  inference request must not overlap (logical ``max_active == 1``) and the
  queue is not repopulated after stop.
- Startup ordering: content LLM load and TTS warmup cannot overlap.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from conftest import FakeConverter, FakeSynthesizer
from notamusicbot.app import Application
from notamusicbot.config import AppSettings, ContentSettings, SpeechSettings
from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.llm import LocalQwenLlmProvider
from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.discord_bot.player import GuildPlayer
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.speech import (
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

BLOCKING_S = 0.3
PROCESSING = AudioProcessingProfile(target_sample_rate_hz=48000, channels=1)
CONVERSION = VoiceConversionProfile(backend="passthrough")


class _SlowBlocking:
    """Shared bookkeeping for a fake blocking GPU function."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.finished = threading.Event()
        self.active = 0
        self.max_active = 0

    def work(self) -> str:
        self.started.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(BLOCKING_S)
        finally:
            self.active -= 1
            self.finished.set()
        return "ok"


async def _wait_thread_event(event: threading.Event, timeout: float = 5.0) -> None:
    if not await asyncio.to_thread(event.wait, timeout):
        pytest.fail("fake blocking work never started/finished")


# --------------------------------------------------------------------------
# run_blocking core semantics
# --------------------------------------------------------------------------


async def test_run_blocking_returns_result_and_propagates_errors() -> None:
    coord = InferenceCoordinator()
    assert await coord.run_blocking(lambda: 7) == 7

    def boom() -> None:
        raise ValueError("gpu exploded")

    with pytest.raises(ValueError, match="gpu exploded"):
        await coord.run_blocking(boom)


async def test_run_blocking_lease_survives_caller_cancellation() -> None:
    """Core Round-4 P0 invariant at the coordinator level."""
    coord = InferenceCoordinator()
    job = _SlowBlocking()
    task = asyncio.create_task(coord.run_blocking(job.work))
    await _wait_thread_event(job.started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The blocking thread is still running here; job B must not enter.
    b_entered_after_finish: bool | None = None

    async def job_b() -> None:
        nonlocal b_entered_after_finish
        async with coord.slot():
            b_entered_after_finish = job.finished.is_set()

    await asyncio.wait_for(job_b(), 10)
    assert b_entered_after_finish is True, (
        "job B entered the coordinator while the cancelled job's blocking thread was still running"
    )
    # The detached blocking work really completed under the lease.
    await _wait_thread_event(job.finished)
    assert coord.pending == 0


async def test_run_blocking_bounded_pending_queue() -> None:
    coord = InferenceCoordinator(max_concurrent=1, queue_size=2)
    release = threading.Event()

    def hold() -> None:
        release.wait(5)

    holder = asyncio.create_task(coord.run_blocking(hold))
    queued = [asyncio.create_task(coord.run_blocking(lambda: None)) for _ in range(2)]
    await asyncio.sleep(0.05)
    # 1 admitted + 2 queued = 3; admission bound is max_concurrent + queue_size.
    assert coord.pending == 3
    overflow_started = asyncio.Event()

    async def overflow() -> None:
        overflow_started.set()
        await coord.run_blocking(lambda: None)

    overflow_task = asyncio.create_task(overflow())
    await asyncio.sleep(0.05)
    assert overflow_started.is_set()
    # Beyond the admission bound the submission waits (bounded queue).
    assert coord.pending == 3
    release.set()
    await asyncio.wait_for(holder, 10)
    await asyncio.gather(*queued)
    await asyncio.wait_for(overflow_task, 10)


async def test_run_blocking_rejects_nested_double_acquire() -> None:
    coord = InferenceCoordinator()
    errors: list[Exception] = []

    async def nested() -> None:
        async with coord.slot():
            try:
                await coord.run_blocking(lambda: None)
            except RuntimeError as exc:
                errors.append(exc)

    await asyncio.wait_for(nested(), 5)
    assert len(errors) == 1


# --------------------------------------------------------------------------
# A-LLM: cancelled LLM generate keeps the shared lease
# --------------------------------------------------------------------------


async def test_cancelled_llm_generate_holds_shared_slot() -> None:
    coord = InferenceCoordinator()
    provider = LocalQwenLlmProvider(gpu_coordinator=coord)
    job = _SlowBlocking()
    provider._generate_sync = lambda _prompt: job.work()  # type: ignore[method-assign]
    request = ContentRequest(kind=ContentKind.SONG, song_name="X", requester_name="Y")

    task = asyncio.create_task(provider.generate(request))
    await _wait_thread_event(job.started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    b_entered_after_finish: bool | None = None

    async def job_b() -> None:
        nonlocal b_entered_after_finish
        async with coord.slot():
            b_entered_after_finish = job.finished.is_set()

    await asyncio.wait_for(job_b(), 10)
    assert b_entered_after_finish is True, (
        "second LLM job entered the shared coordinator while the cancelled "
        "generate's blocking thread was still running"
    )
    assert job.max_active == 1


# --------------------------------------------------------------------------
# A-TTS / A-RVC: cancelled pipeline run keeps the lease for the whole chain
# --------------------------------------------------------------------------


class _SlowSynthesizer(FakeSynthesizer):
    """Fake synthesizer whose blocking model call sleeps (TTS-stage fake)."""

    def __init__(self) -> None:
        FakeSynthesizer.__init__(self)
        self.job = _SlowBlocking()

    async def synthesize(self, request, output_path):
        self.calls.append(request)
        # Blocking GPU work in a thread, exactly like the real Qwen call.
        await asyncio.to_thread(self.job.work)
        import numpy as np
        import soundfile as sf

        n = 2400
        wave = 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / 24000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(sf.write, output_path, wave, 24000, subtype="PCM_16")
        from notamusicbot.domain.speech import AudioArtifact

        return AudioArtifact(path=output_path, sample_rate_hz=24000, duration_s=0.1)


class _SlowConverter(FakeConverter):
    """Fake RVC-style converter: blocking worker-reply wait in a thread."""

    def __init__(self) -> None:
        FakeConverter.__init__(self)
        self.job = _SlowBlocking()

    async def convert(self, source, profile, output_path):
        self.calls.append((source, profile))
        import shutil

        # Emulates RvcVoiceConverter.convert: the worker subprocess reply is
        # awaited in a thread INSIDE the pipeline-held lease.
        await asyncio.to_thread(self.job.work)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source.path, output_path)
        from notamusicbot.domain.speech import AudioArtifact

        return AudioArtifact(
            path=output_path,
            sample_rate_hz=source.sample_rate_hz,
            duration_s=source.duration_s,
        )


def _slow_pipeline(tmp_path: Path, coord: InferenceCoordinator, synth, conv) -> SpeechPipeline:
    return SpeechPipeline(
        synth,
        conv,
        PROCESSING,
        coordinator=coord,
        output_dir=tmp_path / "speech",
        keep_artifacts=True,
    )


async def _cancelled_pipeline_holds_lease(tmp_path: Path, synth, conv, job: _SlowBlocking) -> None:
    coord = InferenceCoordinator()
    pipeline = _slow_pipeline(tmp_path, coord, synth, conv)
    request = TtsRequest(text="Hello there", language="English", speaker="Aiden")

    task = asyncio.create_task(pipeline.run(request, CONVERSION))
    await _wait_thread_event(job.started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    b_entered_after_finish: bool | None = None

    async def job_b() -> None:
        nonlocal b_entered_after_finish
        async with coord.slot():
            b_entered_after_finish = job.finished.is_set()

    await asyncio.wait_for(job_b(), 10)
    assert b_entered_after_finish is True, (
        "a second GPU job entered the coordinator while the cancelled "
        "pipeline's blocking stage thread was still running"
    )
    await _wait_thread_event(job.finished)
    assert job.max_active == 1


async def test_cancelled_pipeline_tts_stage_holds_lease(tmp_path) -> None:
    synth = _SlowSynthesizer()
    await _cancelled_pipeline_holds_lease(tmp_path, synth, FakeConverter(), synth.job)


async def test_cancelled_pipeline_rvc_stage_holds_lease(tmp_path) -> None:
    conv = _SlowConverter()
    await _cancelled_pipeline_holds_lease(tmp_path, FakeSynthesizer(), conv, conv.job)


# --------------------------------------------------------------------------
# A-RVC: orphaned worker replies are discarded safely
# --------------------------------------------------------------------------


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class _FakeProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)
        self.returncode = None

    def poll(self) -> int | None:
        return None


def test_rvc_late_reply_for_cancelled_request_is_discarded() -> None:
    conv = RvcVoiceConverter()
    # Request 7 was cancelled while its reply thread waited; a request 8 is
    # reading now and first receives the late reply for 7.
    conv._orphaned_ids.add(7)
    process = _FakeProcess(
        [
            '{"id": 7, "status": "ok", "output_wav_path": "orphan.wav"}\n',
            '{"id": 8, "status": "ok", "output_wav_path": "mine.wav"}\n',
        ]
    )
    reply = conv._read_reply(process, 8)
    assert reply["id"] == 8
    assert reply["output_wav_path"] == "mine.wav"
    assert 7 not in conv._orphaned_ids


def test_rvc_mismatched_reply_is_skipped() -> None:
    conv = RvcVoiceConverter()
    process = _FakeProcess(
        [
            '{"id": 99, "status": "ok"}\n',
            '{"id": 8, "status": "ok"}\n',
        ]
    )
    reply = conv._read_reply(process, 8)
    assert reply["id"] == 8


# --------------------------------------------------------------------------
# B: stop during post-song generation, then a new inference request
# --------------------------------------------------------------------------


class _FakeVoiceClient:
    def __init__(self) -> None:
        self._after = None

    def is_connected(self) -> bool:
        return True

    def is_playing(self) -> bool:
        return False

    def play(self, source, after) -> None:
        self._after = after
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: after(None))

    def stop(self) -> None:
        if self._after is not None:
            self._after(None)


def _stream_media(title: str) -> PlayableMedia:
    return PlayableMedia(
        media_id=title.lower().replace(" ", "-"),
        kind=MediaKind.STREAM,
        title=title,
        stream_url="https://example.invalid/song",
        requester="tester",
    )


async def test_stop_during_post_song_generation_no_overlap_no_repopulation(tmp_path) -> None:
    coord = InferenceCoordinator()
    synth = _SlowSynthesizer()
    llm_job = _SlowBlocking()
    provider = LocalQwenLlmProvider(gpu_coordinator=coord)
    provider._generate_sync = lambda _prompt: llm_job.work()  # type: ignore[method-assign]
    pipeline = _slow_pipeline(tmp_path, coord, synth, FakeConverter())

    state = GuildState(1)
    vc = _FakeVoiceClient()
    state.queue.put_nowait(_stream_media("Song A"))

    async def post_song_hook(media, outcome) -> None:
        # In-flight post-song generation: fake TTS (pipeline) then fake LLM.
        await pipeline.run(
            TtsRequest(text="Post song", language="English", speaker="Aiden"), CONVERSION
        )
        await provider.generate(
            ContentRequest(kind=ContentKind.SONG, song_name="Song A", requester_name="tester")
        )
        # Would repopulate the queue — must never run after stop().
        state.put_front(_stream_media("POST-STOP SEGMENT"))

    player = GuildPlayer(
        state,
        get_voice_client=lambda: vc,
        source_factory=lambda media: object(),
        on_playback_complete=post_song_hook,
    )
    player_task = player.start()
    # Let the song complete and the hook run through its fake TTS stage, then
    # reach the fake LLM stage: cancellation lands while the LLM await is in
    # flight (fake LLM + fake TTS both exercised in the hook).
    await _wait_thread_event(synth.job.finished)
    await _wait_thread_event(llm_job.started)

    # !gstop: cancels the player task while post-song generation is in flight.
    await player.stop(timeout=0.05)
    with pytest.raises(asyncio.CancelledError):
        await player_task
    assert player_task.cancelled()

    # Immediately issue a second inference request through the SAME shared
    # coordinator; it must wait for the detached generation's blocking work.
    probe_entered_after_finish: bool | None = None

    async def probe() -> None:
        nonlocal probe_entered_after_finish
        async with coord.slot():
            llm_job.active += 1
            llm_job.max_active = max(llm_job.max_active, llm_job.active)
            try:
                probe_entered_after_finish = llm_job.finished.is_set()
            finally:
                llm_job.active -= 1

    await asyncio.wait_for(probe(), 10)
    assert probe_entered_after_finish is True, (
        "a new inference request acquired the GPU lease while the cancelled "
        "post-song generation was still running"
    )
    # The detached generation's blocking work really ended.
    await _wait_thread_event(llm_job.finished)
    # Logical no-overlap invariant across the fake TTS, fake LLM and probe.
    assert llm_job.max_active == 1
    assert synth.job.max_active == 1
    # The queue was NOT repopulated after stop().
    assert state.queue.empty()


# --------------------------------------------------------------------------
# Startup ordering (review section 3): LLM load vs TTS warmup
# --------------------------------------------------------------------------


async def test_startup_load_and_warmup_do_not_overlap(monkeypatch) -> None:
    events: list[str] = []
    lock = threading.Lock()

    def record(event: str) -> None:
        with lock:
            events.append(event)

    class _FakeLlm:
        @classmethod
        def from_settings(cls, settings, gpu_coordinator=None) -> _FakeLlm:
            return cls()

        def is_loaded(self) -> bool:
            return True

        def load(self) -> object:
            record("load-start")
            time.sleep(0.15)
            record("load-end")
            return object()

    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _FakeLlm)

    class _FakeSynth:
        def warmup(self, output_path: Path) -> object:
            record("warmup-start")
            time.sleep(0.15)
            record("warmup-end")
            return object()

        async def synthesize(self, request, output_path):  # pragma: no cover
            raise AssertionError("not used in startup test")

    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            voice_conversion_backends=["passthrough"],
            warmup_on_start=True,
            output_dir="var/test-speech",
        ),
        content=ContentSettings(enabled=True, provider="local_qwen"),
    )
    app = Application(settings)
    app._synthesizer = _FakeSynth()

    await app.startup()
    assert app._warmup_task is not None
    await asyncio.wait_for(app._warmup_task, 10)

    assert events == ["load-start", "load-end", "warmup-start", "warmup-end"], (
        "content LLM load and TTS warmup overlapped on startup"
    )
