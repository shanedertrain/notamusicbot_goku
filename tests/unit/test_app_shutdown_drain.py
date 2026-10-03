"""Review Round-6 §3/B: application inference shutdown — drain, no RVC
restart after shutdown, and the Discord close boundary.

Deterministic fakes only: the TTS synthesizer parks on an explicit asyncio
Event after the job acquired the GPU lease; the RVC converter is a spy that
counts starts/shutdowns.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from notamusicbot.app import Application
from notamusicbot.config import AppSettings, SpeechSettings
from notamusicbot.discord_bot.bot import stop_guild_players
from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.domain.speech import (
    AudioArtifact,
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.coordinator import CoordinatorClosedError
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.voice_conversion.router import RouterClosedError


def _write_wav(path: Path, seconds: float = 0.1) -> None:
    """Write a real (libsndfile-readable) mono WAV so postprocess can read it."""
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    n = int(rate * seconds)
    sf.write(path, 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / rate), rate, subtype="PCM_16")


def _app() -> Application:
    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            commentary_enabled=True,
        )
    )
    return Application(settings)


class _CountingConverter:
    """Spy RVC converter counting starts/shutdowns/converts."""

    def __init__(self) -> None:
        self.starts = 0
        self.shutdowns = 0
        self.converts = 0

    async def start(self) -> None:
        self.starts += 1

    async def shutdown(self) -> None:
        self.shutdowns += 1

    async def convert(
        self, source: AudioArtifact, profile: VoiceConversionProfile, output_path: Path
    ) -> AudioArtifact:
        self.converts += 1
        _write_wav(output_path)
        return AudioArtifact(path=output_path, sample_rate_hz=16000, duration_s=0.1)


class _BlockingSynthesizer:
    """TTS fake that parks on an event AFTER the pipeline acquired the lease.

    ``entered`` is set inside ``synthesize`` — by then the detached holder
    holds the coordinator's GPU lease, so it proves the job started.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        _write_wav(output_path)
        return AudioArtifact(path=output_path, sample_rate_hz=16000, duration_s=0.1)


def _request() -> TtsRequest:
    return TtsRequest(text="hello", language="en", speaker="default")


def _pipeline(
    app: Application, synth: Any, converter: _CountingConverter, tmp_path: Path
) -> SpeechPipeline:
    app._synthesizer = synth  # noqa: SLF001 - test injection boundary
    router = app.converter_router
    router._rvc = converter  # noqa: SLF001
    router._rvc_available = True  # noqa: SLF001
    router._rvc_start_attempted = True  # noqa: SLF001
    return SpeechPipeline(
        synth,
        app.converter,  # the real router; delegates to the injected spy
        AudioProcessingProfile(),
        coordinator=app.coordinator,
        output_dir=tmp_path / "speech",
    )


async def test_pipeline_during_application_shutdown_retires_before_rvc_shutdown(
    tmp_path: Path,
) -> None:
    """A running pipeline job parked in TTS is drained to completion BEFORE
    the RVC router is shut down — and the RVC start count never increases
    after shutdown (no lazy restart of a dead worker)."""
    app = _app()
    converter = _CountingConverter()
    synth = _BlockingSynthesizer()
    pipeline = _pipeline(app, synth, converter, tmp_path)

    job = asyncio.create_task(
        pipeline.run(_request(), VoiceConversionProfile(backend="rvc", model="Goku"))
    )
    # Deterministic: the job holds the GPU lease and is parked inside TTS.
    await synth.entered.wait()

    shutdown_task = asyncio.create_task(app.shutdown())
    # Deterministic: shutdown stopped accepting new work (closed flag set
    # first), but drain cannot finish while the job is parked in TTS.
    while not app.coordinator.closed:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not shutdown_task.done(), "shutdown drained past a still-running job"

    synth.release.set()  # let TTS finish; the chain proceeds to RVC
    result = await asyncio.wait_for(job, 10)
    assert result[0].path.is_file()
    await asyncio.wait_for(shutdown_task, 10)

    # Running job retired BEFORE RVC shutdown: its convert ran to completion,
    # exactly one shutdown, and the start count NEVER increased after
    # shutdown (the spy is pre-marked available, so starts stays 0 — the
    # post-shutdown assertions below prove no lazy restart either).
    assert converter.converts == 1
    assert converter.shutdowns == 1
    starts_before = converter.starts

    # Post-shutdown convert: explicit closed error, no restart.
    with pytest.raises(RouterClosedError):
        await app.converter_router.convert(
            AudioArtifact(path=result[0].path, sample_rate_hz=16000, duration_s=0.1),
            VoiceConversionProfile(backend="rvc", model="Goku"),
            result[0].path.with_name("late.wav"),
        )
    assert converter.starts == starts_before, "RVC restarted after application shutdown"

    # The coordinator now rejects new submissions outright.
    with pytest.raises(CoordinatorClosedError):
        await app.coordinator.run_detached(asyncio.sleep, 0)


async def test_coordinator_drain_via_app_rejects_new_submission() -> None:
    """app.shutdown() leaves the coordinator closed: new work is rejected."""
    app = _app()
    await app.shutdown()
    assert app.coordinator.closed is True
    with pytest.raises(CoordinatorClosedError):
        await app.coordinator.run_detached(asyncio.sleep, 0)
    with pytest.raises(RouterClosedError):
        await app.converter_router.convert(
            AudioArtifact(path="x.wav", sample_rate_hz=16000, duration_s=0.1),
            VoiceConversionProfile(backend="rvc", model="Goku"),
            Path("out.wav"),
        )


# ---------------------------------------------------------------------------
# Discord close boundary (review Round-6 §3/B): players stop before teardown
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self) -> None:
        self.stopped = asyncio.Event()
        self.cancelled = False
        self.done_flag = False

    def done(self) -> bool:
        return self.done_flag

    def cancel(self) -> None:
        self.cancelled = True

    def __await__(self):  # pragma: no cover - only used when done
        return self


async def test_stop_guild_players_signals_and_awaits_player_tasks() -> None:
    """stop_guild_players: request_stop is set, the queue is drained, and a
    still-running player task is awaited to its natural exit."""
    state = GuildState(1)
    exited = asyncio.Event()

    async def player_loop() -> None:
        await state.stop_event.wait()
        exited.set()

    task = asyncio.get_running_loop().create_task(player_loop())
    state.player_task = task  # type: ignore[assignment]
    state.queue.put_nowait(_FakeMedia())

    await stop_guild_players([state])

    assert exited.is_set()
    assert state.queue.empty()
    assert state.player_task is None


async def test_stop_guild_players_cancels_stuck_player_task() -> None:
    """A player task that will not exit on request is cancelled at shutdown."""
    state = GuildState(2)

    async def stuck_loop() -> None:
        await asyncio.Event().wait()  # never set

    task = asyncio.get_running_loop().create_task(stuck_loop())
    state.player_task = task  # type: ignore[assignment]

    await asyncio.wait_for(stop_guild_players([state]), 5)

    assert task.cancelled()
    assert state.player_task is None


class _FakeMedia:  # minimal duck-typed queue item for drain/cleanup
    media_id = "m1"
    title = "t"
    requester = "r"
    kind = None
    source_path = None  # cleanup_media_item() no-ops on unbacked items
    temp_file = False
