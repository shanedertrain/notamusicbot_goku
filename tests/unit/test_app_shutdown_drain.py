"""Review Round-6 §3/B: application inference shutdown — drain, no restart
after shutdown, and the Discord close boundary.

Deterministic fakes only: the TTS synthesizer parks on an explicit asyncio
Event after the job acquired the GPU lease; its shutdown() records teardown.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from notamusicbot.app import Application
from notamusicbot.config import AppSettings, SpeechSettings
from notamusicbot.discord_bot.guild_state import GuildState
from notamusicbot.discord_bot.player import stop_guild_players
from notamusicbot.domain.speech import AudioArtifact, AudioProcessingProfile, TtsRequest
from notamusicbot.speech.coordinator import CoordinatorClosedError
from notamusicbot.speech.pipeline import SpeechPipeline


def _write_wav(path: Path, seconds: float = 0.1) -> None:
    """Write a real (libsndfile-readable) mono WAV so postprocess can read it."""
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16000
    n = int(rate * seconds)
    sf.write(path, 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / rate), rate, subtype="PCM_16")


def _app(synth: Any | None = None) -> Application:
    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            commentary_enabled=True,
        )
    )
    factory = (lambda: synth) if synth is not None else None
    return Application(settings, synthesizer_factory=factory)


class _BlockingSynthesizer:
    """TTS fake that parks on an event AFTER the pipeline acquired the lease.

    ``entered`` is set inside ``synthesize`` — by then the detached holder
    holds the coordinator's GPU lease, so it proves the job started.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.shutdowns = 0

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        _write_wav(output_path)
        return AudioArtifact(path=output_path, sample_rate_hz=16000, duration_s=0.1)

    async def shutdown(self) -> None:
        self.shutdowns += 1


def _pipeline(app: Application, synth: Any, tmp_path: Path) -> SpeechPipeline:
    app.load_speech_model()  # install the injected synthesizer for teardown
    return SpeechPipeline(
        synth,
        AudioProcessingProfile(),
        coordinator=app.coordinator,
        output_dir=tmp_path / "speech",
    )


async def test_pipeline_job_drains_before_synthesizer_teardown(tmp_path: Path) -> None:
    """A running pipeline job parked in TTS is drained to completion BEFORE
    the synthesizer is shut down; afterwards no new job can start."""
    synth = _BlockingSynthesizer()
    app = _app(synth)
    pipeline = _pipeline(app, synth, tmp_path)

    job = asyncio.create_task(pipeline.run(TtsRequest(text="hello")))
    # Deterministic: the job holds the GPU lease and is parked inside TTS.
    await synth.entered.wait()

    shutdown_task = asyncio.create_task(app.shutdown())
    # Deterministic: shutdown stopped accepting new work (closed flag set
    # first), but the drain cannot finish while the job is parked in TTS.
    while not app.coordinator.closed:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not shutdown_task.done(), "shutdown drained past a still-running job"
    assert synth.shutdowns == 0, "synthesizer torn down while a job was running"

    synth.release.set()  # let the job finish
    result = await asyncio.wait_for(job, 10)
    assert result[0].path.is_file()
    await asyncio.wait_for(shutdown_task, 10)
    assert synth.shutdowns == 1

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


# ---------------------------------------------------------------------------
# Discord close boundary (review Round-6 §3/B): players stop before teardown
# ---------------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_shutdown_drain_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live-shutdown hardening: if the coordinator drain stalls (a long TTS
    clip in flight), app.shutdown() must not wait forever — it proceeds to
    synthesizer teardown after the (shortened, injected) drain timeout."""
    synth = _BlockingSynthesizer()
    app = _app(synth)
    app.load_speech_model()  # install the injected synthesizer for teardown

    class _StalledCoordinator:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            await asyncio.Event().wait()  # never completes

        async def run_detached(self, async_fn: Any, *args: Any) -> Any:
            return await async_fn()

    app.coordinator = _StalledCoordinator()  # type: ignore[assignment]
    monkeypatch.setattr(Application, "SHUTDOWN_DRAIN_TIMEOUT_S", 0.2)

    await asyncio.wait_for(app.shutdown(), 10)

    assert app.coordinator.close_calls == 1
    assert synth.shutdowns == 1  # teardown happened despite the stall


@pytest.mark.asyncio
async def test_shutdown_still_drains_when_job_finishes_in_time() -> None:
    """The normal path is unchanged: a coordinator that closes promptly is
    awaited to completion before synthesizer teardown."""
    order: list[str] = []

    class _OrderingSynth(_BlockingSynthesizer):
        async def shutdown(self) -> None:
            order.append("synth-shutdown")
            await super().shutdown()

    synth = _OrderingSynth()
    app = _app(synth)
    app.load_speech_model()  # install the injected synthesizer for teardown

    class _PromptCoordinator:
        async def close(self) -> None:
            order.append("drained")

    app.coordinator = _PromptCoordinator()  # type: ignore[assignment]
    await app.shutdown()
    assert order == ["drained", "synth-shutdown"]
