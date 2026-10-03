"""VoiceConverterRouter: backend routing, lifecycle, and failure policy."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.router import (
    RouterClosedError,
    RvcUnavailableError,
    UnknownBackendError,
    VoiceConverterRouter,
)

PROFILE_PASSTHROUGH = VoiceConversionProfile(backend="passthrough")
PROFILE_RVC = VoiceConversionProfile(backend="rvc", model="Goku")
PROFILE_UNKNOWN = VoiceConversionProfile(backend="skynet")


class SpyRvcConverter:
    """Spy standing in for RvcVoiceConverter at the composition root."""

    def __init__(self, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.starts = 0
        self.shutdowns = 0
        self.calls: list[tuple[AudioArtifact, VoiceConversionProfile, Path]] = []

    async def start(self) -> None:
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("no CUDA device available")

    async def shutdown(self) -> None:
        self.shutdowns += 1

    async def convert(
        self, source: AudioArtifact, profile: VoiceConversionProfile, output_path: Path
    ) -> AudioArtifact:
        self.calls.append((source, profile, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.path, output_path)
        return AudioArtifact(
            path=output_path, sample_rate_hz=source.sample_rate_hz, duration_s=source.duration_s
        )


def make_source(tmp_path: Path) -> AudioArtifact:
    path = tmp_path / "src.wav"
    path.write_bytes(b"fake-pcm")
    return AudioArtifact(path=path, sample_rate_hz=24000, duration_s=0.5)


async def test_unknown_backend_raises_clear_error(tmp_path: Path) -> None:
    router = VoiceConverterRouter(rvc_factory=SpyRvcConverter)
    with pytest.raises(UnknownBackendError, match="backend"):
        await router.convert(make_source(tmp_path), PROFILE_UNKNOWN, tmp_path / "out.wav")


async def test_disabled_backend_raises_even_if_known(tmp_path: Path) -> None:
    router = VoiceConverterRouter(enabled_backends=["passthrough"], rvc_factory=SpyRvcConverter)
    with pytest.raises(UnknownBackendError, match="not enabled"):
        await router.convert(make_source(tmp_path), PROFILE_RVC, tmp_path / "out.wav")


async def test_router_rejects_unknown_backend_names_up_front() -> None:
    with pytest.raises(UnknownBackendError):
        VoiceConverterRouter(enabled_backends=["passthrough", "sovits"])


async def test_passthrough_routing_is_a_no_op_and_starts_nothing(tmp_path: Path) -> None:
    spy = SpyRvcConverter()
    router = VoiceConverterRouter(enabled_backends=["passthrough"], rvc_factory=lambda: spy)
    await router.startup()
    out = tmp_path / "out.wav"
    result = await router.convert(make_source(tmp_path), PROFILE_PASSTHROUGH, out)
    assert result.path == out
    assert spy.starts == 0  # passthrough needs no worker
    assert out.read_bytes() == b"fake-pcm"
    await router.shutdown()
    assert spy.shutdowns == 0


async def test_rvc_routing_uses_the_owned_persistent_adapter(tmp_path: Path) -> None:
    spy = SpyRvcConverter()
    router = VoiceConverterRouter(rvc_factory=lambda: spy)
    await router.startup()
    assert router.rvc_available is True
    assert spy.starts == 1
    source = make_source(tmp_path)
    out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
    await router.convert(source, PROFILE_RVC, out1)
    await router.convert(source, PROFILE_RVC, out2)
    assert len(spy.calls) == 2  # same persistent adapter, no restarts
    assert spy.calls[0][1] is PROFILE_RVC
    await router.shutdown()
    assert spy.shutdowns == 1


async def test_rvc_startup_failure_is_logged_not_raised_and_policy_error_fails_loudly(
    tmp_path: Path,
) -> None:
    spy = SpyRvcConverter(fail_start=True)  # the documented no-GPU failure path
    router = VoiceConverterRouter(rvc_factory=lambda: spy, rvc_unavailable_policy="error")
    await router.startup()  # must not raise
    assert router.rvc_available is False
    with pytest.raises(RvcUnavailableError, match="unavailable"):
        await router.convert(make_source(tmp_path), PROFILE_RVC, tmp_path / "out.wav")


async def test_rvc_unavailable_policy_passthrough_degrades(tmp_path: Path) -> None:
    spy = SpyRvcConverter(fail_start=True)
    router = VoiceConverterRouter(rvc_factory=lambda: spy, rvc_unavailable_policy="passthrough")
    await router.startup()
    out = tmp_path / "out.wav"
    result = await router.convert(make_source(tmp_path), PROFILE_RVC, out)
    assert result.path == out  # degraded: source voice passes through un-converted
    assert spy.calls == []


# ---------------------------------------------------------------------------
# Review Round-6 §3/B: explicit closed/shutting-down state
# ---------------------------------------------------------------------------


async def test_convert_after_shutdown_raises_router_closed_and_does_not_restart(
    tmp_path: Path,
) -> None:
    """After shutdown(), a stray late convert must be REFUSED with an explicit
    error — never lazily restarting the RVC worker (review reproduction:
    starts went 1 -> 2 after shutdown)."""
    spy = SpyRvcConverter()
    router = VoiceConverterRouter(rvc_factory=lambda: spy)
    await router.startup()
    assert spy.starts == 1
    await router.shutdown()
    assert spy.shutdowns == 1

    with pytest.raises(RouterClosedError, match="shut down"):
        await router.convert(make_source(tmp_path), PROFILE_RVC, tmp_path / "out.wav")
    # Defense in depth held: no restart happened behind the caller's back.
    assert spy.starts == 1


async def test_deliberate_startup_cycle_after_shutdown_re_enables(tmp_path: Path) -> None:
    """A deliberate new startup() cycle clears the closed state and serves
    converts again."""
    spy = SpyRvcConverter()
    router = VoiceConverterRouter(rvc_factory=lambda: spy)
    await router.startup()
    await router.shutdown()
    with pytest.raises(RouterClosedError):
        await router.convert(make_source(tmp_path), PROFILE_RVC, tmp_path / "out.wav")

    await router.startup()
    assert router.rvc_available is True
    out = tmp_path / "out.wav"
    await router.convert(make_source(tmp_path), PROFILE_RVC, out)
    assert len(spy.calls) == 1


async def test_shutdown_marks_closed_before_worker_teardown(tmp_path: Path) -> None:
    """The closed flag is set BEFORE the worker is torn down, so a convert
    racing the shutdown already sees the closed router."""
    events: list[str] = []
    release = asyncio.Event()

    class OrderedSpy(SpyRvcConverter):
        async def shutdown(self) -> None:
            await release.wait()  # hold the worker teardown until the test looks
            events.append("worker-shutdown")

    router = VoiceConverterRouter(rvc_factory=lambda: OrderedSpy())
    await router.startup()
    shutdown_task = asyncio.create_task(router.shutdown())
    await asyncio.sleep(0)  # let shutdown enter (closed flag + worker teardown)
    # The closed flag is already latched while the worker teardown is still
    # parked behind `release` — a convert racing the shutdown sees it.
    assert router._closed is True  # noqa: SLF001
    with pytest.raises(RouterClosedError):
        await router.convert(make_source(tmp_path), PROFILE_RVC, tmp_path / "race.wav")
    release.set()
    await shutdown_task
    assert events == ["worker-shutdown"]
