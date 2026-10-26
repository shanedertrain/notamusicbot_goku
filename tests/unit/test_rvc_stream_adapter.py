"""Unit tests for the streaming API of :class:`RvcVoiceConverter` and the router.

A stdlib-only fake worker speaks the line-JSON stream protocol
(``stream_open`` / ``stream_push`` / ``stream_close``). It doubles every
input sample (so 24 kHz in -> 48 kHz out, order checkable) and lags one push
behind so the flush tail is exercised. Mode and request log come via env vars.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from notamusicbot.domain.speech import VoiceConversionProfile
from notamusicbot.speech.voice_conversion import router as router_module
from notamusicbot.speech.voice_conversion import rvc as rvc_module
from notamusicbot.speech.voice_conversion.base import VoiceStreamParams
from notamusicbot.speech.voice_conversion.router import (
    RouterClosedError,
    RvcUnavailableError,
    UnknownBackendError,
    VoiceConverterRouter,
)
from notamusicbot.speech.voice_conversion.rvc import (
    RvcStreamError,
    RvcVoiceConverter,
    WorkerRequestError,
)

FAKE_STREAM_WORKER = (
    textwrap.dedent(
        """
    import base64, json, os, sys, time
    from array import array

    mode = os.environ.get("FAKE_MODE", "ok")
    log_path = os.environ["FAKE_LOG"]
    stream = {"id": None, "in": 0, "out": 0, "pending": b"", "steps": 0}

    def log(obj):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**obj, "t": time.time()}) + "\\n")

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    def double(raw):
        a = array("f")
        a.frombytes(raw)
        o = array("f")
        for v in a:
            o.append(v)
            o.append(v)
        return o.tobytes()

    reply({"id": None, "status": "ok", "action": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        action = req.get("action")
        log({"action": action, **{k: v for k, v in req.items() if k not in ("id", "pcm_b64")}})
        rid = req["id"]
        if action == "health":
            reply({"id": rid, "status": "ok", "device": "cpu"})
        elif action == "load-model":
            reply({"id": rid, "status": "ok", "model": req.get("name"), "device": "cpu"})
        elif action == "stream_open":
            stream.update(id=req["stream_id"], **{"in": 0, "out": 0, "pending": b"", "steps": 0})
            reply({"id": rid, "status": "ok", "stream_id": req["stream_id"],
                   "input_sample_rate": 24000, "output_sample_rate": 48000})
        elif action == "stream_push":
            if mode == "push-error":
                stream["id"] = None
                reply({"id": rid, "status": "error", "error": "boom", "stream_closed": True})
                continue
            if mode == "slow-push":
                time.sleep(0.3)
            raw = base64.b64decode(req["pcm_b64"])
            assert len(raw) // 4 == req["samples"]
            out = stream["pending"]
            stream["pending"] = double(raw)
            stream["in"] += req["samples"]
            stream["out"] += len(out) // 4
            stream["steps"] += 1
            reply({"id": rid, "status": "ok", "stream_id": stream["id"],
                   "pcm_b64": base64.b64encode(out).decode(), "samples": len(out) // 4,
                   "input_total": stream["in"], "output_total": stream["out"],
                   "compute_ms": 1.0})
        elif action == "stream_close":
            closed = stream["id"] == req["stream_id"]
            out = stream["pending"] if (closed and req["flush"]) else b""
            if closed:
                stream["out"] += len(out) // 4
            reply({"id": rid, "status": "ok", "stream_id": req["stream_id"], "closed": closed,
                   "pcm_b64": base64.b64encode(out).decode(), "samples": len(out) // 4,
                   "input_total": stream["in"], "output_total": stream["out"],
                   "compute_ms_total": 2.0, "steps": stream["steps"]})
            stream["id"] = None
        elif action == "shutdown":
            reply({"id": rid, "status": "ok", "action": "shutdown"})
            sys.exit(0)
        else:
            reply({"id": rid, "status": "error", "error": "unknown " + str(action)})
    """
    ).strip()
    + "\n"
)

PARAMS = VoiceStreamParams(enabled=True)
PROFILE = VoiceConversionProfile(backend="rvc", model="Fake", pitch_shift=3)


@pytest.fixture()
def make_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "fake_stream_worker.py"
    script.write_text(FAKE_STREAM_WORKER, encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"
    created: list[RvcVoiceConverter] = []

    def make(mode: str = "ok", **kwargs: Any) -> RvcVoiceConverter:
        monkeypatch.setenv("FAKE_MODE", mode)
        monkeypatch.setenv("FAKE_LOG", str(log_path))
        kwargs.setdefault("stream_params", PARAMS)
        converter = RvcVoiceConverter(
            worker_python=sys.executable,
            worker_script=script,
            start_timeout_s=10.0,
            load_timeout_s=10.0,
            convert_timeout_s=10.0,
            **kwargs,
        )
        created.append(converter)
        return converter

    make.log_path = log_path  # type: ignore[attr-defined]
    yield make
    for converter in created:
        converter._kill_worker_at_exit()  # noqa: SLF001


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stream_calls(path: Path) -> list[dict[str, Any]]:
    return [r for r in read_log(path) if str(r["action"]).startswith("stream_")]


async def source(arrays: list[np.ndarray], delay: float = 0.0) -> AsyncIterator[np.ndarray]:
    for arr in arrays:
        if delay:
            await asyncio.sleep(delay)
        yield arr


def ramp(n: int, start: int = 0) -> np.ndarray:
    return (np.arange(start, start + n, dtype=np.float32) / 1_000_000.0).astype(np.float32)


# --------------------------------------------------------------- supports


def test_supports_streaming_gating(make_converter):
    assert make_converter().supports_streaming(PROFILE) is True
    assert make_converter(stream_params=None).supports_streaming(PROFILE) is False
    off = VoiceStreamParams(enabled=False)
    assert make_converter(stream_params=off).supports_streaming(PROFILE) is False
    other = VoiceConversionProfile(backend="passthrough")
    assert make_converter().supports_streaming(other) is False
    assert make_converter().supports_streaming(VoiceConversionProfile(backend="rvc")) is False


def test_default_stream_timeouts(make_converter, monkeypatch):
    monkeypatch.delenv("RVC_WORKER_STREAM_TIMEOUT", raising=False)
    monkeypatch.delenv("RVC_WORKER_STREAM_OPEN_TIMEOUT", raising=False)
    conv = make_converter()
    assert conv._stream_timeout == 10.0  # noqa: SLF001
    assert conv._stream_open_timeout == 30.0  # noqa: SLF001
    monkeypatch.setenv("RVC_WORKER_STREAM_TIMEOUT", "7")
    assert make_converter()._stream_timeout == 7.0  # noqa: SLF001


async def test_convert_stream_refused_when_unsupported(make_converter):
    conv = make_converter(stream_params=None)
    with pytest.raises(RvcStreamError):
        async for _ in conv.convert_stream(source([ramp(10)]), PROFILE):
            pass


# ------------------------------------------------------------- happy path


async def test_happy_path_split_order_and_totals(make_converter):
    conv = make_converter()
    arrays = [ramp(6000, 0), ramp(120_000, 6000), ramp(6000, 126_000), np.zeros(0, np.float32)]
    arrays.append(ramp(100, 132_000))
    try:
        outs = [o async for o in conv.convert_stream(source(arrays), PROFILE)]
    finally:
        await conv.shutdown()
    total_in = sum(a.size for a in arrays)
    got = np.concatenate(outs)
    assert got.dtype == np.float32
    assert got.size == 2 * total_in
    expected = np.repeat(np.concatenate(arrays), 2)
    np.testing.assert_array_equal(got, expected)

    calls = stream_calls(make_converter.log_path)
    opens = [c for c in calls if c["action"] == "stream_open"]
    pushes = [c for c in calls if c["action"] == "stream_push"]
    closes = [c for c in calls if c["action"] == "stream_close"]
    assert len(opens) == 1 and opens[0]["f0_method"] == "pm" and opens[0]["pitch_shift"] == 3
    assert opens[0]["block_ms"] == 320 and opens[0]["search_ms"] == 20
    assert all(p["samples"] <= 48000 for p in pushes)
    assert sum(p["samples"] for p in pushes) == total_in
    assert len(pushes) == 6  # 6000, 48000+48000+24000, 6000, 100
    assert len(closes) == 1 and closes[0]["flush"] is True


async def test_empty_stream_closes_cleanly(make_converter):
    conv = make_converter()
    try:
        outs = [o async for o in conv.convert_stream(source([]), PROFILE)]
    finally:
        await conv.shutdown()
    assert outs == []
    closes = [c for c in stream_calls(make_converter.log_path) if c["action"] == "stream_close"]
    assert [c["flush"] for c in closes] == [True]


async def test_warm_stream_true_and_false(make_converter):
    conv = make_converter()
    try:
        assert await conv.warm_stream(PROFILE) is True
        calls = stream_calls(make_converter.log_path)
        pushes = [c for c in calls if c["action"] == "stream_push"]
        assert [p["samples"] for p in pushes] == [6000, 6000]
        assert calls[-1]["action"] == "stream_close" and calls[-1]["flush"] is True
    finally:
        await conv.shutdown()
    assert await make_converter(stream_params=None).warm_stream(PROFILE) is False


async def test_warm_stream_failure_returns_false(make_converter):
    conv = make_converter("push-error")
    try:
        assert await conv.warm_stream(PROFILE) is False
    finally:
        await conv.shutdown()


# -------------------------------------------------- cancel / error / locks


async def test_aclose_mid_stream_aborts_and_releases_lock(make_converter):
    conv = make_converter()
    try:
        gen = conv.convert_stream(source([ramp(6000)] * 6), PROFILE)
        first = None
        async for first in gen:  # noqa: B007
            break
        assert first is not None
        await gen.aclose()
        closes = [c for c in stream_calls(make_converter.log_path) if c["action"] == "stream_close"]
        assert [c["flush"] for c in closes] == [False]
        assert not conv._stream_lock.locked()  # noqa: SLF001
        outs = [o async for o in conv.convert_stream(source([ramp(6000)] * 2), PROFILE)]
        assert sum(o.size for o in outs) == 4 * 6000
    finally:
        await conv.shutdown()


async def test_task_cancel_mid_stream_aborts(make_converter):
    conv = make_converter("slow-push")
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in conv.convert_stream(source([ramp(6000)] * 20), PROFILE):
            started.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        closes = [c for c in stream_calls(make_converter.log_path) if c["action"] == "stream_close"]
        assert closes and closes[-1]["flush"] is False
        assert not conv._stream_lock.locked()  # noqa: SLF001
    finally:
        await conv.shutdown()


async def test_producer_exception_aborts_and_propagates(make_converter):
    conv = make_converter()

    async def bad() -> AsyncIterator[np.ndarray]:
        yield ramp(6000)
        msg = "tts exploded"
        raise ValueError(msg)

    try:
        with pytest.raises(ValueError, match="tts exploded"):
            async for _ in conv.convert_stream(bad(), PROFILE):
                pass
        closes = [c for c in stream_calls(make_converter.log_path) if c["action"] == "stream_close"]
        assert [c["flush"] for c in closes] == [False]
        assert not conv._stream_lock.locked()  # noqa: SLF001
    finally:
        await conv.shutdown()


async def test_worker_error_reply_becomes_stream_error(make_converter):
    conv = make_converter("push-error")
    try:
        with pytest.raises(RvcStreamError) as info:
            async for _ in conv.convert_stream(source([ramp(6000)]), PROFILE):
                pass
        assert isinstance(info.value.__cause__, WorkerRequestError)
        assert not conv._stream_lock.locked()  # noqa: SLF001
    finally:
        await conv.shutdown()


async def test_second_concurrent_stream_times_out(make_converter, monkeypatch):
    monkeypatch.setattr(rvc_module, "_STREAM_LOCK_TIMEOUT_S", 0.2)
    conv = make_converter()
    try:
        first = conv.convert_stream(source([ramp(6000)] * 4), PROFILE)
        await anext(first)
        with pytest.raises(RvcStreamError, match="another stream is active"):
            async for _ in conv.convert_stream(source([ramp(6000)]), PROFILE):
                pass
        await first.aclose()
    finally:
        await conv.shutdown()


async def test_second_stream_queues_until_first_finishes(make_converter, monkeypatch):
    monkeypatch.setattr(rvc_module, "_STREAM_LOCK_TIMEOUT_S", 10.0)
    conv = make_converter()

    async def run(n: int) -> int:
        outs = [o async for o in conv.convert_stream(source([ramp(6000)] * n), PROFILE)]
        return sum(o.size for o in outs)

    try:
        a, b = await asyncio.gather(run(3), run(2))
    finally:
        await conv.shutdown()
    assert (a, b) == (3 * 12000, 2 * 12000)
    opens = [c for c in stream_calls(make_converter.log_path) if c["action"] == "stream_open"]
    assert len(opens) == 2


async def test_producer_prefetches_during_slow_push(make_converter):
    conv = make_converter("slow-push")
    pulled: list[float] = []

    async def chunks() -> AsyncIterator[np.ndarray]:
        for _ in range(5):
            pulled.append(time.monotonic())
            yield ramp(6000)

    try:
        t0 = time.monotonic()
        outs = [o async for o in conv.convert_stream(chunks(), PROFILE)]
    finally:
        await conv.shutdown()
    assert sum(o.size for o in outs) == 5 * 12000
    # The 5 pulls happen long before the 5 x 0.3 s of pushes could finish.
    assert pulled[-1] - t0 < 0.9


async def test_convert_stream_logs_non_pm_note(make_converter, caplog):
    conv = make_converter()
    profile = VoiceConversionProfile(backend="rvc", model="Fake", f0_method="harvest")
    try:
        with caplog.at_level("INFO"):
            _ = [o async for o in conv.convert_stream(source([ramp(6000)]), profile)]
    finally:
        await conv.shutdown()
    assert any("force f0 method pm" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- router


class SpyRvc:
    """Fake RVC adapter exposing the streaming surface."""

    def __init__(self, *, supports: bool = True, fail_start: bool = False) -> None:
        self.supports = supports
        self.fail_start = fail_start
        self.closed_streams = 0
        self.warmed = 0

    async def start(self) -> None:
        if self.fail_start:
            msg = "no gpu"
            raise RuntimeError(msg)

    async def shutdown(self) -> None:
        return None

    def supports_streaming(self, profile: VoiceConversionProfile) -> bool:
        return self.supports

    async def warm_stream(self, profile: VoiceConversionProfile) -> bool:
        self.warmed += 1
        return True

    async def convert_stream(
        self, chunks: AsyncIterator[np.ndarray], profile: VoiceConversionProfile
    ) -> AsyncIterator[np.ndarray]:
        try:
            async for c in chunks:
                yield np.repeat(c, 2)
        finally:
            self.closed_streams += 1


def make_router(spy: SpyRvc | None = None, **kwargs: Any) -> tuple[VoiceConverterRouter, SpyRvc]:
    spy = spy or SpyRvc()
    kwargs.setdefault("rvc_stream_params", PARAMS)
    router = VoiceConverterRouter(rvc_factory=lambda: spy, **kwargs)
    return router, spy


async def test_router_streams_through_adapter():
    router, spy = make_router()
    await router.startup()
    assert router.supports_streaming(PROFILE) is True
    outs = [o async for o in router.convert_stream(source([ramp(10), ramp(20)]), PROFILE)]
    assert [o.size for o in outs] == [20, 40]
    assert spy.closed_streams == 1
    assert await router.warm_stream(PROFILE) is True and spy.warmed == 1


async def test_router_early_close_propagates_to_adapter():
    router, spy = make_router()
    await router.startup()
    gen = router.convert_stream(source([ramp(10)] * 5), PROFILE)
    await anext(gen)
    await gen.aclose()
    assert spy.closed_streams == 1


async def test_router_supports_streaming_gates():
    other = VoiceConversionProfile(backend="passthrough")
    router, _ = make_router()
    assert router.supports_streaming(PROFILE) is False  # not started: rvc unavailable
    await router.startup()
    assert router.supports_streaming(PROFILE) is True
    assert router.supports_streaming(other) is False
    await router.shutdown()
    assert router.supports_streaming(PROFILE) is False  # closed

    off, _ = make_router(rvc_stream_params=VoiceStreamParams(enabled=False))
    await off.startup()
    assert off.supports_streaming(PROFILE) is False
    none, _ = make_router(rvc_stream_params=None)
    await none.startup()
    assert none.supports_streaming(PROFILE) is False
    unsupported, _ = make_router(SpyRvc(supports=False))
    await unsupported.startup()
    assert unsupported.supports_streaming(PROFILE) is False
    assert await unsupported.warm_stream(PROFILE) is False
    disabled, _ = make_router(enabled_backends=["passthrough"])
    await disabled.startup()
    assert disabled.supports_streaming(PROFILE) is False


async def test_router_closed_raises():
    router, _ = make_router()
    await router.startup()
    await router.shutdown()
    with pytest.raises(RouterClosedError):
        async for _ in router.convert_stream(source([ramp(10)]), PROFILE):
            pass


async def test_router_rvc_unavailable_no_passthrough_fallback():
    router, _ = make_router(SpyRvc(fail_start=True), rvc_unavailable_policy="passthrough")
    await router.startup()
    with pytest.raises(RvcUnavailableError):
        async for _ in router.convert_stream(source([ramp(10)]), PROFILE):
            pass


async def test_router_disabled_backend_and_unknown():
    router, _ = make_router(enabled_backends=["passthrough"])
    with pytest.raises(UnknownBackendError):
        async for _ in router.convert_stream(source([ramp(10)]), PROFILE):
            pass
    router2, _ = make_router()
    with pytest.raises(UnknownBackendError):
        async for _ in router2.convert_stream(
            source([ramp(10)]), VoiceConversionProfile(backend="nope")
        ):
            pass


def test_router_default_factory_passes_params(monkeypatch):
    captured: dict[str, Any] = {}

    class SpyConverter:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rvc_module, "RvcVoiceConverter", SpyConverter)
    router = VoiceConverterRouter(rvc_stream_params=PARAMS)
    router._rvc_factory()  # noqa: SLF001
    assert captured == {"stream_params": PARAMS}
    assert router_module.VoiceConverterRouter is VoiceConverterRouter
