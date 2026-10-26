"""Spark-first chat with lazy in-process fallback (fakes only, no network, no GPU)."""

from __future__ import annotations

import logging

import httpx
import pytest

from notamusicbot.config import ListenLlmSettings
from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.fallback import FallbackChatProvider, LazyLocalProvider
from notamusicbot.content.spark_chat import EmptyModelText, SparkChatProvider, SparkHTTPError


class FakeSpark:
    """Stands in for SparkChatProvider (same stream/generate/health surface)."""

    def __init__(self, deltas=("Hello ", "there"), error=None, error_after=0, healthy=True):
        self.deltas = list(deltas)
        self.error = error
        self.error_after = error_after
        self.healthy = healthy
        self.calls = 0
        self.health_calls = 0

    async def stream(self, request):
        self.calls += 1
        for i, delta in enumerate(self.deltas):
            if self.error is not None and i == self.error_after:
                raise self.error
            yield delta
        if self.error is not None and self.error_after >= len(self.deltas):
            raise self.error

    async def generate(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "".join(self.deltas)

    async def health(self):
        self.health_calls += 1
        return self.healthy


class FakeLocal:
    """Stands in for LocalQwenLlmProvider."""

    def __init__(self, loaded=False):
        self.loaded = loaded
        self.generate_calls = 0
        self.unloads = 0
        self.gate = None

    def is_loaded(self):
        return self.loaded

    async def generate(self, request):
        self.generate_calls += 1
        self.loaded = True
        if self.gate is not None:
            await self.gate.wait()
        return "local reply"

    def unload(self):
        self.unloads += 1
        self.loaded = False


def _chat() -> ContentRequest:
    return ContentRequest(kind=ContentKind.CHAT, transcript_text="hello there")


def _build(spark, local=None):
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return local

    lazy = LazyLocalProvider(factory) if local is not None else None
    provider = FallbackChatProvider(spark, lazy, 30.0)
    return provider, lazy, factory_calls


ELIGIBLE = [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow"),
    TimeoutError(),
    SparkHTTPError(503, "/chat/completions"),
    SparkHTTPError(500, "/models"),
    EmptyModelText("no text"),
]


@pytest.mark.parametrize("error", ELIGIBLE, ids=lambda e: type(e).__name__)
async def test_spark_failure_falls_back_to_local_exactly_once(error, caplog) -> None:
    spark, local = FakeSpark(error=error), FakeLocal()
    provider, _lazy, factory_calls = _build(spark, local)
    with caplog.at_level(logging.WARNING, logger="notamusicbot.content.fallback"):
        assert await provider.generate(_chat()) == "local reply"
    assert local.generate_calls == 1
    assert factory_calls == [1]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert type(error).__name__ in warnings[0].getMessage()
    assert "cold load" in warnings[0].getMessage()


async def test_stream_fallback_yields_local_text_as_one_chunk() -> None:
    spark, local = FakeSpark(error=httpx.ConnectError("refused")), FakeLocal()
    provider, _, _ = _build(spark, local)
    assert [d async for d in provider.stream(_chat())] == ["local reply"]
    assert local.generate_calls == 1


async def test_healthy_spark_never_touches_local() -> None:
    spark, local = FakeSpark(), FakeLocal()
    provider, _, factory_calls = _build(spark, local)
    assert await provider.generate(_chat()) == "Hello there"
    assert [d async for d in provider.stream(_chat())] == ["Hello ", "there"]
    assert local.generate_calls == 0
    assert factory_calls == []  # the in-process provider is not even built


async def test_fallback_none_reraises_original_error() -> None:
    error = httpx.ConnectError("refused")
    spark = FakeSpark(error=error)
    provider = FallbackChatProvider(spark, None, 30.0)
    with pytest.raises(httpx.ConnectError) as err:
        await provider.generate(_chat())
    assert err.value is error
    with pytest.raises(httpx.ConnectError):
        _ = [d async for d in provider.stream(_chat())]
    assert spark.calls == 2  # no circuit without a fallback: Spark is retried


async def test_non_eligible_errors_are_not_fallen_back() -> None:
    for error in (SparkHTTPError(400, "/chat/completions"), ValueError("bug")):
        spark, local = FakeSpark(error=error), FakeLocal()
        provider, _, _ = _build(spark, local)
        with pytest.raises(type(error)):
            await provider.generate(_chat())
        assert local.generate_calls == 0


async def test_non_chat_kind_never_falls_back() -> None:
    spark, local = FakeSpark(error=httpx.ConnectError("refused")), FakeLocal()
    provider, _, factory_calls = _build(spark, local)
    song = ContentRequest(kind=ContentKind.SONG, song_name="x")
    with pytest.raises(ValueError):
        await provider.generate(song)
    with pytest.raises(ValueError):
        _ = [d async for d in provider.stream(song)]
    assert (spark.calls, local.generate_calls, factory_calls) == (0, 0, [])

    class Other:
        async def generate(self, request):
            return "music text"

    with_other = FallbackChatProvider(spark, LazyLocalProvider(lambda: local), 30.0, other=Other())
    assert await with_other.generate(song) == "music text"
    assert (spark.calls, local.generate_calls) == (0, 0)


async def test_failure_after_partial_text_is_not_spliced_with_local() -> None:
    spark = FakeSpark(deltas=("Hel", "lo"), error=httpx.ReadError("cut"), error_after=1)
    local = FakeLocal()
    provider, _, _ = _build(spark, local)
    seen = []
    with pytest.raises(httpx.ReadError):
        async for delta in provider.stream(_chat()):
            seen.append(delta)
    assert seen == ["Hel"]
    assert local.generate_calls == 0


async def test_early_stop_closes_the_spark_stream_at_once() -> None:
    from contextlib import aclosing

    closed: list[bool] = []

    class ClosingSpark(FakeSpark):
        async def stream(self, request):
            try:
                for delta in ("One. ", "Two. ", "Three."):
                    yield delta
            finally:
                closed.append(True)

    provider, _, _ = _build(ClosingSpark(), FakeLocal())
    async with aclosing(provider.stream(_chat())) as deltas:
        async for delta in deltas:
            assert delta == "One. "
            break
    assert closed == [True]  # closed by aclose, not left to garbage collection


async def test_down_spark_is_skipped_until_health_interval_passes(monkeypatch) -> None:
    import notamusicbot.content.fallback as module

    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    spark, local = FakeSpark(error=httpx.ConnectError("refused")), FakeLocal()
    provider, _, _ = _build(spark, local)
    await provider.generate(_chat())
    await provider.generate(_chat())  # inside the 30 s window: straight to local
    assert (spark.calls, local.generate_calls) == (1, 2)
    now[0] += 31.0
    await provider.generate(_chat())  # window over: Spark retried
    assert (spark.calls, local.generate_calls) == (2, 3)


async def test_ensure_spark_preferred_unloads_idle_loaded_local_when_healthy() -> None:
    spark, local = FakeSpark(healthy=True), FakeLocal(loaded=True)
    provider, lazy, _ = _build(spark, local)
    await provider.generate(_chat())  # builds the lazy provider (Spark answers, local untouched)
    lazy._provider = local  # the model is resident from an earlier fallback
    assert lazy.loaded is True
    assert await provider.ensure_spark_preferred() is True
    assert local.unloads == 1
    assert lazy.loaded is False
    assert await provider.ensure_spark_preferred() is False  # nothing left to unload
    assert local.unloads == 1


async def test_ensure_spark_preferred_keeps_model_when_spark_unhealthy() -> None:
    spark, local = FakeSpark(healthy=False), FakeLocal(loaded=True)
    provider, lazy, _ = _build(spark, local)
    lazy._provider = local
    assert await provider.ensure_spark_preferred() is False
    assert spark.health_calls == 1
    assert local.unloads == 0
    assert lazy.loaded is True


async def test_ensure_spark_preferred_does_not_unload_a_busy_model() -> None:
    import asyncio

    spark, local = FakeSpark(healthy=True), FakeLocal(loaded=True)
    local.gate = asyncio.Event()
    provider, lazy, _ = _build(spark, local)
    task = asyncio.create_task(lazy.generate(_chat()))
    await asyncio.sleep(0)
    assert lazy.idle is False
    assert await provider.ensure_spark_preferred() is False
    assert local.unloads == 0
    local.gate.set()
    assert await task == "local reply"
    assert await provider.ensure_spark_preferred() is True
    assert local.unloads == 1


async def test_ensure_spark_preferred_without_fallback_is_a_no_op() -> None:
    spark = FakeSpark()
    provider = FallbackChatProvider(spark, None, 30.0)
    assert await provider.ensure_spark_preferred() is False
    assert spark.health_calls == 0


async def test_lazy_unload_clears_localqwen_attributes() -> None:
    class Weights:
        _model = object()
        _tokenizer = object()
        load_info = object()

        def is_loaded(self):
            return self._model is not None

    weights = Weights()
    lazy = LazyLocalProvider(lambda: weights)
    lazy._provider = weights
    assert lazy.loaded is True
    assert await lazy.unload() is True
    assert (weights._model, weights._tokenizer, weights.load_info) == (None, None, None)
    assert lazy.loaded is False
    assert await lazy.unload() is False


def test_fallback_wraps_a_real_spark_provider_type() -> None:
    primary = SparkChatProvider(ListenLlmSettings())
    provider = FallbackChatProvider(primary, None, 30.0)
    assert provider.primary is primary
    assert provider.fallback is None
