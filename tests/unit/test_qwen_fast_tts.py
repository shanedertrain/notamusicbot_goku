"""Fast (faster-qwen3-tts) TTS backend: selection, fallback and streaming (fakes only)."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from notamusicbot.domain.speech import TtsRequest
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.tts import qwen_custom_voice as mod
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer

_RATE = 24000


class FakeFastModel:
    """Stands in for FasterQwen3TTS; never loads weights."""

    instances: list[FakeFastModel] = []
    fail_warmup = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.gen_closed = threading.Event()
        self.chunks_made = 0
        self.n_chunks = 6
        self.chunk_len = 12000  # 0.5 s
        self.gate: threading.Event | None = None

    @classmethod
    def from_pretrained(cls, path: str, **kw: Any) -> FakeFastModel:
        m = cls()
        m.calls.append(("from_pretrained", {"path": path, **kw}))
        cls.instances.append(m)
        return m

    def warmup(self) -> None:
        if self.fail_warmup:
            raise RuntimeError("graph capture failed")
        self.calls.append(("warmup", {}))

    def generate_custom_voice(self, **kw: Any) -> tuple[list[np.ndarray], int]:
        self.calls.append(("generate", kw))
        return [np.full(_RATE, 0.1, dtype=np.float32)], _RATE

    def generate_custom_voice_streaming(self, **kw: Any):
        self.calls.append(("stream", kw))
        try:
            for i in range(self.n_chunks):
                if self.gate is not None:
                    self.gate.wait(5)
                self.chunks_made += 1
                yield np.full(self.chunk_len, float(i + 1), dtype=np.float64), _RATE, {}
        finally:
            self.gen_closed.set()


@pytest.fixture
def fake_fast(monkeypatch: pytest.MonkeyPatch) -> type[FakeFastModel]:
    FakeFastModel.instances = []
    FakeFastModel.fail_warmup = False
    monkeypatch.setitem(
        sys.modules,
        "faster_qwen3_tts",
        types.SimpleNamespace(FasterQwen3TTS=FakeFastModel),
    )
    monkeypatch.setattr(
        "notamusicbot.content.llm.resolve_local_model_path", lambda model_id: "/local/snap"
    )
    return FakeFastModel


class FakeStock:
    @classmethod
    def from_pretrained(cls, path: str, **kw: Any) -> FakeStock:
        return cls()


@pytest.fixture
def fake_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeStock))


def _req(text: str, max_audio_seconds: float | None = None) -> TtsRequest:
    return TtsRequest(
        text=text, language="English", speaker="Ryan", max_audio_seconds=max_audio_seconds
    )


def _build(**kw: Any) -> QwenCustomVoiceSynthesizer:
    return QwenCustomVoiceSynthesizer.from_settings("Qwen/x", "cpu", "bfloat16", "sdpa", **kw)


def test_fast_path_selected(fake_fast: type[FakeFastModel], fake_stock: None) -> None:
    synth = _build(stream_chunk_size=8)
    assert synth.supports_streaming is True
    (model,) = fake_fast.instances
    kind, kw = model.calls[0]
    assert kind == "from_pretrained"
    assert kw["path"] == "/local/snap"
    assert kw["device"] == "cpu"
    assert kw["attn_implementation"] == "sdpa"
    assert kw["max_seq_len"] == 1024
    assert [c[0] for c in model.calls[1:3]] == ["warmup", "generate"]


def test_fast_disabled_uses_stock(fake_fast: type[FakeFastModel], fake_stock: None) -> None:
    synth = _build(fast=False)
    assert synth.supports_streaming is False
    assert fake_fast.instances == []


def test_fallback_when_import_fails(
    monkeypatch: pytest.MonkeyPatch, fake_stock: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", None)  # import raises ImportError
    monkeypatch.setattr(
        "notamusicbot.content.llm.resolve_local_model_path", lambda model_id: "/local/snap"
    )
    with caplog.at_level("WARNING"):
        synth = _build()
    assert synth.supports_streaming is False
    assert isinstance(synth._model, FakeStock)
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_fallback_when_capture_fails(fake_fast: type[FakeFastModel], fake_stock: None) -> None:
    fake_fast.fail_warmup = True
    synth = _build()
    assert synth.supports_streaming is False
    assert isinstance(synth._model, FakeStock)


async def test_synthesize_on_fast_model_keeps_contract(
    fake_fast: type[FakeFastModel], fake_stock: None, tmp_path: Path
) -> None:
    synth = _build()
    out = tmp_path / "a.wav"
    art = await synth.synthesize(_req("hi there", 2.0), out)
    assert out.exists()
    assert art.sample_rate_hz == _RATE
    kw = fake_fast.instances[0].calls[-1][1]
    assert kw["max_new_tokens"] == mod.max_new_tokens_for(2.0)


async def test_stream_in_order_and_dtype(fake_fast: type[FakeFastModel], fake_stock: None) -> None:
    synth = _build(stream_chunk_size=5)
    got = [c async for c in synth.stream_pcm(_req("hello"))]
    assert [float(c[0]) for c in got] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert all(c.dtype == np.float32 and c.ndim == 1 for c in got)
    kw = fake_fast.instances[0].calls[-1][1]
    assert kw["chunk_size"] == 5
    assert kw["speaker"] == "Ryan"
    assert fake_fast.instances[0].gen_closed.is_set()


async def test_stream_obeys_audio_cap(fake_fast: type[FakeFastModel], fake_stock: None) -> None:
    synth = _build()
    req = _req("hello", 1.25)
    got = [c async for c in synth.stream_pcm(req)]
    assert sum(len(c) for c in got) == int(1.25 * _RATE)
    assert len(got) == 3
    assert len(got[-1]) == 6000
    assert fake_fast.instances[0].gen_closed.is_set()


async def test_stream_cancel_releases_lease_and_worker(
    fake_fast: type[FakeFastModel], fake_stock: None
) -> None:
    coord = InferenceCoordinator(max_concurrent=1, queue_size=2)
    synth = _build(coordinator=coord)
    model = fake_fast.instances[0]
    model.n_chunks = 1000
    agen = synth.stream_pcm(_req("hello"))
    first = await agen.__anext__()
    assert len(first) == 12000
    assert coord.pending == 1  # lease held mid-stream
    await agen.aclose()
    assert model.gen_closed.is_set()
    assert coord.pending == 0
    # Backpressure bounded the worker well short of the full generation.
    assert model.chunks_made <= 1 + mod.STREAM_QUEUE_DEPTH + 1
    # Lease is free again: a new stream runs to completion.
    model.n_chunks = 2
    model.gen_closed.clear()
    got = [c async for c in synth.stream_pcm(_req("again"))]
    assert len(got) == 2


async def test_stream_task_cancel_while_waiting_for_chunk(
    fake_fast: type[FakeFastModel], fake_stock: None
) -> None:
    coord = InferenceCoordinator(max_concurrent=1, queue_size=2)
    synth = _build(coordinator=coord)
    model = fake_fast.instances[0]
    model.gate = threading.Event()  # worker blocks before the first chunk

    async def consume() -> None:
        async for _ in synth.stream_pcm(_req("hello")):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    assert coord.pending == 1
    task.cancel()
    model.gate.set()  # let the blocked generator step so the worker sees stop
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.gen_closed.is_set()
    assert coord.pending == 0


async def test_stream_error_propagates_and_releases(
    fake_fast: type[FakeFastModel], fake_stock: None
) -> None:
    coord = InferenceCoordinator(max_concurrent=1, queue_size=2)
    synth = _build(coordinator=coord)
    model = fake_fast.instances[0]

    def boom(**kw: Any):
        yield np.zeros(10, dtype=np.float32), _RATE, {}
        raise RuntimeError("cuda oops")

    model.generate_custom_voice_streaming = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cuda oops"):
        _ = [c async for c in synth.stream_pcm(_req("hello"))]
    assert coord.pending == 0


async def test_stream_unsupported_on_stock(
    fake_fast: type[FakeFastModel], fake_stock: None
) -> None:
    synth = _build(fast=False)
    with pytest.raises(RuntimeError, match="streaming"):
        _ = [c async for c in synth.stream_pcm(_req("hello"))]
