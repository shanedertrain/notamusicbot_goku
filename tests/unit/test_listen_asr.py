from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from notamusicbot.config import ListenAsrSettings
from notamusicbot.domain.listen import AudioChunk
from notamusicbot.speech.asr import factory
from notamusicbot.speech.asr._common import AsrModelMissingError, AsrTimeoutError
from notamusicbot.speech.asr.parakeet_onnx import ParakeetOnnxTranscriber
from notamusicbot.speech.asr.whisper_hf import WhisperHfTranscriber
from notamusicbot.speech.coordinator import InferenceCoordinator


class FakeOnnxModel:
    def __init__(self, text: str = "what is up", delay_s: float = 0.0) -> None:
        self.text = text
        self.delay_s = delay_s
        self.calls: list[tuple[int, int]] = []

    def recognize(self, samples: np.ndarray, sample_rate: int) -> str:
        self.calls.append((len(samples), sample_rate))
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.text


def _audio(seconds: float = 1.0) -> AudioChunk:
    return AudioChunk(samples=np.zeros(int(16000 * seconds), dtype=np.float32))


def _parakeet(model: FakeOnnxModel, **kw: object) -> ParakeetOnnxTranscriber:
    t = ParakeetOnnxTranscriber(ListenAsrSettings(**kw))
    t._model = model
    return t


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", ""), (".", ""), ("Thank you.", ""), ("hello", ""), ("what is up", "what is up")],
)
async def test_parakeet_normalizes_noise(raw: str, expected: str) -> None:
    t = _parakeet(FakeOnnxModel(raw))
    result = await t.transcribe(_audio())
    assert result.text == expected
    assert result.language == "en"
    assert result.elapsed_s >= 0.0


async def test_parakeet_passes_samples_and_rate() -> None:
    model = FakeOnnxModel()
    await _parakeet(model).transcribe(_audio(2.0))
    assert model.calls == [(32000, 16000)]


async def test_parakeet_timeout_raises_and_loop_stays_free() -> None:
    t = _parakeet(FakeOnnxModel(delay_s=0.5), timeout_s=0.05)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    task = asyncio.create_task(ticker())
    start = time.perf_counter()
    with pytest.raises(AsrTimeoutError):
        await t.transcribe(_audio())
    elapsed = time.perf_counter() - start
    task.cancel()
    assert isinstance(AsrTimeoutError("x"), TimeoutError)
    assert elapsed < 0.3  # returned at the timeout, not after the 0.5 s call
    assert ticks >= 3


async def test_parakeet_uses_a_worker_thread_not_the_loop() -> None:
    t = _parakeet(FakeOnnxModel(delay_s=0.2))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await t.transcribe(_audio())
    task.cancel()
    assert ticks >= 8  # a blocking call on the loop would leave ~0


async def test_parakeet_shutdown_releases_model() -> None:
    t = _parakeet(FakeOnnxModel())
    await t.shutdown()
    assert t._model is None


def test_parakeet_missing_cache_names_setup_step(monkeypatch) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def _absent(*args: object, **kwargs: object) -> str:
        assert kwargs["local_files_only"] is True  # never a download
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _absent)
    pytest.importorskip("onnx_asr")
    t = ParakeetOnnxTranscriber.from_settings(
        ListenAsrSettings(model_id="istupakov/not-a-real-model-xyz")
    )
    with pytest.raises(AsrModelMissingError, match=r"hf download istupakov/not-a-real-model-xyz"):
        t.load()


class FakeWhisper(WhisperHfTranscriber):
    def load(self) -> None:  # no real model in unit tests
        pass

    def _recognize(self, audio: AudioChunk) -> str:
        time.sleep(self.delay_s)
        return self.text

    delay_s = 0.0
    text = "what is up"


async def test_whisper_goes_through_coordinator_and_normalizes() -> None:
    seen: list[object] = []

    class SpyCoordinator(InferenceCoordinator):
        async def run_blocking(self, fn, *args, **kwargs):  # type: ignore[override]
            seen.append(fn)
            return await super().run_blocking(fn, *args, **kwargs)

    t = FakeWhisper(ListenAsrSettings(backend="whisper_hf"), SpyCoordinator())
    assert (await t.transcribe(_audio())).text == "what is up"
    assert len(seen) == 1
    t.text = "Thank you."
    assert (await t.transcribe(_audio())).text == ""


async def test_whisper_timeout() -> None:
    t = FakeWhisper(ListenAsrSettings(backend="whisper_hf", timeout_s=0.05), InferenceCoordinator())
    t.delay_s = 0.3
    with pytest.raises(AsrTimeoutError):
        await t.transcribe(_audio())
    await asyncio.sleep(0.4)  # let the detached lease holder finish


def test_factory_picks_class_per_backend() -> None:
    coordinator = InferenceCoordinator()
    p = factory.build_transcriber(ListenAsrSettings(), coordinator)
    assert type(p) is ParakeetOnnxTranscriber
    w = factory.build_transcriber(ListenAsrSettings(backend="whisper_hf"), coordinator)
    assert type(w) is WhisperHfTranscriber
    assert w._coordinator is coordinator


def test_factory_calls_constructors(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        ParakeetOnnxTranscriber, "from_settings", classmethod(lambda c, s: calls.append("p"))
    )
    monkeypatch.setattr(
        WhisperHfTranscriber, "from_settings", classmethod(lambda c, s, co: calls.append("w"))
    )
    factory.build_transcriber(ListenAsrSettings(), InferenceCoordinator())
    factory.build_transcriber(ListenAsrSettings(backend="whisper_hf"), InferenceCoordinator())
    assert calls == ["p", "w"]


_CLIP = Path.home() / ".cache" / "listen-asr-bakeoff" / "clips" / "clip1.wav"
_CACHE = (
    Path.home() / ".cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v2-onnx/snapshots"
)


@pytest.mark.integration
async def test_parakeet_real_model_on_synthetic_clip() -> None:
    """Needs the Parakeet HF cache and a TTS clip from tools/listen_bench_asr.py clips."""
    if not (_CACHE.exists() and _CLIP.exists()):
        pytest.skip("Parakeet cache or bake-off TTS clip (no speech fixture in repo) absent")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("onnx_asr")
    samples, rate = sf.read(_CLIP, dtype="float32")
    assert rate == 16000
    t = ParakeetOnnxTranscriber.from_settings(ListenAsrSettings())
    t.load()
    await t.transcribe(AudioChunk(samples=samples))  # warm-up
    result = await t.transcribe(AudioChunk(samples=samples))
    await t.shutdown()
    assert result.text
    assert result.elapsed_s < 2.0


# --- device cuda: fp32 weights, torch first, silent self-check, cpu int8 fallback ---


def _patch_loader(monkeypatch, tmp_path, build) -> list[tuple[str | None, list[str]]]:
    """Fake snapshot (all files present), fake torch, CUDA provider 'available'."""
    import sys
    import types

    import huggingface_hub
    import onnx_asr
    import onnxruntime as ort

    seen: list[tuple[str | None, list[str]]] = []
    for name in (*_ALL_FILES, "nemo128.onnx", "vocab.txt", "config.json"):
        (tmp_path / name).write_text("x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])

    class _Torch(types.ModuleType):
        pass

    monkeypatch.setitem(sys.modules, "torch", _Torch("torch"))

    def _load_model(name, path, quantization=None, sess_options=None, providers=None):
        seen.append((quantization, list(providers)))
        return build(quantization, providers)

    monkeypatch.setattr(onnx_asr, "load_model", _load_model)
    return seen


_ALL_FILES = [
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
]


def test_parakeet_cuda_overrides_int8_with_one_warning(monkeypatch, tmp_path, caplog) -> None:
    pytest.importorskip("onnx_asr")
    fake = FakeOnnxModel()
    seen = _patch_loader(monkeypatch, tmp_path, lambda q, p: fake)
    t = ParakeetOnnxTranscriber(ListenAsrSettings(device="cuda", quantization="int8"))
    with caplog.at_level("WARNING"):
        t.load()
    assert t._model is fake
    assert seen == [(None, ["CUDAExecutionProvider", "CPUExecutionProvider"])]  # fp32
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    assert fake.calls == [(8000, 16000)]  # the silent self-check ran at load


def test_parakeet_cuda_fp32_has_no_warning(monkeypatch, tmp_path, caplog) -> None:
    pytest.importorskip("onnx_asr")
    seen = _patch_loader(monkeypatch, tmp_path, lambda q, p: FakeOnnxModel())
    t = ParakeetOnnxTranscriber(ListenAsrSettings(device="cuda", quantization="fp32"))
    with caplog.at_level("WARNING"):
        t.load()
    assert seen[0][0] is None
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_parakeet_cuda_failed_self_check_falls_back_to_cpu_int8(
    monkeypatch, tmp_path, caplog
) -> None:
    pytest.importorskip("onnx_asr")
    cpu = FakeOnnxModel()

    class _NoCudnn(FakeOnnxModel):
        def recognize(self, samples: np.ndarray, sample_rate: int) -> str:
            raise RuntimeError("RequireCudnnHandle")

    seen = _patch_loader(monkeypatch, tmp_path, lambda q, p: _NoCudnn() if q is None else cpu)
    t = ParakeetOnnxTranscriber(ListenAsrSettings(device="cuda", quantization="fp32"))
    with caplog.at_level("WARNING"):
        t.load()
    assert t._model is cpu
    assert seen[-1] == ("int8", ["CPUExecutionProvider"])
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_parakeet_cuda_without_cuda_provider_falls_back(monkeypatch, tmp_path) -> None:
    import onnxruntime as ort

    pytest.importorskip("onnx_asr")
    cpu = FakeOnnxModel()
    seen = _patch_loader(monkeypatch, tmp_path, lambda q, p: cpu)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    t = ParakeetOnnxTranscriber(ListenAsrSettings(device="cuda"))
    t.load()
    assert seen == [("int8", ["CPUExecutionProvider"])]


def test_parakeet_cuda_imports_torch_before_session(monkeypatch, tmp_path) -> None:
    import sys

    import onnx_asr

    pytest.importorskip("onnx_asr")
    order: list[str] = []
    _patch_loader(monkeypatch, tmp_path, lambda q, p: FakeOnnxModel())
    monkeypatch.delitem(sys.modules, "torch")

    class _Finder:
        def find_spec(self, name, path=None, target=None):
            if name == "torch":
                order.append("torch")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])
    real = onnx_asr.load_model
    monkeypatch.setattr(
        onnx_asr, "load_model", lambda *a, **k: (order.append("session"), real(*a, **k))[1]
    )
    ParakeetOnnxTranscriber(ListenAsrSettings(device="cuda", quantization="fp32")).load()
    assert order == ["torch", "session"]


def test_parakeet_cpu_default_path_unchanged(monkeypatch, tmp_path) -> None:
    pytest.importorskip("onnx_asr")
    cpu = FakeOnnxModel()
    seen = _patch_loader(monkeypatch, tmp_path, lambda q, p: cpu)
    t = ParakeetOnnxTranscriber(ListenAsrSettings())
    t.load()
    assert seen == [("int8", ["CPUExecutionProvider"])]
    assert cpu.calls == []  # no self-check on cpu
