"""QwenCloneSynthesizer: clone prompt wiring, cache and validation (fake model).

No GPU and no weights: the model object is injected directly and records the
calls the backend makes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from notamusicbot.domain.speech import TtsRequest
from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer


class FakeCloneModel:
    """Records create_voice_clone_prompt/generate_voice_clone calls."""

    def __init__(self, seconds: float = 0.1, sample_rate: int = 24000) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.prompt_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.prompt_items = [object()]

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[object]:
        self.prompt_calls.append(kwargs)
        return self.prompt_items

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
        self.generate_calls.append(kwargs)
        n = int(self.sample_rate * self.seconds)
        return [np.zeros(n, dtype=np.float32)], self.sample_rate


def _reference(tmp_path: Path, *, seconds: float = 0.2) -> Path:
    path = tmp_path / "ref.wav"
    n = int(24000 * seconds)
    sf.write(path, 0.3 * np.ones(n, dtype=np.float32), 24000, subtype="PCM_16")
    return path


def _request(reference: Path, **kwargs: Any) -> TtsRequest:
    fields: dict[str, Any] = {"text": "Hello there.", "reference_audio": str(reference)}
    fields.update(kwargs)
    return TtsRequest(**fields)


def test_xvec_clone_builds_prompt_once_and_reuses_it(tmp_path: Path) -> None:
    model = FakeCloneModel()
    synth = QwenCloneSynthesizer(model)
    reference = _reference(tmp_path)

    async def run() -> None:
        first = await synth.synthesize(_request(reference), tmp_path / "a.wav")
        second = await synth.synthesize(_request(reference), tmp_path / "b.wav")
        assert first.path.is_file() and second.path.is_file()

    import asyncio

    asyncio.run(run())
    # One prompt build for repeated calls with the same voice.
    assert len(model.prompt_calls) == 1
    prompt = model.prompt_calls[0]
    assert prompt["x_vector_only_mode"] is True
    assert prompt["ref_text"] == ""
    wav, rate = prompt["ref_audio"]
    assert rate == 24000
    assert isinstance(wav, np.ndarray) and wav.ndim == 1 and wav.dtype == np.float32
    # Two generations, both reusing the cached prompt items.
    assert len(model.generate_calls) == 2
    for call in model.generate_calls:
        assert call["voice_clone_prompt"] is model.prompt_items
        assert call["text"] == "Hello there."
        assert call["language"] == "English"


def test_prompt_cache_is_keyed_by_voice_and_mode(tmp_path: Path) -> None:
    model = FakeCloneModel()
    synth = QwenCloneSynthesizer(model)
    first = _reference(tmp_path)
    second = tmp_path / "ref2.wav"
    sf.write(second, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")

    import asyncio

    async def run() -> None:
        await synth.synthesize(_request(first), tmp_path / "a.wav")
        await synth.synthesize(_request(second), tmp_path / "b.wav")
        await synth.synthesize(
            _request(first, clone_mode="icl", reference_text="Hello."), tmp_path / "c.wav"
        )

    asyncio.run(run())
    assert len(model.prompt_calls) == 3
    modes = [call["x_vector_only_mode"] for call in model.prompt_calls]
    assert modes == [True, True, False]


def test_icl_wires_the_reference_transcript(tmp_path: Path) -> None:
    model = FakeCloneModel()
    synth = QwenCloneSynthesizer(model)
    reference = _reference(tmp_path)

    import asyncio

    asyncio.run(
        synth.synthesize(
            _request(reference, clone_mode="icl", reference_text="  Hello there.  "),
            tmp_path / "out.wav",
        )
    )
    prompt = model.prompt_calls[0]
    assert prompt["x_vector_only_mode"] is False
    assert prompt["ref_text"] == "Hello there."  # trimmed


def test_artifact_is_written_with_the_model_sample_rate(tmp_path: Path) -> None:
    model = FakeCloneModel(seconds=0.25, sample_rate=24000)
    synth = QwenCloneSynthesizer(model)
    reference = _reference(tmp_path)
    out = tmp_path / "nested" / "out.wav"

    import asyncio

    artifact = asyncio.run(synth.synthesize(_request(reference), out))
    assert artifact.path == out
    assert artifact.sample_rate_hz == 24000
    assert artifact.duration_s == pytest.approx(0.25)
    info = sf.info(out)
    assert info.samplerate == 24000
    assert info.frames == 6000


def test_max_new_tokens_is_omitted_without_cap(tmp_path: Path) -> None:
    model = FakeCloneModel()
    synth = QwenCloneSynthesizer(model)
    reference = _reference(tmp_path)

    import asyncio

    asyncio.run(synth.synthesize(_request(reference), tmp_path / "out.wav"))
    assert "max_new_tokens" not in model.generate_calls[0]


def test_empty_text_raises(tmp_path: Path) -> None:
    synth = QwenCloneSynthesizer(FakeCloneModel())
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="text is empty"):
        import asyncio

        asyncio.run(synth.synthesize(_request(reference, text="   "), tmp_path / "o.wav"))


def test_missing_reference_audio_raises(tmp_path: Path) -> None:
    synth = QwenCloneSynthesizer(FakeCloneModel())
    with pytest.raises(ValueError, match="reference audio not found"):
        import asyncio

        asyncio.run(synth.synthesize(_request(tmp_path / "missing.wav"), tmp_path / "o.wav"))


def test_request_without_reference_raises(tmp_path: Path) -> None:
    synth = QwenCloneSynthesizer(FakeCloneModel())
    with pytest.raises(ValueError, match="no reference audio"):
        import asyncio

        asyncio.run(synth.synthesize(TtsRequest(text="hi"), tmp_path / "o.wav"))


def test_icl_without_transcript_raises(tmp_path: Path) -> None:
    synth = QwenCloneSynthesizer(FakeCloneModel())
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="reference transcript"):
        import asyncio

        asyncio.run(synth.synthesize(_request(reference, clone_mode="icl"), tmp_path / "o.wav"))


def test_supports_streaming_is_false() -> None:
    assert QwenCloneSynthesizer.supports_streaming is False


def test_shutdown_drops_the_model_and_prompts(tmp_path: Path) -> None:
    model = FakeCloneModel()
    synth = QwenCloneSynthesizer(model)
    reference = _reference(tmp_path)

    import asyncio

    async def run() -> None:
        await synth.synthesize(_request(reference), tmp_path / "out.wav")
        assert synth._prompts  # noqa: SLF001 - cache populated by the call
        await synth.shutdown()
        assert synth._model is None  # noqa: SLF001 - model released

    asyncio.run(run())
