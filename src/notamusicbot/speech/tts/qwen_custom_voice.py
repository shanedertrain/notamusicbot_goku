"""Qwen3-TTS 0.6B CustomVoice backend.

The model is loaded once when the synthesizer is constructed (never per
utterance) and inference is serialized by the caller's coordinator. The
sample rate returned by ``generate_custom_voice`` is used verbatim — it is
never hard-coded.

The 0.6B CustomVoice checkpoint has no instruction control: ``TtsRequest
.instruction`` is accepted but ignored here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, TtsRequest

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)

CUSTOM_VOICE_SPEAKERS = (
    "Aiden",
    "Ryan",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Sohee",
)


def normalize_text(text: str) -> str:
    """Minimal, explicit text normalization.

    Punctuation and quotes are preserved (the old pipeline stripped quotes
    indiscriminately). Only surrounding whitespace is trimmed; empty input is
    rejected by the caller before reaching the model.
    """
    return " ".join(text.split())


class QwenCustomVoiceSynthesizer:
    """Persistent Qwen3-TTS synthesizer.

    ``model`` must be an already-constructed ``qwen_tts.Qwen3TTSModel``; this
    class never constructs heavyweight objects at import time.
    """

    def __init__(
        self,
        model: Any,
        *,
        default_speaker: str = "Aiden",
        default_language: str = "English",
    ) -> None:
        self._model = model
        self.default_speaker = default_speaker
        self.default_language = default_language

    @classmethod
    def from_settings(
        cls,
        model_id: str,
        device: str,
        dtype: str,
        attn_implementation: str,
    ) -> QwenCustomVoiceSynthesizer:
        """Construct the persistent model from settings (call at startup only)."""
        import torch
        from qwen_tts import Qwen3TTSModel

        torch_dtype = getattr(torch, dtype, None)
        kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except (ImportError, ValueError, RuntimeError):
            # FlashAttention 2 (or the requested attention backend) may be
            # unavailable; SDPA is the supported fallback and must not fail.
            if attn_implementation != "sdpa":
                logger.warning(
                    "Attention backend %r unavailable; falling back to sdpa",
                    attn_implementation,
                )
                model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, **kwargs)
            else:
                raise
        return cls(model)

    async def synthesize(
        self,
        request: TtsRequest,
        output_path: Path,
    ) -> AudioArtifact:
        """Generate speech and write an uncompressed PCM WAV file.

        Heavy model inference is offloaded to a thread so the event loop is not
        blocked; the coordinator still serializes actual GPU access.
        """
        text = normalize_text(request.text)
        if not text:
            msg = "TTS request text is empty"
            raise ValueError(msg)
        speaker = request.speaker or self.default_speaker
        language = request.language or self.default_language

        start = time.perf_counter()
        wavs, sample_rate = await _generate(self._model, text, language, speaker)
        elapsed = time.perf_counter() - start

        waveform: npt.NDArray[np.floating[Any]] = wavs[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await _write_wav(output_path, waveform, int(sample_rate))
        duration = float(len(waveform) / int(sample_rate))
        logger.info(
            "TTS done speaker=%s language=%s sample_rate=%d duration=%.2fs latency=%.2fs",
            speaker,
            language,
            int(sample_rate),
            duration,
            elapsed,
        )
        return AudioArtifact(path=output_path, sample_rate_hz=int(sample_rate), duration_s=duration)

    def warmup(self, output_path: Path) -> AudioArtifact:
        """Run one short generation at startup; timings are logged separately.

        Safe to call from anywhere: with no running event loop it runs a fresh
        loop; from inside a running loop it executes on a worker thread so the
        caller's loop is never blocked or re-entered.
        """
        import asyncio
        import concurrent.futures

        request = TtsRequest(
            text="Warmup.",
            language=self.default_language,
            speaker=self.default_speaker,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            artifact = asyncio.run(self.synthesize(request, output_path))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                artifact = pool.submit(asyncio.run, self.synthesize(request, output_path)).result()
        logger.info("Warmup synthesis complete duration=%.2fs", artifact.duration_s)
        return artifact


async def _generate(
    model: Any,
    text: str,
    language: str,
    speaker: str,
) -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
    """Run model inference off the event loop; returns (wavs, sample_rate)."""
    import asyncio

    def _run() -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        return wavs, int(sample_rate)

    return await asyncio.to_thread(_run)


async def _write_wav(path: Path, waveform: npt.NDArray[np.floating[Any]], sample_rate: int) -> None:
    import asyncio

    def _run() -> None:
        sf.write(path, waveform, sample_rate, subtype="PCM_16")

    await asyncio.to_thread(_run)
