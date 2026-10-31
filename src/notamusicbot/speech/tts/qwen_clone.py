"""Qwen3-TTS 0.6B Base voice-cloning backend.

The model is loaded once when the synthesizer is constructed (never per
utterance) and inference is serialized by the caller's coordinator. The sample
rate returned by ``generate_voice_clone`` is used verbatim — it is never
hard-coded.

A voice is a reference clip plus (for ICL cloning) its transcript: the clone
prompt is built once per voice and cached in memory, and every later utterance
for that voice reuses it (the documented production pattern). There is no
streaming in v1: ``supports_streaming`` is False and callers synthesize whole
sentences through the speech pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, TtsRequest
from notamusicbot.speech.tts._common import max_new_tokens_for, normalize_text

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# The clone model's reference audio rate (speaker encoder input).
REFERENCE_SAMPLE_RATE_HZ = 24000

# faster-qwen3-tts CUDA-graph static cache length (prompt + generated frames).
FAST_MAX_SEQ_LEN = 1024

# xvec clones only the speaker embedding; icl additionally conditions on the
# reference speech codes and therefore needs the reference transcript.
CLONE_MODES = ("xvec", "icl")


def _load_reference_audio(path: Path) -> npt.NDArray[np.float32]:
    """Decode ``path`` to mono float32 24 kHz (soundfile + soxr, no network)."""
    import numpy as np
    import soxr

    wav, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = np.asarray(wav, dtype=np.float32).mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    if int(sample_rate) != REFERENCE_SAMPLE_RATE_HZ:
        mono = soxr.resample(np.asarray(mono, dtype=np.float32), int(sample_rate), 24000)
    return np.asarray(mono, dtype=np.float32).reshape(-1)


class QwenCloneSynthesizer:
    """Persistent Qwen3-TTS clone synthesizer.

    ``model`` must be an already-constructed ``qwen_tts.Qwen3TTSModel``; this
    class never constructs heavyweight objects at import time.
    """

    # Whole-sentence only in v1 (no CUDA-graph streaming backend).
    supports_streaming = False

    def __init__(
        self,
        model: Any,
        *,
        default_language: str = "English",
        fast: bool = False,
    ) -> None:
        self._model = model
        self._fast = fast
        # Prompt extraction lives on the stock Qwen3TTSModel; the faster
        # wrapper exposes it as ``.model``. Both paths pass the resulting
        # prompt items to ``generate_voice_clone``.
        self._prompt_model = getattr(model, "model", model) if fast else model
        self.default_language = default_language
        # (resolved reference path, reference text, clone mode) -> prompt items.
        self._prompts: dict[tuple[str, str, str], Any] = {}
        self._prompt_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        model_id: str,
        device: str,
        dtype: str,
        attn_implementation: str,
        *,
        fast: bool = True,
    ) -> QwenCloneSynthesizer:
        """Construct the persistent clone model from settings (startup only).

        ``fast`` prefers the faster-qwen3-tts CUDA-graph wrapper (measured
        about 3x faster generation on the 4070 here); any load failure logs one
        warning and falls back to the stock model. CPU installs always use the
        stock model.
        """
        import torch

        from notamusicbot.content.llm import resolve_local_model_path

        # Local snapshot path, never the hub id (offline-safe; see helper).
        model_id = resolve_local_model_path(model_id)
        torch_dtype = getattr(torch, dtype, None)

        if fast and str(device).startswith("cuda"):
            try:
                fast_model = _load_fast_model(model_id, device, torch_dtype, attn_implementation)
            except Exception as exc:  # noqa: BLE001 - any failure must fall back
                logger.warning(
                    "faster-qwen3-tts unavailable (%s: %s); using stock Qwen3-TTS",
                    type(exc).__name__,
                    exc,
                )
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            else:
                return cls(fast_model, fast=True)

        from qwen_tts import Qwen3TTSModel

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

        Heavy model inference (including a first-time prompt build) is
        offloaded to a thread so the event loop is not blocked; the caller's
        coordinator still serializes actual GPU access.

        Review Round-4 P0 (cancellation safety): the blocking generation call
        always executes INSIDE the speech pipeline's lease-held chain
        (``SpeechPipeline.run`` wraps the whole TTS -> postprocess chain in
        ``coordinator.run_detached``), so the GPU lease is held until the
        blocking call actually finishes even if the pipeline caller is
        cancelled.
        """
        text = normalize_text(request.text)
        if not text:
            msg = "TTS request text is empty"
            raise ValueError(msg)
        reference_audio = request.reference_audio
        if not reference_audio:
            msg = "TTS request has no reference audio (clone backend)"
            raise ValueError(msg)
        reference_path = Path(reference_audio)
        if not reference_path.is_file():
            msg = f"Voice reference audio not found: {reference_path}"
            raise ValueError(msg)
        clone_mode = request.clone_mode
        if clone_mode not in CLONE_MODES:
            msg = f"Unknown clone mode {clone_mode!r} (expected one of {CLONE_MODES})"
            raise ValueError(msg)
        reference_text = (request.reference_text or "").strip()
        if clone_mode == "icl" and not reference_text:
            msg = "clone_mode 'icl' requires the reference transcript"
            raise ValueError(msg)
        language = request.language or self.default_language

        start = time.perf_counter()
        wavs, sample_rate = await self._generate(
            text,
            language,
            reference_path.resolve(),
            reference_text,
            clone_mode,
            max_new_tokens=max_new_tokens_for(request.max_audio_seconds),
        )
        elapsed = time.perf_counter() - start

        waveform: npt.NDArray[np.floating[Any]] = wavs[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await _write_wav(output_path, waveform, int(sample_rate))
        duration = float(len(waveform) / int(sample_rate))
        logger.info(
            "TTS clone done voice=%s mode=%s language=%s sample_rate=%d duration=%.2fs "
            "latency=%.2fs",
            reference_path.name,
            clone_mode,
            language,
            int(sample_rate),
            duration,
            elapsed,
        )
        return AudioArtifact(path=output_path, sample_rate_hz=int(sample_rate), duration_s=duration)

    def _prompt_items(
        self,
        reference_path: Path,
        reference_text: str,
        clone_mode: str,
    ) -> Any:
        """Clone prompt for one voice, built once and cached for the process.

        ``xvec`` needs no transcript (only the speaker embedding is used), so
        the reference text is ignored there; ``icl`` passes it through.
        """
        key = (str(reference_path), reference_text, clone_mode)
        with self._prompt_lock:
            cached = self._prompts.get(key)
            if cached is not None:
                return cached
            wav = _load_reference_audio(reference_path)
            items = self._prompt_model.create_voice_clone_prompt(
                ref_audio=(wav, REFERENCE_SAMPLE_RATE_HZ),
                ref_text=reference_text,
                x_vector_only_mode=(clone_mode != "icl"),
            )
            self._prompts[key] = items
        logger.info("Voice clone prompt ready voice=%s mode=%s", reference_path.name, clone_mode)
        return items

    async def _generate(
        self,
        text: str,
        language: str,
        reference_path: Path,
        reference_text: str,
        clone_mode: str,
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
        """Run clone inference off the event loop; returns (wavs, sample_rate).

        ``max_new_tokens=None`` omits the kwarg so the library default applies.
        """
        extra: dict[str, Any] = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}

        def _run() -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
            items = self._prompt_items(reference_path, reference_text, clone_mode)
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=items,
                **extra,
            )
            return wavs, int(sample_rate)

        return await asyncio.to_thread(_run)

    async def shutdown(self) -> None:
        """Drop the model and return its VRAM to the driver."""
        self._model = None
        self._prompt_model = None
        with self._prompt_lock:
            self._prompts.clear()

        def _empty_cache() -> None:
            import gc

            import torch

            # The faster wrapper holds CUDA-graph state behind cyclic
            # references; collect before emptying the allocator cache.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        await asyncio.to_thread(_empty_cache)


def _load_fast_model(
    model_path: str,
    device: str,
    torch_dtype: Any,
    attn_implementation: str,
) -> Any:
    """Build FasterQwen3TTS and capture its CUDA graphs (a few seconds, once)."""
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        model_path,
        device=device,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_seq_len=FAST_MAX_SEQ_LEN,
    )
    model.warmup()
    return model


async def _write_wav(path: Path, waveform: npt.NDArray[np.floating[Any]], sample_rate: int) -> None:
    def _run() -> None:
        sf.write(path, waveform, sample_rate, subtype="PCM_16")

    await asyncio.to_thread(_run)
