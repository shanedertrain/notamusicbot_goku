"""Qwen3-TTS 0.6B CustomVoice backend.

The model is loaded once when the synthesizer is constructed (never per
utterance) and inference is serialized by the caller's coordinator. The
sample rate returned by ``generate_custom_voice`` is used verbatim — it is
never hard-coded.

The 0.6B CustomVoice checkpoint has no instruction control: ``TtsRequest
.instruction`` is accepted but ignored here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, TtsRequest

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from notamusicbot.speech.coordinator import InferenceCoordinator

logger = logging.getLogger(__name__)

# The "12Hz" speech codec runs at 12.5 frames (talker tokens) per second of
# audio (speech_tokenizer config: 24000 Hz / 1920 = 12.5). The generation bound
# gets about one extra second of slack so a cap that lands mid-word is trimmed
# (with a fade) by the caller instead of being cut dead by the model.
CODEC_TOKENS_PER_SECOND = 12.5
MAX_NEW_TOKENS_SLACK = math.ceil(CODEC_TOKENS_PER_SECOND)


def max_new_tokens_for(max_audio_seconds: float | None) -> int | None:
    """``max_new_tokens`` bound for an audio cap, or None for no bound."""
    if max_audio_seconds is None:
        return None
    return math.ceil(max_audio_seconds * CODEC_TOKENS_PER_SECOND) + MAX_NEW_TOKENS_SLACK


# faster-qwen3-tts CUDA-graph static cache length (prompt + generated frames).
FAST_MAX_SEQ_LEN = 1024
# Chunks the streaming worker may run ahead of the consumer before it blocks.
STREAM_QUEUE_DEPTH = 4
# Poll interval for the worker's stop checks while it waits on the consumer.
_STOP_POLL_S = 0.05

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
        fast: bool = False,
        stream_chunk_size: int = 8,
        coordinator: InferenceCoordinator | None = None,
    ) -> None:
        self._model = model
        self.default_speaker = default_speaker
        self.default_language = default_language
        # ``fast`` means ``model`` is a faster_qwen3_tts.FasterQwen3TTS (same
        # generate_custom_voice contract plus streaming).
        self._fast = fast
        self._stream_chunk_size = stream_chunk_size
        # Lease for stream_pcm, which (unlike synthesize) is not called from
        # inside the speech pipeline's lease. Without one, streams are only
        # serialized against each other by a private lock.
        self._coordinator = coordinator
        self._stream_lock: asyncio.Lock | None = None

    @property
    def supports_streaming(self) -> bool:
        """True only when the fast (faster-qwen3-tts) model is loaded."""
        return self._fast

    @classmethod
    def from_settings(
        cls,
        model_id: str,
        device: str,
        dtype: str,
        attn_implementation: str,
        *,
        fast: bool = True,
        stream_chunk_size: int = 8,
        coordinator: InferenceCoordinator | None = None,
    ) -> QwenCustomVoiceSynthesizer:
        """Construct the persistent model from settings (call at startup only).

        With ``fast`` the faster-qwen3-tts CUDA-graph model is tried first; any
        failure to import it or capture its graphs logs one warning and falls
        back to the stock ``Qwen3TTSModel`` path below.
        """
        import torch

        from notamusicbot.content.llm import resolve_local_model_path

        # Local snapshot path, never the hub id (offline-safe; see helper).
        model_id = resolve_local_model_path(model_id)
        torch_dtype = getattr(torch, dtype, None)
        if fast:
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
                return cls(
                    fast_model,
                    fast=True,
                    stream_chunk_size=stream_chunk_size,
                    coordinator=coordinator,
                )
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
        return cls(model, stream_chunk_size=stream_chunk_size, coordinator=coordinator)

    async def synthesize(
        self,
        request: TtsRequest,
        output_path: Path,
    ) -> AudioArtifact:
        """Generate speech and write an uncompressed PCM WAV file.

        Heavy model inference is offloaded to a thread so the event loop is not
        blocked; the coordinator still serializes actual GPU access.

        Review Round-4 P0 (cancellation safety): this synthesizer's blocking
        ``model.generate_custom_voice`` call always executes INSIDE the speech
        pipeline's lease-held chain (``SpeechPipeline.run`` wraps the whole
        TTS -> VC -> postprocess chain in ``coordinator.run_blocking``), so the
        GPU lease is held until this blocking call actually finishes even if
        the pipeline caller is cancelled. No separate coordination is needed
        here, and calling ``synthesize`` outside the pipeline's lease would
        bypass arbitration — the pipeline is the only production entry point.
        """
        text = normalize_text(request.text)
        if not text:
            msg = "TTS request text is empty"
            raise ValueError(msg)
        speaker = request.speaker or self.default_speaker
        language = request.language or self.default_language

        start = time.perf_counter()
        wavs, sample_rate = await _generate(
            self._model,
            text,
            language,
            speaker,
            max_new_tokens=max_new_tokens_for(request.max_audio_seconds),
        )
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

    async def stream_pcm(self, request: TtsRequest) -> AsyncIterator[np.ndarray]:
        """Yield mono float32 24 kHz chunks as they are generated (fast path only).

        Holds the GPU lease for the whole generation. The blocking generator
        runs on a worker thread that may run ``STREAM_QUEUE_DEPTH`` chunks
        ahead of the consumer. Stops after ``request.max_audio_seconds`` of
        audio (last chunk trimmed). Closing or cancelling this generator stops
        the worker at its next chunk boundary and releases the lease only once
        the worker has really finished. Raises ``RuntimeError`` when streaming
        is unsupported and ``ValueError`` on empty text.
        """
        if not self._fast:
            msg = "streaming requires the faster-qwen3-tts backend"
            raise RuntimeError(msg)
        text = normalize_text(request.text)
        if not text:
            msg = "TTS request text is empty"
            raise ValueError(msg)
        inner = self._stream_locked(
            text,
            request.language or self.default_language,
            request.speaker or self.default_speaker,
            request.max_audio_seconds,
        )
        # aclosing: closing this generator must close the inner one (and so
        # stop the worker) before the lease is released.
        async with self._stream_lease(), contextlib.aclosing(inner):
            async for chunk in inner:
                yield chunk

    def _stream_lease(self) -> contextlib.AbstractAsyncContextManager[None]:
        if self._coordinator is not None:
            return self._coordinator.slot()
        if self._stream_lock is None:
            self._stream_lock = asyncio.Lock()
        return _locked(self._stream_lock)

    async def _stream_locked(
        self, text: str, language: str, speaker: str, max_audio_seconds: float | None
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        credits = threading.Semaphore(STREAM_QUEUE_DEPTH)
        stop = threading.Event()
        extra: dict[str, Any] = {}
        max_new = max_new_tokens_for(max_audio_seconds)
        if max_new is not None:
            extra["max_new_tokens"] = max_new

        def _emit(kind: str, value: Any) -> None:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(out.put_nowait, (kind, value))

        def _worker() -> None:
            import numpy as np

            gen = None
            try:
                gen = self._model.generate_custom_voice_streaming(
                    text=text,
                    language=language,
                    speaker=speaker,
                    chunk_size=self._stream_chunk_size,
                    **extra,
                )
                emitted = 0
                for item in gen:
                    if stop.is_set():
                        return
                    audio, rate = item[0], int(item[1])
                    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
                    done = False
                    if max_audio_seconds is not None:
                        room = max(0, int(max_audio_seconds * rate) - emitted)
                        if len(chunk) >= room:
                            chunk, done = chunk[:room], True
                    emitted += len(chunk)
                    if len(chunk):
                        # Backpressure: wait for the consumer, but never past stop.
                        while not credits.acquire(timeout=_STOP_POLL_S):
                            if stop.is_set():
                                return
                        _emit("chunk", chunk)
                    if done:
                        return
            except BaseException as exc:  # noqa: BLE001 - relayed to the consumer
                _emit("error", exc)
            finally:
                if gen is not None:
                    with contextlib.suppress(Exception):
                        gen.close()
                _emit("end", None)

        worker = loop.run_in_executor(None, _worker)
        try:
            while True:
                kind, value = await out.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                credits.release()
                yield value
        finally:
            stop.set()
            # Wait the worker out even if cancelled again, so the lease is
            # never released while CUDA-graph state is still in use.
            cancelled = False
            while not worker.done():
                try:
                    await asyncio.wait({worker})
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError

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


@contextlib.asynccontextmanager
async def _locked(lock: asyncio.Lock) -> AsyncIterator[None]:
    async with lock:
        yield


def _load_fast_model(
    model_path: str, device: str, torch_dtype: Any, attn_implementation: str
) -> Any:
    """Build FasterQwen3TTS and capture its CUDA graphs (about 3 s, once)."""
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        model_path,
        device=device,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_seq_len=FAST_MAX_SEQ_LEN,
    )
    model.warmup()
    # One short real generation finishes graph capture for the decode path.
    model.generate_custom_voice(
        text="Warmup, one two three.", speaker="Ryan", language="English", max_new_tokens=60
    )
    return model


async def _generate(
    model: Any,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int | None = None,
) -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
    """Run model inference off the event loop; returns (wavs, sample_rate).

    ``max_new_tokens=None`` omits the kwarg so the library default applies.
    """
    import asyncio

    extra: dict[str, Any] = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}

    def _run() -> tuple[list[npt.NDArray[np.floating[Any]]], int]:
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            **extra,
        )
        return wavs, int(sample_rate)

    return await asyncio.to_thread(_run)


async def _write_wav(path: Path, waveform: npt.NDArray[np.floating[Any]], sample_rate: int) -> None:
    import asyncio

    def _run() -> None:
        sf.write(path, waveform, sample_rate, subtype="PCM_16")

    await asyncio.to_thread(_run)
