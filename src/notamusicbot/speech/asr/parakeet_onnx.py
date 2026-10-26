"""Parakeet TDT 0.6B v2 via onnx-asr (the default listen ASR).

CPU int8 by default (``intra_op_num_threads`` from settings,
``CPUExecutionProvider`` only). ``device: cuda`` runs the fp32 weights on the
GPU, self-checks at load and falls back to CPU int8 if CUDA/cuDNN is broken.
The blocking call runs in ``asyncio.to_thread`` and does NOT take the GPU
:class:`InferenceCoordinator` lease. Weights are read from the local HF cache
only. onnx-asr is imported lazily (``listen`` extra).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from notamusicbot.config import ListenAsrSettings
from notamusicbot.domain.listen import AudioChunk, Transcript
from notamusicbot.speech.asr._common import (
    AsrModelMissingError,
    AsrTimeoutError,
    missing_model_message,
    normalize_transcript,
)

logger = logging.getLogger(__name__)

# onnx-asr's name for the model whose files live in the default repo.
ONNX_ASR_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"
_FILES = {
    "int8": ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"],
    "fp32": ["encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"],
}
_COMMON_FILES = ["nemo128.onnx", "vocab.txt", "config.json"]
_SELF_CHECK_SAMPLES = 8000  # 0.5 s of silence at 16 kHz


def _real_files_dir(snapshot: str, names: list[str]) -> str:
    """A sibling dir of hard links to the snapshot's blobs.

    onnxruntime refuses external data (``encoder-model.onnx.data``) whose
    resolved path leaves the model directory, which an HF snapshot symlink
    always does. Hard links cost no disk; a copy is the cross-device fallback.
    """
    src = Path(snapshot)
    dst = src.parents[1] / "notamusicbot-parakeet-fp32"
    dst.mkdir(exist_ok=True)
    for name in names:
        target = dst / name
        blob = (src / name).resolve()
        if target.exists() and target.stat().st_ino == blob.stat().st_ino:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(blob, target)
        except OSError:
            shutil.copyfile(blob, target)
    return str(dst)


class ParakeetOnnxTranscriber:
    def __init__(self, settings: ListenAsrSettings) -> None:
        self._settings = settings
        self._model: Any = None

    @classmethod
    def from_settings(cls, settings: ListenAsrSettings) -> ParakeetOnnxTranscriber:
        return cls(settings)

    def load(self) -> None:
        """Build the model from the local HF cache (blocking; never downloads)."""
        if self._model is not None:
            return
        s = self._settings
        if s.device == "cuda":
            self._model = self._load_cuda()
            return
        self._model = self._build(s.quantization, "cpu")

    def _build(self, quantization: str, device: str) -> Any:
        """One onnx-asr model for ``quantization`` on ``device`` (local cache only)."""
        import onnx_asr
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        s = self._settings
        try:
            snapshot = snapshot_download(
                s.model_id,
                local_files_only=True,
                allow_patterns=[*_FILES[quantization], *_COMMON_FILES],
            )
        except Exception as exc:
            raise AsrModelMissingError(missing_model_message(s.model_id)) from exc
        missing = [f for f in _FILES[quantization] if not (Path(snapshot) / f).exists()]
        if missing:
            raise AsrModelMissingError(missing_model_message(s.model_id))

        if quantization == "fp32":
            snapshot = _real_files_dir(snapshot, [*_FILES["fp32"], *_COMMON_FILES])

        options = ort.SessionOptions()
        options.intra_op_num_threads = s.threads
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        model = onnx_asr.load_model(
            ONNX_ASR_MODEL_NAME,
            snapshot,
            quantization="int8" if quantization == "int8" else None,
            sess_options=options,
            providers=providers,
        )
        logger.info(
            "parakeet loaded quant=%s threads=%d device=%s", quantization, s.threads, device
        )
        return model

    def _load_cuda(self) -> Any:
        """fp32 on CUDA, proven by one silent inference; CPU int8 if that fails.

        The int8 graph falls back to the CPU on CUDA (742 Memcpy nodes, no
        speedup), so cuda always runs the fp32 weights. torch is imported first
        because that is what puts the pip cuDNN on the loader path.
        """
        import onnxruntime as ort

        s = self._settings
        if s.quantization == "int8":
            logger.warning("parakeet device=cuda needs fp32 weights; ignoring quantization=int8")
        try:
            import torch  # noqa: F401  (loads cuDNN before the ONNX session exists)

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("onnxruntime has no CUDAExecutionProvider")
            model = self._build("fp32", "cuda")
            # A missing cuDNN raises RequireCudnnHandle at the first inference.
            model.recognize(np.zeros(_SELF_CHECK_SAMPLES, dtype=np.float32), sample_rate=16000)
            return model
        except AsrModelMissingError:
            raise
        except Exception as exc:
            logger.warning("parakeet cuda unavailable (%s); falling back to cpu int8", exc)
            return self._build("int8", "cpu")

    def _recognize(self, audio: AudioChunk) -> str:
        if self._model is None:
            self.load()
        return str(self._model.recognize(audio.samples, sample_rate=audio.sample_rate_hz))

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._recognize, audio), self._settings.timeout_s
            )
        except TimeoutError as exc:
            raise AsrTimeoutError(
                f"parakeet exceeded listen.asr.timeout_s={self._settings.timeout_s}"
            ) from exc
        return normalize_transcript(
            text, language=self._settings.language, elapsed_s=time.perf_counter() - started
        )

    async def shutdown(self) -> None:
        self._model = None
