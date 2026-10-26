"""Whisper small.en via transformers (the GPU fallback listen ASR).

fp16 on cuda, local HF cache only. The blocking call goes through the shared
:class:`InferenceCoordinator` so it never overlaps TTS/RVC on the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from notamusicbot.config import ListenAsrSettings
from notamusicbot.domain.listen import AudioChunk, Transcript
from notamusicbot.speech.asr._common import (
    AsrModelMissingError,
    AsrTimeoutError,
    missing_model_message,
    normalize_transcript,
)
from notamusicbot.speech.coordinator import InferenceCoordinator

logger = logging.getLogger(__name__)


class WhisperHfTranscriber:
    def __init__(self, settings: ListenAsrSettings, coordinator: InferenceCoordinator) -> None:
        self._settings = settings
        self._coordinator = coordinator
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    @classmethod
    def from_settings(
        cls, settings: ListenAsrSettings, coordinator: InferenceCoordinator
    ) -> WhisperHfTranscriber:
        return cls(settings, coordinator)

    def load(self) -> None:
        """Load from the local HF cache (blocking; never downloads)."""
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = self._settings.whisper_model_id
        cuda = self._settings.device == "cuda"
        try:
            processor = WhisperProcessor.from_pretrained(model_id, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                local_files_only=True,
                torch_dtype=torch.float16 if cuda else torch.float32,
            )
        except OSError as exc:
            raise AsrModelMissingError(missing_model_message(model_id)) from exc
        self._model = model.to("cuda" if cuda else "cpu").eval()
        self._processor = processor
        self._torch = torch
        logger.info("whisper loaded model=%s device=%s", model_id, self._settings.device)

    def _recognize(self, audio: AudioChunk) -> str:
        self.load()
        torch = self._torch
        device = "cuda" if self._settings.device == "cuda" else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        feats = self._processor(
            audio.samples, sampling_rate=audio.sample_rate_hz, return_tensors="pt"
        ).input_features
        with torch.inference_mode():
            ids = self._model.generate(feats.to(device, dtype=dtype))
        return str(self._processor.batch_decode(ids, skip_special_tokens=True)[0])

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                self._coordinator.run_blocking(self._recognize, audio), self._settings.timeout_s
            )
        except TimeoutError as exc:
            raise AsrTimeoutError(
                f"whisper exceeded listen.asr.timeout_s={self._settings.timeout_s}"
            ) from exc
        return normalize_transcript(
            text, language=self._settings.language, elapsed_s=time.perf_counter() - started
        )

    async def shutdown(self) -> None:
        self._model = None
        self._processor = None
