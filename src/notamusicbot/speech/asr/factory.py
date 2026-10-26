"""Pick the listen ASR backend from settings."""

from __future__ import annotations

from notamusicbot.config import ListenAsrSettings
from notamusicbot.speech.asr.base import Transcriber
from notamusicbot.speech.asr.parakeet_onnx import ParakeetOnnxTranscriber
from notamusicbot.speech.asr.whisper_hf import WhisperHfTranscriber
from notamusicbot.speech.coordinator import InferenceCoordinator


def build_transcriber(
    settings: ListenAsrSettings, coordinator: InferenceCoordinator
) -> Transcriber:
    """Parakeet (CPU, ignores the coordinator) or Whisper (GPU, serialized by it)."""
    if settings.backend == "whisper_hf":
        return WhisperHfTranscriber.from_settings(settings, coordinator)
    return ParakeetOnnxTranscriber.from_settings(settings)
