"""Silero VAD (ONNX, CPU) behind :class:`VoiceActivityDetector`.

Needs the optional ``listen`` extra (``pip install notamusicbot[listen]``).
The import is lazy so the package, and every test that does not construct
this class, loads without silero-vad.
"""

from __future__ import annotations

import numpy as np

from notamusicbot.domain.listen import LISTEN_SAMPLE_RATE_HZ
from notamusicbot.speech.vad.base import FRAME_SAMPLES


class SileroVad:
    """One stateful silero stream: use one instance per speaker."""

    def __init__(self) -> None:
        try:
            import torch

            # Importing silero_vad calls torch.set_num_threads(1) process-wide;
            # restore it so the LLM/TTS torch CPU work is not throttled.
            threads = torch.get_num_threads()
            from silero_vad import load_silero_vad

            torch.set_num_threads(threads)
        except ImportError as exc:  # pragma: no cover - exercised only when missing
            raise RuntimeError(
                "SileroVad needs the 'listen' extra (silero-vad); "
                "install it with: pip install 'notamusicbot[listen]'"
            ) from exc
        self._torch = torch
        # onnx=True runs on onnxruntime (CPU, 1 intra-op thread); no GPU, no
        # coordinator slot. The model file ships inside the wheel
        # (silero_vad/data/silero_vad.onnx): loading makes no network call.
        # Verified against silero-vad 6.2.3: the model takes exactly 512
        # samples at 16 kHz and keeps its own LSTM state and 64-sample context
        # between calls, so one instance per speaker and reset() between
        # utterances is the whole contract.
        self._model = load_silero_vad(onnx=True)

    def process(self, frame: np.ndarray) -> float:
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {len(frame)}")
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        return float(self._model(tensor, LISTEN_SAMPLE_RATE_HZ).item())

    def reset(self) -> None:
        self._model.reset_states()
