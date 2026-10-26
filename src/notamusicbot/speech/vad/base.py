"""VoiceActivityDetector protocol.

Backends (silero ONNX, energy) implement :class:`VoiceActivityDetector`. They
run on the listen VAD thread on the CPU, never on the GPU or through the
inference coordinator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from notamusicbot.domain.listen import LISTEN_SAMPLE_RATE_HZ

if TYPE_CHECKING:
    import numpy as np

# One VAD frame: 32 ms at 16 kHz (the frame size silero expects).
FRAME_MS = 32
FRAME_SAMPLES = LISTEN_SAMPLE_RATE_HZ * FRAME_MS // 1000


class VoiceActivityDetector(Protocol):
    def process(self, frame: np.ndarray) -> float:
        """Return the speech probability in [0, 1] for one float32 frame."""
        ...

    def reset(self) -> None:
        """Drop any internal state (new speaker or new session)."""
        ...
