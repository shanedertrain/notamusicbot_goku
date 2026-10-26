"""Energy (RMS) voice-activity detector: no dependencies, CPU only.

Used by unit tests and as the degraded-mode fallback when silero is missing.
It cannot tell speech from other loud sound, so it is a gate, not a classifier.
"""

from __future__ import annotations

import math

import numpy as np

# Floor for log10 so digital silence maps to a finite, very low dBFS value.
_MIN_RMS = 1e-10


class EnergyVad:
    """Maps frame RMS (in dBFS) to a 0..1 pseudo-probability.

    The probability is 0.5 exactly at ``threshold_db`` and ramps linearly to
    0 at ``threshold_db - ramp_db`` and to 1 at ``threshold_db + ramp_db``,
    so the default endpointer threshold of 0.5 means "RMS above threshold_db".
    """

    def __init__(self, threshold_db: float = -40.0, ramp_db: float = 10.0) -> None:
        if ramp_db <= 0.0:
            raise ValueError("ramp_db must be > 0")
        self.threshold_db = threshold_db
        self.ramp_db = ramp_db

    @staticmethod
    def rms_dbfs(frame: np.ndarray) -> float:
        """RMS level of ``frame`` in dBFS (full-scale sine is about -3 dBFS)."""
        if frame.size == 0:
            return 20.0 * math.log10(_MIN_RMS)
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, _MIN_RMS))

    def process(self, frame: np.ndarray) -> float:
        level = self.rms_dbfs(frame)
        p = 0.5 + (level - self.threshold_db) / (2.0 * self.ramp_db)
        return min(1.0, max(0.0, p))

    def reset(self) -> None:
        """Stateless: nothing to drop."""
