"""Review additions for the listen speed build: stream limiting."""

from __future__ import annotations

import numpy as np

from notamusicbot.listen.reply import STREAM_GAIN_DB, _soft_limit


def test_soft_limit_is_linear_below_the_knee_and_stays_within_full_scale() -> None:
    low = np.array([-0.5, -0.1, 0.0, 0.3, 0.79], dtype=np.float32)
    assert np.allclose(_soft_limit(low), low)
    loud = np.array([-4.0, -1.0, 1.0, 1.5, 4.0], dtype=np.float32)
    out = _soft_limit(loud)
    assert np.all(np.abs(out) <= 1.0)
    assert np.all(np.sign(out) == np.sign(loud))
    assert np.all(np.diff(_soft_limit(np.linspace(0.0, 3.0, 300, dtype=np.float32))) >= 0)


def test_stream_gain_matches_the_live_measurement() -> None:
    # 8 real fast-TTS sentences vs the postprocessed WAV path (2026-10-25):
    # median RMS difference -3.7 dB at +3 dB gain, -0.6 dB at +5 dB.
    assert 3.5 <= STREAM_GAIN_DB <= 5.0
