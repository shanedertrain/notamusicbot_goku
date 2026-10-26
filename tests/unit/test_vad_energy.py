from __future__ import annotations

import numpy as np
import pytest

from notamusicbot.speech.vad.base import FRAME_SAMPLES
from notamusicbot.speech.vad.energy import EnergyVad


def _tone(dbfs: float, n: int = FRAME_SAMPLES) -> np.ndarray:
    # A sine's RMS is amplitude / sqrt(2); scale so the RMS is exactly dbfs.
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    t = np.arange(n) / 16000
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_silence_is_zero() -> None:
    assert EnergyVad().process(np.zeros(FRAME_SAMPLES, dtype=np.float32)) == 0.0


def test_loud_tone_is_one() -> None:
    assert EnergyVad().process(_tone(-20.0)) == 1.0


def test_probability_is_half_at_threshold() -> None:
    p = EnergyVad(threshold_db=-40.0).process(_tone(-40.0))
    assert p == pytest.approx(0.5, abs=0.01)


def test_ramp_values() -> None:
    vad = EnergyVad(threshold_db=-40.0, ramp_db=10.0)
    assert vad.process(_tone(-50.0)) == pytest.approx(0.0, abs=0.01)
    assert vad.process(_tone(-45.0)) == pytest.approx(0.25, abs=0.01)
    assert vad.process(_tone(-35.0)) == pytest.approx(0.75, abs=0.01)


def test_rms_dbfs_of_scaled_tone() -> None:
    assert EnergyVad.rms_dbfs(_tone(-20.0)) == pytest.approx(-20.0, abs=0.05)


def test_empty_frame_and_reset_are_safe() -> None:
    vad = EnergyVad()
    assert vad.process(np.zeros(0, dtype=np.float32)) == 0.0
    vad.reset()


def test_bad_ramp_rejected() -> None:
    with pytest.raises(ValueError):
        EnergyVad(ramp_db=0.0)


def test_silero_integration_skipped_if_missing() -> None:
    pytest.importorskip("silero_vad")
    from notamusicbot.speech.vad.silero import SileroVad

    vad = SileroVad()
    assert vad.process(np.zeros(FRAME_SAMPLES, dtype=np.float32)) < 0.5
    vad.reset()
