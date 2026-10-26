"""SileroVad against the real silero-vad 6.2.3 ONNX model (CPU).

Skipped when silero-vad is not installed. Needs no network: the model ships
inside the wheel.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import FakeClock
from notamusicbot.config import ListenVadSettings
from notamusicbot.domain.listen import UtteranceEvent
from notamusicbot.listen.endpointer import Endpointer
from notamusicbot.speech.vad.base import FRAME_SAMPLES

pytestmark = pytest.mark.integration

SR = 16000
FRAME_S = FRAME_SAMPLES / SR


def _make_vad():
    pytest.importorskip("silero_vad")
    from notamusicbot.speech.vad.silero import SileroVad

    return SileroVad()


def _voiced(seconds: float = 1.0) -> np.ndarray:
    """120 Hz sawtooth (5 Hz vibrato) through three formant resonators, 4 Hz syllable AM."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 120 + 5 * np.sin(2 * np.pi * 3 * t)
    saw = 2 * ((np.cumsum(f0) / SR) % 1) - 1
    out = np.zeros(n)
    for fc, bw in ((700, 100), (1200, 120), (2600, 160)):
        r = np.exp(-np.pi * bw / SR)
        a1, a2 = -2 * r * np.cos(2 * np.pi * fc / SR), r * r
        y1 = y2 = 0.0
        res = np.empty(n)
        for i in range(n):
            y = (1 - r) * saw[i] - a1 * y1 - a2 * y2
            res[i] = y
            y2, y1 = y1, y
        out += res
    out = out / np.abs(out).max() * 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return out.astype(np.float32)


def _probs(vad, signal: np.ndarray) -> list[float]:
    return [
        vad.process(signal[i : i + FRAME_SAMPLES])
        for i in range(0, len(signal) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]


def test_silence_scores_low_and_in_range() -> None:
    probs = _probs(_make_vad(), np.zeros(SR, dtype=np.float32))
    assert len(probs) == 31  # 1 s / 32 ms, whole frames only
    assert all(0.0 <= p <= 1.0 for p in probs)
    # measured: max 0.009 on silero-vad 6.2.3
    assert max(probs) < 0.05


def test_synthetic_voiced_signal_scores_high_and_in_range() -> None:
    probs = _probs(_make_vad(), _voiced())
    assert len(probs) == 31
    assert all(0.0 <= p <= 1.0 for p in probs)
    # measured: min 0.159, mean 0.617, max 0.894. The mean is the stable
    # figure; a synthetic source is not real speech, so confirm with a real
    # voice before trusting thresholds.
    assert np.mean(probs) > 0.4
    assert max(probs) > 0.5


def test_reset_makes_runs_identical() -> None:
    vad = _make_vad()
    sig = _voiced(0.5)
    first = _probs(vad, sig)
    vad.reset()
    assert _probs(vad, sig) == first


def test_fresh_instances_are_independent() -> None:
    a, b = _make_vad(), _make_vad()
    _probs(a, _voiced(0.5))  # a now carries state; b must be unaffected
    assert _probs(b, np.zeros(SR // 2, dtype=np.float32)) == _probs(
        _make_vad(), np.zeros(SR // 2, dtype=np.float32)
    )


def test_wrong_frame_size_rejected() -> None:
    with pytest.raises(ValueError):
        _make_vad().process(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32))


def test_per_frame_latency_is_small() -> None:
    vad = _make_vad()
    frame = _voiced(0.1)[:FRAME_SAMPLES]
    vad.process(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(100):
        vad.process(frame)
    mean_ms = (time.perf_counter() - start) * 10
    assert mean_ms < 5.0  # measured ~0.1 ms; wide margin for a busy box


def test_does_not_leave_torch_single_threaded(monkeypatch) -> None:
    import sys

    torch = pytest.importorskip("torch")
    pytest.importorskip("silero_vad")
    # Force a fresh import so the module-level set_num_threads(1) runs again
    # here, whatever order the tests ran in (an earlier test already imported it).
    for name in [m for m in sys.modules if m == "silero_vad" or m.startswith("silero_vad.")]:
        monkeypatch.delitem(sys.modules, name)
    from notamusicbot.speech.vad.silero import SileroVad

    before = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        SileroVad()  # not _make_vad(): its importorskip would import silero_vad first
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)


def test_endpointer_with_silero_factory_makes_one_event() -> None:
    _make_vad()  # skip early when silero is missing
    from notamusicbot.speech.vad.silero import SileroVad

    clock = FakeClock()
    events: list = []
    ep = Endpointer(SileroVad, ListenVadSettings(), clock, events.append)
    sig = np.concatenate([np.zeros(int(0.3 * SR)), _voiced(1.0), np.zeros(int(1.0 * SR))]).astype(
        np.float32
    )
    for i in range(0, len(sig) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        clock.advance(FRAME_S)
        ep.feed(7, sig[i : i + FRAME_SAMPLES])
    utts = [e for e in events if isinstance(e, UtteranceEvent)]
    assert len(utts) == 1
    assert utts[0].user_id == 7
