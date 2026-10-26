"""Tests for legacy/rvc_f0.py (pitch extraction + post-processing)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy"))
import rvc_f0  # noqa: E402
from rvc_f0 import F0PostConfig, auto_pitch_shift, median_f0, postprocess_f0  # noqa: E402

SAMPLE = Path("/home/cbash23/.cache/listen-speed/qwen-fast/samples/fast_nonstream_s0.wav")
LEGACY_PY = ROOT / ".venv" / "bin" / "python"


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


needs_pm = pytest.mark.skipif(not _have("parselmouth"), reason="parselmouth not installed")
needs_rmvpe = pytest.mark.skipif(
    not (rvc_f0.rmvpe_available() and _have("torch") and _have("librosa") and SAMPLE.exists()),
    reason="rmvpe weights / torch / librosa / sample unavailable",
)


def test_median_f0():
    assert median_f0(np.zeros(5)) == 0.0
    assert median_f0(np.array([0, 100, 0, 200, 300.0])) == 200.0


def test_auto_pitch_shift():
    assert auto_pitch_shift(125, 108) == -3
    assert auto_pitch_shift(125, 108, strength=0.5) == -1
    assert auto_pitch_shift(100, 800) == 12
    assert auto_pitch_shift(800, 100) == -12
    assert auto_pitch_shift(800, 100, max_abs=5) == -5
    assert auto_pitch_shift(0, 108) == 0
    assert auto_pitch_shift(125, 0) == 0
    assert auto_pitch_shift(100, 100) == 0


def test_postprocess_noop_and_copy():
    f0 = np.array([0, 100, 100, 0], dtype=np.float32)
    out = postprocess_f0(f0, F0PostConfig(), 50, 1100)
    assert out.dtype == np.float32 and np.array_equal(out, f0) and out is not f0


def test_median_keeps_unvoiced_zeros():
    f0 = np.zeros(30, dtype=np.float32)
    f0[5:15] = 100.0
    f0[10] = 300.0  # spike
    f0[20:28] = 150.0
    out = postprocess_f0(f0, F0PostConfig(median_width=5), None, None)
    assert np.array_equal(out > 0, f0 > 0)
    assert out[10] == pytest.approx(100.0)
    assert np.allclose(out[20:28], 150.0)


def test_octave_repair_folds_spike():
    f0 = np.full(40, 120.0, dtype=np.float32)
    f0[20] = 240.0
    f0[30] = 60.0
    out = postprocess_f0(f0, F0PostConfig(octave_repair=True), None, None)
    assert out[20] == pytest.approx(120.0)
    assert out[30] == pytest.approx(120.0)
    assert np.allclose(out[[0, 10, 39]], 120.0)
    # a genuine 5-semitone step is left alone
    g = np.full(40, 120.0, dtype=np.float32)
    g[20] = 160.0
    assert postprocess_f0(g, F0PostConfig(octave_repair=True), None, None)[20] == 160.0


def test_clamp_fold_or_zero():
    f0 = np.array([0, 100, 600, 40, 100000, 0.5], dtype=np.float32)
    out = postprocess_f0(f0, F0PostConfig(clamp_min_hz=50, clamp_max_hz=500), None, None)
    assert out[1] == 100.0
    assert out[2] == pytest.approx(300.0)
    assert out[3] == pytest.approx(80.0)
    assert out[0] == 0.0
    assert out[4] == 0.0  # cannot fold into range within the octave budget
    assert out[5] == 0.0


def test_fill_gap():
    f0 = np.zeros(20, dtype=np.float32)
    f0[2:5] = 100.0
    f0[7:10] = 200.0  # gap of 2 frames
    f0[15:18] = 100.0  # gap of 5 frames
    out = postprocess_f0(f0, F0PostConfig(fill_gap_frames=2), None, None)
    assert out[5] == pytest.approx(100.0 * 2 ** (1 / 3))
    assert out[6] == pytest.approx(100.0 * 2 ** (2 / 3))
    assert np.all(out[10:15] == 0) and out[0] == 0 and out[19] == 0


def test_hard_bounds():
    f0 = np.array([30, 100, 2000], dtype=np.float32)
    assert list(postprocess_f0(f0, None, 50, 1100)) == [0, 100, 0]


def test_unknown_method_and_missing_weights(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        rvc_f0.extract_f0(np.zeros(1600, np.float32), 11, "nope")
    monkeypatch.setattr(rvc_f0, "RMVPE_DEFAULT_PATH", tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="rmvpe weights not found"):
        rvc_f0.extract_f0(np.zeros(1600, np.float32), 11, "rmvpe")


@needs_pm
def test_pm_tone_shape_dtype_median():
    t = np.arange(32000) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    p_len = len(x) // 160
    f0 = rvc_f0.extract_f0(x, p_len, "pm")
    assert f0.shape == (p_len,) and f0.dtype == np.float32
    assert abs(median_f0(f0) - 150.0) < 3.0
    short = rvc_f0.extract_f0(x, p_len + 7, "pm")
    assert short.shape == (p_len + 7,)


@needs_rmvpe
def test_rmvpe_state_dict_strict():
    import torch

    net = rvc_f0._rmvpe_classes()()
    state = torch.load(str(rvc_f0.RMVPE_DEFAULT_PATH), map_location="cpu", weights_only=True)
    res = net.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys


@needs_rmvpe
def test_rmvpe_vs_harvest_on_speech():
    if not (_have("pyworld") and _have("librosa")):
        pytest.skip("pyworld/librosa missing")
    import librosa

    x, _ = librosa.load(str(SAMPLE), sr=16000)
    p_len = len(x) // 160 + 1
    t0 = time.perf_counter()
    f = rvc_f0.extract_f0(x, p_len, "rmvpe")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    f = rvc_f0.extract_f0(x, p_len, "rmvpe")
    warm = time.perf_counter() - t0
    h = rvc_f0.extract_f0(x, p_len, "harvest")
    print(f"rmvpe first={first:.3f}s warm={warm:.3f}s for {len(x) / 16000:.2f}s audio")
    assert f.shape == (p_len,) and f.dtype == np.float32
    mf, mh = median_f0(f), median_f0(h)
    assert abs(mf - mh) / mh < 0.06
    both = (f > 0) & (h > 0)
    assert both.sum() > 50
    assert np.corrcoef(np.log(f[both]), np.log(h[both]))[0, 1] >= 0.9
