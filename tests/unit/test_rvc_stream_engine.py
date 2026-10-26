"""Unit tests for the streaming RVC engine (legacy/rvc_stream.py): CPU, fake converters."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "legacy"))

import rvc_stream  # noqa: E402
from rvc_stream import StreamEngine, StreamParams  # noqa: E402

IN_PER_MS = 24
SR_IN = 24000


class PhaseJumpVoice:
    """Deterministic fake voice: 2x upsample of the input plus a sine whose phase restarts at
    each window start (the phase break real NSF windows have). Records the window lengths."""

    def __init__(self, freq: float = 220.0, amp: float = 0.2) -> None:
        self.freq = freq
        self.amp = amp
        self.calls = 0
        self.window_lengths: list[int] = []

    def __call__(self, win: np.ndarray) -> np.ndarray:
        self.window_lengths.append(win.shape[0])
        phase0 = 0.7 * self.calls  # differs per window, deterministic
        self.calls += 1
        n = 2 * win.shape[0]
        t = np.arange(n) / 48000.0
        sine = self.amp * np.sin(2 * np.pi * self.freq * t + phase0)
        return (np.repeat(win, 2) + sine).astype(np.float32)


class PeriodicVoice:
    """Pure periodic voice with a different phase in every window."""

    def __init__(self, freq: float = 220.0) -> None:
        self.freq = freq
        self.calls = 0

    def __call__(self, win: np.ndarray) -> np.ndarray:
        phase0 = 1.3 * self.calls + 0.4 * self.calls**2
        self.calls += 1
        t = np.arange(2 * win.shape[0]) / 48000.0
        return (0.5 * np.sin(2 * np.pi * self.freq * t + phase0)).astype(np.float32)


def make_input(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR_IN
    x = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(n)
    return x.astype(np.float32)


def run(x: np.ndarray, chunks, params: StreamParams | None = None, voice=None) -> np.ndarray:
    engine = StreamEngine(params or StreamParams(), voice or PhaseJumpVoice())
    outs = []
    i = 0
    for size in chunks:
        outs.append(engine.push(x[i : i + size]))
        i += size
    if i < x.shape[0]:
        outs.append(engine.push(x[i:]))
    outs.append(engine.flush())
    return np.concatenate(outs)


def fixed_chunks(n: int, size: int) -> list[int]:
    return [size] * (n // size) + ([n % size] if n % size else [])


# (a) exact length accounting ----------------------------------------------------------------


@pytest.mark.parametrize("total", [0, 1, 2, 480, 1200, 5760, 5761, 24 * 240, 79200, 150000])
def test_output_total_is_exactly_twice_input_after_flush(total: int) -> None:
    x = make_input(total)
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    outs = [engine.push(x[i : i + 6000]) for i in range(0, total, 6000)]
    outs.append(engine.flush())
    y = np.concatenate(outs) if outs else np.zeros(0, np.float32)
    assert engine.input_total == total
    assert engine.output_total == 2 * total
    assert y.shape[0] == 2 * total
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_total_zero_flush_returns_empty() -> None:
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    assert engine.flush().shape == (0,)
    assert engine.output_total == 0


def test_short_total_without_search() -> None:
    params = StreamParams(search_ms=0)
    y = run(make_input(1200), [], params)
    assert y.shape[0] == 2400


# (b) chunk invariance -------------------------------------------------------------------------


def test_chunk_invariance_bit_identical() -> None:
    n = 24000 * 3 + 1234
    x = make_input(n)
    ref = run(x, fixed_chunks(n, 6000))
    for size in (1, 480, 24000):
        if size == 1:
            sub = x[:30000]
            assert np.array_equal(run(sub, fixed_chunks(30000, 1)), run(sub, [30000]))
            continue
        assert np.array_equal(run(x, fixed_chunks(n, size)), ref)
    rng = np.random.default_rng(7)
    sizes = []
    left = n
    while left > 0:
        s = int(min(left, rng.integers(0, 9000)))
        sizes.append(s)
        left -= s
    assert np.array_equal(run(x, sizes), ref)
    assert np.array_equal(run(x, [n]), ref)


def test_two_streams_share_no_state() -> None:
    """A second engine (fresh voice) must reproduce the first bit for bit: no stale tail/buffer."""
    x = make_input(24000 * 2 + 77)
    other = make_input(24000, seed=9)
    first = run(x, fixed_chunks(x.shape[0], 6000))
    run(other, fixed_chunks(other.shape[0], 6000))  # an unrelated stream in between
    second = run(x, fixed_chunks(x.shape[0], 6000))
    assert np.array_equal(first, second)


# (c) first output timing ----------------------------------------------------------------------


def test_first_output_exactly_at_min_input() -> None:
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    need = engine.min_input_samples_for_first_output
    assert need == (100 + 40 + 100) * 24 == 5760
    x = make_input(need + 10)
    assert engine.push(x[: need - 1]).shape[0] == 0
    out = engine.push(x[need - 1 : need])
    assert out.shape[0] == 2 * 100 * 24
    assert engine.steps == 1


# (d) SOLA continuity --------------------------------------------------------------------------


def test_sola_join_continuity_with_phase_breaks() -> None:
    params = StreamParams()
    n = 24000 * 4
    x = make_input(n)
    voice = PeriodicVoice()
    y = run(x, fixed_chunks(n, 6000), params, voice)
    d2 = np.abs(np.diff(y.astype(np.float64), 2))
    median = float(np.median(d2))
    joins = [2 * params.first_block_ms * IN_PER_MS]
    while joins[-1] + 2 * params.block_ms * IN_PER_MS < y.shape[0] - 2400:
        joins.append(joins[-1] + 2 * params.block_ms * IN_PER_MS)
    assert len(joins) >= 8
    worst = max(float(d2[j - 40 : j + 40].max()) for j in joins[1:])
    # Without SOLA a phase break shows up as a >10x jump; measured worst is ~1.2x the median
    # (a pure sine's second difference peaks at ~1.4x its median). Require <= 3x.
    assert worst <= 3.0 * median, (worst, median)

    # Phase-breaks are real: without alignment the crossfade region loses energy (cancellation,
    # measured 0.72 of the stream RMS); with SOLA it stays at ~0.99.
    def crossfade_rms_ratio(out: np.ndarray) -> float:
        width = 2 * 2 * params.crossfade_ms * IN_PER_MS // 2
        rms = [float(np.sqrt(np.mean(out[j : j + width] ** 2))) for j in joins[1:]]
        return min(rms) / float(np.sqrt(np.mean(out**2)))

    unaligned = run(x, fixed_chunks(n, 6000), StreamParams(search_ms=0), PeriodicVoice())
    assert crossfade_rms_ratio(y) > 0.95
    assert crossfade_rms_ratio(unaligned) < 0.85


# (e) trimming --------------------------------------------------------------------------------


def test_internal_buffer_stays_bounded_for_30s_stream() -> None:
    p = StreamParams()
    max_push = 6000
    bound = (
        p.search_ms + p.context_ms + p.block_ms + p.crossfade_ms + p.lookahead_ms
    ) * IN_PER_MS + max_push
    engine = StreamEngine(p, PhaseJumpVoice())
    x = make_input(30 * SR_IN)
    peak = 0
    for i in range(0, x.shape[0], max_push):
        engine.push(x[i : i + max_push])
        peak = max(peak, engine.buffered_samples)
    assert peak <= bound, (peak, bound)
    assert engine.input_total == 30 * SR_IN
    engine.flush()
    assert engine.output_total == 2 * 30 * SR_IN


# (f) validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"block_ms": 330},
        {"first_block_ms": 90},
        {"context_ms": 1010},
        {"lookahead_ms": 50},
        {"crossfade_ms": 45},
        {"search_ms": 10},
        {"f0_method": "harvest"},
        {"first_block_ms": 0},
        {"first_block_ms": 1020},
        {"block_ms": 80},
        {"block_ms": 2020},
        {"context_ms": 160},
        {"context_ms": 4020},
        {"lookahead_ms": 20},
        {"lookahead_ms": 520},
        {"crossfade_ms": 0},
        {"crossfade_ms": 220},
        {"search_ms": 120},
        {"block_ms": 100, "crossfade_ms": 120},
        {"block_ms": 320.0},
        {"index_rate": 1.5},
    ],
)
def test_params_validate_rejects(overrides: dict) -> None:
    with pytest.raises(ValueError):
        StreamParams(**overrides).validate()
    with pytest.raises(ValueError):
        StreamEngine(StreamParams(**overrides), PhaseJumpVoice())


def test_params_validate_accepts_defaults_and_edges() -> None:
    StreamParams().validate()
    StreamParams(search_ms=0).validate()
    StreamParams(first_block_ms=20, block_ms=100, context_ms=200, lookahead_ms=40).validate()
    StreamParams(crossfade_ms=100, block_ms=100).validate()
    assert StreamParams().f0_method == "pm"


def test_harvest_message() -> None:
    with pytest.raises(ValueError, match="streaming supports only pm"):
        StreamParams(f0_method="harvest").validate()


def test_from_fields_picks_known_keys_only() -> None:
    p = StreamParams.from_fields(
        {
            "id": 3,
            "action": "stream_open",
            "block_ms": 400,
            "pitch_shift": 2,
            "first_block_ms": None,
        }
    )
    assert p.block_ms == 400 and p.pitch_shift == 2 and p.first_block_ms == 100
    with pytest.raises(ValueError):
        StreamParams.from_fields({"block_ms": 333})
    with pytest.raises(ValueError):
        StreamParams.from_fields({"f0_method": "harvest"})


# (g) abort / closed / bad input ----------------------------------------------------------------


def test_abort_drops_state_and_closes() -> None:
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    engine.push(make_input(10000))
    engine.abort()
    with pytest.raises(RuntimeError):
        engine.push(make_input(10))
    with pytest.raises(RuntimeError):
        engine.flush()
    engine.abort()  # idempotent


def test_post_flush_errors() -> None:
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    engine.push(make_input(1000))
    engine.flush()
    with pytest.raises(RuntimeError):
        engine.push(make_input(10))
    with pytest.raises(RuntimeError):
        engine.flush()


def test_push_rejects_non_finite_and_bad_shape() -> None:
    engine = StreamEngine(StreamParams(), PhaseJumpVoice())
    bad = make_input(100)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        engine.push(bad)
    bad[5] = np.inf
    with pytest.raises(ValueError):
        engine.push(bad)
    with pytest.raises(ValueError):
        engine.push(np.zeros((2, 10), np.float32))
    assert engine.input_total == 0
    assert engine.push(np.zeros(0, np.float32)).shape == (0,)
    assert engine.push(np.zeros(1, np.float32)).shape == (0,)


def test_converter_wrong_length_is_an_error() -> None:
    engine = StreamEngine(StreamParams(), lambda w: np.zeros(w.shape[0], np.float32))
    with pytest.raises(RuntimeError):
        engine.push(make_input(6000))


# (h) constant window shapes -------------------------------------------------------------------


def test_windows_are_constant_length_after_first_block() -> None:
    p = StreamParams()
    voice = PhaseJumpVoice()
    engine = StreamEngine(p, voice)
    x = make_input(5 * SR_IN)
    for i in range(0, x.shape[0], 6000):
        engine.push(x[i : i + 6000])
    engine.flush()
    first = (p.search_ms + p.context_ms + p.first_block_ms + p.crossfade_ms + p.lookahead_ms) * 24
    steady = (p.search_ms + p.context_ms + p.block_ms + p.crossfade_ms + p.lookahead_ms) * 24
    assert voice.window_lengths[0] == first
    assert len(voice.window_lengths) > 4
    assert set(voice.window_lengths[1:]) == {steady}


# optional GPU test ------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RVC_STREAM_GPU_TESTS") != "1",
    reason="set RVC_STREAM_GPU_TESTS=1 (GPU, legacy venv)",
)
def test_gpu_real_model_smoke() -> None:  # pragma: no cover - needs the legacy venv + GPU
    saved = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        import rvc_worker

        worker = rvc_worker.Worker()
        worker.load_model("Goku")
    finally:
        sys.argv = saved
    engine = rvc_stream.open_engine(worker._vc, worker._vc_args, StreamParams())
    x = make_input(2 * SR_IN)
    outs = [engine.push(x[i : i + 6000]) for i in range(0, x.shape[0], 6000)]
    outs.append(engine.flush())
    y = np.concatenate(outs)
    assert y.shape[0] == 2 * x.shape[0]
    assert np.isfinite(y).all()
