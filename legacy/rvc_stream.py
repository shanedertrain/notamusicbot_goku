"""Streaming RVC engine: block + left context + right lookahead + SOLA join.

Port of the reference prototype described in docs/RVC_STREAMING_DESIGN.md (sections 2, 3, 9).

Input is mono float32 at 24 kHz (TTS chunks); output is mono float32 at 48 kHz. The engine
is a pure function of the input sample sequence given a deterministic window converter and is
independent of how the input is chunked. ``output_total == 2 * input_total`` exactly after
``flush()``.

Import-time dependencies are numpy + scipy only; torch / faiss / rvc_cli are imported lazily
inside ``make_rvc_window_converter`` so the engine can be tested with fake converters.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

IN_SAMPLE_RATE_HZ = 24000
OUT_SAMPLE_RATE_HZ = 48000

_IN_PER_MS = IN_SAMPLE_RATE_HZ // 1000  # 24 input samples per ms
_FADE_IN_SAMPLES = 96  # 2 ms @ 48 kHz
_FADE_OUT_SAMPLES = 480  # 10 ms @ 48 kHz

# (min, max) in ms for each tunable; all must also be multiples of 20 ms.
_RANGES_MS = {
    "first_block_ms": (20, 1000),
    "block_ms": (100, 2000),
    "context_ms": (200, 4000),
    "lookahead_ms": (40, 500),
    "crossfade_ms": (10, 200),
    "search_ms": (0, 100),
}


@dataclass(frozen=True)
class StreamParams:
    """Tuning of one stream. All ``*_ms`` values are multiples of 20 ms."""

    first_block_ms: int = 100
    block_ms: int = 320
    context_ms: int = 1000
    lookahead_ms: int = 100
    crossfade_ms: int = 40
    search_ms: int = 20
    pitch_shift: int = 0
    f0_method: str = "pm"
    use_index: bool = False
    index_rate: float | None = None

    def validate(self) -> None:
        for name, (low, high) in _RANGES_MS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value % 20 != 0:
                raise ValueError(f"{name} must be a multiple of 20 ms, got {value}")
            if not low <= value <= high:
                raise ValueError(f"{name} must be in {low}..{high} ms, got {value}")
        if self.crossfade_ms > self.block_ms:
            raise ValueError("crossfade_ms must be <= block_ms")
        if self.search_ms > self.context_ms:
            raise ValueError("search_ms must be <= context_ms")
        if isinstance(self.pitch_shift, bool) or not isinstance(self.pitch_shift, int):
            raise ValueError(f"pitch_shift must be an int, got {self.pitch_shift!r}")
        if not -48 <= self.pitch_shift <= 48:
            raise ValueError(f"pitch_shift must be in -48..48, got {self.pitch_shift}")
        if self.f0_method != "pm":
            raise ValueError("streaming supports only pm")
        if self.index_rate is not None and not 0.0 <= float(self.index_rate) <= 1.0:
            raise ValueError("index_rate must be in 0..1")

    @classmethod
    def from_fields(cls, fields_: dict[str, Any]) -> StreamParams:
        """Build from a dict (e.g. a protocol request): known keys only, None = default."""
        known = {f.name for f in fields(cls)}
        picked = {k: v for k, v in fields_.items() if k in known and v is not None}
        params = cls(**picked)
        params.validate()
        return params


WindowConverter = Callable[[np.ndarray], np.ndarray]


def _fade_in(block: np.ndarray) -> np.ndarray:
    n = min(_FADE_IN_SAMPLES, block.shape[0])
    out = block.copy()
    if n:
        out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def _fade_out(block: np.ndarray) -> np.ndarray:
    n = min(_FADE_OUT_SAMPLES, block.shape[0])
    out = block.copy()
    if n:
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


class StreamEngine:
    """Block-wise streaming conversion with SOLA joins (see module docstring)."""

    def __init__(self, params: StreamParams, convert_window: WindowConverter) -> None:
        params.validate()
        self._params = params
        self._convert = convert_window
        self._first = params.first_block_ms * _IN_PER_MS
        self._block = params.block_ms * _IN_PER_MS
        self._ctx = params.context_ms * _IN_PER_MS
        self._look = params.lookahead_ms * _IN_PER_MS
        self._xfade = params.crossfade_ms * _IN_PER_MS
        self._search = params.search_ms * _IN_PER_MS
        # Crossfade ramp over 2*xfade output samples; strictly inside (0, 1).
        fo = 2 * self._xfade
        self._ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(fo) + 0.5) / fo)).astype(np.float32)
        self._buf = np.zeros(0, np.float32)  # input from absolute index self._buf_start
        self._buf_start = 0
        self._pos = 0  # nominal absolute input position of the next block
        self._tail: np.ndarray | None = None
        self._input_total = 0
        self._output_total = 0
        self._steps = 0
        self._compute_s = 0.0
        self._closed = False

    # -- accounting ---------------------------------------------------------------------------

    @property
    def min_input_samples_for_first_output(self) -> int:
        return self._first + self._xfade + self._look

    @property
    def input_total(self) -> int:
        return self._input_total

    @property
    def output_total(self) -> int:
        return self._output_total

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def compute_s(self) -> float:
        return self._compute_s

    @property
    def buffered_samples(self) -> int:
        """Length of the internal (trimmed) input buffer, for tests and diagnostics."""
        return int(self._buf.shape[0])

    # -- public API ---------------------------------------------------------------------------

    def push(self, pcm24: np.ndarray) -> np.ndarray:
        self._check_open()
        x = np.asarray(pcm24, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("pcm must be a 1-D mono array")
        if x.shape[0] and not np.isfinite(x).all():
            raise ValueError("pcm contains NaN or Inf")
        if x.shape[0]:
            self._buf = np.concatenate([self._buf, x])
            self._input_total += x.shape[0]
        outs: list[np.ndarray] = []
        while self._pos + self._blen() + self._xfade + self._look <= self._input_total:
            outs.append(self._step())
        return self._join(outs)

    def flush(self) -> np.ndarray:
        self._check_open()
        self._closed = True
        remaining = 2 * self._input_total - self._output_total
        if self._input_total == 0 or remaining <= 0:
            self._release()
            return np.zeros(0, np.float32)
        seg = self._convert_segment(self._blen())
        if self._tail is None:
            body = _fade_in(seg[2 * self._search :][:remaining])
        else:
            s2 = self._align(seg)
            fo = 2 * self._xfade
            xf = self._tail * (1.0 - self._ramp) + s2[:fo] * self._ramp
            body = np.concatenate([xf, s2[fo:]])[:remaining]
        body = _fade_out(body)
        if body.shape[0] != remaining:  # cannot happen with a valid window; keep exactness
            body = np.pad(body, (0, remaining - body.shape[0]))[:remaining]
        self._output_total += body.shape[0]
        self._pos = self._input_total
        self._release()
        return body.astype(np.float32, copy=False)

    def abort(self) -> None:
        self._closed = True
        self._release()

    # -- internals ----------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("stream engine is closed")

    def _release(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._tail = None

    def _blen(self) -> int:
        return self._first if self._pos == 0 else self._block

    @staticmethod
    def _join(outs: list[np.ndarray]) -> np.ndarray:
        if not outs:
            return np.zeros(0, np.float32)
        return outs[0] if len(outs) == 1 else np.concatenate(outs)

    def _window(self, blen: int) -> np.ndarray:
        """input[pos-search-context : pos+blen+xfade+look], reflect padded outside the data."""
        start = self._pos - self._search - self._ctx
        end = self._pos + blen + self._xfade + self._look
        have_end = self._buf_start + self._buf.shape[0]
        lo, hi = max(start, 0), min(end, have_end)
        w = self._buf[lo - self._buf_start : hi - self._buf_start]
        pad_left = max(0, -start)
        pad_right = max(0, end - have_end)
        if pad_left or pad_right:
            w = np.pad(w, (pad_left, pad_right), mode="reflect")
        return w

    def _convert_segment(self, blen: int) -> np.ndarray:
        win = self._window(blen)
        t0 = time.perf_counter()
        y = self._convert(win)
        self._compute_s += time.perf_counter() - t0
        self._steps += 1
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (2 * win.shape[0],):
            raise RuntimeError(
                f"window converter returned {y.shape}, expected ({2 * win.shape[0]},)"
            )
        # seg starts at nominal position pos - search.
        return y[2 * self._ctx :]

    def _align(self, seg: np.ndarray) -> np.ndarray:
        """Return seg shifted by the offset that best matches the saved tail."""
        so, fo = 2 * self._search, 2 * self._xfade
        if so == 0:
            return seg
        cand = seg[: so + fo].astype(np.float64)
        tail = self._tail.astype(np.float64)
        num = np.correlate(cand, tail, mode="valid")  # so + 1 values
        energy = np.sqrt(np.convolve(cand * cand, np.ones(fo), mode="valid") + 1e-9)
        d = int(np.argmax(num / energy))
        return seg[d:]

    def _step(self) -> np.ndarray:
        blen = self._blen()
        seg = self._convert_segment(blen)
        so, fo = 2 * self._search, 2 * self._xfade
        if self._tail is None:
            out = _fade_in(seg[so : so + 2 * blen])
            self._tail = seg[so + 2 * blen : so + 2 * blen + fo].copy()
        else:
            s2 = self._align(seg)
            xf = self._tail * (1.0 - self._ramp) + s2[:fo] * self._ramp
            out = np.concatenate([xf, s2[fo : 2 * blen]])
            self._tail = s2[2 * blen : 2 * blen + fo].copy()
        if out.shape[0] != 2 * blen or self._tail.shape[0] != fo:
            raise RuntimeError("internal block size mismatch")
        self._pos += blen
        self._output_total += out.shape[0]
        self._trim()
        return out.astype(np.float32, copy=False)

    def _trim(self) -> None:
        keep_from = max(0, self._pos - self._search - self._ctx)
        drop = keep_from - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = keep_from


# -- real RVC window converter --------------------------------------------------------------


def _resolve_index_path(vc: Any) -> str | None:
    """Locate the faiss index file; the stored path is relative to rvc_cli/."""
    raw = getattr(vc, "index_path_stripped", None)
    if not raw:
        return None
    candidates = [raw]
    if not os.path.isabs(raw):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "..", "rvc_cli", raw))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def make_rvc_window_converter(vc: Any, vc_args: Any, params: StreamParams) -> WindowConverter:
    """Build ``window24 -> window48`` around a loaded rvc_cli ``VoiceConverter``.

    Mirrors VC.pipeline / VC.vc of vc_infer_pipeline.py for one window, without the per-call
    print, ``empty_cache`` or index re-read.
    """
    import torch
    from scipy import signal

    params.validate()
    pipe = vc.vc  # rvc_cli VC object: device, is_half, get_f0_pm_computation
    device = pipe.device
    is_half = bool(pipe.is_half)
    is_v1 = vc.version == "v1"
    layer = 9 if is_v1 else 12
    sid_value = int(vc_args.sid)
    pitch_shift = params.pitch_shift
    tgt_sr = int(vc.tgt_sr)
    hp_b, hp_a = signal.butter(N=5, Wn=48, btype="high", fs=16000)
    f0_min, f0_max = 50, 1100
    mel_min = 1127 * np.log(1 + f0_min / 700)
    mel_max = 1127 * np.log(1 + f0_max / 700)

    index = None
    index_rows = None
    index_rate = float(params.index_rate if params.index_rate is not None else vc_args.index_rate)
    if params.use_index and index_rate:
        index_path = _resolve_index_path(vc)
        if index_path is not None:
            import faiss

            index = faiss.read_index(index_path)
            index_rows = index.reconstruct_n(0, index.ntotal)

    if tgt_sr == OUT_SAMPLE_RATE_HZ:
        up = down = 1
    else:
        g = math.gcd(tgt_sr, OUT_SAMPLE_RATE_HZ)
        up, down = OUT_SAMPLE_RATE_HZ // g, tgt_sr // g

    def convert(win24: np.ndarray) -> np.ndarray:
        want = 2 * win24.shape[0]
        x16 = signal.resample_poly(win24, 2, 3).astype(np.float32)
        x16 = signal.filtfilt(hp_b, hp_a, x16).astype(np.float32)
        p_len = x16.shape[0] // 160
        # f0 (pm) + the mel quantisation of VC.get_f0, minus its print().
        f0 = pipe.get_f0_pm_computation(x16, 10.0, f0_min, f0_max, p_len)
        f0 = f0 * pow(2, pitch_shift / 12)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        coarse = np.rint(f0_mel).astype(int)
        with torch.no_grad():
            pitch = torch.tensor(coarse[:p_len], device=device).unsqueeze(0).long()
            pitchf = torch.tensor(f0[:p_len], device=device).unsqueeze(0).float()
            feats = torch.from_numpy(x16)
            feats = feats.half() if is_half else feats.float()
            feats = feats.view(1, -1).to(device)
            mask = torch.zeros_like(feats, dtype=torch.bool, device=device)
            logits = vc.hubert_model.extract_features(
                source=feats, padding_mask=mask, output_layer=layer
            )
            feats = vc.hubert_model.final_proj(logits[0]) if is_v1 else logits[0]
            if index is not None:
                npy = feats[0].float().cpu().numpy()
                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(index_rows[ix] * np.expand_dims(weight, 2), axis=1)
                if is_half:
                    npy = npy.astype("float16")
                blended = torch.from_numpy(npy).unsqueeze(0).to(device)
                feats = blended * index_rate + (1 - index_rate) * feats
            feats = torch.nn.functional.interpolate(feats.permute(0, 2, 1), scale_factor=2)
            feats = feats.permute(0, 2, 1)
            n = min(p_len, feats.shape[1])
            lengths = torch.tensor([n], device=device).long()
            sid = torch.tensor(sid_value, device=device).unsqueeze(0).long()
            out = vc.net_g.infer(feats, lengths, pitch[:, :n], pitchf[:, :n], sid)[0][0, 0]
            y = out.data.cpu().float().numpy()
        if up != down:
            y = signal.resample_poly(y, up, down)
        y = np.asarray(y, dtype=np.float32)
        if y.shape[0] < want:
            y = np.pad(y, (0, want - y.shape[0]))
        y = y[:want]
        if not np.isfinite(y).all():
            raise RuntimeError("RVC produced non-finite samples")
        return y

    return convert


def open_engine(vc: Any, vc_args: Any, params: StreamParams) -> StreamEngine:
    """Engine over a real loaded model. ``params.pitch_shift`` overrides ``vc_args.f0_pitch``."""
    return StreamEngine(params, make_rvc_window_converter(vc, vc_args, params))
