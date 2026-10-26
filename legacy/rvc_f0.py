"""F0 (pitch) extraction and post-processing for the legacy RVC runtime.

Pure helpers shared by ``rvc_worker.py`` and ``rvc_stream.py``.  Module top
imports only numpy/scipy/stdlib; torch, librosa, pyworld, parselmouth and
torchcrepe are imported lazily so the pure helpers (``postprocess_f0``,
``median_f0``, ``auto_pitch_shift``) work in any venv.

Contract: every extractor returns ``float32[p_len]`` in Hz on a 10 ms hop
(160 samples at 16 kHz); 0 means unvoiced.

RMVPE is the standard RVC "E2E" network (mel front-end, deep U-Net, BiGRU,
360-bin cents salience).  Weights: see ``legacy/rvc_f0_weights.md``.
"""

from __future__ import annotations

import functools
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

SR = 16000
HOP = 160

RMVPE_DEFAULT_PATH = Path(
    os.environ.get("RVC_RMVPE_PATH")
    or "/home/cbash23/.cache/listen-speed/rvc-quality/weights/rmvpe.pt"
)

METHODS = ("rmvpe", "pm", "dio", "harvest", "crepe")


@dataclass(frozen=True)
class F0PostConfig:
    """Post-processing of an f0 track (all off by default)."""

    median_width: int = 0  # odd >= 3 enables a median filter on voiced runs
    octave_repair: bool = False
    clamp_min_hz: float | None = None
    clamp_max_hz: float | None = None
    fill_gap_frames: int = 0  # interpolate unvoiced gaps of <= N frames


# ---------------------------------------------------------------- helpers


def _fit(f0: np.ndarray, p_len: int) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    if len(f0) >= p_len:
        return np.ascontiguousarray(f0[:p_len])
    out = np.zeros(p_len, dtype=np.float32)
    out[: len(f0)] = f0
    return out


def median_f0(f0: np.ndarray) -> float:
    """Median of the voiced (>0) frames, 0.0 if there are none."""
    f0 = np.asarray(f0)
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 0.0


def auto_pitch_shift(
    source_median_hz: float,
    target_median_hz: float,
    *,
    max_abs: int = 12,
    strength: float = 1.0,
) -> int:
    """Whole semitones moving ``source`` toward ``target`` (scaled by strength)."""
    if source_median_hz <= 0 or target_median_hz <= 0:
        return 0
    st = round(strength * 12.0 * float(np.log2(target_median_hz / source_median_hz)))
    return int(max(-max_abs, min(max_abs, st)))


def _voiced_runs(voiced: np.ndarray):
    """Yield (start, end) of consecutive True runs (end exclusive)."""
    if not len(voiced):
        return
    d = np.diff(np.concatenate(([0], voiced.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    yield from zip(starts.tolist(), ends.tolist(), strict=True)


def _median_voiced(f0: np.ndarray, width: int) -> np.ndarray:
    if width < 3:
        return f0
    width |= 1
    out = f0.copy()
    for s, e in _voiced_runs(f0 > 0):
        if e - s >= 2:
            out[s:e] = ndimage.median_filter(f0[s:e], size=width, mode="nearest")
    return out


# octave-like jumps (semitones) and the ratio that folds each one back
_FOLDS = ((12.0, 2.0), (19.0196, 3.0), (24.0, 4.0))


def _octave_repair(f0: np.ndarray, radius: int = 10) -> np.ndarray:
    idx = np.flatnonzero(f0 > 0)
    if len(idx) < 3:
        return f0
    vals = f0[idx].astype(np.float64)
    out = f0.copy()
    n = len(idx)
    for j in range(n):
        lo, hi = max(0, j - radius), min(n, j + radius + 1)
        nb = np.concatenate((vals[lo:j], vals[j + 1 : hi]))
        if not len(nb):
            continue
        local = float(np.median(nb))
        d = 12.0 * np.log2(vals[j] / local)
        for st, ratio in _FOLDS:
            if abs(abs(d) - st) < 1.5:
                out[idx[j]] = vals[j] / ratio if d > 0 else vals[j] * ratio
                break
    return out


def _clamp(f0: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    if lo is None and hi is None:
        return f0
    out = f0.copy()
    for i in np.flatnonzero(f0 > 0):
        v = float(f0[i])
        for _ in range(6):
            if hi is not None and v > hi:
                v /= 2.0
            elif lo is not None and v < lo:
                v *= 2.0
            else:
                break
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        out[i] = v if ok else 0.0
    return out


def _fill_gaps(f0: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return f0
    out = f0.copy()
    runs = list(_voiced_runs(f0 > 0))
    for (_, e1), (s2, _) in zip(runs[:-1], runs[1:], strict=True):
        gap = s2 - e1
        if 0 < gap <= n:
            a, b = np.log(f0[e1 - 1]), np.log(f0[s2])
            t = np.arange(1, gap + 1) / (gap + 1)
            out[e1:s2] = np.exp(a + (b - a) * t)
    return out


def postprocess_f0(
    f0: np.ndarray,
    post: F0PostConfig | None,
    f0_min: float | None = None,
    f0_max: float | None = None,
) -> np.ndarray:
    """Octave repair -> clamp -> voiced-run median -> gap fill; returns float32.

    ``f0_min`` / ``f0_max`` (if given) are hard limits: voiced frames outside
    them are set to 0 after the configured steps.
    """
    out = np.asarray(f0, dtype=np.float32).copy()
    if post is not None:
        if post.octave_repair:
            out = _octave_repair(out)
        out = _clamp(out, post.clamp_min_hz, post.clamp_max_hz)
        if post.median_width >= 3:
            out = _median_voiced(out, int(post.median_width))
        if post.fill_gap_frames > 0:
            out = _fill_gaps(out, int(post.fill_gap_frames))
    if f0_min is not None:
        out[(out > 0) & (out < f0_min)] = 0.0
    if f0_max is not None:
        out[out > f0_max] = 0.0
    return out.astype(np.float32)


# ------------------------------------------------------------ extractors


def _f0_pm(x: np.ndarray, p_len: int, f0_min: float, f0_max: float) -> np.ndarray:
    import parselmouth

    f0 = (
        parselmouth.Sound(x.astype(np.float64), SR)
        .to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )
    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    return f0


def _f0_pyworld(x: np.ndarray, kind: str, f0_min: float, f0_max: float) -> np.ndarray:
    import pyworld
    from scipy import signal

    xd = x.astype(np.double)
    fn = pyworld.harvest if kind == "harvest" else pyworld.dio
    f0, t = fn(xd, fs=SR, f0_ceil=f0_max, f0_floor=f0_min, frame_period=10)
    f0 = pyworld.stonemask(xd, f0, t, SR)
    return signal.medfilt(f0, 3)


def _f0_crepe(x: np.ndarray, p_len: int, f0_min: float, f0_max: float, device: str | None):
    import torch
    import torchcrepe

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio = torch.from_numpy(x.astype(np.float32)).to(dev).unsqueeze(0)
    pitch = torchcrepe.predict(
        audio, SR, HOP, f0_min, f0_max, "full", batch_size=512, device=dev, pad=True
    )
    f0 = pitch.squeeze(0).cpu().float().numpy()
    f0[f0 < 0.001] = 0.0
    return f0


# ---------------------------------------------------------------- RMVPE


@functools.lru_cache(maxsize=1)
def _rmvpe_classes():
    """Build the RMVPE modules lazily (torch is only imported here)."""
    import torch
    from torch import nn

    class ConvBlockRes(nn.Module):
        def __init__(self, cin, cout, momentum=0.01):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cout, (3, 3), (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(cout, momentum=momentum),
                nn.ReLU(),
                nn.Conv2d(cout, cout, (3, 3), (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(cout, momentum=momentum),
                nn.ReLU(),
            )
            self.is_shortcut = cin != cout
            if self.is_shortcut:
                self.shortcut = nn.Conv2d(cin, cout, (1, 1))

        def forward(self, x):
            if self.is_shortcut:
                return self.conv(x) + self.shortcut(x)
            return self.conv(x) + x

    class ResEncoderBlock(nn.Module):
        def __init__(self, cin, cout, kernel_size, n_blocks=1, momentum=0.01):
            super().__init__()
            self.conv = nn.ModuleList(
                [ConvBlockRes(cin, cout, momentum)]
                + [ConvBlockRes(cout, cout, momentum) for _ in range(n_blocks - 1)]
            )
            self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

        def forward(self, x):
            for layer in self.conv:
                x = layer(x)
            if self.pool is not None:
                return x, self.pool(x)
            return x

    class Encoder(nn.Module):
        def __init__(self, cin, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
            super().__init__()
            self.bn = nn.BatchNorm2d(cin, momentum=momentum)
            self.layers = nn.ModuleList()
            for _ in range(n_encoders):
                self.layers.append(
                    ResEncoderBlock(cin, out_channels, kernel_size, n_blocks, momentum)
                )
                cin = out_channels
                out_channels *= 2
            self.out_channel = out_channels

        def forward(self, x):
            x = self.bn(x)
            concat = []
            for layer in self.layers:
                t, x = layer(x)
                concat.append(t)
            return x, concat

    class Intermediate(nn.Module):
        def __init__(self, cin, cout, n_inters, n_blocks, momentum=0.01):
            super().__init__()
            self.layers = nn.ModuleList(
                [ResEncoderBlock(cin, cout, None, n_blocks, momentum)]
                + [
                    ResEncoderBlock(cout, cout, None, n_blocks, momentum)
                    for _ in range(n_inters - 1)
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class ResDecoderBlock(nn.Module):
        def __init__(self, cin, cout, stride, n_blocks=1, momentum=0.01):
            super().__init__()
            out_padding = (0, 1) if stride == (1, 2) else (1, 1)
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    cin, cout, (3, 3), stride, (1, 1), output_padding=out_padding, bias=False
                ),
                nn.BatchNorm2d(cout, momentum=momentum),
                nn.ReLU(),
            )
            self.conv2 = nn.ModuleList(
                [ConvBlockRes(cout * 2, cout, momentum)]
                + [ConvBlockRes(cout, cout, momentum) for _ in range(n_blocks - 1)]
            )

        def forward(self, x, concat):
            x = self.conv1(x)
            x = torch.cat((x, concat), dim=1)
            for layer in self.conv2:
                x = layer(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, cin, n_decoders, stride, n_blocks, momentum=0.01):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(n_decoders):
                cout = cin // 2
                self.layers.append(ResDecoderBlock(cin, cout, stride, n_blocks, momentum))
                cin = cout

        def forward(self, x, concat):
            for i, layer in enumerate(self.layers):
                x = layer(x, concat[-1 - i])
            return x

    class DeepUnet(nn.Module):
        def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, cin=1, en_out=16):
            super().__init__()
            self.encoder = Encoder(cin, en_de_layers, kernel_size, n_blocks, en_out)
            self.intermediate = Intermediate(
                self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
            )
            self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

        def forward(self, x):
            x, concat = self.encoder(x)
            x = self.intermediate(x)
            return self.decoder(x, concat)

    class BiGRU(nn.Module):
        def __init__(self, input_features, hidden_features, num_layers):
            super().__init__()
            self.gru = nn.GRU(
                input_features, hidden_features, num_layers=num_layers,
                batch_first=True, bidirectional=True,
            )  # fmt: skip

        def forward(self, x):
            return self.gru(x)[0]

    class E2E(nn.Module):
        def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5,
                     inter_layers=4, cin=1, en_out=16):  # fmt: skip
            super().__init__()
            self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, cin, en_out)
            self.cnn = nn.Conv2d(en_out, 3, (3, 3), padding=(1, 1))
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

        def forward(self, mel):
            # mel: (B, 128, T) with T a multiple of 32
            mel = mel.transpose(-1, -2).unsqueeze(1)
            x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
            return self.fc(x)

    return E2E


class _Rmvpe:
    """Loaded RMVPE model + mel front-end + decoder for one (path, device)."""

    THRESHOLD = 0.03

    def __init__(self, path: Path, device: str):
        import librosa
        import torch

        self.torch = torch
        self.device = device
        net = _rmvpe_classes()()
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
        net.eval().to(device)
        self.net = net
        self.half = False
        if device.startswith("cuda"):
            try:
                net.half()
                self.half = True
            except Exception:  # pragma: no cover - fp32 fallback
                net.float()
        basis = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        self.mel_basis = torch.from_numpy(basis).float().to(device)
        self.window = torch.hann_window(1024).to(device)
        self.cents = (20.0 * torch.arange(360, dtype=torch.float32) + 1997.3794084376191).to(device)
        self.lock = threading.Lock()

    def _mel(self, audio):
        torch = self.torch
        spec = torch.stft(
            audio, 1024, HOP, 1024, window=self.window, center=True, return_complex=True
        )
        mag = spec.abs()
        return torch.log(torch.clamp(self.mel_basis @ mag, min=1e-5))

    def _run(self, mel):
        torch = self.torch
        n = mel.shape[-1]
        pad = 32 * ((n - 1) // 32 + 1) - n
        if pad:
            mel = torch.nn.functional.pad(mel, (0, pad), mode="constant")
        hidden = self.net(mel.half() if self.half else mel)
        return hidden[:, :n].float()

    def infer(self, x16: np.ndarray) -> np.ndarray:
        torch = self.torch
        audio = torch.from_numpy(np.ascontiguousarray(x16, dtype=np.float32)).to(self.device)
        with self.lock, torch.no_grad():
            try:
                sal = self._run(self._mel(audio.unsqueeze(0)))[0]
            except RuntimeError:
                if not self.half:
                    raise
                self.net.float()
                self.half = False
                sal = self._run(self._mel(audio.unsqueeze(0)))[0]
            return self._decode(sal).cpu().numpy()

    def _decode(self, sal):
        torch = self.torch
        center = sal.argmax(dim=1)
        padded = torch.nn.functional.pad(sal, (4, 4))
        cents = torch.nn.functional.pad(self.cents, (4, 4))
        idx = center.unsqueeze(1) + torch.arange(9, device=sal.device).unsqueeze(0)
        w = padded.gather(1, idx)
        c = cents[idx]
        avg = (w * c).sum(1) / w.sum(1).clamp_min(1e-9)
        avg = torch.where(sal.max(dim=1).values > self.THRESHOLD, avg, torch.zeros_like(avg))
        f0 = 10.0 * torch.pow(2.0, avg / 1200.0)
        return torch.where(avg > 0, f0, torch.zeros_like(f0))


_RMVPE_CACHE: dict[tuple[str, str], _Rmvpe] = {}
_RMVPE_LOCK = threading.Lock()


def rmvpe_available() -> bool:
    return RMVPE_DEFAULT_PATH.is_file()


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _get_rmvpe(device: str | None) -> _Rmvpe:
    path = RMVPE_DEFAULT_PATH
    if not path.is_file():
        raise RuntimeError(f"rmvpe weights not found: {path}")
    dev = device or _default_device()
    key = (str(path), dev)
    with _RMVPE_LOCK:
        m = _RMVPE_CACHE.get(key)
        if m is None:
            m = _Rmvpe(path, dev)
            _RMVPE_CACHE[key] = m
        return m


def _f0_rmvpe(x: np.ndarray, device: str | None) -> np.ndarray:
    return _get_rmvpe(device).infer(x)


# ------------------------------------------------------------------ API


def extract_f0(
    x16: np.ndarray,
    p_len: int,
    method: str,
    *,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    device: str | None = None,
    post: F0PostConfig | None = None,
) -> np.ndarray:
    """f0 in Hz as float32[p_len] on a 10 ms hop; 0 = unvoiced."""
    if method not in METHODS:
        raise ValueError(f"unknown f0 method: {method!r} (expected one of {METHODS})")
    x = np.asarray(x16, dtype=np.float32).reshape(-1)
    if method == "rmvpe":
        f0 = _f0_rmvpe(x, device)
        f0 = np.where((f0 >= f0_min) & (f0 <= f0_max), f0, 0.0)
    elif method == "pm":
        f0 = _f0_pm(x, p_len, f0_min, f0_max)
    elif method in ("harvest", "dio"):
        f0 = _f0_pyworld(x, method, f0_min, f0_max)
    else:
        f0 = _f0_crepe(x, p_len, f0_min, f0_max, device)
    f0 = _fit(f0, p_len)
    if post is not None:
        f0 = postprocess_f0(f0, post, None, None)
    return f0.astype(np.float32)
