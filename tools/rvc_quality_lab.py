#!/usr/bin/env python
"""RVC output-quality lab: render variants through the REAL engine and score them with proxies.

Plan: docs/RVC_QUALITY_PLAN.md (sections 3-5). Run with the legacy venv, GPU work under the lock::

    flock /home/cbash23/.cache/listen-speed/gpu.lock \\
        .venv/bin/python tools/rvc_quality_lab.py --model Obama --stage 0
    ... --stage A
    ... --stage B --base-variant a4_rmvpe_post_idx05
    ... --stage C --base-variant a4_rmvpe_post_idx05 --base-shift -3
    ... --model Obama --variants a4_rmvpe_post_idx05,c1_idx03 --clips 0,1

Renders land in ``<out>/samples/<model>/<variant>/s{0..5}.wav`` (whole path: model rate,
streams: 48 kHz). ``<out>/results.json`` merges by (model, variant); ``<out>/index.md`` is
rewritten from it with BEST flags. A variant that needs a feature the installed worker/f0 module
does not have (rmvpe weights, index/protect/rms params, stream f0_algo ...) is recorded as
``{"status": "skipped", "reason": ...}``.

The metric functions at the top are pure numpy/scipy (librosa / pyworld / torch / faiss are
imported lazily) so they can be unit-tested without a GPU. All metrics are proxies; the owner's
ears decide.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import inspect
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path("/home/cbash23/.cache/listen-speed/rvc-quality")
RYAN_DIR = Path("/home/cbash23/.cache/listen-speed/qwen-fast/samples")
N_CLIPS = 6
F0_HOP = 160  # samples @16 kHz = 10 ms
FRAME_S = 0.01

# Target registers (median f0, Hz) used for register_err_st and auto shift.
TARGET_HZ = {"Obama": 108.0, "Goku": 145.0}
# Voice variant (yaml stem) per model.
MODEL_VARIANT = {"Obama": "obama", "Goku": "goku"}
# Reference variant for mel_dist_ref_db.
REF_VARIANT = "ref_harvest_idx05"
DEFAULT_BASE_VARIANT = "a4_rmvpe_post_idx05"

# --------------------------------------------------------------------------------------------
# Pure metric functions
# --------------------------------------------------------------------------------------------


def semitone_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """12*log2(a/b) elementwise (both must be > 0)."""
    return 12.0 * np.log2(np.asarray(a, np.float64) / np.asarray(b, np.float64))


def _align_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)


def f0_octave_err_rate(f0_out: np.ndarray, f0_ref: np.ndarray) -> float:
    """Fraction of commonly voiced frames that deviate >= 8 st from the reference AND sit within
    1.5 st of +-12 / +-24 st (i.e. an octave error rather than a genuine different pitch)."""
    o, r = _align_pair(f0_out, f0_ref)
    both = (o > 0) & (r > 0)
    if not both.any():
        return float("nan")
    dev = semitone_diff(o[both], r[both])
    mag = np.abs(dev)
    near_oct = np.minimum(np.abs(mag - 12.0), np.abs(mag - 24.0)) <= 1.5
    return float(np.mean((mag >= 8.0) & near_oct))


def f0_jump_rate(f0: np.ndarray, jump_st: float = 3.0) -> float:
    """Frame-to-frame jumps > ``jump_st`` between adjacent voiced frames, per second of voiced
    speech (wobble / glitch detector)."""
    f = np.asarray(f0, np.float64)
    voiced = f > 0
    n_voiced = int(voiced.sum())
    if n_voiced < 2:
        return float("nan")
    pair = voiced[:-1] & voiced[1:]
    if not pair.any():
        return 0.0
    d = np.abs(semitone_diff(f[1:][pair], f[:-1][pair]))
    return float((d > jump_st).sum() / (n_voiced * FRAME_S))


def voiced_runs(f0: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """[start, stop) index pairs of consecutive voiced frames with length >= min_len."""
    v = np.concatenate([[False], np.asarray(f0) > 0, [False]])
    edges = np.flatnonzero(v[1:] != v[:-1])
    runs = [(int(s), int(e)) for s, e in zip(edges[0::2], edges[1::2], strict=False)]
    return [(s, e) for s, e in runs if e - s >= min_len]


def f0_smooth_cents(f0: np.ndarray) -> float:
    """Std (cents) of the second difference of log2 f0, pooled over voiced runs of >= 3 frames."""
    f = np.asarray(f0, np.float64)
    chunks = []
    for s, e in voiced_runs(f, 3):
        chunks.append(np.diff(1200.0 * np.log2(f[s:e]), 2))
    if not chunks:
        return float("nan")
    return float(np.std(np.concatenate(chunks)))


def f0_track_corr(f0_out: np.ndarray, f0_ref: np.ndarray) -> float:
    """Pearson correlation of log f0 on commonly voiced frames."""
    o, r = _align_pair(f0_out, f0_ref)
    both = (o > 0) & (r > 0)
    if both.sum() < 6:
        return float("nan")
    a, b = np.log(o[both]), np.log(r[both])
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def f0_median_hz(f0: np.ndarray) -> float:
    f = np.asarray(f0, np.float64)
    f = f[f > 0]
    return float(np.median(f)) if f.size else float("nan")


def register_err_st(median_hz: float, target_hz: float) -> float:
    """Absolute semitone distance of a median f0 to the target register."""
    if not (median_hz > 0 and target_hz > 0) or not math.isfinite(median_hz):
        return float("nan")
    return float(abs(12.0 * math.log2(median_hz / target_hz)))


def voiced_frac(f0: np.ndarray) -> float:
    f = np.asarray(f0)
    return float((f > 0).mean()) if f.size else float("nan")


def voiced_frac_ratio(f0_out: np.ndarray, f0_src: np.ndarray) -> float:
    s = voiced_frac(f0_src)
    if not s or not math.isfinite(s):
        return float("nan")
    return voiced_frac(f0_out) / s


def hf_hiss_metrics(
    y: np.ndarray,
    sr: int,
    voiced_mask: np.ndarray,
    *,
    cutoff_hz: float = 12000.0,
    n_fft: int = 1024,
) -> dict[str, float]:
    """Energy above ``cutoff_hz`` / total energy summed over unvoiced frames, plus the mean
    spectral flatness of those frames. ``voiced_mask`` is per 10 ms frame (harvest voicing)."""
    import librosa

    y = np.asarray(y, np.float32)
    if sr <= 2 * cutoff_hz:
        return {"hf_hiss_ratio": float("nan"), "unvoiced_flatness": float("nan")}
    hop = int(round(sr * FRAME_S))
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    t = min(power.shape[1], len(voiced_mask))
    unv = ~np.asarray(voiced_mask[:t], bool)
    power = power[:, :t][:, unv]
    if power.shape[1] == 0:
        return {"hf_hiss_ratio": float("nan"), "unvoiced_flatness": float("nan")}
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = float(power.sum())
    ratio = float(power[freqs >= cutoff_hz].sum() / total) if total > 0 else float("nan")
    flat = librosa.feature.spectral_flatness(S=np.sqrt(power), n_fft=n_fft)
    return {"hf_hiss_ratio": ratio, "unvoiced_flatness": float(np.mean(flat))}


def sanity(y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, np.float64)
    finite = np.isfinite(y)
    yy = np.where(finite, y, 0.0)
    return {
        "peak": float(np.abs(yy).max()) if yy.size else 0.0,
        "clip_frac": float(np.mean(np.abs(yy) >= 0.999)) if yy.size else 0.0,
        "nan": float((~finite).sum()),
    }


def mel_db(x16: np.ndarray) -> np.ndarray:
    """80-mel log power (dB) at 16 kHz, n_fft 1024, hop 160."""
    import librosa

    spec = librosa.feature.melspectrogram(
        y=np.asarray(x16, np.float32), sr=16000, n_fft=1024, hop_length=F0_HOP, n_mels=80
    )
    return 10.0 * np.log10(spec + 1e-10)


def align_lag_frames(ma: np.ndarray, mb: np.ndarray, max_lag: int = 30) -> int:
    """Lag (frames) to apply so that ``mb`` lines up with ``ma``: positive lag means ``mb`` must
    be advanced (b[lag:] matches a[:-lag])... Returned so that ``a[i] ~ b[i + lag]``. Uses the
    cross-correlation of the per-frame mean mel level."""
    ea, eb = ma.mean(axis=0), mb.mean(axis=0)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = ea[: n - lag], eb[lag:]
        else:
            x, y = ea[-lag:], eb[: n + lag]
        if len(x) < 10:
            continue
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best:
            best, best_lag = c, lag
    return best_lag


def mel_dist_db(a16: np.ndarray, b16: np.ndarray) -> float:
    """Mean |log-mel| difference (dB, 80 mels) after cross-correlation alignment."""
    ma, mb = mel_db(a16), mel_db(b16)
    lag = align_lag_frames(ma, mb)
    if lag >= 0:
        ma_, mb_ = ma, mb[:, lag:]
    else:
        ma_, mb_ = ma[:, -lag:], mb
    k = min(ma_.shape[1], mb_.shape[1])
    if k < 5:
        return float("nan")
    return float(np.mean(np.abs(ma_[:, :k] - mb_[:, :k])))


def join_ratio(y: np.ndarray, joins: Sequence[int]) -> dict[str, float]:
    """As legacy/rvc_stream_eval.py::_jump_ratio: max |2nd diff| near each join over the local
    95th percentile."""
    d2 = np.abs(np.diff(np.asarray(y, np.float64), 2))
    out = []
    for j in joins:
        seg = d2[max(0, j - 48) : j + 48]
        if len(seg):
            ref = np.percentile(d2[max(0, j - 2400) : j + 2400], 95) + 1e-9
            out.append(seg.max() / ref)
    if not out:
        return {}
    return {"join_max": float(max(out)), "join_med": float(np.median(out))}


def stream_join_positions(n_out: int, first_block_ms: int, block_ms: int) -> list[int]:
    """48 kHz output sample indices of SOLA joins (as rvc_stream_eval: 2 * block in 24 kHz)."""
    first_b = first_block_ms * 24
    block_b = block_ms * 24
    joins = [2 * first_b]
    while joins[-1] + 2 * block_b < n_out:
        joins.append(joins[-1] + 2 * block_b)
    return joins


def knn_mean_l2(dist_sq: np.ndarray) -> float:
    """faiss L2 indexes return SQUARED distances; mean of the root over all rows/neighbours."""
    d = np.sqrt(np.maximum(np.asarray(dist_sq, np.float64), 0.0))
    return float(d.mean()) if d.size else float("nan")


def source_f0_ref(src_f0: np.ndarray, shift_st: float) -> np.ndarray:
    """Reference f0 = harvest(source) * 2^(shift/12)."""
    return np.asarray(src_f0, np.float64) * 2.0 ** (shift_st / 12.0)


def f0_metrics(f0_out: np.ndarray, f0_src: np.ndarray, shift_st: float, target_hz: float) -> dict:
    ref = source_f0_ref(f0_src, shift_st)
    med = f0_median_hz(f0_out)
    return {
        "f0_octave_err_rate": f0_octave_err_rate(f0_out, ref),
        "f0_jump_rate": f0_jump_rate(f0_out),
        "f0_smooth_cents": f0_smooth_cents(f0_out),
        "f0_track_corr": f0_track_corr(f0_out, ref),
        "f0_median_hz": med,
        "register_err_st": register_err_st(med, target_hz),
        "voiced_frac_ratio": voiced_frac_ratio(f0_out, f0_src),
    }


# --------------------------------------------------------------------------------------------
# Ranking / flags
# --------------------------------------------------------------------------------------------

RANK_METRICS = ("f0_octave_err_rate", "f0_jump_rate", "retrieval_dist")
# metric -> True when lower is better
BEST_METRICS = {
    "f0_octave_err_rate": True,
    "f0_jump_rate": True,
    "f0_smooth_cents": True,
    "f0_track_corr": False,
    "register_err_st": True,
    "retrieval_dist": True,
    "mel_dist_ref_db": True,
    "hf_hiss_ratio": True,
    "rtf": True,
}
VOICED_GUARD = 0.85


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _mean_of(row: dict, key: str) -> float:
    v = (row.get("mean") or {}).get(key)
    return float(v) if _finite(v) else float("nan")


def eligible_rows(rows: Iterable[dict]) -> list[dict]:
    """Rows that may win BEST_OVERALL: status ok, voiced ratio >= 0.85, hiss not in the worst
    quartile (strictly above the 75th percentile of the candidates' hiss), all ranked metrics
    finite."""
    cand = [
        r
        for r in rows
        if r.get("status") == "ok"
        and _finite(_mean_of(r, "voiced_frac_ratio"))
        and _mean_of(r, "voiced_frac_ratio") >= VOICED_GUARD
        and all(_finite(_mean_of(r, m)) for m in RANK_METRICS)
    ]
    hiss = [_mean_of(r, "hf_hiss_ratio") for r in cand]
    finite_hiss = [h for h in hiss if math.isfinite(h)]
    if len(finite_hiss) >= 4:
        thr = float(np.percentile(finite_hiss, 75))
        cand = [r for r, h in zip(cand, hiss, strict=True) if not (math.isfinite(h) and h > thr)]
    return cand


def rank_average(rows: Sequence[dict]) -> dict[str, float]:
    """variant -> mean rank (1 = best) over RANK_METRICS (average ranks for ties)."""
    from scipy.stats import rankdata

    if not rows:
        return {}
    ranks = np.zeros(len(rows))
    for m in RANK_METRICS:
        vals = np.array([_mean_of(r, m) for r in rows])
        ranks += rankdata(vals, method="average")
    ranks /= len(RANK_METRICS)
    return {r["variant"]: float(k) for r, k in zip(rows, ranks, strict=True)}


def compute_flags(rows: Sequence[dict]) -> dict[str, list[str]]:
    """variant -> flags. BEST_OVERALL over whole-path variants, BEST_STREAM over stream variants,
    BEST_<metric> per metric over all ok rows of the same render mode (excluding the mel
    reference itself for mel_dist_ref_db)."""
    flags: dict[str, list[str]] = {r["variant"]: [] for r in rows}
    for mode, label in (("whole", "BEST_OVERALL"), ("stream", "BEST_STREAM")):
        group = [r for r in rows if r.get("mode", "whole") == mode]
        elig = eligible_rows(group)
        avg = rank_average(elig)
        if avg:
            best = min(avg.values())
            for v, k in avg.items():
                if k == best:
                    flags[v].append(label)
        for metric, lower in BEST_METRICS.items():
            vals = []
            for r in group:
                if r.get("status") != "ok":
                    continue
                if metric == "mel_dist_ref_db" and r["variant"] == REF_VARIANT:
                    continue
                v = _mean_of(r, metric)
                if math.isfinite(v):
                    vals.append((v, r["variant"]))
            if not vals:
                continue
            target = min(vals)[0] if lower else max(vals)[0]
            for v, name in vals:
                if v == target:
                    flags[name].append(f"BEST_{metric}")
    return flags


# --------------------------------------------------------------------------------------------
# Variant table (plan section 5)
# --------------------------------------------------------------------------------------------

# Every key that may appear in Variant.fields (worker `convert` request fields + pitch_shift).
WORKER_FIELDS = frozenset(
    {
        "f0_method",
        "pitch_shift",
        "use_index",
        "index_rate",
        "index_nprobe",
        "protect",
        "rms_mix_rate",
        "f0_post",
        "auto_pitch",
    }
)
# Keys of Variant.stream (StreamParams fields).
STREAM_FIELDS = frozenset(
    {
        "context_ms",
        "block_ms",
        "first_block_ms",
        "f0_algo",
        "use_index",
        "index_rate",
        "index_nprobe",
        "protect",
        "f0_post",
        "pitch_shift",
    }
)
F0_POST_KEYS = frozenset(
    {"median_width", "octave_repair", "clamp_min_hz", "clamp_max_hz", "fill_gap_frames"}
)
SPEAKERS = ("ryan", "aiden", "eric", "dylan")

POST_DEFAULT = {
    "median_width": 5,
    "octave_repair": True,
    "clamp_min_hz": 50.0,
    "clamp_max_hz": 500.0,
    "fill_gap_frames": 2,
}


@dataclasses.dataclass
class Variant:
    id: str
    stage: str
    desc: str
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)
    source: str = "ryan"
    mode: str = "whole"  # whole | stream
    stream: dict[str, Any] = dataclasses.field(default_factory=dict)
    env: dict[str, str] = dataclasses.field(default_factory=dict)


def _v(id_: str, stage: str, desc: str, **kw: Any) -> Variant:
    return Variant(id=id_, stage=stage, desc=desc, **kw)


def _auto(model: str, strength: float = 1.0) -> dict[str, Any]:
    return {"target_median_hz": TARGET_HZ[model], "max_abs": 12, "strength": strength}


def stage_zero_a(model: str) -> list[Variant]:
    """Stage 0 and A. Stage A ids are the base for B-E."""
    out = [
        _v("v00_baseline", "0", "pm, shift 0, no index (today)", fields={"f0_method": "pm"}),
        _v(
            REF_VARIANT,
            "0",
            "harvest, index 0.5 nprobe 16, auto shift (mel reference)",
            fields={
                "f0_method": "harvest",
                "use_index": True,
                "index_rate": 0.5,
                "index_nprobe": 16,
                "auto_pitch": _auto(model),
            },
        ),
        _v(
            "a1_pm_idx05",
            "A",
            "pm + index 0.5",
            fields={"f0_method": "pm", "use_index": True, "index_rate": 0.5, "index_nprobe": 16},
        ),
        _v("a2_rmvpe", "A", "rmvpe, no index", fields={"f0_method": "rmvpe"}),
        _v(
            "a3_rmvpe_post",
            "A",
            "rmvpe + f0 post (median5, octave repair, clamp 50-500, gap fill 2)",
            fields={"f0_method": "rmvpe", "f0_post": dict(POST_DEFAULT)},
        ),
        _v(
            "a4_rmvpe_post_idx05",
            "A",
            "a3 + index 0.5",
            fields={
                "f0_method": "rmvpe",
                "f0_post": dict(POST_DEFAULT),
                "use_index": True,
                "index_rate": 0.5,
                "index_nprobe": 16,
            },
        ),
        _v(
            "a5_harvest_idx05",
            "A",
            "harvest + index 0.5",
            fields={
                "f0_method": "harvest",
                "use_index": True,
                "index_rate": 0.5,
                "index_nprobe": 16,
            },
        ),
    ]
    if model == "Goku":
        out = [v for v in out if v.id not in ("a2_rmvpe", "a3_rmvpe_post")]
    return out


def _derive(base: Variant, id_: str, stage: str, desc: str, **over: Any) -> Variant:
    fields = {**copy.deepcopy(base.fields), **over.pop("fields", {})}
    return Variant(
        id=id_,
        stage=stage,
        desc=desc,
        fields=fields,
        source=over.pop("source", base.source),
        mode=over.pop("mode", base.mode),
        stream={**base.stream, **over.pop("stream", {})},
        env={**base.env, **over.pop("env", {})},
    )


def build_variants(
    model: str,
    *,
    base_id: str = DEFAULT_BASE_VARIANT,
    base_shift: int = 0,
    base_auto: float | None = None,
) -> dict[str, Variant]:
    """The full variant table for ``model``, in run order. Stage B-E variants start from the
    stage-A variant ``base_id`` with the chosen fixed shift ``base_shift`` (or auto-shift strength
    ``base_auto``) applied; B variants always start from the unshifted base."""
    if model not in TARGET_HZ:
        raise ValueError(f"unknown model {model!r}")
    table: dict[str, Variant] = {v.id: v for v in stage_zero_a(model)}
    if base_id not in table:
        base_id = "a4_rmvpe_post_idx05" if "a4_rmvpe_post_idx05" in table else next(iter(table))
    base = table[base_id]
    goku = model == "Goku"

    # Stage B: register.
    shifts = (
        [("b1_shift_fixed_0", 0), ("b2_shift_fixed_p2", 2)]
        if goku
        else [
            ("b1_shift_fixed_m3", -3),
            ("b2_shift_fixed_m4", -4),
            ("b3_shift_fixed_m5", -5),
        ]
    )
    for id_, sh in shifts:
        table[id_] = _derive(
            base, id_, "B", f"{base_id} with fixed shift {sh:+d}", fields={"pitch_shift": sh}
        )
    table["b4_autoshift"] = _derive(
        base,
        "b4_autoshift",
        "B",
        f"{base_id} with auto shift to {TARGET_HZ[model]:g} Hz, strength 1.0",
        fields={"auto_pitch": _auto(model, 1.0)},
    )
    table["b5_autoshift_half"] = _derive(
        base,
        "b5_autoshift_half",
        "B",
        f"{base_id} with auto shift to {TARGET_HZ[model]:g} Hz, strength 0.5",
        fields={"auto_pitch": _auto(model, 0.5)},
    )

    # B/C/D/E "best" base: the carried shift.
    best = _derive(base, "_best", "C", "best")
    if base_auto is not None:
        best.fields["auto_pitch"] = _auto(model, base_auto)
        best.fields.pop("pitch_shift", None)
    elif base_shift:
        best.fields["pitch_shift"] = int(base_shift)
    carry = (
        f"{base_id}, shift {base_shift:+d}" if base_auto is None else f"{base_id}, auto {base_auto}"
    )

    def c(id_: str, desc: str, **fields: Any) -> None:
        env = fields.pop("_env", None)
        table[id_] = _derive(
            best, id_, "C", f"{carry}: {desc}", fields=fields, **({"env": env} if env else {})
        )

    c("c1_idx03", "index_rate 0.3", use_index=True, index_rate=0.3)
    c("c2_idx075", "index_rate 0.75", use_index=True, index_rate=0.75)
    if not goku:
        c("c3_nprobe1", "index nprobe 1", index_nprobe=1)
        c("c4_nprobe32", "index nprobe 32", index_nprobe=32)
    c("c5_protect033", "protect 0.33", protect=0.33)
    c("c6_protect05", "protect 0.5 (off)", protect=0.5)
    if not goku:
        c("c7_rms025", "rms_mix_rate 0.25", rms_mix_rate=0.25)
        c("c8_rms10", "rms_mix_rate 1.0 (off)", rms_mix_rate=1.0)
        c("c9_fp32", "fp32 synthesis", _env={"RVC_WORKER_FP32": "1"})

    # Stage D: other source speakers (shift re-derived by auto).
    if not goku:
        for id_, spk in (("d1_aiden", "aiden"), ("d2_eric", "eric"), ("d3_dylan", "dylan")):
            v = _derive(best, id_, "D", f"{carry}: source speaker {spk}, auto shift", source=spk)
            v.fields.pop("pitch_shift", None)
            v.fields["auto_pitch"] = _auto(model, 1.0)
            table[id_] = v

    # Stage E: streams (index + protect, no rms_mix, no auto: fixed shift only).
    stream_base = {
        "pitch_shift": int(base_shift) if base_auto is None else int(base_shift),
        "use_index": bool(best.fields.get("use_index", False)),
        "index_rate": best.fields.get("index_rate"),
        "index_nprobe": best.fields.get("index_nprobe"),
        "protect": best.fields.get("protect", 0.33),
        "f0_post": best.fields.get("f0_post"),
    }
    e_rows = [
        ("e1_stream_pm_ctx1000", "pm", 1000),
        ("e2_stream_pm_ctx2000", "pm", 2000),
        ("e3_stream_rmvpe_ctx1000", "rmvpe", 1000),
        ("e4_stream_rmvpe_ctx2000", "rmvpe", 2000),
    ]
    if goku:
        e_rows = [e_rows[3]]
    for id_, algo, ctx in e_rows:
        stream = {k: v for k, v in stream_base.items() if v is not None}
        stream.update({"f0_algo": algo, "context_ms": ctx})
        if algo == "pm":
            stream.pop("f0_post", None)
        table[id_] = Variant(
            id=id_,
            stage="E",
            desc=f"stream {algo} ctx {ctx} ms ({carry}, index+protect, no rms_mix)",
            fields={},
            mode="stream",
            stream=stream,
        )
    if goku:
        keep_b = {"b1_shift_fixed_0", "b2_shift_fixed_p2", "b4_autoshift", "b5_autoshift_half"}
        table = {k: v for k, v in table.items() if v.stage != "B" or k in keep_b}
    return table


def stage_ids(table: dict[str, Variant], stage: str) -> list[str]:
    if stage == "all":
        return list(table)
    return [k for k, v in table.items() if v.stage == stage]


# --------------------------------------------------------------------------------------------
# Feature availability
# --------------------------------------------------------------------------------------------

_NEW_CONVERT_FIELDS = (
    "use_index",
    "index_rate",
    "index_nprobe",
    "protect",
    "rms_mix_rate",
    "f0_post",
    "auto_pitch",
)


def _convert_accepts(worker: Any, name: str) -> bool:
    try:
        params = inspect.signature(worker.convert).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _rmvpe_ok() -> bool:
    try:
        import rvc_f0

        return bool(rvc_f0.rmvpe_available())
    except Exception:
        return False


def skip_reason(
    variant: Variant, worker: Any, stream_fields: set[str], worker_src: str
) -> str | None:
    """Why ``variant`` cannot run with the installed Q1/Q2 code (None = runnable)."""
    if variant.mode == "whole":
        for name in _NEW_CONVERT_FIELDS:
            if name in variant.fields and not _convert_accepts(worker, name):
                return f"worker.convert has no {name!r} yet"
        if variant.fields.get("f0_method") == "rmvpe" and not _rmvpe_ok():
            return "rmvpe unavailable (rvc_f0 or weights missing)"
        if "RVC_WORKER_FP32" in variant.env and "RVC_WORKER_FP32" not in worker_src:
            return "worker has no RVC_WORKER_FP32 switch"
    else:
        for name in variant.stream:
            if name not in stream_fields:
                return f"StreamParams has no {name!r} yet"
        if variant.stream.get("f0_algo") == "rmvpe" and not _rmvpe_ok():
            return "rmvpe unavailable (rvc_f0 or weights missing)"
    return None


# --------------------------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------------------------


def source_wav(speaker: str, clip: int, out_dir: Path) -> Path:
    if speaker == "ryan":
        return RYAN_DIR / f"fast_nonstream_s{clip}.wav"
    return out_dir / "sources" / speaker / f"s{clip}.wav"


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float32), int(sr)


def resample_to(x: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return x.astype(np.float32)
    from scipy import signal

    g = math.gcd(sr, to)
    return signal.resample_poly(x, to // g, sr // g).astype(np.float32)


def harvest_f0(x16: np.ndarray, f0_floor: float = 50.0, f0_ceil: float = 1100.0) -> np.ndarray:
    import pyworld

    return pyworld.harvest(
        np.asarray(x16, np.float64),
        16000,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=10.0,
    )[0].astype(np.float64)


# --------------------------------------------------------------------------------------------
# Rendering + scoring
# --------------------------------------------------------------------------------------------


class Lab:
    def __init__(self, model: str, out: Path) -> None:
        self.model = model
        self.out = out
        self.workers: dict[tuple, Any] = {}
        self._src_f0: dict[tuple[str, int], np.ndarray] = {}
        self._index: Any = None
        self._index_rows: dict[str, Any] = {}
        self._retr_src: dict[tuple[str, int], float] = {}
        self.rvc_worker: Any = None
        self.rvc_stream: Any = None

    # -- engine -----------------------------------------------------------------------------

    def _import_engine(self) -> None:
        if self.rvc_worker is not None:
            return
        sys.argv = sys.argv[:1]  # rvc_cli's Config parses argv at import
        sys.path.insert(0, str(REPO_ROOT / "legacy"))
        sys.path.insert(0, str(REPO_ROOT / "rvc_cli"))
        sys.path.insert(0, str(REPO_ROOT))
        import rvc_stream
        import rvc_worker

        self.rvc_worker = rvc_worker
        self.rvc_stream = rvc_stream
        os.chdir(REPO_ROOT)

    def worker_for(self, env: dict[str, str]) -> Any:
        self._import_engine()
        key = tuple(sorted(env.items()))
        if key in self.workers:
            return self.workers[key]
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                w = self.rvc_worker.Worker()
                info = w.load_model(self.model, MODEL_VARIANT[self.model])
            print(f"loaded {info}", file=sys.stderr)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.workers[key] = w
        self._warm(w)
        return w

    def _warm(self, worker: Any) -> None:
        work = self.out / "work"
        work.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(sys.stderr):
            worker.convert(
                str(source_wav("ryan", 0, self.out)),
                str(work / "warm.wav"),
                pitch_shift=0,
                f0_method="pm",
            )

    # -- source / index features --------------------------------------------------------------

    def src_f0(self, speaker: str, clip: int) -> np.ndarray:
        key = (speaker, clip)
        if key not in self._src_f0:
            x, sr = read_mono(source_wav(speaker, clip, self.out))
            self._src_f0[key] = harvest_f0(resample_to(x, sr, 16000))
        return self._src_f0[key]

    def get_index(self, worker: Any) -> tuple[Any, str] | None:
        if self._index is not None:
            return self._index
        try:
            import faiss
        except ImportError:
            return None
        vc = worker._vc
        path = None
        getter = getattr(vc, "get_index", None)
        if callable(getter):
            try:
                got = getter()
                if got is not None:
                    idx = got[0] if isinstance(got, tuple) else got
                    if hasattr(idx, "search"):
                        self._index = (idx, "vc.get_index()")
                        return self._index
            except Exception:
                pass
        raw = getattr(vc, "index_path_stripped", None)
        for cand in (raw, os.path.join(REPO_ROOT, "rvc_cli", raw or "")):
            if cand and os.path.isfile(cand):
                path = cand
                break
        if path is None:
            return None
        idx = faiss.read_index(path)
        self._index = (idx, path)
        return self._index

    def hubert_features(self, worker: Any, x16: np.ndarray) -> np.ndarray:
        import torch

        vc = worker._vc
        pipe = vc.vc
        is_v1 = vc.version == "v1"
        layer = 9 if is_v1 else 12
        feats = torch.from_numpy(np.asarray(x16, np.float32))
        feats = feats.half() if pipe.is_half else feats.float()
        feats = feats.view(1, -1).to(pipe.device)
        mask = torch.zeros_like(feats, dtype=torch.bool, device=pipe.device)
        with torch.no_grad():
            logits = vc.hubert_model.extract_features(
                source=feats, padding_mask=mask, output_layer=layer
            )
            f = vc.hubert_model.final_proj(logits[0]) if is_v1 else logits[0]
        return f[0].float().cpu().numpy()

    def retrieval(self, worker: Any, x16: np.ndarray) -> float:
        got = self.get_index(worker)
        if got is None:
            return float("nan")
        index = got[0]
        with contextlib.suppress(Exception):
            index.nprobe = 32
        feats = np.ascontiguousarray(self.hubert_features(worker, x16), dtype=np.float32)
        dist, _ = index.search(feats, 8)
        return knn_mean_l2(dist)

    def retrieval_src(self, worker: Any, speaker: str, clip: int) -> float:
        key = (speaker, clip)
        if key not in self._retr_src:
            x, sr = read_mono(source_wav(speaker, clip, self.out))
            self._retr_src[key] = self.retrieval(worker, resample_to(x, sr, 16000))
        return self._retr_src[key]

    # -- render ---------------------------------------------------------------------------

    def render(self, variant: Variant, clip: int, wav_path: Path, worker: Any) -> dict[str, Any]:
        """Render one clip; returns {render_s, audio_s, reply?, sr, extra}."""
        src = source_wav(variant.source, clip, self.out)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        x24, sr_in = read_mono(src)
        audio_s = len(x24) / sr_in
        if variant.mode == "whole":
            fields = dict(variant.fields)
            fields.setdefault("pitch_shift", 0)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                reply = worker.convert(str(src), str(wav_path), **fields)
            render_s = time.perf_counter() - t0
            return {"render_s": render_s, "audio_s": audio_s, "reply": reply, "extra": {}}
        return self._render_stream(variant, x24, audio_s, wav_path, worker)

    def _render_stream(
        self, variant: Variant, x24: np.ndarray, audio_s: float, wav_path: Path, worker: Any
    ) -> dict[str, Any]:
        import soundfile as sf

        rs = self.rvc_stream
        vc_args = copy.copy(worker._vc_args)
        vc_args.f0_method = "pm"
        params = rs.StreamParams.from_fields(dict(variant.stream))
        with contextlib.redirect_stdout(sys.stderr):
            engine = rs.open_engine(worker._vc, vc_args, params)
            chunk = 12 * 480  # 240 ms = 12 x 20 ms
            outs = []
            for i in range(0, len(x24), chunk):
                outs.append(engine.push(x24[i : i + chunk]))
            outs.append(engine.flush())
        y = np.concatenate(outs) if outs else np.zeros(0, np.float32)
        sf.write(str(wav_path), y, 48000, subtype="FLOAT")
        joins = stream_join_positions(len(y), params.first_block_ms, params.block_ms)
        extra = join_ratio(y, joins)
        return {
            "render_s": float(engine.compute_s),
            "audio_s": audio_s,
            "reply": {"pitch_shift_applied": int(params.pitch_shift)},
            "extra": extra,
        }

    def companion_whole(self, variant: Variant, clip: int, worker: Any) -> Path | None:
        """Whole-path render with the stream variant's settings (for stream_vs_whole_mel_db)."""
        s = variant.stream
        fields: dict[str, Any] = {"f0_method": "rmvpe" if s.get("f0_algo") == "rmvpe" else "pm"}
        for k in ("use_index", "index_rate", "index_nprobe", "protect", "f0_post", "pitch_shift"):
            if s.get(k) is not None:
                fields[k] = s[k]
        probe = Variant(id=variant.id + "_whole", stage=variant.stage, desc="", fields=fields)
        if skip_reason(probe, worker, set(), "") is not None:
            return None
        path = self.out / "samples" / self.model / variant.id / "_whole" / f"s{clip}.wav"
        try:
            self.render(probe, clip, path, worker)
        except Exception as e:
            print(f"companion whole failed: {e}", file=sys.stderr)
            return None
        return path

    # -- scoring ----------------------------------------------------------------------------

    def score_clip(
        self,
        variant: Variant,
        clip: int,
        wav_path: Path,
        rendered: dict[str, Any],
        worker: Any,
    ) -> dict[str, Any]:
        y, sr = read_mono(wav_path)
        y16 = resample_to(y, sr, 16000)
        src_f0 = self.src_f0(variant.source, clip)
        reply = rendered.get("reply") or {}
        shift = float(reply.get("pitch_shift_applied", variant.fields.get("pitch_shift", 0) or 0))
        f0_out = harvest_f0(y16)
        m: dict[str, Any] = f0_metrics(f0_out, src_f0, shift, TARGET_HZ[self.model])
        m.update(hf_hiss_metrics(y, sr, f0_out > 0))
        m.update(sanity(y))
        m["rtf"] = rendered["render_s"] / rendered["audio_s"]
        m["out_ratio"] = (len(y) / sr) / rendered["audio_s"]
        m["pitch_shift_applied"] = shift
        m["source_f0_median_hz"] = reply.get("source_f0_median_hz")
        m["index_used"] = reply.get("index_used")
        m["retrieval_dist"] = self.retrieval(worker, y16)
        m["retrieval_dist_src"] = self.retrieval_src(worker, variant.source, clip)
        ref_path = self.out / "samples" / self.model / REF_VARIANT / f"s{clip}.wav"
        if variant.id == REF_VARIANT:
            m["mel_dist_ref_db"] = 0.0
        elif ref_path.is_file():
            r, rsr = read_mono(ref_path)
            m["mel_dist_ref_db"] = mel_dist_db(resample_to(r, rsr, 16000), y16)
        else:
            m["mel_dist_ref_db"] = float("nan")
        m.update(rendered.get("extra") or {})
        if variant.mode == "stream":
            comp = self.companion_whole(variant, clip, worker)
            if comp is not None:
                c, csr = read_mono(comp)
                m["stream_vs_whole_mel_db"] = mel_dist_db(resample_to(c, csr, 16000), y16)
        return m


def _clean(obj: Any) -> Any:
    """JSON-safe: nan/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


MEAN_KEYS = (
    "f0_octave_err_rate",
    "f0_jump_rate",
    "f0_smooth_cents",
    "f0_track_corr",
    "f0_median_hz",
    "register_err_st",
    "voiced_frac_ratio",
    "hf_hiss_ratio",
    "unvoiced_flatness",
    "peak",
    "clip_frac",
    "nan",
    "rtf",
    "out_ratio",
    "pitch_shift_applied",
    "retrieval_dist",
    "retrieval_dist_src",
    "mel_dist_ref_db",
    "join_max",
    "join_med",
    "stream_vs_whole_mel_db",
)


def mean_metrics(per_clip: Sequence[dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in MEAN_KEYS:
        vals = [c[k] for c in per_clip if _finite(c.get(k))]
        if vals:
            out[k] = float(np.mean(vals)) if k != "nan" else float(np.sum(vals))
    return out


def run_variant(lab: Lab, variant: Variant, clips: Sequence[int]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": lab.model,
        "variant": variant.id,
        "stage": variant.stage,
        "desc": variant.desc,
        "mode": variant.mode,
        "source": variant.source,
        "settings": _clean(
            {"fields": variant.fields, "stream": variant.stream, "env": variant.env}
        ),
    }
    lab._import_engine()
    worker = lab.worker_for(variant.env)
    stream_fields = {f.name for f in dataclasses.fields(lab.rvc_stream.StreamParams)}
    wsrc = Path(lab.rvc_worker.__file__).read_text(errors="ignore")
    vc_py = Path(lab.rvc_worker.__file__).resolve().parent.parent / "rvc_cli" / "voice_converter.py"
    if vc_py.exists():
        wsrc += vc_py.read_text(errors="ignore")
    reason = skip_reason(variant, worker, stream_fields, wsrc)
    if variant.source != "ryan":
        for c in clips:
            if not source_wav(variant.source, c, lab.out).is_file():
                reason = (
                    f"source clips for {variant.source!r} missing "
                    "(run tools/rvc_quality_sources.py)"
                )
                break
    if reason:
        row.update(status="skipped", reason=reason, per_clip=[], mean={}, wavs=[])
        return row
    per_clip, wavs = [], []
    try:
        for clip in clips:
            wav_path = lab.out / "samples" / lab.model / variant.id / f"s{clip}.wav"
            rendered = lab.render(variant, clip, wav_path, worker)
            metrics = lab.score_clip(variant, clip, wav_path, rendered, worker)
            metrics["clip"] = clip
            metrics["wav"] = str(wav_path)
            per_clip.append(metrics)
            wavs.append(str(wav_path))
    except Exception as e:
        import traceback

        traceback.print_exc()
        row.update(
            status="error",
            reason=f"{type(e).__name__}: {e}",
            per_clip=_clean(per_clip),
            mean={},
            wavs=wavs,
        )
        return row
    row.update(
        status="ok", per_clip=_clean(per_clip), mean=_clean(mean_metrics(per_clip)), wavs=wavs
    )
    return row


# --------------------------------------------------------------------------------------------
# results.json / index.md
# --------------------------------------------------------------------------------------------


def merge_results(existing: list[dict], new: Iterable[dict]) -> list[dict]:
    """Merge by (model, variant): replace in place, append otherwise."""
    rows = list(existing)
    pos = {(r["model"], r["variant"]): i for i, r in enumerate(rows)}
    for r in new:
        key = (r["model"], r["variant"])
        if key in pos:
            rows[pos[key]] = r
        else:
            pos[key] = len(rows)
            rows.append(r)
    return rows


def attach_flags(rows: list[dict]) -> None:
    for model in {r["model"] for r in rows}:
        group = [r for r in rows if r["model"] == model]
        flags = compute_flags(group)
        for r in group:
            r["flags"] = flags.get(r["variant"], [])


def _fmt(v: Any, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if _finite(v) else "-"


def _settings_str(row: dict) -> str:
    s = row.get("settings") or {}
    parts = []
    for k, v in (s.get("fields") or {}).items():
        if k == "f0_post" and isinstance(v, dict):
            v = "post"
        elif k == "auto_pitch" and isinstance(v, dict):
            v = f"{v.get('target_median_hz'):g}Hz@{v.get('strength')}"
        parts.append(f"{k}={v}")
    for k, v in (s.get("stream") or {}).items():
        if k == "f0_post" and isinstance(v, dict):
            v = "post"
        parts.append(f"{k}={v}")
    for k, v in (s.get("env") or {}).items():
        parts.append(f"{k}={v}")
    if row.get("source") and row["source"] != "ryan":
        parts.append(f"src={row['source']}")
    return ", ".join(parts).replace("|", "/")


def render_index_md(rows: Sequence[dict]) -> str:
    lines = [
        "# RVC quality lab",
        "",
        "Proxies only; the owner's ears decide. Flags: BEST_OVERALL = lowest "
        "rank-average of octave_err, jump_rate, retrieval_dist among variants with "
        "voiced_frac_ratio >= 0.85 and hiss not in the worst quartile; BEST_STREAM is the same "
        "for stream renders. Compare retrieval_dist at equal index_rate (index-blended renders "
        "score better by construction).",
        "",
    ]
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        lines += [f"## {model} (target {TARGET_HZ.get(model, 0):g} Hz)", ""]
        lines += [
            "| variant | settings | octave_err | jump_rate | f0_corr | register_err_st | "
            "retrieval_dist (src) | mel_dist_ref | hiss | rtf | flags | wavs |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        srcs: dict[str, list[float]] = {}
        for r in group:
            if r.get("status") != "ok":
                lines.append(
                    f"| {r['variant']} | {_settings_str(r)} | **{r.get('status')}**: "
                    f"{str(r.get('reason', '')).replace('|', '/')} | | | | | | | | | |"
                )
                continue
            m = r["mean"]
            if _finite(m.get("retrieval_dist_src")):
                srcs.setdefault(r.get("source", "ryan"), []).append(m["retrieval_dist_src"])
            wavs = " ".join(f"[s{i}]({p})" for i, p in enumerate(r.get("wavs", [])))
            lines.append(
                (
                    "| {v} | {s} | {oe} | {jr} | {fc} | {rg} | {rd} ({rs}) | {md} | {hi} "
                    "| {rtf} | {fl} | {w} |"
                ).format(
                    v=r["variant"],
                    s=_settings_str(r),
                    oe=_fmt(m.get("f0_octave_err_rate"), 4),
                    jr=_fmt(m.get("f0_jump_rate"), 2),
                    fc=_fmt(m.get("f0_track_corr")),
                    rg=_fmt(m.get("register_err_st"), 2),
                    rd=_fmt(m.get("retrieval_dist"), 2),
                    rs=_fmt(m.get("retrieval_dist_src"), 2),
                    md=_fmt(m.get("mel_dist_ref_db"), 2),
                    hi=_fmt(m.get("hf_hiss_ratio"), 5),
                    rtf=_fmt(m.get("rtf"), 3),
                    fl=", ".join(r.get("flags", [])),
                    w=wavs,
                )
            )
        for spk, vals in sorted(srcs.items()):
            lines.append(
                f"| _src control ({spk})_ | unconverted source | | | | | "
                f"{_fmt(float(np.mean(vals)), 2)} | | | | | |"
            )
        stream_rows = [r for r in group if r.get("status") == "ok" and r.get("mode") == "stream"]
        if stream_rows:
            lines += [
                "",
                f"Stream joins ({model}): join_max (target <= ~2), stream_vs_whole_mel_db",
                "",
            ]
            lines += [
                "| variant | join_max | join_med | stream_vs_whole_mel_db | rtf |",
                "|---|---|---|---|---|",
            ]
            for r in stream_rows:
                m = r["mean"]
                lines.append(
                    f"| {r['variant']} | {_fmt(m.get('join_max'), 2)} | "
                    f"{_fmt(m.get('join_med'), 2)} | "
                    f"{_fmt(m.get('stream_vs_whole_mel_db'), 2)} | {_fmt(m.get('rtf'), 3)} |"
                )
        lines.append("")
    return "\n".join(lines)


def write_outputs(out: Path, new_rows: Iterable[dict]) -> list[dict]:
    path = out / "results.json"
    existing: list[dict] = []
    if path.is_file():
        with contextlib.suppress(Exception):
            existing = json.loads(path.read_text())
    rows = merge_results(existing, new_rows)
    attach_flags(rows)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(rows), indent=1))
    (out / "index.md").write_text(render_index_md(_clean(rows)))
    return rows


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def parse_clips(text: str) -> list[int]:
    clips: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            clips.extend(range(int(a), int(b) + 1))
        else:
            clips.append(int(part))
    bad = [c for c in clips if not 0 <= c < N_CLIPS]
    if bad:
        raise ValueError(f"clips out of range 0..{N_CLIPS - 1}: {bad}")
    return clips


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Obama", choices=sorted(TARGET_HZ))
    p.add_argument("--stage", default="all", choices=["0", "A", "B", "C", "D", "E", "all"])
    p.add_argument("--variants", default=None, help="comma-separated ids (overrides --stage)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--clips", default="0..5", help="e.g. 0..5 or 0,2,3")
    p.add_argument("--base-variant", default=DEFAULT_BASE_VARIANT)
    p.add_argument("--base-shift", type=int, default=0, help="fixed shift carried into stages C-E")
    p.add_argument(
        "--base-auto", type=float, default=None, help="auto-shift strength carried into C"
    )
    p.add_argument("--list", action="store_true", help="print the variant table and exit")
    args = p.parse_args(argv)

    table = build_variants(
        args.model, base_id=args.base_variant, base_shift=args.base_shift, base_auto=args.base_auto
    )
    if args.list:
        for v in table.values():
            print(f"{v.stage:>3} {v.id:28s} {v.mode:6s} {v.desc}")
        return 0
    ids = (
        [s.strip() for s in args.variants.split(",") if s.strip()]
        if args.variants
        else stage_ids(table, args.stage)
    )
    unknown = [i for i in ids if i not in table]
    if unknown:
        p.error(f"unknown variants for {args.model}: {unknown}")
    clips = parse_clips(args.clips)
    out = Path(args.out).resolve()
    lab = Lab(args.model, out)
    # Reference first (mel distance needs it on disk).
    need_ref = REF_VARIANT in table and REF_VARIANT not in ids
    if need_ref and any(
        (out / "samples" / args.model / REF_VARIANT / f"s{c}.wav").is_file() is False for c in clips
    ):
        ids = [REF_VARIANT, *ids]
    elif REF_VARIANT in ids:
        ids = [REF_VARIANT, *[i for i in ids if i != REF_VARIANT]]
    for vid in ids:
        v = table[vid]
        print(f"== {args.model} {vid} ({v.desc})", file=sys.stderr, flush=True)
        row = run_variant(lab, v, clips)
        write_outputs(out, [row])  # write after each variant so partial runs are kept
        print(f"   {row['status']} {row.get('reason', '')}", file=sys.stderr, flush=True)
    rows = json.loads((out / "results.json").read_text())
    best = [
        r["variant"]
        for r in rows
        if r["model"] == args.model and "BEST_OVERALL" in r.get("flags", [])
    ]
    print(json.dumps({"index": str(out / "index.md"), "BEST_OVERALL": best}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
