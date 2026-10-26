#!/usr/bin/env python
"""Reviewer CLI: compare streaming RVC against whole-utterance conversion (f0 pm).

Run with the legacy venv and a GPU lease (the GPU is shared with the live bot, keep it short):

    flock /home/cbash23/.cache/listen-speed/gpu.lock \\
        .venv/bin/python legacy/rvc_stream_eval.py a.wav b.wav \\
        --cfg '[{}, {"block_ms": 400}]' --write-dir /home/cbash23/.cache/listen-speed/rvc-stream/out

Inputs are 24 kHz mono WAVs (other rates / channel counts are resampled / downmixed). Prints one
JSON list: per file a ``noise_floor_whole_vs_whole`` row, then one row per ``--cfg`` override with
out_ratio, rtf, first_latency_ms, step_ms, f0_corr, f0_cents_med, voiced_agree, mel_l1_db,
join_max, join_med and jump_ref_whole (same join positions in the whole-utterance audio).
Acceptance thresholds: docs/RVC_STREAMING_DESIGN.md section 9.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CHUNK_MS = 250


def _metrics(ref, y):  # noqa: ANN001
    import librosa
    import numpy as np
    import parselmouth
    import scipy.signal as sig

    def f0_of(y48):  # noqa: ANN001
        y16 = sig.resample_poly(y48, 1, 3).astype(np.float64)
        pitch = parselmouth.Sound(y16, 16000).to_pitch_ac(
            time_step=0.01, voicing_threshold=0.6, pitch_floor=50, pitch_ceiling=1100
        )
        return pitch.selected_array["frequency"]

    def align(a, b, maxlag=2400):  # noqa: ANN001
        n = min(len(a), len(b))
        ea = sig.resample_poly(np.abs(sig.hilbert(a[:n])), 1, 48)
        eb = sig.resample_poly(np.abs(sig.hilbert(b[:n])), 1, 48)
        c = np.correlate(ea - ea.mean(), eb - eb.mean(), "full")
        lag = (np.argmax(c) - (len(eb) - 1)) * 48
        return int(np.clip(lag, -maxlag, maxlag))

    lag = align(ref, y)
    a, b = (ref[lag:], y) if lag > 0 else (ref, y[-lag:])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    fa, fb = f0_of(a), f0_of(b)
    m = min(len(fa), len(fb))
    fa, fb = fa[:m], fb[:m]
    voiced = (fa > 0) & (fb > 0)
    cents = np.abs(1200 * np.log2(fb[voiced] / fa[voiced]))
    if voiced.sum() > 5:
        corr = float(np.corrcoef(np.log(fa[voiced]), np.log(fb[voiced]))[0, 1])
    else:
        corr = float("nan")

    def mel(sig_):  # noqa: ANN001
        spec = librosa.feature.melspectrogram(
            y=sig_, sr=48000, n_fft=2048, hop_length=480, n_mels=80
        )
        return librosa.power_to_db(spec + 1e-10)

    ma, mb = mel(a), mel(b)
    k = min(ma.shape[1], mb.shape[1])
    union = max(1, int(((fa > 0) | (fb > 0)).sum()))
    return {
        "lag_ms": lag / 48,
        "f0_corr": round(corr, 4),
        "f0_cents_med": round(float(np.median(cents)), 1) if cents.size else None,
        "voiced_agree": round(float(voiced.sum() / union), 3),
        "mel_l1_db": round(float(np.mean(np.abs(ma[:, :k] - mb[:, :k]))), 2),
    }


def _jump_ratio(y, joins):  # noqa: ANN001
    import numpy as np

    d2 = np.abs(np.diff(y, 2))
    out = []
    for j in joins:
        seg = d2[max(0, j - 48) : j + 48]
        if len(seg):
            ref = np.percentile(d2[max(0, j - 2400) : j + 2400], 95) + 1e-9
            out.append(seg.max() / ref)
    if not out:
        return {}
    return {"join_max": round(float(max(out)), 2), "join_med": round(float(np.median(out)), 2)}


def _read_24k_mono(path: str):  # noqa: ANN202
    import numpy as np
    import scipy.signal as sig
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if sr != 24000:
        g = math.gcd(sr, 24000)
        x = sig.resample_poly(x, 24000 // g, sr // g).astype(np.float32)
    return x


def _to_48k(y, sr: int):  # noqa: ANN001, ANN202
    import numpy as np
    import scipy.signal as sig

    if sr == 48000:
        return y.astype(np.float32)
    g = math.gcd(sr, 48000)
    return sig.resample_poly(y, 48000 // g, sr // g).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wavs", nargs="+", help="24 kHz mono WAV files")
    parser.add_argument("--cfg", default="[{}]", help="JSON list of StreamParams overrides")
    parser.add_argument("--write-dir", default=None, help="save streamed/whole WAVs here")
    parser.add_argument("--chunk-ms", type=int, default=CHUNK_MS, help="input chunk size")
    parser.add_argument("--model", default="Goku")
    args = parser.parse_args()
    cfgs = json.loads(args.cfg)
    if not isinstance(cfgs, list) or not all(isinstance(c, dict) for c in cfgs):
        parser.error("--cfg must be a JSON list of objects")
    wavs = [str(Path(w).resolve()) for w in args.wavs]  # rvc_worker chdirs to the repo root
    write_dir = Path(args.write_dir).resolve() if args.write_dir else None
    if write_dir:
        write_dir.mkdir(parents=True, exist_ok=True)

    # rvc_cli's Config parses sys.argv at import: clear it BEFORE importing the worker.
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(HERE))
    import numpy as np
    import rvc_stream
    import rvc_worker
    import soundfile as sf

    # The model/whole-path code prints on stdout; keep stdout JSON-only.
    with contextlib.redirect_stdout(sys.stderr):
        worker = rvc_worker.Worker()
        info = worker.load_model(args.model)
        print(info, file=sys.stderr)
        worker._vc_args = copy.copy(worker._vc_args)
        worker._vc_args.f0_method = "pm"
        tgt_sr = int(worker._vc.tgt_sr)

        def whole(path: str):  # noqa: ANN202
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "in.wav"
                src.write_bytes(Path(path).read_bytes())
                out = worker._vc.convert_voice(src, worker._vc_args)
                y, sr = sf.read(out, dtype="float32")
            return _to_48k(y, sr if sr else tgt_sr)

        results: list[dict[str, Any]] = []
        for path in wavs:
            x = _read_24k_mono(path)
            seconds = len(x) / 24000
            t0 = time.perf_counter()
            ref = whole(path)
            print(f"whole {seconds:.2f}s took {time.perf_counter() - t0:.3f}s", file=sys.stderr)
            ref2 = whole(path)
            results.append(
                {"file": path, "kind": "noise_floor_whole_vs_whole", **_metrics(ref, ref2)}
            )
            for cidx, over in enumerate(cfgs):
                params = rvc_stream.StreamParams.from_fields({**over, "f0_method": "pm"})
                engine = rvc_stream.open_engine(worker._vc, worker._vc_args, params)
                chunk = int(args.chunk_ms * 24)
                outs: list = []
                first = None
                for i in range(0, len(x), chunk):
                    before = engine.compute_s
                    got = engine.push(x[i : i + chunk])
                    if got.shape[0] and first is None:
                        first = engine.compute_s - before
                    outs.append(got)
                outs.append(engine.flush())
                y = np.concatenate(outs) if outs else np.zeros(0, np.float32)
                first_b = params.first_block_ms * 24
                block_b = params.block_ms * 24
                joins = [2 * first_b]
                while joins[-1] + 2 * block_b < len(y):
                    joins.append(joins[-1] + 2 * block_b)
                row: dict[str, Any] = {
                    "file": path,
                    "cfg": over,
                    "out_ratio": round(len(y) / (2 * len(x)), 4),
                    "rtf": round(engine.compute_s / seconds, 3),
                    "first_latency_ms": round(1000 * (first or 0.0), 1),
                    "step_ms": round(1000 * engine.compute_s / max(1, engine.steps), 1),
                    "steps": engine.steps,
                }
                row.update(_metrics(ref, y))
                row.update(_jump_ratio(y, joins))
                row["jump_ref_whole"] = _jump_ratio(ref, joins).get("join_max")
                results.append(row)
                if write_dir:
                    stem = Path(path).stem
                    sf.write(write_dir / f"{stem}_cfg{cidx}.wav", y, 48000)
                    if cidx == 0:
                        sf.write(write_dir / f"{stem}_whole.wav", ref, 48000)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
