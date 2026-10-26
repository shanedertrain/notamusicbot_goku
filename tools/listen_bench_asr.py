"""Listen-mode P0c: ASR bake-off (no Discord).

Candidates, all measured on the same synthetic clips:
  whisper_small_en, whisper_base_en   transformers fp16 cuda, .venv-main
  parakeet_v2_cpu*, parakeet_v3_cpu   onnx-asr + onnxruntime (CPU), scratch venv
  parakeet_v2_gpu                     onnx-asr + onnxruntime-gpu, SEPARATE scratch venv
  qwen3_asr_1_7b                      qwen-asr (bf16 cuda), voiceover/.venv-asr

Nothing is installed into .venv-main or voiceover/.venv-asr. The scratch venvs
live under SCRATCH (default ~/.cache/listen-asr-bakeoff) and were made with:

    uv venv --python 3.12 cpu
    uv pip install -p cpu "onnx-asr[hub]" onnxruntime soundfile numpy
    uv venv --python 3.12 gpu
    uv pip install -p gpu "onnx-asr[hub]" onnxruntime-gpu soundfile numpy nvidia-cudnn-cu13
    (GPU run also needs LD_LIBRARY_PATH to that venv's site-packages/nvidia/*/lib dirs)

Steps (each writes one partial JSON under SCRATCH/results; `merge` combines them):

    .venv-main/bin/python tools/listen_bench_asr.py clips
    .venv-main/bin/python tools/listen_bench_asr.py whisper
    SCRATCH/cpu/bin/python tools/listen_bench_asr.py onnx --model v2 --device cpu \\
        [--quant int8] [--threads N]
    SCRATCH/gpu/bin/python tools/listen_bench_asr.py onnx --model v2 --device cuda
    voiceover/.venv-asr/bin/python tools/listen_bench_asr.py qwen
    .venv-main/bin/python tools/listen_bench_asr.py merge

Clips are TTS (goku profile, voice conversion passthrough) of known text,
resampled to 16 kHz mono float32. Synthetic clean speech is optimistic compared
with real microphones; no real recordings are used. The clip WAVs are synthetic
and live only under SCRATCH/clips. Result: evidence/listen_asr_bakeoff.json.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("LISTEN_ASR_SCRATCH", Path.home() / ".cache" / "listen-asr-bakeoff"))
CLIPS_DIR = SCRATCH / "clips"
RESULTS_DIR = SCRATCH / "results"
DEFAULT_OUT = REPO_ROOT / "evidence" / "listen_asr_bakeoff.json"

# Three ~5 s sentences (same as listen_bench_models) plus one ~15 s clip, the
# maximum utterance length.
ASR_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "Please play some music and turn the volume up a little.",
    "What time does the next train leave for the city tomorrow morning?",
    (
        "Yesterday I walked down to the old market with my brother, and we bought "
        "fresh bread, two jars of honey, and a small basket of strawberries "
        "before the afternoon rain started to fall over the whole town."
    ),
]
SAMPLE_RATE = 16000
SILENCE_S = 5.0
NOISE_DBFS = -50.0
REPS = 3  # timed repetitions per clip after one untimed warm-up; median reported
NOTE = (
    "ASR clips are synthetic clean TTS speech (goku profile, no voice conversion); "
    "word error rates are optimistic compared with real microphones. No real "
    "recordings were used."
)


def smi() -> dict[str, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used, free, total = (int(x) for x in out.strip().split(","))
    return {"used_mib": used, "free_mib": free, "total_mib": total}


def wer(reference: str, hypothesis: str) -> float:
    """Word-level edit distance / reference length (same as listen_bench_models)."""
    ref = re.sub(r"[^a-z0-9' ]+", " ", reference.lower()).split()
    hyp = re.sub(r"[^a-z0-9' ]+", " ", hypothesis.lower()).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref) if ref else 0.0


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def silence_clips():
    import numpy as np

    n = int(SILENCE_S * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    rms = 10 ** (NOISE_DBFS / 20)
    noise = (rng.standard_normal(n) * rms).astype(np.float32)
    return [
        ("silence_5s", np.zeros(n, dtype=np.float32)),
        ("white_noise_-50dbfs_5s", noise),
    ]


def load_clips():
    import numpy as np
    import soundfile as sf

    manifest = json.loads((CLIPS_DIR / "manifest.json").read_text())
    clips = []
    for row in manifest:
        audio, rate = sf.read(CLIPS_DIR / row["file"], dtype="float32")
        assert rate == SAMPLE_RATE and audio.ndim == 1
        clips.append((row["text"], np.ascontiguousarray(audio)))
    return clips


def run_clips_and_silence(transcribe, clips, *, cpu_candidate: bool) -> dict:
    """One untimed warm-up, then REPS timed runs per clip (median), then silence tests."""
    transcribe(clips[0][1])  # warm-up, not reported
    rows = []
    for text, audio in clips:
        lats, cpu_ts, hyp = [], [], ""
        for _ in range(REPS):
            c0, t0 = time.process_time(), time.perf_counter()
            hyp = transcribe(audio)
            lats.append(time.perf_counter() - t0)
            cpu_ts.append(time.process_time() - c0)
        dur = len(audio) / SAMPLE_RATE
        lat = statistics.median(lats)
        row = {
            "reference": text,
            "hypothesis": hyp,
            "clip_s": round(dur, 3),
            "latency_s": round(lat, 3),
            "latency_all_s": [round(x, 3) for x in lats],
            "rtf": round(lat / dur, 4),
            "wer": round(wer(text, hyp), 4),
        }
        if cpu_candidate:
            row["cpu_time_s"] = round(statistics.median(cpu_ts), 3)
            row["effective_cores"] = round(statistics.median(cpu_ts) / lat, 2)
        rows.append(row)
    silence = []
    for name, audio in silence_clips():
        t0 = time.perf_counter()
        hyp = transcribe(audio)
        silence.append(
            {
                "clip": name,
                "text": hyp,
                "emitted_text": bool(hyp.strip()),
                "latency_s": round(time.perf_counter() - t0, 3),
            }
        )
    long_row = max(rows, key=lambda r: r["clip_s"])
    return {
        "per_clip": rows,
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        "mean_wer": round(sum(r["wer"] for r in rows) / len(rows), 4),
        "long_clip": {
            "clip_s": long_row["clip_s"],
            "latency_s": long_row["latency_s"],
            "rtf": long_row["rtf"],
            "wer": long_row["wer"],
        },
        "silence": silence,
    }


def save_partial(name: str, data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"candidate": name, "baseline_note": NOTE, **data}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


# --- step: clips (.venv-main) -------------------------------------------------


async def make_clips() -> None:
    import numpy as np
    import soundfile as sf
    import soxr

    from notamusicbot.app import Application
    from notamusicbot.config import load_app_settings
    from notamusicbot.domain.speech import (
        AudioProcessingProfile,
        TtsRequest,
        VoiceConversionProfile,
    )
    from notamusicbot.domain.voice_profiles import load_voice_profiles

    settings = load_app_settings()
    profile = load_voice_profiles(Path(settings.speech.voices_dir))["goku"]
    app = Application(settings)
    app.load_speech_model()
    pipeline = app.build_pipeline()
    processing = AudioProcessingProfile(
        target_sample_rate_hz=settings.speech.sample_rate_hz,
        channels=settings.speech.channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )
    conversion = VoiceConversionProfile(
        backend="passthrough",
        model=profile.voice_conversion.model,
        pitch_shift=profile.voice_conversion.pitch_shift,
    )
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest, created = [], []
    try:
        for i, text in enumerate(ASR_SENTENCES):
            request = TtsRequest(
                text=text, language=profile.tts.language, speaker=profile.tts.speaker
            )
            final, _ = await pipeline.run(request, conversion, processing)
            created.append(final.path)
            data, rate = sf.read(final.path, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            audio = np.ascontiguousarray(soxr.resample(mono, rate, SAMPLE_RATE), dtype=np.float32)
            name = f"clip{i}.wav"
            sf.write(CLIPS_DIR / name, audio, SAMPLE_RATE, subtype="FLOAT")
            manifest.append(
                {"file": name, "text": text, "duration_s": round(len(audio) / SAMPLE_RATE, 3)}
            )
    finally:
        for p in created:
            p.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass
    (CLIPS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    await app.shutdown()


# --- step: whisper (.venv-main) ----------------------------------------------


def bench_whisper() -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    clips = load_clips()
    for model_id, name in (
        ("openai/whisper-small.en", "whisper_small_en"),
        ("openai/whisper-base.en", "whisper_base_en"),
    ):
        gc.collect()
        torch.cuda.empty_cache()
        before = smi()
        t0 = time.perf_counter()
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        model.to("cuda").eval()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0

        def transcribe(audio, processor=processor, model=model) -> str:
            feats = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = model.generate(feats.to("cuda", dtype=torch.float16))
            torch.cuda.synchronize()
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        res = run_clips_and_silence(transcribe, clips, cpu_candidate=False)
        after = smi()
        save_partial(
            name,
            {
                "model": model_id,
                "device": "cuda",
                "dtype": "fp16",
                "runtime": "transformers",
                "load_s": round(load_s, 2),
                "vram_before_mib": before["used_mib"],
                "vram_after_mib": after["used_mib"],
                "vram_delta_mib": after["used_mib"] - before["used_mib"],
                "install": "none (already in .venv-main and the HF cache)",
                **res,
            },
        )
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


# --- step: onnx (scratch venvs) ----------------------------------------------

ONNX_REPOS = {
    "v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
}
ONNX_MODEL_IDS = {"v2": "nemo-parakeet-tdt-0.6b-v2", "v3": "nemo-parakeet-tdt-0.6b-v3"}


def _ort_sessions(obj, seen=None, depth=0):
    """Find onnxruntime InferenceSession objects inside an onnx-asr model (best effort)."""
    import onnxruntime as ort

    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    if isinstance(obj, ort.InferenceSession):
        return [obj]
    found = []
    attrs = getattr(obj, "__dict__", {})
    for v in list(attrs.values()):
        found += _ort_sessions(v, seen, depth + 1)
    return found


def materialize(model_key: str) -> Path:
    """Hard-link the HF snapshot into a plain directory.

    onnxruntime 1.31 rejects external-data files (encoder-model.onnx.data) whose
    symlink target escapes the model directory, which is what the HF cache layout
    does for the fp32 export. A real directory of hard links loads fine.
    """
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(ONNX_REPOS[model_key]))
    dest = SCRATCH / "models" / model_key
    dest.mkdir(parents=True, exist_ok=True)
    for f in snap.iterdir():
        target = dest / f.name
        if not target.exists():
            os.link(f.resolve(), target)
    return dest


def bench_onnx(args) -> None:
    import onnx_asr
    import onnxruntime as ort

    clips = load_clips()
    name = f"parakeet_{args.model}_{args.device}" + (f"_{args.quant}" if args.quant else "")
    if args.threads:
        name += f"_t{args.threads}"
    gpu = args.device == "cuda"
    before = smi()
    sess_options = ort.SessionOptions()
    if args.threads:
        sess_options.intra_op_num_threads = args.threads
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    )
    model_dir = materialize(args.model)
    t0 = time.perf_counter()
    model = onnx_asr.load_model(
        ONNX_MODEL_IDS[args.model],
        model_dir,
        quantization=args.quant,
        sess_options=sess_options,
        providers=providers,
    )
    load_s = time.perf_counter() - t0
    sessions = _ort_sessions(model)
    session_providers = sorted({tuple(s.get_providers()) for s in sessions})
    after_load = smi()

    def transcribe(audio) -> str:
        return str(model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()

    res = run_clips_and_silence(transcribe, clips, cpu_candidate=not gpu)
    after = smi()
    default_threads = os.cpu_count()
    out = {
        "model": ONNX_REPOS[args.model],
        "device": args.device,
        "quantization": args.quant or "fp32",
        "runtime": f"onnx-asr {onnx_asr.__version__ if hasattr(onnx_asr, '__version__') else ''} "
        f"+ onnxruntime {ort.__version__}",
        "available_providers": ort.get_available_providers(),
        "session_providers": [list(p) for p in session_providers],
        "load_s": round(load_s, 2),
        "vram_before_mib": before["used_mib"],
        "vram_after_load_mib": after_load["used_mib"],
        "vram_after_mib": after["used_mib"],
        "vram_delta_mib": after["used_mib"] - before["used_mib"],
        "intra_op_threads": args.threads or "ort default (physical cores)",
        "logical_cpus": default_threads,
        "cpu_model": cpu_model(),
        **res,
    }
    save_partial(name, out)


# --- step: qwen (voiceover/.venv-asr) ----------------------------------------


def bench_qwen() -> None:
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    clips = load_clips()
    before = smi()
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda"
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after_load = smi()

    def transcribe(audio) -> str:
        out = model.transcribe(
            (np.asarray(audio, dtype=np.float32), SAMPLE_RATE), language="English"
        )
        torch.cuda.synchronize()
        return out[0].text.strip()

    res = run_clips_and_silence(transcribe, clips, cpu_candidate=False)
    after = smi()
    save_partial(
        "qwen3_asr_1_7b",
        {
            "model": "Qwen/Qwen3-ASR-1.7B",
            "device": "cuda",
            "dtype": "bf16",
            "runtime": "qwen-asr 0.0.6 (voiceover/.venv-asr)",
            "load_s": round(load_s, 2),
            "vram_before_mib": before["used_mib"],
            "vram_after_load_mib": after_load["used_mib"],
            "vram_after_mib": after["used_mib"],
            "vram_delta_mib": after["used_mib"] - before["used_mib"],
            **res,
        },
    )


# --- step: merge -------------------------------------------------------------


def merge(out_path: Path, extra_path: Path | None) -> None:
    extra = json.loads(extra_path.read_text()) if extra_path else {}
    candidates = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))}
    manifest = json.loads((CLIPS_DIR / "manifest.json").read_text())
    result = {
        "note": NOTE,
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "clips": manifest,
        "silence_tests": [
            f"{SILENCE_S:.0f} s digital silence",
            f"{SILENCE_S:.0f} s white noise at {NOISE_DBFS:.0f} dBFS (rms, seeded)",
        ],
        "method": (
            f"one untimed warm-up on clip 0, then {REPS} timed runs per clip (median reported); "
            "silence tests run once after the clips"
        ),
        "candidates": candidates,
        **extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("clips")
    sub.add_parser("whisper")
    sub.add_parser("qwen")
    p = sub.add_parser("onnx")
    p.add_argument("--model", choices=["v2", "v3"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--quant", default=None, help="e.g. int8 (default fp32)")
    p.add_argument("--threads", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--out", type=Path, default=DEFAULT_OUT)
    m.add_argument("--extra", type=Path, default=None, help="JSON of extra top-level keys")
    args = parser.parse_args()
    if args.step == "clips":
        asyncio.run(make_clips())
    elif args.step == "whisper":
        bench_whisper()
    elif args.step == "qwen":
        bench_qwen()
    elif args.step == "onnx":
        bench_onnx(args)
    else:
        merge(args.out, args.extra)


if __name__ == "__main__":
    sys.exit(main())
