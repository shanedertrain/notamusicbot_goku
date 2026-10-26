"""Listen-mode P0b: local GPU micro-benchmark (no Discord).

Measures, in one process, on the real production components:
  1. TTS + RVC through the real SpeechPipeline (goku profile, RVC), cold first
     job then warm-up job then 5 one-sentence lines.
  2. ASR with openai/whisper-small.en and openai/whisper-base.en (transformers,
     fp16, cuda): load time, VRAM delta, per-clip latency, word error rate.
     Clips are plain-TTS (no voice conversion) of three known sentences,
     resampled to 16 kHz mono float32 in memory. Synthetic clean speech is
     optimistic compared with real microphones.
  3. VRAM (nvidia-smi used/free MiB) at each load step: LLM, TTS, RVC worker,
     Whisper small.en.

Run from the repo root (needs the two Whisper models in the HF cache; this
tool never downloads):

    .venv-main/bin/python tools/listen_bench_models.py [--out PATH]

Result: evidence/listen_models_benchmark.json. Per-job WAVs under var/speech
are deleted after reading; nothing is written elsewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import subprocess
import time
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "evidence" / "listen_models_benchmark.json"

LINES = [
    "Hey there, it's me, Goku!",
    "That fight sounds really intense, tell me more.",
    "I'm getting hungry, let's grab some food.",
    "Wow, you're getting stronger every single day.",
    "Let's train together and see how far we can go.",
]
ASR_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "Please play some music and turn the volume up a little.",
    "What time does the next train leave for the city tomorrow morning?",
]
WHISPER_MODELS = ["openai/whisper-small.en", "openai/whisper-base.en"]
NOTE = (
    "ASR clips are synthetic clean TTS speech; word error rates are optimistic "
    "compared with real microphones."
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
    """Word-level edit distance / reference length (lowercase, no punctuation)."""
    ref = re.sub(r"[^a-z0-9' ]+", " ", reference.lower()).split()
    hyp = re.sub(r"[^a-z0-9' ]+", " ", hypothesis.lower()).split()
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref) if ref else 0.0


async def run_job(pipeline, text, profile, processing, *, vc: bool):
    conversion = VoiceConversionProfile(
        backend=profile.voice_conversion.backend if vc else "passthrough",
        model=profile.voice_conversion.model,
        pitch_shift=profile.voice_conversion.pitch_shift,
    )
    request = TtsRequest(text=text, language=profile.tts.language, speaker=profile.tts.speaker)
    return await pipeline.run(request, conversion, processing)


def job_row(text, metrics) -> dict:
    return {
        "text": text,
        "words": len(text.split()),
        "tts_s": round(metrics.tts_s, 3),
        "vc_s": round(metrics.voice_conversion_s, 3),
        "pp_s": round(metrics.postprocess_s, 3),
        "total_s": round(metrics.total_s, 3),
        "audio_s": round(metrics.output_duration_s, 3),
        "rtf": round(metrics.total_rtf, 3),
    }


def load_clip_16k(path: Path) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return np.ascontiguousarray(soxr.resample(mono, rate, 16000), dtype=np.float32)


def bench_whisper(model_id, clips, steps_key, vram_steps):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    gc.collect()
    torch.cuda.empty_cache()
    before = smi()
    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to("cuda").eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after = smi()
    vram_steps[steps_key] = after

    def transcribe(audio: np.ndarray) -> str:
        feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats.to("cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    transcribe(clips[0][1])  # warm-up, not reported
    rows = []
    for text, audio in clips:
        t = time.perf_counter()
        hyp = transcribe(audio)
        lat = time.perf_counter() - t
        rows.append(
            {
                "reference": text,
                "hypothesis": hyp,
                "clip_s": round(len(audio) / 16000, 3),
                "latency_s": round(lat, 3),
                "wer": round(wer(text, hyp), 4),
            }
        )
    result = {
        "model": model_id,
        "load_s": round(load_s, 2),
        "vram_delta_mib": after["used_mib"] - before["used_mib"],
        "per_clip": rows,
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        "mean_wer": round(sum(r["wer"] for r in rows) / len(rows), 4),
    }
    return result, (processor, model)


async def main_async(out_path: Path) -> dict:
    import torch

    result: dict = {"note": NOTE, "vram_steps": {}}
    steps = result["vram_steps"]
    steps["0_baseline"] = smi()

    settings = load_app_settings()
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles["goku"]
    app = Application(settings)

    # Same order as Application.startup(): LLM, then (TTS) and the RVC worker.
    t = time.perf_counter()
    await asyncio.to_thread(app.load_content_model)
    result["llm_load_s"] = round(time.perf_counter() - t, 2)
    steps["1_after_llm"] = smi()

    t = time.perf_counter()
    app.load_speech_model()
    pipeline = app.build_pipeline()
    # The synthesizer constructor loads the weights (speech features enabled).
    result["tts_load_s"] = round(time.perf_counter() - t, 2)
    steps["2_after_tts"] = smi()

    t = time.perf_counter()
    await app.converter_router.startup()
    result["rvc_startup_s"] = round(time.perf_counter() - t, 2)
    steps["3_after_rvc_startup"] = smi()

    processing = AudioProcessingProfile(
        target_sample_rate_hz=settings.speech.sample_rate_hz,
        channels=settings.speech.channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )
    created: list[Path] = []

    async def job(text, *, vc):
        final, metrics = await run_job(pipeline, text, profile, processing, vc=vc)
        created.append(final.path)
        return final, metrics

    try:
        # 1. TTS + RVC. Cold first job, then warm-up job, then the 5 lines.
        _, m = await job("Hello, this is the first cold job.", vc=True)
        result["tts_rvc_cold_first_job"] = job_row("Hello, this is the first cold job.", m)
        steps["4_after_first_rvc_job"] = smi()
        _, m = await job("Warm up job, here we go.", vc=True)
        result["tts_rvc_warmup_job"] = job_row("Warm up job, here we go.", m)
        rows = []
        for text in LINES:
            _, m = await job(text, vc=True)
            rows.append(job_row(text, m))
        result["tts_rvc_warm_lines"] = rows
        result["tts_rvc_warm_mean"] = {
            k: round(sum(r[k] for r in rows) / len(rows), 3)
            for k in ("tts_s", "vc_s", "total_s", "audio_s", "rtf")
        }

        # ASR clips: plain TTS, no voice conversion, 16 kHz mono float32.
        clips = []
        for text in ASR_SENTENCES:
            final, _ = await job(text, vc=False)
            clips.append((text, load_clip_16k(final.path)))
    finally:
        for p in created:
            p.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass

    # 3 (final step) + 2: Whisper small.en with LLM+TTS+RVC still resident.
    small, small_objs = bench_whisper(WHISPER_MODELS[0], clips, "5_after_whisper_small", steps)
    steps["5_after_whisper_small"] = smi()
    base, base_objs = bench_whisper(WHISPER_MODELS[1], clips, "6_after_whisper_base_also", steps)
    steps["6_after_whisper_base_also"] = smi()
    result["whisper_small_en"] = small
    result["whisper_base_en"] = base
    result["everything_resident_free_mib"] = steps["5_after_whisper_small"]["free_mib"]
    result["everything_resident_note"] = (
        "step 5 = LLM + TTS + RVC worker + whisper small.en; step 6 additionally holds base.en"
    )
    del small_objs, base_objs

    await app.shutdown()
    del torch
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = asyncio.run(main_async(args.out))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
