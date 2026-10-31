"""Real-GPU smoke test: end-to-end Qwen3-TTS clone speech pipeline.

Loads the production clone synthesizer and speech pipeline from
``config/app.yaml`` (offline, local HF snapshot), runs one Obama-profile
request through the full TTS -> postprocess chain, and prints timings, the
output WAV info and ``nvidia-smi`` VRAM before/after.

    .venv-main/bin/python tools/smoke_clone_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "var" / "clone_smoke"
TEXT = "Look, let me be clear. That last track was something else, and I mean that."


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


async def main() -> int:
    from notamusicbot.config import load_app_settings
    from notamusicbot.domain.speech import AudioProcessingProfile
    from notamusicbot.domain.voice_profiles import load_voice_profiles, tts_request_for
    from notamusicbot.speech.coordinator import InferenceCoordinator
    from notamusicbot.speech.pipeline import SpeechPipeline
    from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

    settings = load_app_settings()
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles["obama"]
    processing = AudioProcessingProfile(
        target_sample_rate_hz=settings.speech.sample_rate_hz,
        channels=settings.speech.channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vram_before = smi()
    print(f"[smoke] GPU before load: {json.dumps(vram_before)}", flush=True)

    print(f"[smoke] loading {settings.speech.tts_model_id} (offline snapshot)", flush=True)
    t0 = time.perf_counter()
    synthesizer = QwenCloneSynthesizer.from_settings(
        model_id=settings.speech.tts_model_id,
        device=settings.speech.tts_device,
        dtype=settings.speech.tts_dtype,
        attn_implementation=settings.speech.tts_attn_implementation,
    )
    load_s = time.perf_counter() - t0
    vram_loaded = smi()

    pipeline = SpeechPipeline(
        synthesizer,
        processing,
        coordinator=InferenceCoordinator(max_concurrent=1, queue_size=4),
        output_dir=OUT_DIR / "jobs",
        keep_artifacts=True,
    )
    request = tts_request_for(profile, TEXT)
    print(
        f"[smoke] request: profile={profile.id} reference={request.reference_audio} "
        f"mode={request.clone_mode}",
        flush=True,
    )
    t0 = time.perf_counter()
    artifact, metrics = await pipeline.run(request, processing)
    pipeline_s = time.perf_counter() - t0
    vram_after = smi()
    info = sf.info(artifact.path)
    print(f"[smoke] model_load_s={load_s:.2f}", flush=True)
    print(f"[smoke] tts_s={metrics.tts_s:.2f} postprocess_s={metrics.postprocess_s:.2f}", flush=True)
    print(f"[smoke] pipeline_s={pipeline_s:.2f} (metrics total={metrics.total_s:.2f})", flush=True)
    print(
        f"[smoke] output={artifact.path} samplerate={info.samplerate} "
        f"channels={info.channels} frames={info.frames} duration_s={info.frames / info.samplerate:.2f}",
        flush=True,
    )
    print(f"[smoke] VRAM after load: {json.dumps(vram_loaded)}", flush=True)
    print(f"[smoke] VRAM after job:  {json.dumps(vram_after)}", flush=True)

    await synthesizer.shutdown()
    vram_shutdown = smi()
    print(f"[smoke] VRAM after synthesizer shutdown: {json.dumps(vram_shutdown)}", flush=True)

    headroom_mib = vram_after["total_mib"] - vram_after["used_mib"]
    peak_over_baseline = vram_after["used_mib"] - vram_before["used_mib"]
    summary = {
        "profile": profile.id,
        "text": TEXT,
        "model_id": settings.speech.tts_model_id,
        "model_load_s": round(load_s, 2),
        "tts_s": round(metrics.tts_s, 2),
        "postprocess_s": round(metrics.postprocess_s, 2),
        "pipeline_s": round(pipeline_s, 2),
        "output": {
            "path": str(artifact.path),
            "sample_rate_hz": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_s": round(info.frames / info.samplerate, 2),
        },
        "vram": {
            "before": vram_before,
            "after_load": vram_loaded,
            "after_job": vram_after,
            "after_shutdown": vram_shutdown,
            "peak_over_baseline_mib": peak_over_baseline,
            "headroom_mib": headroom_mib,
        },
    }
    (OUT_DIR / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[smoke] wrote {OUT_DIR / 'smoke.json'}", flush=True)
    # Sanity: the job produced audio at the configured output format, and VRAM
    # stayed inside the 16 GB budget.
    assert info.frames > 0, "clone TTS produced no audio"
    assert info.samplerate == settings.speech.sample_rate_hz, (
        f"unexpected output rate {info.samplerate} Hz"
    )
    assert headroom_mib > 512, f"VRAM headroom too small: {headroom_mib} MiB"
    print("[smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
