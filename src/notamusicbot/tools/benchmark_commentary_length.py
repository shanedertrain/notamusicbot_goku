"""Benchmark: commentary sentence cap on/off through the real LLM + TTS.

Evidence tool for the commentary-pipeline performance pass. Generates song
commentary twice with the real local LLM (Qwen3-4B) and synthesizes both
texts with the real Qwen clone TTS — once uncapped, once trimmed to
``--max-sentences`` sentences — and reports the timing delta (LLM generation
time is sample-dependent; TTS cost scales with audio duration).

    .venv-main/bin/python -m notamusicbot.tools.benchmark_commentary_length \
        --song "Bohemian Rhapsody" --artist Queen --max-sentences 3

Writes ``evidence/commentary_length_benchmark.json``. Run ALONE: it owns the
GPU for its duration (LLM + TTS are serialized through the shared
coordinator).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = REPO_ROOT / "evidence" / "commentary_length_benchmark.json"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _voice_profile(profile_id: str):
    from notamusicbot.config import load_app_settings
    from notamusicbot.domain.voice_profiles import load_voice_profiles

    settings = load_app_settings()
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles.get(profile_id)
    if profile is None:
        raise SystemExit(f"Unknown voice profile {profile_id!r} (available: {sorted(profiles)})")
    return settings, profile


async def run_once(
    *,
    song: str,
    artist: str,
    requester: str,
    max_sentences: int | None,
    output_dir: Path,
    tag: str,
    profile_id: str = "obama",
) -> dict[str, Any]:
    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.content.trim import trim_sentences
    from notamusicbot.domain.speech import AudioProcessingProfile
    from notamusicbot.domain.voice_profiles import tts_request_for
    from notamusicbot.speech.coordinator import shared_gpu_coordinator
    from notamusicbot.speech.pipeline import SpeechPipeline
    from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

    settings, profile = _voice_profile(profile_id)

    request = ContentRequest(
        kind=ContentKind.SONG,
        song_name=song,
        artist_name=artist,
        requester_name=requester,
        persona_description="an upbeat radio host",
    )
    provider = LocalQwenLlmProvider(gpu_coordinator=shared_gpu_coordinator())
    provider.load()
    llm_start = time.perf_counter()
    raw_text = await provider.generate(request)
    llm_s = time.perf_counter() - llm_start

    text = trim_sentences(raw_text, max_sentences)

    synthesizer = QwenCloneSynthesizer.from_settings(
        model_id=settings.speech.tts_model_id,
        device=settings.speech.tts_device,
        dtype=settings.speech.tts_dtype,
        attn_implementation=settings.speech.tts_attn_implementation,
    )
    pipeline = SpeechPipeline(
        synthesizer,
        AudioProcessingProfile(),
        coordinator=shared_gpu_coordinator(),
        output_dir=output_dir,
        keep_artifacts=True,
    )
    tts_start = time.perf_counter()
    artifact, metrics = await pipeline.run(tts_request_for(profile, text), None)
    tts_s = time.perf_counter() - tts_start
    row = {
        "tag": tag,
        "max_sentences": max_sentences,
        "llm_seconds": round(llm_s, 3),
        "raw_text_chars": len(raw_text),
        "spoken_text": text,
        "spoken_sentences": len(trim_sentences(text, None).split(".")),
        "tts_seconds": round(tts_s, 3),
        "audio_duration_s": round(artifact.duration_s, 3),
        "tts_rtf": round(metrics.tts_rtf, 3),
        "final_wav": str(artifact.path),
    }
    print(
        f"[{tag}] llm={llm_s:.2f}s tts={tts_s:.2f}s audio={artifact.duration_s:.2f}s text={text!r}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song", default="Bohemian Rhapsody")
    parser.add_argument("--artist", default="Queen")
    parser.add_argument("--requester", default="Alice")
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--profile", default="obama", help="Voice profile id to clone")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "var" / "commentary_len_bench"
    )
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args()

    async def _all() -> dict[str, Any]:
        uncapped = await run_once(
            song=args.song,
            artist=args.artist,
            requester=args.requester,
            max_sentences=None,
            output_dir=args.output_dir / "uncapped",
            tag="uncapped",
            profile_id=args.profile,
        )
        capped = await run_once(
            song=args.song,
            artist=args.artist,
            requester=args.requester,
            max_sentences=args.max_sentences,
            output_dir=args.output_dir / "capped",
            tag=f"capped_{args.max_sentences}",
            profile_id=args.profile,
        )
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "commit": git_commit(),
            "gpu": gpu_name(),
            "profile": args.profile,
            "uncapped": uncapped,
            "capped": capped,
        }

    report = asyncio.run(_all())
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.evidence}")


if __name__ == "__main__":
    main()
