"""Benchmark: RVC f0 extraction method A/B (harvest vs pm vs crepe).

Evidence tool for the commentary-pipeline performance pass. The shipped model
YAMLs set ``f0_method: harvest`` — the slow CPU pitch extractor (the user's
real run: RVC RTF 1.0). This tool pushes the SAME input through the real RVC
worker once per f0 method (via the production adapter + IPC path, exercising
the ``convert`` ``f0_method`` override) and records wall-clock conversion
times, so the speed ratio can be compared and the outputs listened to.

    .venv-main/bin/python -m notamusicbot.tools.benchmark_f0 \
        --methods harvest pm crepe \
        --input rvc_cli/input/hello.wav --model Obama

Writes ``evidence/f0_benchmark.json`` (timestamp, commit, GPU, per-method
timings) and keeps every converted WAV for subjective quality comparison.
Run ALONE: it owns the GPU/RVC worker for its duration.
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

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = REPO_ROOT / "evidence" / "f0_benchmark.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "var" / "f0_benchmark"


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


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


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate if info.samplerate else 0.0


async def run_benchmark(
    methods: list[str],
    *,
    input_wav: Path,
    model: str,
    variant: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    if not input_wav.is_file():
        raise SystemExit(f"input not found: {input_wav}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = AudioArtifact(
        path=input_wav,
        sample_rate_hz=0,  # probed from the output; input format handled by the worker
        duration_s=audio_duration(input_wav),
    )
    converter = RvcVoiceConverter()
    results: list[dict[str, Any]] = []
    try:
        for method in methods:
            dst = output_dir / f"out_{model.lower()}_{method}.wav"
            profile = VoiceConversionProfile(
                backend="rvc",
                model=model,
                variant=variant,
                pitch_shift=0,
                f0_method=method,
            )
            start = time.perf_counter()
            artifact = await converter.convert(source, profile, dst)
            wall = time.perf_counter() - start
            duration = artifact.duration_s or audio_duration(dst)
            results.append(
                {
                    "method": method,
                    "wall_seconds": round(wall, 3),
                    "output_wav": str(dst),
                    "output_duration_s": round(duration, 3),
                    "rtf": round(wall / duration, 3) if duration > 0 else None,
                }
            )
            print(
                f"f0_method={method:>10s}  wall={wall:6.2f}s  "
                f"audio={duration:5.2f}s  rtf={wall / duration if duration else 0:.2f}"
            )
    finally:
        await converter.shutdown()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "commit": git_commit(),
        "gpu": gpu_name(),
        "input_wav": str(input_wav),
        "input_duration_s": round(source.duration_s, 3),
        "model": model,
        "variant": variant,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["harvest", "pm", "crepe"])
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "rvc_cli" / "input" / "hello.wav")
    parser.add_argument("--model", default="Obama")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args()

    report = asyncio.run(
        run_benchmark(
            args.methods,
            input_wav=args.input,
            model=args.model,
            variant=args.variant,
            output_dir=args.output_dir,
        )
    )
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.evidence}")


if __name__ == "__main__":
    main()
