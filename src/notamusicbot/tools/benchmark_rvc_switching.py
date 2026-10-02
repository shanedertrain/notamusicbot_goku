"""Benchmark: alternating RVC model profiles through the legacy worker.

Measures the per-switch reload cost (worker restart + model load) when a
profile change forces the persistent worker to swap models, versus a
same-model "warm" conversion. Review Round-2 P1 evidence tool:

    .venv-main/bin/python -m notamusicbot.tools.benchmark_rvc_switching \
        --models Goku Obama --turns 2

Alternation pattern per pass: M1, M2, M1, M2 ... measuring every conversion;
switches (model != worker's loaded model) are reported separately from warm
same-model conversions.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "rvc_cli" / "models"


def make_source_wav(path: Path, seconds: float = 2.0, rate: int = 24000) -> AudioArtifact:
    """Write a deterministic sine-wave input for conversion."""
    n = int(rate * seconds)
    wave = 0.4 * np.sin(2 * np.pi * 220 * np.arange(n) / rate)
    sf.write(path, wave, rate, subtype="PCM_16")
    return AudioArtifact(path=path, sample_rate_hz=rate, duration_s=seconds)


def available_models(candidates: list[str]) -> list[str]:
    ok = []
    for name in candidates:
        d = MODELS_DIR / name
        if d.is_dir() and list(d.glob("*.pth")) and list(d.glob("*.index")):
            ok.append(name)
    return ok


async def run_alternation(
    models: list[str],
    *,
    output_dir: Path,
    turns: int = 1,
    seconds: float = 2.0,
) -> list[dict[str, float | str | bool]]:
    """Alternate conversions and record per-call timings.

    Returns one row per conversion: model, switch (model changed vs the
    worker's previously loaded model), seconds (full convert call).
    """
    converter = RvcVoiceConverter()
    rows: list[dict[str, float | str | bool]] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for turn in range(turns):
            for model in models:
                src = make_source_wav(out_dir / f"src_{turn}_{model}.wav", seconds)
                dst = out_dir / f"out_{turn}_{model}.wav"
                profile = VoiceConversionProfile(backend="rvc", model=model, pitch_shift=0)
                switch = converter.loaded_model != model
                start = time.perf_counter()
                await converter.convert(src, profile, dst)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "turn": turn,
                        "model": model,
                        "switch": switch,
                        "seconds": round(elapsed, 3),
                    }
                )
                src.path.unlink(missing_ok=True)
    finally:
        await converter.shutdown()
    return rows


def summarize(rows: list[dict[str, float | str | bool]]) -> dict[str, float]:
    switch_times = [r["seconds"] for r in rows if r["switch"]]
    warm_times = [r["seconds"] for r in rows if not r["switch"]]
    return {
        "switches": len(switch_times),
        "switch_avg_s": round(sum(switch_times) / len(switch_times), 3) if switch_times else 0.0,
        "warm_converts": len(warm_times),
        "warm_avg_s": round(sum(warm_times) / len(warm_times), 3) if warm_times else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["Goku", "Obama"])
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "var" / "rvc_switch_bench")
    args = parser.parse_args()

    models = available_models(args.models)
    if len(models) < 2:
        raise SystemExit(f"need at least two complete models under {MODELS_DIR}: got {models}")
    rows = asyncio.run(
        run_alternation(models, output_dir=args.output_dir, turns=args.turns, seconds=args.seconds)
    )
    for r in rows:
        print(f"turn={r['turn']} model={r['model']} switch={r['switch']} time={r['seconds']}s")
    stats = summarize(rows)
    print(
        f"summary: {stats['switches']} switches avg {stats['switch_avg_s']}s | "
        f"{stats['warm_converts']} warm converts avg {stats['warm_avg_s']}s"
    )


if __name__ == "__main__":
    main()
