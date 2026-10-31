"""Voice benchmark CLI: clone voices through the production TTS path.

Synthesizes a fixed corpus with each selected voice profile (Qwen3-TTS Base
zero-shot cloning) and writes labeled audition WAVs plus machine-readable
timings so a human — not a metric — picks the best voice.

Example:
    python -m notamusicbot.tools.benchmark_voices \
        --profiles obama goku \
        --corpus tests/fixtures/voice_benchmark.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from notamusicbot.config import load_app_settings
from notamusicbot.domain.speech import AudioArtifact, TtsRequest
from notamusicbot.domain.voice_profiles import load_voice_profiles, tts_request_for

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path("tests/fixtures/voice_benchmark.txt")
CSV_COLUMNS = (
    "profile",
    "line",
    "text",
    "tts_time",
    "total_time",
    "duration_s",
    "tts_rtf",
    "total_rtf",
    "peak_dbfs",
    "error",
)


@dataclass
class BenchmarkRow:
    """One corpus line through one voice profile."""

    profile: str
    line: int
    text: str
    tts_time: float = 0.0
    total_time: float = 0.0
    duration_s: float = 0.0
    tts_rtf: float = 0.0
    total_rtf: float = 0.0
    peak_dbfs: float = 0.0
    error: str = ""


def peak_dbfs(path: Path) -> float:
    """Peak level of a WAV in dBFS (0 for silence, positive on clipping)."""
    data, _rate = sf.read(path, dtype="float64", always_2d=True)
    peak = float(abs(data).max()) if data.size else 0.0
    if peak <= 0.0:
        return 0.0
    return 20.0 * math.log10(peak)


def load_corpus(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def write_results_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            writer.writerow({column: data[column] for column in CSV_COLUMNS})


def write_metadata(
    metadata: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


async def run_benchmark(
    synthesizer: object,
    targets: dict[str, TtsRequest],
    corpus: list[str],
    out_dir: Path,
) -> list[BenchmarkRow]:
    """Run every (profile, line) pair; keep audition WAVs; record timings.

    ``synthesizer`` is the production ``SpeechSynthesizer`` implementation;
    ``targets`` maps a profile id to a prototype clone request whose text is
    replaced per corpus line.
    """
    rows: list[BenchmarkRow] = []
    for profile_id, prototype in targets.items():
        target_dir = out_dir / profile_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, text in enumerate(corpus, start=1):
            row = BenchmarkRow(profile=profile_id, line=index, text=text)
            rows.append(row)
            start = time.perf_counter()
            try:
                request = replace(prototype, text=text)
                wav_path = target_dir / f"line_{index:02d}.wav"
                artifact: AudioArtifact = await synthesizer.synthesize(  # type: ignore[attr-defined]
                    request, wav_path
                )
                row.tts_time = time.perf_counter() - start
                row.total_time = row.tts_time
                row.duration_s = artifact.duration_s
                if row.duration_s <= 0:
                    row.duration_s = _probe_duration(artifact)
                if row.duration_s > 0:
                    row.tts_rtf = row.tts_time / row.duration_s
                    row.total_rtf = row.total_time / row.duration_s
                row.peak_dbfs = peak_dbfs(artifact.path)
            except Exception as exc:  # noqa: BLE001 - one bad line must not kill the run
                row.total_time = time.perf_counter() - start
                row.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Benchmark line failed (profile=%s, line=%d)", profile_id, index)
    return rows


def _probe_duration(artifact: AudioArtifact) -> float:
    try:
        info = sf.info(artifact.path)
        return info.frames / info.samplerate if info.samplerate else 0.0
    except Exception:  # noqa: BLE001 - probing must never raise through the CLI
        return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m notamusicbot.tools.benchmark_voices",
        description="Benchmark voice profiles through the production clone TTS path.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["obama"],
        help="Voice profile ids from config/voices/ (default: obama)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Text file with one line per benchmark utterance",
    )
    parser.add_argument("--app-config", default=None, help="Optional app YAML config path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Benchmark root directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def amain(args: argparse.Namespace) -> int:
    from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

    settings = load_app_settings(Path(args.app_config) if args.app_config else None)
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    selected: dict[str, Any] = {}
    for profile_id in args.profiles:
        profile = profiles.get(profile_id)
        if profile is None:
            logger.error("Unknown voice profile %r (available: %s)", profile_id, sorted(profiles))
            return 2
        selected[profile_id] = profile

    corpus = load_corpus(args.corpus)
    if not corpus:
        logger.error("Corpus %s is empty", args.corpus)
        return 2

    synthesizer = QwenCloneSynthesizer.from_settings(
        model_id=settings.speech.tts_model_id,
        device=settings.speech.tts_device,
        dtype=settings.speech.tts_dtype,
        attn_implementation=settings.speech.tts_attn_implementation,
    )
    targets = {
        profile_id: tts_request_for(profile, corpus[0]) for profile_id, profile in selected.items()
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or Path("benchmark") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        device = settings.speech.tts_device
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
    except Exception:  # noqa: BLE001 - metadata must not depend on torch
        device, gpu_name, torch_version = settings.speech.tts_device, None, None

    metadata: dict[str, object] = {
        "started_at_utc": timestamp,
        "profiles": sorted(selected),
        "corpus_path": str(args.corpus),
        "corpus_lines": corpus,
        "tts_model_id": settings.speech.tts_model_id,
        "tts_device": device,
        "gpu_name": gpu_name,
        "torch_version": torch_version,
    }

    try:
        rows = await run_benchmark(synthesizer, targets, corpus, out_dir)
    finally:
        await synthesizer.shutdown()

    failures = sum(1 for row in rows if row.error)
    metadata["failures"] = failures
    write_results_csv(rows, out_dir / "results.csv")
    write_metadata(metadata, out_dir / "metadata.json")
    logger.info("Benchmark complete: %d rows, %d failures -> %s", len(rows), failures, out_dir)
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.DEBUG if (argv and "-v" in argv) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
