"""Voice benchmark CLI (REWORK_HANDOFF.md sections 9 and 23).

Synthesizes a fixed corpus with candidate Qwen source speakers, passes every
result through the selected VC target (or passthrough with ``--no-vc``), and
writes labeled audition WAVs plus machine-readable timings so a human — not a
metric — picks the best source speaker per target voice.

Example:
    python -m notamusicbot.tools.benchmark_voices \
        --profile goku --speakers Aiden Ryan \
        --corpus tests/fixtures/voice_benchmark.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

from notamusicbot.config import load_app_settings
from notamusicbot.domain.speech import (
    AudioArtifact,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.domain.voice_profiles import load_voice_profiles
from notamusicbot.speech.tts.qwen_custom_voice import CUSTOM_VOICE_SPEAKERS
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter

logger = logging.getLogger(__name__)

DEFAULT_SPEAKERS = ("Aiden", "Ryan")
DEFAULT_CORPUS = Path("tests/fixtures/voice_benchmark.txt")
CSV_COLUMNS = (
    "speaker",
    "line",
    "text",
    "tts_time",
    "vc_time",
    "total_time",
    "duration_s",
    "tts_rtf",
    "vc_rtf",
    "total_rtf",
    "peak_dbfs",
    "error",
)


@dataclass
class BenchmarkRow:
    """One corpus line through one source speaker."""

    speaker: str
    line: int
    text: str
    tts_time: float = 0.0
    vc_time: float = 0.0
    total_time: float = 0.0
    duration_s: float = 0.0
    tts_rtf: float = 0.0
    vc_rtf: float = 0.0
    total_rtf: float = 0.0
    peak_dbfs: float = 0.0
    error: str = ""


def peak_dbfs(path: Path) -> float:
    """Peak level of a WAV in dBFS (0 for silence, positive on clipping)."""
    data, _rate = sf.read(path, dtype="float64", always_2d=True)
    peak = float(abs(data).max()) if data.size else 0.0
    if peak <= 0.0:
        return 0.0
    import math

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
    converter: object,
    speakers: list[str],
    corpus: list[str],
    out_dir: Path,
    *,
    conversion: VoiceConversionProfile,
) -> list[BenchmarkRow]:
    """Run every (speaker, line) pair; keep audition WAVs; record timings.

    ``synthesizer``/``converter`` are the production
    ``SpeechSynthesizer``/``VoiceConverter`` implementations.
    """
    rows: list[BenchmarkRow] = []
    for speaker in speakers:
        source_dir = out_dir / speaker / "source"
        converted_dir = out_dir / speaker / "converted"
        source_dir.mkdir(parents=True, exist_ok=True)
        converted_dir.mkdir(parents=True, exist_ok=True)
        for index, text in enumerate(corpus, start=1):
            row = BenchmarkRow(speaker=speaker, line=index, text=text)
            rows.append(row)
            start = time.perf_counter()
            try:
                request = TtsRequest(text=text, language="English", speaker=speaker)
                source_wav = source_dir / f"line_{index:02d}.wav"
                source_artifact = await synthesizer.synthesize(  # type: ignore[attr-defined]
                    request, source_wav
                )
                row.tts_time = time.perf_counter() - start
                converted_wav = converted_dir / f"line_{index:02d}.wav"
                vc_start = time.perf_counter()
                converted = await converter.convert(  # type: ignore[attr-defined]
                    source_artifact, conversion, converted_wav
                )
                row.vc_time = time.perf_counter() - vc_start
                row.total_time = time.perf_counter() - start
                row.duration_s = converted.duration_s
                if row.duration_s <= 0:
                    row.duration_s = _probe_duration(converted)
                if row.duration_s > 0:
                    row.tts_rtf = row.tts_time / row.duration_s
                    row.vc_rtf = row.vc_time / row.duration_s
                    row.total_rtf = row.total_time / row.duration_s
                row.peak_dbfs = peak_dbfs(converted.path)
            except Exception as exc:  # noqa: BLE001 - one bad line must not kill the run
                row.total_time = time.perf_counter() - start
                row.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Benchmark line failed (speaker=%s, line=%d)", speaker, index)
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
        description="Benchmark Qwen source speakers through a VC target voice.",
    )
    parser.add_argument("--profile", default="goku", help="VC target voice profile id")
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=list(DEFAULT_SPEAKERS),
        choices=CUSTOM_VOICE_SPEAKERS,
        help="Candidate Qwen source speakers (default: Aiden Ryan; up to all nine)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Text file with one line per benchmark utterance",
    )
    parser.add_argument(
        "--no-vc", action="store_true", help="Use passthrough conversion (Qwen only)"
    )
    parser.add_argument("--app-config", default=None, help="Optional app YAML config path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Benchmark root directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def amain(args: argparse.Namespace) -> int:
    from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
    from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

    settings = load_app_settings(Path(args.app_config) if args.app_config else None)
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles.get(args.profile)
    if profile is None:
        logger.error("Unknown voice profile %r (available: %s)", args.profile, sorted(profiles))
        return 2

    corpus = load_corpus(args.corpus)
    if not corpus:
        logger.error("Corpus %s is empty", args.corpus)
        return 2

    model = QwenCustomVoiceSynthesizer.from_settings(
        model_id=settings.speech.tts_model_id,
        device=settings.speech.tts_device,
        dtype=settings.speech.tts_dtype,
        attn_implementation=settings.speech.tts_attn_implementation,
    )
    conversion = VoiceConversionProfile(
        backend="passthrough" if args.no_vc else profile.voice_conversion.backend,
        model=profile.voice_conversion.model,
        pitch_shift=profile.voice_conversion.pitch_shift,
    )
    if args.no_vc or conversion.backend == "passthrough":
        converter: object = PassthroughVoiceConverter()
    else:
        rvc = RvcVoiceConverter()
        await rvc.start()
        converter = rvc

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
        "profile": args.profile,
        "no_vc": args.no_vc,
        "vc_backend": conversion.backend,
        "vc_model": conversion.model,
        "vc_pitch_shift": conversion.pitch_shift,
        "speakers": list(args.speakers),
        "corpus_path": str(args.corpus),
        "corpus_lines": corpus,
        "tts_model_id": settings.speech.tts_model_id,
        "tts_device": device,
        "gpu_name": gpu_name,
        "torch_version": torch_version,
    }

    rows = await run_benchmark(
        model, converter, list(args.speakers), corpus, out_dir, conversion=conversion
    )

    failures = sum(1 for row in rows if row.error)
    metadata["failures"] = failures
    write_results_csv(rows, out_dir / "results.csv")
    write_metadata(metadata, out_dir / "metadata.json")
    logger.info("Benchmark complete: %d rows, %d failures -> %s", len(rows), failures, out_dir)

    if not args.no_vc and isinstance(converter, RvcVoiceConverter):
        await converter.shutdown()
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
