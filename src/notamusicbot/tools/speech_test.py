"""Local speech test CLI.

Initializes the same production speech pipeline components (no duplicated
logic) and generates one line through clone TTS -> postprocess.

Example:
    python -m notamusicbot.tools.speech_test --profile obama \
        --text "Hey, it's me, Goku!" --output out/goku.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from notamusicbot.config import load_app_settings
from notamusicbot.domain.speech import AudioProcessingProfile
from notamusicbot.domain.voice_profiles import (
    VoiceProfileError,
    load_voice_profiles,
    tts_request_for,
)
from notamusicbot.speech.pipeline import SpeechPipeline

if TYPE_CHECKING:
    from notamusicbot.config import AppSettings
    from notamusicbot.speech.tts.base import SpeechSynthesizer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m notamusicbot.tools.speech_test",
        description="Generate one line through the production speech pipeline.",
    )
    parser.add_argument("--profile", default="goku", help="Voice profile id from config/voices/")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default=None, help="Override the profile's language")
    parser.add_argument("--output", default=None, help="Final WAV output path")
    parser.add_argument("--app-config", default=None, help="Optional app YAML config path")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_app_settings(Path(args.app_config) if args.app_config else None)
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles.get(args.profile)
    if profile is None:
        logger.error("Unknown voice profile %r (available: %s)", args.profile, sorted(profiles))
        return 2

    if args.language:
        profile.tts.language = args.language
    request = tts_request_for(profile, args.text)
    processing = AudioProcessingProfile(
        target_sample_rate_hz=settings.speech.sample_rate_hz,
        channels=settings.speech.channels,
        speed=profile.audio.speed,
        target_peak_dbfs=profile.audio.target_peak_dbfs,
    )

    output_dir = Path(settings.speech.output_dir)
    pipeline = SpeechPipeline(
        synthesizer=_build_synthesizer(settings),
        processing=processing,
        output_dir=output_dir,
    )
    final, metrics = asyncio.run(pipeline.run(request, processing))
    if args.output:
        final.path.replace(Path(args.output))
    logger.info(
        "Generated %s (duration=%.2fs total=%.2fs rtf=%.2f)",
        args.output or final.path,
        metrics.output_duration_s,
        metrics.total_s,
        metrics.total_rtf,
    )
    return 0


def _build_synthesizer(settings: AppSettings) -> SpeechSynthesizer:
    """Load the persistent Qwen clone model through the production path."""
    from notamusicbot.app import Application

    return Application(settings).load_speech_model()


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (VoiceProfileError, FileNotFoundError) as exc:
        logger.error("speech_test failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
