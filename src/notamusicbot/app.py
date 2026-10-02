"""Application composition root (skeleton).

Startup order follows REWORK_HANDOFF.md section 19: settings -> validation ->
logging -> speech components -> (later phases: content, Discord). Nothing
expensive is constructed at import time.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from notamusicbot.config import AppSettings, SecretsSettings, load_app_settings
from notamusicbot.domain.speech import AudioProcessingProfile
from notamusicbot.domain.users import UserProfile, UserRegistry, load_users_yaml
from notamusicbot.domain.voice_profiles import load_voice_profiles
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
from notamusicbot.speech.voice_conversion.base import VoiceConverter
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class Application:
    """Owns all long-lived components; constructed once per process."""

    def __init__(self, settings: AppSettings, secrets: SecretsSettings | None = None) -> None:
        self.settings = settings
        self.secrets = secrets or SecretsSettings()
        self.voice_profiles = load_voice_profiles(Path(settings.speech.voices_dir))
        users: dict[str, UserProfile] = {}
        users_path = Path(settings.speech.users_config)
        if users_path.is_file():
            users = load_users_yaml(users_path)
        self.users = UserRegistry(users)
        self.coordinator = InferenceCoordinator(
            max_concurrent=settings.speech.max_concurrent_inference,
            queue_size=settings.speech.queue_size,
        )
        self._synthesizer: QwenCustomVoiceSynthesizer | None = None
        self.converter: VoiceConverter = PassthroughVoiceConverter()
        self.pipeline: SpeechPipeline | None = None

    def load_speech_model(self) -> QwenCustomVoiceSynthesizer:
        """Load the persistent Qwen model once (call during startup, not import)."""
        if self._synthesizer is None:
            s = self.settings.speech
            self._synthesizer = QwenCustomVoiceSynthesizer.from_settings(
                model_id=s.tts_model_id,
                device=s.tts_device,
                dtype=s.tts_dtype,
                attn_implementation=s.tts_attn_implementation,
            )
        return self._synthesizer

    def build_pipeline(self) -> SpeechPipeline:
        s = self.settings.speech
        if self._synthesizer is None:
            self.load_speech_model()
        assert self._synthesizer is not None
        processing = AudioProcessingProfile(
            target_sample_rate_hz=s.sample_rate_hz,
            channels=s.channels,
            speed=s.default_speed,
            target_peak_dbfs=s.default_target_peak_dbfs,
        )
        self.pipeline = SpeechPipeline(
            self._synthesizer,
            self.converter,
            processing,
            coordinator=self.coordinator,
            output_dir=Path(s.output_dir),
        )
        return self.pipeline

    async def warmup(self) -> None:
        if self.settings.speech.warmup_on_start and self._synthesizer is not None:
            self._synthesizer.warmup(Path(self.settings.speech.output_dir) / "warmup.wav")

    async def run(self) -> None:
        """Main event loop entry point; Discord wiring arrives in Phase E."""
        self.load_speech_model()
        self.build_pipeline()
        await self.warmup()
        logger.info("Speech stack initialized; Discord layer pending (Phase E)")
        await asyncio.Event().wait()  # run until cancelled (placeholder)


def main() -> None:
    configure_logging()
    app = Application(load_app_settings())
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Shutting down")
