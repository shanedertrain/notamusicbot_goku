"""Application composition root (skeleton).

Startup order follows REWORK_HANDOFF.md section 19: settings -> validation ->
logging -> speech components -> (later phases: content, Discord). Nothing
expensive is constructed at import time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from notamusicbot.config import AppSettings, SecretsSettings, load_app_settings
from notamusicbot.content.base import ContentProvider
from notamusicbot.domain.speech import AudioProcessingProfile
from notamusicbot.domain.users import UserProfile, UserRegistry, load_users_yaml
from notamusicbot.domain.voice_profiles import load_voice_profiles
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
from notamusicbot.speech.voice_conversion.base import VoiceConverter
from notamusicbot.speech.voice_conversion.router import VoiceConverterRouter

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class Application:
    """Owns all long-lived components; constructed once per process."""

    def __init__(
        self,
        settings: AppSettings,
        secrets: SecretsSettings | None = None,
        *,
        rvc_factory: Callable[[], VoiceConverter] | None = None,
    ) -> None:
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
        # Composition-root dependency injection: tests install a spy/fake RVC
        # adapter factory here instead of the real worker-spawning adapter.
        self.converter: VoiceConverter = VoiceConverterRouter(
            enabled_backends=settings.speech.voice_conversion_backends,
            rvc_factory=rvc_factory,
            rvc_unavailable_policy=settings.speech.rvc_unavailable_policy,
        )
        self.pipeline: SpeechPipeline | None = None
        self._content_provider: ContentProvider | None = None
        self._warmup_task: asyncio.Task[None] | None = None

    @property
    def converter_router(self) -> VoiceConverterRouter:
        """The typed router behind :attr:`converter` (lifecycle entry point)."""
        assert isinstance(self.converter, VoiceConverterRouter)
        return self.converter

    def build_content_provider(self) -> ContentProvider | None:
        """Construct the content provider from settings (lazy; no model load here).

        Returns ``None`` when content generation is disabled; commentary then
        uses the static template (pre-Phase-F behavior).
        """
        if self._content_provider is not None:
            return self._content_provider
        content = self.settings.content
        if not content.enabled:
            return None
        if content.provider == "local_qwen":
            from notamusicbot.content.llm import LocalQwenLlmProvider

            self._content_provider = LocalQwenLlmProvider.from_settings(content)
        else:
            from notamusicbot.content.base import StaticTemplateProvider

            self._content_provider = StaticTemplateProvider()
        return self._content_provider

    def load_content_model(self) -> None:
        """Load the local LLM off the event loop (call during startup)."""
        provider = self.build_content_provider()
        if provider is None:
            return
        from notamusicbot.content.llm import LocalQwenLlmProvider

        if isinstance(provider, LocalQwenLlmProvider):
            provider.load()

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

    async def startup(self) -> None:
        """Application lifecycle entry point (call from the bot's ``setup_hook``).

        Starts the voice-conversion router (spawns + health-checks the RVC
        worker when the ``rvc`` backend is enabled) and schedules the optional
        non-blocking warmup task. RVC start failure is logged, not raised, so
        a missing GPU/worker runtime degrades per the configured
        ``rvc_unavailable_policy`` instead of killing startup.
        """
        await self.converter_router.startup()
        if self.settings.speech.warmup_on_start:
            self._warmup_task = asyncio.create_task(self.warmup(), name="speech-warmup")

    async def shutdown(self) -> None:
        """Application lifecycle teardown (call when the bot closes)."""
        task = self._warmup_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._warmup_task = None
        await self.converter_router.shutdown()

    async def warmup(self) -> None:
        """Best-effort model warmup; failures are logged and never fatal."""
        if self._synthesizer is None:
            return
        try:
            await asyncio.to_thread(
                self._synthesizer.warmup,
                Path(self.settings.speech.output_dir) / "warmup.wav",
            )
        except Exception:
            logger.exception("Speech warmup failed; continuing without a warmed model")

    def run_discord(self) -> None:
        """Startup path for the Discord bot (handoff section 19).

        Model loading and pipeline construction happen in the bot's
        ``setup_hook`` (inside the bot's event loop, off the event-loop thread).
        Heavy work is never done at import time.
        """
        token = self.secrets.discord_token
        if not token:
            logger.error("DISCORD_TOKEN is not set; refusing to start the Discord bot")
            return
        # Imported here so non-Discord tooling does not require discord.py.
        from notamusicbot.discord_bot.bot import build_bot

        bot = build_bot(self)
        bot.run(token)


def main() -> None:
    configure_logging()
    app = Application(load_app_settings())
    try:
        app.run_discord()
    except KeyboardInterrupt:
        logger.info("Shutting down")
