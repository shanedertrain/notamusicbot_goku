"""Application composition root (skeleton).

Startup order follows REWORK_HANDOFF.md section 19: settings -> validation ->
logging -> speech components -> (later phases: content, Discord). Nothing
expensive is constructed at import time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

from notamusicbot.config import AppSettings, SecretsSettings, load_app_settings
from notamusicbot.content.base import ContentProvider
from notamusicbot.domain.speech import AudioArtifact, AudioProcessingProfile, TtsRequest
from notamusicbot.domain.users import UserProfile, UserRegistry, load_users_yaml
from notamusicbot.domain.voice_profiles import load_voice_profiles
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.pipeline import SpeechPipeline
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
from notamusicbot.speech.voice_conversion.base import VoiceConverter, VoiceStreamParams
from notamusicbot.speech.voice_conversion.router import VoiceConverterRouter

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class _LazySpeechSynthesizer:
    """Deferred-loading stand-in for the real synthesizer.

    Installed when no enabled feature needs speech (review Round-2 P1: skip
    expensive component loading). The real Qwen model is only constructed on
    the first actual use; ``warmup`` is never scheduled for it (guarded in
    :meth:`Application.startup`).
    """

    def __init__(self, app: Application) -> None:
        self._app = app
        self._real: QwenCustomVoiceSynthesizer | None = None

    def _realize(self) -> QwenCustomVoiceSynthesizer:
        if self._real is None:
            s = self._app.settings.speech
            logger.info("Lazy speech model load on first use (%s)", s.tts_model_id)
            self._real = QwenCustomVoiceSynthesizer.from_settings(
                model_id=s.tts_model_id,
                device=s.tts_device,
                dtype=s.tts_dtype,
                attn_implementation=s.tts_attn_implementation,
                fast=s.tts_fast,
                stream_chunk_size=s.tts_stream_chunk_size,
                coordinator=self._app.coordinator,
            )
        return self._real

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        return await self._realize().synthesize(request, output_path)

    def warmup(self, output_path: Path) -> AudioArtifact:
        return self._realize().warmup(output_path)

    def __getattr__(self, name: str):
        return getattr(self._realize(), name)


class Application:
    # Live-shutdown hardening: bound the inference drain so Ctrl+C stays
    # responsive even when a TTS clip (~45 s) is in flight.
    SHUTDOWN_DRAIN_TIMEOUT_S = 30.0
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
        self._synthesizer: QwenCustomVoiceSynthesizer | _LazySpeechSynthesizer | None = None
        # Composition-root dependency injection: tests install a spy/fake RVC
        # adapter factory here instead of the real worker-spawning adapter.
        self.converter: VoiceConverter = VoiceConverterRouter(
            enabled_backends=settings.speech.voice_conversion_backends,
            rvc_factory=rvc_factory,
            rvc_unavailable_policy=settings.speech.rvc_unavailable_policy,
            rvc_stream_params=VoiceStreamParams.from_settings(settings.speech.voice_conversion),
        )
        self.pipeline: SpeechPipeline | None = None
        self._content_provider: ContentProvider | None = None
        self._listen_provider: ContentProvider | None = None
        self._warmup_task: asyncio.Task[None] | None = None

    @property
    def converter_router(self) -> VoiceConverterRouter:
        """The typed router behind :attr:`converter` (lifecycle entry point)."""
        assert isinstance(self.converter, VoiceConverterRouter)
        return self.converter

    @property
    def speech_features_enabled(self) -> bool:
        """True when any enabled feature needs the speech pipeline or the LLM.

        Review Round-2 P1: when neither commentary, content, interstitial nor
        podcast is enabled, the expensive local models are not loaded at all.
        Listen mode replies through the same LLM and speech pipeline.
        """
        s = self.settings
        return (
            s.speech.commentary_enabled
            or s.content.enabled
            or s.interstitial.enabled
            or s.podcast.enabled
            or s.listen.enabled
        )

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

            # Review Round-3 P1: hand the LLM the SAME shared GPU coordinator
            # the speech pipeline uses, so LLM generation and TTS/RVC inference
            # are serialized on the one GPU.
            self._content_provider = LocalQwenLlmProvider.from_settings(
                content, gpu_coordinator=self.coordinator
            )
        else:
            from notamusicbot.content.base import StaticTemplateProvider

            self._content_provider = StaticTemplateProvider()
        return self._content_provider

    @property
    def listen_uses_spark(self) -> bool:
        """True when listen mode takes its reply text from the Spark."""
        listen = self.settings.listen
        return listen.enabled and listen.llm.provider == "spark"

    @property
    def listen_content_provider(self) -> ContentProvider | None:
        """The provider listen mode replies through (``None`` unless listen is on)."""
        return self.build_listen_content_provider()

    def _build_local_chat_provider(self) -> Any:
        """The in-process Qwen for listen: shared with music if that exists."""
        from notamusicbot.content.llm import LocalQwenLlmProvider

        shared = self.build_content_provider()
        if isinstance(shared, LocalQwenLlmProvider):
            return shared
        return LocalQwenLlmProvider.from_settings(
            self.settings.content, gpu_coordinator=self.coordinator
        )

    def _build_listen_fallback_provider(self) -> Any:
        """Spark-down fallback: a dedicated Qwen capped at ``listen.reply.max_new_tokens``.

        ``ContentRequest`` has no per-request token limit, so the shared music
        provider (``content.max_new_tokens``, 120) cannot serve voice chat at
        the 60-token cap. A dedicated instance costs a second model load only
        if music commentary also runs on the local model at the same time.
        """
        from notamusicbot.content.llm import LocalQwenLlmProvider

        content = self.settings.content.model_copy(
            update={"max_new_tokens": self.settings.listen.reply.max_new_tokens}
        )
        return LocalQwenLlmProvider.from_settings(content, gpu_coordinator=self.coordinator)

    def build_reply_synthesizer(self, voice_profile: Any) -> Any:
        """Listen-mode ``ReplySynthesizer`` with the configured TTS length caps.

        Needs ``build_pipeline()`` to have run. Job directories of failed calls
        are cleaned under the pipeline's output dir.
        """
        from notamusicbot.listen.reply import ReplySynthesizer

        if self.pipeline is None:
            raise RuntimeError("speech pipeline is not built")
        reply = self.settings.listen.reply
        return ReplySynthesizer(
            self.pipeline,
            voice_profile,
            artifact_root=Path(self.settings.speech.output_dir),
            tts_max_seconds_per_word=reply.tts_max_seconds_per_word,
            tts_cap_floor_seconds=reply.tts_cap_floor_seconds,
            stream_converter=self.converter,
        )

    def build_listen_content_provider(self) -> ContentProvider | None:
        """Construct the listen-mode chat provider (no model load, no network).

        Independent of ``content.enabled``: listen must never be left without an
        LLM because music commentary is off. With ``llm.provider: spark`` the
        in-process model is only a lazy fallback and is never loaded at startup.
        """
        listen = self.settings.listen
        if not listen.enabled:
            return None
        if self._listen_provider is not None:
            return self._listen_provider
        if self.listen_uses_spark:
            from notamusicbot.content.fallback import FallbackChatProvider, LazyLocalProvider
            from notamusicbot.content.spark_chat import SparkChatProvider

            local = None
            if listen.llm.fallback == "local":
                local = LazyLocalProvider(
                    self._build_listen_fallback_provider, gpu_coordinator=self.coordinator
                )
            self._listen_provider = FallbackChatProvider(
                SparkChatProvider(listen.llm, max_new_tokens=listen.reply.max_new_tokens),
                local,
                listen.llm.health_interval_s,
            )
        else:
            self._listen_provider = self._build_local_chat_provider()
        return self._listen_provider

    def load_content_model(self) -> None:
        """Load the local LLM off the event loop (call during startup).

        Fail-soft (review Round-2 P1): any load failure (OOM, corrupted cache,
        missing weights, CUDA error) is logged and a static fallback provider
        is installed instead, so the bot still starts and content requests
        return fallback text. Skipped entirely when no enabled feature needs
        the LLM.
        """
        if not self.speech_features_enabled:
            logger.info("Content/speech features disabled; skipping content model load")
            return
        if self.listen_uses_spark:
            # Listen replies come from the Spark; the in-process model is a lazy
            # fallback (and music commentary loads it on first use if needed).
            logger.info("Listen mode uses the Spark; skipping eager in-process LLM load")
            return
        provider = self.build_content_provider()
        if provider is None:
            return
        from notamusicbot.content.base import StaticTemplateProvider
        from notamusicbot.content.llm import LocalQwenLlmProvider

        if isinstance(provider, LocalQwenLlmProvider):
            try:
                provider.load()
            except Exception:
                logger.exception(
                    "Local content model failed to load; installing static fallback provider"
                )
                self._content_provider = StaticTemplateProvider()

    def load_speech_model(self) -> QwenCustomVoiceSynthesizer | _LazySpeechSynthesizer:
        """Load the persistent Qwen model once (call during startup, not import).

        When no enabled feature needs speech, a lazy proxy is installed that
        only loads the real model on first use (review Round-2 P1: skip
        expensive components).
        """
        if self._synthesizer is None:
            if not self.speech_features_enabled:
                self._synthesizer = _LazySpeechSynthesizer(self)
                logger.info("Speech features disabled; TTS model load deferred to first use")
            else:
                s = self.settings.speech
                self._synthesizer = QwenCustomVoiceSynthesizer.from_settings(
                    model_id=s.tts_model_id,
                    device=s.tts_device,
                    dtype=s.tts_dtype,
                    attn_implementation=s.tts_attn_implementation,
                    fast=s.tts_fast,
                    stream_chunk_size=s.tts_stream_chunk_size,
                    coordinator=self.coordinator,
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
            keep_artifacts=s.keep_artifacts,
            failed_job_retention_hours=s.failed_job_retention_hours,
            f0_method_override=s.voice_conversion.f0_method_override,
            quality=s.voice_conversion,
        )
        return self.pipeline

    async def startup(self) -> None:
        """Application lifecycle entry point (call from the bot's ``setup_hook``).

        Review Round-4 P1 (startup ordering): the content LLM model is loaded
        to completion BEFORE the voice-conversion router starts (RVC worker
        spawn + health check + model load) and before the warmup task is
        scheduled. LLM weight loading is GPU/VRAM-heavy and is NOT itself
        arbitrated through the coordinator (it happens once, outside any
        slot), so with ``warmup_on_start: true`` the warmup or RVC startup
        could otherwise acquire the GPU while the LLM weights are still
        streaming onto it. Reordering makes load → RVC start → warmup
        strictly sequential; the startup-order test
        (``test_startup_load_and_warmup_do_not_overlap``) pins this.

        RVC start failure is logged, not raised, so a missing GPU/worker
        runtime degrades per the configured ``rvc_unavailable_policy`` instead
        of killing startup.
        """
        # Content LLM loads first (fail-soft, off the event loop).
        await asyncio.to_thread(self.load_content_model)
        await self.converter_router.startup()
        # Warmup only when a feature will actually use the synthesizer and the
        # setting asks for it; scheduled as a task so it never blocks startup.
        # At most one warmup task is scheduled per startup cycle.
        if (
            self.settings.speech.warmup_on_start
            and self.speech_features_enabled
            and self._warmup_task is None
        ):
            self._warmup_task = asyncio.create_task(self.warmup(), name="speech-warmup")

    async def shutdown(self) -> None:
        """Application lifecycle teardown (call when the bot closes).

        Review Round-6 §3/B inference shutdown order:

        1. cancel the warmup task (its detached GPU holder is retired by the
           drain below);
        2. ``coordinator.close()`` — stop accepting new inference submissions
           (CoordinatorClosedError), cancel queued-not-started jobs, and AWAIT
           already-started detached GPU jobs to completion;
        3. ``converter_router.shutdown()`` — only now is the RVC worker torn
           down, and since no pipeline job can be running, no late
           ``convert()`` can lazily restart it (the router also refuses
           converts with RouterClosedError as defense in depth);
        4. remaining resources (none yet at this layer).
        """
        task = self._warmup_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._warmup_task = None
        # Live-shutdown hardening (user report: Ctrl+C looked hung): the drain
        # waits for in-flight GPU work (a TTS clip can run ~45 s). Bound it so
        # interactive shutdown stays responsive; the coordinator is still
        # closed (flag set first inside close), we just stop waiting for
        # stragglers before tearing down the backends.
        try:
            await asyncio.wait_for(self.coordinator.close(), timeout=self.SHUTDOWN_DRAIN_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "Inference drain timed out after %ss; proceeding with backend teardown",
                self.SHUTDOWN_DRAIN_TIMEOUT_S,
            )
        # Use the backing attribute: tests inject protocol-level converter
        # spies that satisfy shutdown() but are not VoiceConverterRouter.
        await self.converter.shutdown()

    async def warmup(self) -> None:
        """Best-effort model warmup; failures are logged and never fatal."""
        if self._synthesizer is None:
            return
        try:
            # Review Round-3 P1 startup case: TTS warmup is GPU inference too,
            # so it arbitrates through the same shared coordinator as LLM
            # generation and pipeline jobs (never overlaps them on the GPU).
            # Review Round-4 P0: run_blocking (not slot + to_thread) so a
            # cancelled warmup task cannot release the GPU lease while the
            # warmup synthesis thread is still running.
            await self.coordinator.run_blocking(
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
    # Live-shutdown hardening (user report): a second Ctrl+C during the
    # bounded close means the user wants OUT now. Force-exit instead of
    # unwinding through teardown that may itself block; the RVC worker is
    # reaped by its atexit hook either way.
    _sigint_count = 0

    def _on_sigint(signum: int, frame: Any) -> None:
        nonlocal _sigint_count
        _sigint_count += 1
        if _sigint_count >= 2:
            logger.warning("Second Ctrl+C received; forcing exit")
            os._exit(130)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        app.run_discord()
    except KeyboardInterrupt:
        logger.info("Shutting down (press Ctrl+C again to force exit)")
