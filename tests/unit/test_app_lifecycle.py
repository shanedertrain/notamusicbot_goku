"""Round-2 review fixes: fail-soft content load, skip-disabled loading,
keep_artifacts wiring, and warmup-once/non-blocking semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import FakeSynthesizer
from notamusicbot.app import Application, _LazySpeechSynthesizer
from notamusicbot.config import AppSettings, ContentSettings, SpeechSettings
from notamusicbot.content.base import ContentKind, ContentRequest, StaticTemplateProvider


def _app(**kwargs) -> Application:
    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            **kwargs,
        )
    )
    return Application(settings)


def _disabled_both(**speech_kwargs) -> AppSettings:
    return AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            commentary_enabled=False,
            **speech_kwargs,
        ),
        content=ContentSettings(enabled=False),
    )


class _CountingSynthesizer:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    @classmethod
    def from_settings(cls, **kwargs) -> _CountingSynthesizer:
        return cls(**kwargs)

    async def synthesize(self, request, output_path):  # pragma: no cover - not used here
        raise AssertionError

    def warmup(self, output_path):  # pragma: no cover - not used here
        raise AssertionError


class _ExplodingLlm:
    instances = 0
    loads = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).instances += 1

    @classmethod
    def from_settings(cls, settings, **kwargs) -> _ExplodingLlm:
        return cls(settings)

    def is_loaded(self) -> bool:
        return False

    def load(self) -> None:
        type(self).loads += 1
        raise RuntimeError("CUDA OOM while loading Qwen3-4B")

    async def generate(self, request) -> str:  # pragma: no cover - not used here
        raise AssertionError


class _OkLlm(_ExplodingLlm):
    loads = 0

    def load(self) -> None:
        type(self).loads += 1


# --------------------------------------------------------------------- P1 fail-soft


def test_content_load_failure_installs_static_fallback(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _ExplodingLlm)
    app = _app(commentary_enabled=True)
    settings = AppSettings(content=ContentSettings(enabled=True, provider="local_qwen"))
    app.settings = settings

    # Must not raise: startup continues with fallback text available.
    app.load_content_model()

    provider = app.build_content_provider()
    assert isinstance(provider, StaticTemplateProvider)
    text = asyncio.run(
        provider.generate(
            ContentRequest(kind=ContentKind.SONG, song_name="Song", requester_name="Req")
        )
    )
    assert text  # deterministic static commentary
    assert _ExplodingLlm.loads == 1


def test_content_load_failure_logs_and_app_stays_usable(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _ExplodingLlm)
    app = _app()
    app.settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
        ),
        content=ContentSettings(enabled=True),
    )
    app.load_content_model()
    # Usable state: pipeline can still be composed without the LLM.
    app.settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
        ),
        content=ContentSettings(enabled=False),
    )
    app.build_content_provider()  # cached fallback provider returned
    assert isinstance(app.build_content_provider(), StaticTemplateProvider)


def test_successful_content_load_keeps_llm_provider(monkeypatch) -> None:
    class OkLlm(_OkLlm):
        pass

    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", OkLlm)
    app = _app()
    app.settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
        ),
        content=ContentSettings(enabled=True),
    )
    app.load_content_model()
    assert OkLlm.loads == 1
    assert not isinstance(app.build_content_provider(), StaticTemplateProvider)


# ------------------------------------------------------------- P1 skip-disabled


def test_no_llm_or_tts_construction_when_all_disabled(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.app.QwenCustomVoiceSynthesizer", _CountingSynthesizer)
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _ExplodingLlm)
    _CountingSynthesizer.instances = 0
    _ExplodingLlm.instances = 0

    app = Application(_disabled_both())
    app.load_content_model()
    app.load_speech_model()
    app.build_pipeline()

    assert _CountingSynthesizer.instances == 0  # TTS never constructed
    assert _ExplodingLlm.instances == 0  # LLM never constructed
    assert isinstance(app.load_speech_model(), _LazySpeechSynthesizer)
    assert app.speech_features_enabled is False


def test_listen_alone_enables_speech_features() -> None:
    from notamusicbot.config import ListenSettings

    settings = _disabled_both()
    assert Application(settings).speech_features_enabled is False
    settings.listen = ListenSettings(enabled=True)
    assert Application(settings).speech_features_enabled is True


def test_no_warmup_scheduled_when_all_disabled() -> None:
    async def main() -> None:
        app = Application(_disabled_both(warmup_on_start=True))
        await app.startup()
        assert app._warmup_task is None  # never scheduled
        await app.shutdown()

    asyncio.run(main())


def test_tts_loaded_but_no_llm_when_only_commentary_enabled(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.app.QwenCustomVoiceSynthesizer", _CountingSynthesizer)
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _ExplodingLlm)
    _CountingSynthesizer.instances = 0
    _ExplodingLlm.instances = 0

    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            commentary_enabled=True,
        ),
        content=ContentSettings(enabled=False),
    )
    app = Application(settings)
    app.load_content_model()
    app.load_speech_model()
    assert _CountingSynthesizer.instances == 1  # TTS loaded eagerly
    assert _ExplodingLlm.instances == 0  # no LLM needed for static commentary


# ------------------------------------------------------------- P2 keep_artifacts


def test_build_pipeline_passes_keep_artifacts_and_retention(tmp_path) -> None:
    app = _app(
        commentary_enabled=True,
        keep_artifacts=True,
        failed_job_retention_hours=2.5,
        output_dir=tmp_path / "speech",
    )
    app.settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=Path("tests/fixtures"),
            users_config=Path("config/does-not-exist.yaml"),
            commentary_enabled=True,
            keep_artifacts=True,
            failed_job_retention_hours=2.5,
            output_dir=tmp_path / "speech",
        )
    )
    # Review Round-3 P1: unit tests must never construct the real Qwen
    # synthesizer (which imports qwen_tts); inject the fake boundary.
    app._synthesizer = FakeSynthesizer()
    pipeline = app.build_pipeline()
    assert pipeline._keep_artifacts is True
    assert pipeline._failed_job_retention_hours == pytest.approx(2.5)


def test_build_pipeline_defaults_keep_artifacts_off(tmp_path) -> None:
    app = _app(commentary_enabled=True, output_dir=tmp_path / "speech")
    app._synthesizer = FakeSynthesizer()  # never construct the real Qwen backend
    pipeline = app.build_pipeline()
    assert pipeline._keep_artifacts is False


# ------------------------------------------------------------------ P2 warmup


def test_warmup_scheduled_exactly_once_and_non_blocking(monkeypatch) -> None:
    import threading

    warmup_calls: list[Path] = []
    release = threading.Event()

    class SlowSynth:
        def __init__(self) -> None:
            self.calls = 0

        def warmup(self, output_path: Path):
            self.calls += 1
            warmup_calls.append(output_path)
            # Block until the test observes startup already returned.
            release.wait(timeout=5)
            return None

    async def main() -> None:
        app = _app(commentary_enabled=True, warmup_on_start=True)
        synth = SlowSynth()
        app._synthesizer = synth
        await app.startup()
        # Non-blocking: startup returned while warmup is still parked on the
        # release event (startup itself never waited on the synthesis).
        for _ in range(50):  # wait until the worker thread entered warmup
            if synth.calls:
                break
            await asyncio.sleep(0.01)
        assert app._warmup_task is not None
        assert not app._warmup_task.done()  # parked: proves non-blocking startup
        release.set()
        await asyncio.wait_for(app._warmup_task, timeout=5)
        assert len(warmup_calls) == 1
        await app.shutdown()
        assert synth.calls == 1  # shutdown does not re-run warmup

    asyncio.run(main())


def test_warmup_scheduled_only_once_across_double_startup() -> None:
    class FastSynth:
        calls = 0

        def warmup(self, output_path: Path):
            type(self).calls += 1

    async def main() -> None:
        app = _app(commentary_enabled=True, warmup_on_start=True)
        app._synthesizer = FastSynth()
        await app.startup()
        task = app._warmup_task
        await app.startup()  # second call must not schedule another warmup
        assert app._warmup_task is task
        await asyncio.wait_for(task, timeout=5)
        await app.shutdown()
        assert FastSynth.calls == 1

    asyncio.run(main())


# ------------------------------------------------- listen mode: Spark chat provider


def _listen_settings(*, content_enabled: bool, **llm) -> AppSettings:
    from notamusicbot.config import ListenLlmSettings, ListenSettings

    settings = _disabled_both()
    settings.content = ContentSettings(enabled=content_enabled)
    settings.listen = ListenSettings(enabled=True, llm=ListenLlmSettings(**llm))
    return settings


def test_listen_spark_builds_fallback_provider_without_loading_llm(monkeypatch) -> None:
    from notamusicbot.content.fallback import FallbackChatProvider, LazyLocalProvider
    from notamusicbot.content.spark_chat import SparkChatProvider

    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _ExplodingLlm)
    _ExplodingLlm.instances = 0
    _ExplodingLlm.loads = 0
    app = Application(_listen_settings(content_enabled=True))
    app.load_content_model()  # content.enabled, but listen takes its text from the Spark
    provider = app.listen_content_provider
    assert isinstance(provider, FallbackChatProvider)
    assert isinstance(provider.primary, SparkChatProvider)
    assert isinstance(provider.fallback, LazyLocalProvider)
    assert app.listen_content_provider is provider  # cached
    assert (_ExplodingLlm.instances, _ExplodingLlm.loads) == (0, 0)
    assert provider.fallback.loaded is False


def test_listen_spark_with_content_disabled_still_has_an_llm() -> None:
    from notamusicbot.content.fallback import FallbackChatProvider

    app = Application(_listen_settings(content_enabled=False))
    assert app.build_content_provider() is None  # music commentary stays static
    assert isinstance(app.listen_content_provider, FallbackChatProvider)


def test_listen_fallback_none_has_no_local_provider() -> None:
    app = Application(_listen_settings(content_enabled=False, fallback="none"))
    assert app.listen_content_provider.fallback is None


def test_listen_provider_is_none_when_listen_disabled() -> None:
    app = Application(_disabled_both())
    assert app.listen_content_provider is None
    assert app.listen_uses_spark is False


def test_listen_local_provider_uses_the_in_process_llm(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _OkLlm)
    _OkLlm.loads = 0
    app = Application(_listen_settings(content_enabled=False, provider="local"))
    provider = app.listen_content_provider
    assert isinstance(provider, _OkLlm)
    assert app.listen_uses_spark is False


def test_listen_spark_fallback_factory_shares_the_music_llm(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _OkLlm)
    app = Application(_listen_settings(content_enabled=True))
    music = app.build_content_provider()
    assert isinstance(music, _OkLlm)
    assert app._build_local_chat_provider() is music  # one model, not two


def test_content_load_unchanged_when_listen_disabled(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _OkLlm)
    _OkLlm.loads = 0
    settings = _listen_settings(content_enabled=True)
    settings.listen.enabled = False
    settings.speech.commentary_enabled = True
    Application(settings).load_content_model()
    assert _OkLlm.loads == 1


def test_listen_fallback_local_model_uses_the_listen_token_cap(monkeypatch) -> None:
    monkeypatch.setattr("notamusicbot.content.llm.LocalQwenLlmProvider", _CapRecordingLlm)
    app = Application(_listen_settings(content_enabled=True))
    assert app.settings.content.max_new_tokens == 120
    provider = app._build_listen_fallback_provider()
    assert provider.max_new_tokens == app.settings.listen.reply.max_new_tokens == 60
    assert app.settings.content.max_new_tokens == 120  # music's own cap untouched


def test_build_reply_synthesizer_passes_the_tts_caps(tmp_path) -> None:
    from notamusicbot.config import ListenReplySettings, ListenSettings

    settings = _disabled_both()
    settings.listen = ListenSettings(
        enabled=True,
        reply=ListenReplySettings(tts_max_seconds_per_word=0.7, tts_cap_floor_seconds=1.5),
    )
    app = Application(settings)
    profile = next(iter(app.voice_profiles.values()))
    with pytest.raises(RuntimeError):
        app.build_reply_synthesizer(profile)  # pipeline not built yet
    app.pipeline = object()
    synth = app.build_reply_synthesizer(profile)
    assert synth._tts_seconds_per_word == 0.7
    assert synth._tts_cap_floor_s == 1.5
    assert synth._artifact_root == Path(settings.speech.output_dir)


class _CapRecordingLlm:
    @classmethod
    def from_settings(cls, settings, gpu_coordinator=None):
        self = cls()
        self.max_new_tokens = settings.max_new_tokens
        return self
