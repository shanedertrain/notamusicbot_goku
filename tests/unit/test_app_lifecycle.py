"""Round-2 review fixes: fail-soft content load, skip-disabled loading,
keep_artifacts wiring, and warmup-once/non-blocking semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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
    def from_settings(cls, settings) -> _ExplodingLlm:
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
    pipeline = app.build_pipeline()
    assert pipeline._keep_artifacts is True
    assert pipeline._failed_job_retention_hours == pytest.approx(2.5)


def test_build_pipeline_defaults_keep_artifacts_off(tmp_path) -> None:
    app = _app(commentary_enabled=True, output_dir=tmp_path / "speech")
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
