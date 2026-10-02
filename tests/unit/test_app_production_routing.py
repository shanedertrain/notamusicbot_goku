"""Production composition root: Gate R2 evidence.

Proves that the real Application + Discord commentary path

    _enqueue_commentary -> ContentProvider -> SpeechPipeline
                        -> VoiceConverterRouter -> RVC adapter
                        -> profile postprocess -> guild queue

routes by ``voice_conversion.backend`` and passes the voice profile's
``AudioProcessingProfile`` (speed / target peak) into the pipeline — using a
spy RVC adapter injected through the composition root (no GPU, no weights).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("discord", reason="this module drives the discord command layer")

import soundfile as sf
import yaml

from conftest import FakeSynthesizer
from notamusicbot.app import Application
from notamusicbot.config import AppSettings, SpeechSettings
from notamusicbot.discord_bot.commands import _enqueue_commentary
from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.domain.users import UserProfile

PROFILE_ID = "goku_rvc"

PROFILE_YAML = {
    "id": PROFILE_ID,
    "persona": {"display_name": "Goku (RVC)"},
    "tts": {"speaker": "Aiden", "language": "English"},
    "voice_conversion": {"backend": "rvc", "model": "Goku", "pitch_shift": 0},
    # Non-default audio settings: speed 0.5 doubles output duration.
    "audio": {"speed": 0.5, "target_peak_dbfs": -6.0},
}

PASSTHROUGH_PROFILE_YAML = {
    "id": "plain",
    "persona": {"display_name": "Plain"},
    "tts": {"speaker": "Aiden", "language": "English"},
    "voice_conversion": {"backend": "passthrough"},
    "audio": {"speed": 0.5, "target_peak_dbfs": -6.0},
}


class SpyRvcConverter:
    """Spy RVC adapter installed via the composition root."""

    def __init__(self) -> None:
        self.starts = 0
        self.shutdowns = 0
        self.calls: list[tuple[AudioArtifact, VoiceConversionProfile]] = []

    async def start(self) -> None:
        self.starts += 1

    async def shutdown(self) -> None:
        self.shutdowns += 1

    async def convert(
        self, source: AudioArtifact, profile: VoiceConversionProfile, output_path: Path
    ) -> AudioArtifact:
        self.calls.append((source, profile))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.path, output_path)
        return AudioArtifact(
            path=output_path, sample_rate_hz=source.sample_rate_hz, duration_s=source.duration_s
        )


class WarmupSpySynthesizer(FakeSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.warmup_paths: list[Path] = []

    def warmup(self, output_path: Path) -> AudioArtifact:
        self.warmup_paths.append(output_path)
        return AudioArtifact(path=output_path, sample_rate_hz=self.sample_rate, duration_s=0.1)


class FailingWarmupSynthesizer(WarmupSpySynthesizer):
    def warmup(self, output_path: Path) -> AudioArtifact:
        raise RuntimeError("CUDA OOM during warmup")


def make_app(
    tmp_path: Path,
    spy: SpyRvcConverter,
    *,
    profile_yaml: dict | None = None,
    **speech_overrides,
) -> Application:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    (voices_dir / f"{(profile_yaml or PROFILE_YAML)['id']}.yaml").write_text(
        yaml.safe_dump(profile_yaml or PROFILE_YAML), encoding="utf-8"
    )
    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=voices_dir,
            users_config=tmp_path / "users.yaml",
            output_dir=tmp_path / "var" / "speech",
            commentary_enabled=True,
            warmup_on_start=speech_overrides.pop("warmup_on_start", False),
            **speech_overrides,
        )
    )
    app = Application(settings, rvc_factory=lambda: spy)
    app._synthesizer = FakeSynthesizer(sample_rate=24000, seconds=1.0)
    return app


def make_bot_ctx(app: Application):
    queue: list = []

    bot = SimpleNamespace(
        app=app,
        guild_state=lambda guild_id: SimpleNamespace(
            queue=SimpleNamespace(put_nowait=queue.append)
        ),
    )
    ctx = SimpleNamespace(guild=SimpleNamespace(id=42))
    user = UserProfile(user_id="1", screen_name="alice", voice_profile=PROFILE_ID)
    return bot, ctx, user, queue


async def test_production_commentary_routes_rvc_and_applies_profile_audio(
    tmp_path: Path,
) -> None:
    """Gate R2: rvc profile -> spy RVC adapter + profile audio postprocess."""
    spy = SpyRvcConverter()
    app = make_app(tmp_path, spy)
    app.build_pipeline()
    await app.startup()  # production setup_hook path: router startup (worker start)
    assert spy.starts == 1

    bot, ctx, user, queue = make_bot_ctx(app)
    await _enqueue_commentary(bot, ctx, user, song_name="Dragon Ball theme")

    # 1. The RVC adapter was invoked (NOT passthrough) with the rvc profile.
    assert len(spy.calls) == 1
    _source, conversion = spy.calls[0]
    assert conversion.backend == "rvc"
    assert conversion.model == "Goku"

    # 2. The profile's non-default AudioProcessingProfile reached the pipeline:
    #    source is 1.0 s, speed 0.5 -> final ~2.0 s at the configured rate.
    assert len(queue) == 1
    final_path = queue[0].source_path
    info = sf.info(final_path)
    assert info.samplerate == app.settings.speech.sample_rate_hz
    assert info.frames / info.samplerate == pytest.approx(2.0, rel=0.1)

    await app.shutdown()
    assert spy.shutdowns == 1


async def test_production_commentary_with_passthrough_profile_skips_rvc(
    tmp_path: Path,
) -> None:
    spy = SpyRvcConverter()
    app = make_app(tmp_path, spy, profile_yaml=PASSTHROUGH_PROFILE_YAML)
    app.build_pipeline()
    await app.startup()
    bot, ctx, user, queue = make_bot_ctx(app)
    user = UserProfile(user_id="1", screen_name="alice", voice_profile="plain")
    await _enqueue_commentary(bot, ctx, user, song_name="A song")
    assert spy.calls == []  # passthrough profile: RVC adapter never invoked
    assert len(queue) == 1
    await app.shutdown()


async def test_rvc_unavailable_error_policy_skips_commentary(tmp_path: Path) -> None:
    """No-GPU path: policy 'error' skips commentary instead of emitting
    the un-converted source voice; startup and playback are unaffected."""

    class NoGpuSpy(SpyRvcConverter):
        async def start(self) -> None:
            self.starts += 1
            raise RuntimeError("no CUDA device available")

    spy = NoGpuSpy()
    app = make_app(tmp_path, spy, rvc_unavailable_policy="error")
    app.build_pipeline()
    await app.startup()  # soft failure: logged, not raised
    assert spy.starts == 1
    bot, ctx, user, queue = make_bot_ctx(app)
    await _enqueue_commentary(bot, ctx, user, song_name="A song")
    assert queue == []  # commentary skipped; the caller's media still plays
    await app.shutdown()


async def test_lifecycle_startup_shutdown_and_warmup(tmp_path: Path) -> None:
    spy = SpyRvcConverter()
    app = make_app(tmp_path, spy, warmup_on_start=True)
    app._synthesizer = WarmupSpySynthesizer()
    await app.startup()
    assert spy.starts == 1
    assert app._warmup_task is not None
    await app._warmup_task
    assert app._synthesizer.warmup_paths  # warmup ran, non-blocking, success
    await app.shutdown()
    assert spy.shutdowns == 1


async def test_warmup_failure_is_logged_not_fatal(tmp_path: Path) -> None:
    spy = SpyRvcConverter()
    app = make_app(tmp_path, spy, warmup_on_start=True)
    app._synthesizer = FailingWarmupSynthesizer()
    await app.startup()
    await app._warmup_task  # must not raise
    await app.shutdown()
    assert spy.shutdowns == 1


async def test_warmup_disabled_starts_no_task(tmp_path: Path) -> None:
    spy = SpyRvcConverter()
    app = make_app(tmp_path, spy, warmup_on_start=False)
    await app.startup()
    assert app._warmup_task is None
    await app.shutdown()
