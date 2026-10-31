"""Production composition root: commentary path evidence (no GPU, no weights).

Proves that the real Application + Discord commentary path

    _enqueue_commentary -> ContentProvider -> SpeechPipeline
                        -> clone TTS synthesizer
                        -> profile postprocess -> guild queue

builds clone requests from the configured voice profile (absolute reference
path, language, clone mode) and passes the profile's
``AudioProcessingProfile`` (speed / target peak) into the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("discord", reason="this module drives the discord command layer")

import soundfile as sf
import yaml

from conftest import FakeSynthesizer
from notamusicbot.app import Application
from notamusicbot.config import AppSettings, SpeechSettings
from notamusicbot.discord_bot.commands import _enqueue_commentary
from notamusicbot.domain.users import UserProfile

PROFILE_ID = "goku_clone"


class SpySynthesizer(FakeSynthesizer):
    """Writes deterministic audio and records the clone requests it received."""

    def __init__(self, seconds: float = 1.0, *, fail: bool = False) -> None:
        super().__init__(sample_rate=24000, seconds=seconds)
        self.fail = fail
        self.shutdowns = 0

    async def synthesize(self, request, output_path):
        if self.fail:
            raise RuntimeError("model exploded")
        return await super().synthesize(request, output_path)

    async def shutdown(self) -> None:
        self.shutdowns += 1


def _profile_yaml(reference: str) -> dict:
    return {
        "id": PROFILE_ID,
        "persona": {"display_name": "Goku (clone)"},
        "tts": {
            "language": "English",
            "reference_audio": reference,
            "reference_text": "Hey, it's me, Goku!",
            "clone_mode": "xvec",
        },
        # Non-default audio settings: speed 0.5 doubles output duration.
        "audio": {"speed": 0.5, "target_peak_dbfs": -6.0},
    }


def make_app(
    tmp_path: Path,
    synth: SpySynthesizer,
    **speech_overrides,
) -> Application:
    reference = tmp_path / "ref.wav"
    n = 2400
    sf.write(reference, 0.4 * np.ones(n, dtype=np.float32), 24000, subtype="PCM_16")
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    (voices_dir / f"{PROFILE_ID}.yaml").write_text(
        yaml.safe_dump(_profile_yaml(str(reference))), encoding="utf-8"
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
    return Application(settings, synthesizer_factory=lambda: synth)


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


async def test_production_commentary_uses_profile_clone_and_audio(tmp_path: Path) -> None:
    """The commentary path runs the clone request built from the user profile
    and applies the profile's non-default audio settings."""
    synth = SpySynthesizer(seconds=1.0)
    app = make_app(tmp_path, synth)
    app.build_pipeline()
    await app.startup()

    bot, ctx, user, queue = make_bot_ctx(app)
    await _enqueue_commentary(bot, ctx, user, song_name="Dragon Ball theme")

    # 1. The clone request carries the profile's reference/mode (not a speaker).
    assert len(synth.calls) == 1
    request = synth.calls[0]
    assert request.reference_audio is not None
    assert Path(request.reference_audio).is_absolute()
    assert Path(request.reference_audio).name == "ref.wav"
    assert request.clone_mode == "xvec"
    assert request.reference_text == "Hey, it's me, Goku!"
    assert request.language == "English"
    assert request.speaker is None

    # 2. The profile's non-default AudioProcessingProfile reached the pipeline:
    #    source is 1.0 s, speed 0.5 -> final ~2.0 s at the configured rate.
    assert len(queue) == 1
    final_path = queue[0].source_path
    info = sf.info(final_path)
    assert info.samplerate == app.settings.speech.sample_rate_hz
    assert info.frames / info.samplerate == pytest.approx(2.0, rel=0.1)

    await app.shutdown()
    assert synth.shutdowns == 1


async def test_synthesis_failure_skips_commentary(tmp_path: Path) -> None:
    """A speech failure skips the commentary; playback continues unaffected."""
    synth = SpySynthesizer(fail=True)
    app = make_app(tmp_path, synth)
    app.build_pipeline()
    await app.startup()
    bot, ctx, user, queue = make_bot_ctx(app)
    await _enqueue_commentary(bot, ctx, user, song_name="A song")
    assert queue == []  # commentary skipped; the caller's media still plays
    await app.shutdown()


async def test_lifecycle_startup_shutdown_and_warmup(tmp_path: Path) -> None:
    synth = SpySynthesizer(seconds=0.1)
    app = make_app(tmp_path, synth, warmup_on_start=True)
    await app.startup()
    assert app._warmup_task is not None
    await app._warmup_task
    assert len(synth.calls) == 1  # the warmup request ran through synthesize
    assert synth.calls[0].text == "Warmup."
    await app.shutdown()
    assert synth.shutdowns == 1


async def test_warmup_failure_is_logged_not_fatal(tmp_path: Path) -> None:
    synth = SpySynthesizer(fail=True)
    app = make_app(tmp_path, synth, warmup_on_start=True)
    await app.startup()
    await app._warmup_task  # must not raise
    await app.shutdown()


async def test_warmup_disabled_starts_no_task(tmp_path: Path) -> None:
    synth = SpySynthesizer()
    app = make_app(tmp_path, synth, warmup_on_start=False)
    await app.startup()
    assert app._warmup_task is None
    await app.shutdown()
