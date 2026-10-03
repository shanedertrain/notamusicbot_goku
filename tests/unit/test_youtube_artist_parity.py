"""Review Round-4 P1: YouTube uploader parity in song commentary.

Fake yt-dlp result (title 'Song', uploader 'Artist') resolved through the
YouTube resolver must reach the commentary ContentRequest as
``song_name='Song'`` / ``artist_name='Artist'`` via the ``!gplay`` command
path — instead of "an unknown artist".

Self-contained harness (no cross-test-module imports): builds a minimal
Application with an rvc voice profile and fakes at the content/speech
boundaries — no GPU, no network, no weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("discord", reason="this module drives the discord command layer")

import yaml

from notamusicbot.app import Application
from notamusicbot.config import AppSettings, SpeechSettings
from notamusicbot.discord_bot import commands as commands_mod
from notamusicbot.discord_bot.commands import _enqueue_youtube
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.users import UserProfile

YOUTUBE_URL = "https://www.youtube.com/watch?v=abc123"

PROFILE_ID = "goku_rvc"
PROFILE_YAML = {
    "id": PROFILE_ID,
    "persona": {"display_name": "Goku (RVC)"},
    "tts": {"speaker": "Aiden", "language": "English"},
    "voice_conversion": {"backend": "passthrough"},
    "audio": {"speed": 1.0, "target_peak_dbfs": -1.0},
}


class NoopConverter:
    """Passthrough VC adapter — never invoked here (commentary is faked)."""

    async def start(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def convert(self, source, profile, output_path):  # noqa: ANN001
        shutil.copyfile(source.path, output_path)
        return source


def make_app(tmp_path: Path) -> Application:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    (voices_dir / f"{PROFILE_ID}.yaml").write_text(yaml.safe_dump(PROFILE_YAML), encoding="utf-8")
    settings = AppSettings(
        speech=SpeechSettings(
            voices_dir=voices_dir,
            users_config=tmp_path / "users.yaml",
            output_dir=tmp_path / "var" / "speech",
            commentary_enabled=True,
            warmup_on_start=False,
        )
    )
    return Application(settings, rvc_factory=NoopConverter)


def make_bot_ctx(app: Application):
    queue: list = []
    bot = SimpleNamespace(
        app=app,
        guild_state=lambda guild_id: SimpleNamespace(
            queue=SimpleNamespace(put_nowait=queue.append)
        ),
    )
    ctx = SimpleNamespace(guild=SimpleNamespace(id=42))
    return bot, ctx, queue


def fake_resolver_with(title: str, uploader: str):
    """Stands in for YouTubeResolver.resolve() (extraction already faked)."""

    async def resolve(url: str, *, media_id: str | None = None) -> PlayableMedia:
        return PlayableMedia(
            media_id=media_id or "deadbeef",
            kind=MediaKind.STREAM,
            title=title,
            artist=uploader,
            source_url=url,
        )

    return SimpleNamespace(resolve=resolve)


def setup_commentary_spy(monkeypatch, captured: list, tmp_path: Path) -> None:
    async def fake_resolve_text(provider, request):  # noqa: ANN001
        captured.append(request)
        return "Next up!"

    async def fake_generate_media(*args, **kwargs):  # noqa: ANN002, ANN003
        return PlayableMedia(
            media_id="commentary",
            kind=MediaKind.LOCAL,
            title="commentary",
            source_path=tmp_path / "c.wav",
        )

    monkeypatch.setattr(commands_mod, "resolve_commentary_text", fake_resolve_text)
    monkeypatch.setattr(commands_mod, "generate_commentary_media", fake_generate_media)


async def test_youtube_uploader_flows_into_commentary(tmp_path, monkeypatch) -> None:
    app = make_app(tmp_path)
    app.build_pipeline()
    app.pipeline = app.pipeline or object()  # commentary enabled, pipeline present

    captured: list = []
    setup_commentary_spy(monkeypatch, captured, tmp_path)

    bot, ctx, queue = make_bot_ctx(app)
    bot.resolver = fake_resolver_with(title="Song", uploader="Artist")
    bot.app.users = SimpleNamespace(
        lookup=lambda _id: UserProfile(user_id="1", screen_name="alice", voice_profile=PROFILE_ID)
    )
    ctx.author = SimpleNamespace(id="42", name="alice")

    await _enqueue_youtube(bot, ctx, YOUTUBE_URL, quick=False)

    assert len(captured) == 1
    request = captured[0]
    assert request.song_name == "Song"
    assert request.artist_name == "Artist"
    # The queued media itself carries the uploader too.
    assert len(queue) == 2  # commentary first, then the song
    assert queue[-1].artist == "Artist"
    assert queue[-1].title == "Song"


async def test_youtube_unknown_uploader_falls_back_gracefully(tmp_path, monkeypatch) -> None:
    """No uploader ('Unknown' placeholder) still produces a song request —
    legacy behavior did not crash here either."""
    app = make_app(tmp_path)
    app.build_pipeline()
    app.pipeline = app.pipeline or object()

    captured: list = []
    setup_commentary_spy(monkeypatch, captured, tmp_path)

    bot, ctx, _queue = make_bot_ctx(app)
    bot.resolver = fake_resolver_with(title="Song", uploader="Unknown")
    bot.app.users = SimpleNamespace(
        lookup=lambda _id: UserProfile(user_id="1", screen_name="alice", voice_profile=PROFILE_ID)
    )
    ctx.author = SimpleNamespace(id="42", name="alice")

    await _enqueue_youtube(bot, ctx, YOUTUBE_URL, quick=False)

    assert captured[0].song_name == "Song"
    assert captured[0].artist_name == "Unknown"
