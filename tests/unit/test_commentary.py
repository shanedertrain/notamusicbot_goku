"""Unit tests for commentary generation through the shared speech pipeline."""

from __future__ import annotations

from pathlib import Path

from conftest import FailingSynthesizer, FakeConverter, FakeSynthesizer
from notamusicbot.discord_bot.commentary import generate_commentary_media
from notamusicbot.domain.media import MediaKind
from notamusicbot.domain.speech import (
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.pipeline import SpeechPipeline


def make_pipeline(synthesizer, tmp_path: Path) -> SpeechPipeline:
    return SpeechPipeline(
        synthesizer,
        FakeConverter(),
        AudioProcessingProfile(),
        output_dir=tmp_path / "speech",
    )


REQUEST = TtsRequest(text="Next up: a song", language="English", speaker="Aiden")
CONVERSION = VoiceConversionProfile(backend="passthrough")


async def test_commentary_success_returns_local_media(tmp_path: Path) -> None:
    pipeline = make_pipeline(FakeSynthesizer(), tmp_path)
    media = await generate_commentary_media(
        pipeline, REQUEST, CONVERSION, guild_id=42, requester="alice"
    )
    assert media is not None
    assert media.kind is MediaKind.LOCAL
    assert media.source_path is not None and media.source_path.exists()
    assert media.guild_id == 42
    assert media.requester == "alice"


async def test_commentary_failure_is_swallowed(tmp_path: Path) -> None:
    pipeline = make_pipeline(FailingSynthesizer(), tmp_path)
    media = await generate_commentary_media(pipeline, REQUEST, CONVERSION, guild_id=42)
    assert media is None  # speech failure must not crash the media queue
