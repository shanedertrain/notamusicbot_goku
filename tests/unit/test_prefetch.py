"""Unit tests for the post-song commentary prefetch registry/lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from notamusicbot.discord_bot.prefetch import CommentaryPrefetcher
from notamusicbot.domain.media import MediaKind, PlaybackOutcome, PlayableMedia
from notamusicbot.domain.speech import AudioArtifact, TtsRequest, VoiceConversionProfile


# ------------------------------------------------------------------- fakes


class FakeSynth:
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize(self, request: TtsRequest, output_path: Path) -> AudioArtifact:
        self.calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-wav")
        return AudioArtifact(path=output_path, sample_rate_hz=24000, duration_s=1.0)


class FakePipeline:
    def __init__(self, tmp_path: Path) -> None:
        self.synth = FakeSynth()
        self.tmp_path = tmp_path
        self.calls = 0
        self.texts: list[str] = []
        self.hold: asyncio.Event | None = None

    async def run(self, request, conversion, processing):  # noqa: ANN001
        from notamusicbot.speech.artifacts import mark_temp_artifact

        self.calls += 1
        self.synth.calls += 1
        self.texts.append(request.text)
        if self.hold is not None:
            await self.hold.wait()
        path = self.tmp_path / f"job-{self.calls}" / "final.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-wav")
        mark_temp_artifact(path)
        return AudioArtifact(path=path, sample_rate_hz=24000, duration_s=1.0), object()


@dataclass
class FakeTurn:
    voice_profile_id: str
    text: str


class FakeGuildState:
    def __init__(self) -> None:
        self.front: list[Any] = []

    def put_front(self, media) -> None:  # noqa: ANN001
        self.front.insert(0, media)


class FakeProfiles:
    def __init__(self) -> None:
        self.profile = type(
            "P",
            (),
            {
                "tts": type("T", (), {"language": "English", "speaker": "Aiden"})(),
                "persona": type("D", (), {"description": "host"})(),
                "voice_conversion": type(
                    "V", (), {"backend": "passthrough", "model": None, "pitch_shift": 0}
                )(),
                "audio": type("A", (), {"speed": 1.0, "target_peak_dbfs": -1.0})(),
            },
        )()

    def get(self, pid: str):  # noqa: ANN001
        return self.profile


class FakeContentSettings:
    prefetch_enabled: bool = True
    max_sentences: int | None = 3
    enabled: bool = True


class FakeSpeechSettings:
    commentary_enabled: bool = True
    sample_rate_hz: int = 48000
    channels: int = 2


class FakeSettings:
    def __init__(self) -> None:
        self.content = FakeContentSettings()
        self.speech = FakeSpeechSettings()
        self.interstitial = type(
            "I", (), {"voice_profile": "goku", "enabled": True, "probability": 1.0}
        )()


class FakeApp:
    def __init__(self, tmp_path: Path) -> None:
        self.settings = FakeSettings()
        self.voice_profiles = FakeProfiles()
        self.pipeline = FakePipeline(tmp_path)
        self.provider_calls = 0

    def build_content_provider(self):  # noqa: ANN001
        self.provider_calls += 1

        class _Provider:
            async def generate(self, request):  # noqa: ANN001
                return "First sentence. Second one. Third. Fourth. Fifth."

        return _Provider()


class FakeHost:
    def __init__(self, tmp_path: Path) -> None:
        self.app = FakeApp(tmp_path)
        self.guild_state_obj = FakeGuildState()
        self.podcasts = type("P", (), {"active": lambda self, gid: False})()
        self.interstitials = type(
            "I",
            (),
            {
                "should_fire": lambda self, guild_id: True,
                "build_request": lambda self, topic, persona: object(),
                "topic_source": type(
                    "T", (), {"pick": staticmethod(_pick_topic)}
                )(),
            },
        )()

    def guild_state(self, guild_id: int) -> FakeGuildState:
        return self.guild_state_obj


async def _pick_topic():
    return object()


def make_stream_media(media_id: str = "song1") -> PlayableMedia:
    return PlayableMedia(
        media_id=media_id,
        kind=MediaKind.STREAM,
        title="song",
        requester="alice",
        guild_id=42,
        source_path=None,
        source_url="https://example.invalid/song",
    )


# ------------------------------------------------------------------- tests


async def test_prefetch_ready_at_natural_completion_enqueues_without_second_run(
    tmp_path: Path,
) -> None:
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    media = make_stream_media()
    assert prefetcher.start_prefetch(host, 42, media) is True
    # Let the prefetch task finish.
    await asyncio.sleep(0.05)
    assert host.app.pipeline.calls == 1  # exactly one generation

    enqueued = await prefetcher.consume(host, 42, media)
    assert enqueued is True
    assert len(host.guild_state_obj.front) == 1
    assert host.guild_state_obj.front[0].kind is MediaKind.LOCAL
    # No second LLM/TTS run at completion.
    assert host.app.pipeline.calls == 1
    assert host.app.provider_calls == 1
    assert prefetcher.pending_media_ids() == {}


async def test_prefetch_text_trimmed_to_max_sentences(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media())
    await asyncio.sleep(0.05)
    await prefetcher.consume(host, 42, make_stream_media("song1"))
    # Provider produced 5 sentences; TTS must have seen only the first 3.
    assert host.app.pipeline.texts == ["First sentence. Second one. Third."]


async def test_skipped_song_discards_prefetched_artifact(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    media = make_stream_media()
    prefetcher.start_prefetch(host, 42, media)
    await asyncio.sleep(0.05)
    artifact_path = host.app.pipeline.tmp_path / "job-1" / "final.wav"
    assert artifact_path.exists()
    # Song was SKIPPED: the completion hook never fires for it; the NEXT song
    # start discards the stale entry and cleans its artifact per policy.
    prefetcher.start_prefetch(host, 42, make_stream_media("song2"))
    await asyncio.sleep(0.05)
    assert host.guild_state_obj.front == []  # nothing enqueued for the skip
    assert not artifact_path.exists()  # prefetched artifact cleaned up
    # The new song legitimately started its own prefetch.
    assert host.app.pipeline.calls == 2


async def test_prefetch_not_finished_in_time_falls_back(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    host.app.pipeline.hold = asyncio.Event()
    prefetcher = CommentaryPrefetcher()
    media = make_stream_media()
    prefetcher.start_prefetch(host, 42, media)
    await asyncio.sleep(0.02)
    # Completion arrives while generation is still held.
    assert await prefetcher.consume(host, 42, media) is False
    assert host.guild_state_obj.front == []
    # The pending task is marked discarded and can never enqueue later.
    host.app.pipeline.hold.set()
    await asyncio.sleep(0.05)
    assert host.guild_state_obj.front == []


async def test_discarded_pending_generation_cleans_its_artifacts(
    tmp_path: Path,
) -> None:
    host = FakeHost(tmp_path)
    host.app.pipeline.hold = asyncio.Event()
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media())
    await asyncio.sleep(0.02)
    assert await prefetcher.consume(host, 42, make_stream_media()) is False
    host.app.pipeline.hold.set()
    await asyncio.sleep(0.05)
    # The generated artifact (if any) was cleaned, never enqueued.
    assert host.guild_state_obj.front == []
    for path in (host.app.pipeline.tmp_path).glob("job-*/final.wav"):
        from notamusicbot.speech.artifacts import is_temp_speech_artifact

        assert not is_temp_speech_artifact(path)


async def test_podcast_turns_prefetched_in_reverse_put_front_order(
    tmp_path: Path,
) -> None:
    host = FakeHost(tmp_path)
    host.podcasts = type(
        "P",
        (),
        {
            "active": lambda self, gid: True,
            "segment_turns": async_turns,
        },
    )()
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media())
    await asyncio.sleep(0.05)
    assert await prefetcher.consume(host, 42, make_stream_media()) is True
    # Two turns; put_front in stored order keeps the conversation turn order.
    assert len(host.guild_state_obj.front) == 2
    assert "first" in host.guild_state_obj.front[0].title
    assert "second" in host.guild_state_obj.front[1].title


async def async_turns(_self, _guild_id: int) -> list[FakeTurn]:
    return [FakeTurn("goku", "first turn."), FakeTurn("goku", "second turn.")]


async def test_interstitial_not_fired_falls_back_without_enqueue(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    host.interstitials = type(
        "I", (), {"should_fire": lambda self, guild_id: False}
    )()
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media())
    await asyncio.sleep(0.05)
    assert host.app.pipeline.calls == 0  # scheduler said no: nothing generated
    assert await prefetcher.consume(host, 42, make_stream_media()) is False
    assert host.guild_state_obj.front == []


async def test_prefetch_disabled_never_starts(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    host.app.settings.content.prefetch_enabled = False
    prefetcher = CommentaryPrefetcher()
    assert prefetcher.start_prefetch(host, 42, make_stream_media()) is False
    await asyncio.sleep(0.02)
    assert host.app.pipeline.calls == 0
    assert await prefetcher.consume(host, 42, make_stream_media()) is False


async def test_non_stream_media_never_prefetched(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    media = make_stream_media()
    media.kind = MediaKind.LOCAL
    assert prefetcher.start_prefetch(host, 42, media) is False


async def test_double_generation_prevented_registry_checked_once(tmp_path: Path) -> None:
    """consume pops the entry: a second consume cannot re-enqueue or rerun."""
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    media = make_stream_media()
    prefetcher.start_prefetch(host, 42, media)
    await asyncio.sleep(0.05)
    assert await prefetcher.consume(host, 42, media) is True
    assert await prefetcher.consume(host, 42, media) is False
    assert host.app.pipeline.calls == 1


async def test_consume_for_different_media_falls_back(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media("song1"))
    await asyncio.sleep(0.05)
    assert await prefetcher.consume(host, 42, make_stream_media("other")) is False


async def test_generation_failure_falls_back(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)

    async def fail_run(request, conversion, processing):  # noqa: ANN001
        raise RuntimeError("tts boom")

    host.app.pipeline.run = fail_run  # type: ignore[method-assign]
    prefetcher = CommentaryPrefetcher()
    prefetcher.start_prefetch(host, 42, make_stream_media())
    await asyncio.sleep(0.05)
    assert await prefetcher.consume(host, 42, make_stream_media()) is False
