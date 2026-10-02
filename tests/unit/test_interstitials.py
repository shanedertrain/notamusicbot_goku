"""Unit tests for the interstitial scheduler and post-song playback glue."""

from __future__ import annotations

import asyncio
import random
import types
from dataclasses import dataclass, field
from pathlib import Path

from notamusicbot.content.base import ContentKind, StaticTemplateProvider
from notamusicbot.content.interstitials import InterstitialManager, TopicSource
from notamusicbot.discord_bot.interlude_playback import (
    enqueue_profile_commentary,
    fire_post_song_content,
)


# ----------------------------------------------------------------------
# Fakes (no network, no GPU)
# ----------------------------------------------------------------------
@dataclass
class FakeNewsFetcher:
    articles: list = field(default_factory=list)
    fail: bool = False
    calls: int = 0

    async def get_random_article(self, country="us", category=None):
        self.calls += 1
        if self.fail or not self.articles:
            return None
        return self.articles.pop()


@dataclass
class FakeRedditFetcher:
    posts: list = field(default_factory=list)
    fail: bool = False
    calls: int = 0

    async def get_random_post(self):
        self.calls += 1
        if self.fail or not self.posts:
            return None
        return self.posts.pop()


class FakePersona:
    def __init__(self, display_name: str, description: str) -> None:
        self.display_name = display_name
        self.description = description


class FakeProfile:
    def __init__(self, pid: str, description: str = "a witty host") -> None:
        self.id = pid
        self.persona = FakePersona(pid, description)

        class _Tts:
            language = "English"
            speaker = "speaker-x"

        class _Audio:
            speed = 1.0
            target_peak_dbfs = -1.0

        class _Vc:
            backend = "passthrough"
            model = None
            pitch_shift = 0

        self.tts = _Tts()
        self.audio = _Audio()
        self.voice_conversion = _Vc()


class FakeVoiceProfiles:
    def __init__(self, profiles: dict) -> None:
        self._profiles = profiles

    def get(self, pid):
        return self._profiles.get(pid)


class FakeArtifact:
    def __init__(self, path: Path) -> None:
        self.path = path


class FakePipeline:
    def __init__(self, tmp_path: Path, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list = []
        self._counter = 0
        self._tmp = tmp_path

    async def run(self, request, conversion, processing):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("gpu exploded")
        self._counter += 1
        path = self._tmp / f"job{self._counter}" / "final.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        return FakeArtifact(path), None


class FakeGuildState:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()


@dataclass
class FakeInterstitialSettings:
    enabled: bool = True
    probability: float = 1.0
    min_interval_s: float = 0.0
    voice_profile: str = "goku"
    news_country: str = "us"
    news_category: str | None = None


class FakeSpeechSettings:
    commentary_enabled = True
    sample_rate_hz = 48000
    channels = 2


class FakeSettings:
    def __init__(self, interstitial: FakeInterstitialSettings) -> None:
        self.speech = FakeSpeechSettings()
        self.interstitial = interstitial


class FakePodcasts:
    def __init__(self, active: bool = False, turns: list | None = None) -> None:
        self._active = active
        self._turns = turns or []

    def active(self, guild_id: int) -> bool:
        return self._active

    async def segment_turns(self, guild_id: int) -> list:
        return self._turns


class FakeBot:
    """Duck-typed stand-in for NotAMusicBot."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        interstitial: FakeInterstitialSettings | None = None,
        pipeline: FakePipeline | None = None,
        podcasts: FakePodcasts | None = None,
        provider=None,
    ) -> None:
        self.interstitial_settings = interstitial or FakeInterstitialSettings()
        self.settings = FakeSettings(self.interstitial_settings)
        self.app = types.SimpleNamespace(
            settings=self.settings,
            pipeline=pipeline,
            voice_profiles=FakeVoiceProfiles({"goku": FakeProfile("goku")}),
            build_content_provider=lambda: provider,
        )
        self.pipeline = pipeline
        self.podcasts = podcasts or FakePodcasts()
        self.interstitials = InterstitialManager(
            self.interstitial_settings,
            TopicSource(FakeNewsFetcher(), FakeRedditFetcher(), rng=random.Random(1)),
            rng=random.Random(1),
        )
        self.states: dict[int, FakeGuildState] = {}

    def guild_state(self, guild_id: int) -> FakeGuildState:
        return self.states.setdefault(guild_id, FakeGuildState())


# ----------------------------------------------------------------------
# TopicSource / InterstitialManager
# ----------------------------------------------------------------------
def test_topic_source_prefers_available_source() -> None:
    news = FakeNewsFetcher()
    reddit = FakeRedditFetcher(posts=[types.SimpleNamespace(title="post", post_text="body")])
    source = TopicSource(news, reddit, rng=random.Random(0))
    topic = asyncio.run(source.pick())
    assert topic is not None and topic.kind is ContentKind.REDDIT
    assert topic.title == "post"


def test_topic_source_falls_back_to_news() -> None:
    news = FakeNewsFetcher(articles=[types.SimpleNamespace(title="T", commentary_text="T. D")])
    reddit = FakeRedditFetcher()
    source = TopicSource(news, reddit, rng=random.Random(0))
    topic = asyncio.run(source.pick())
    assert topic is not None and topic.kind is ContentKind.NEWS


def test_topic_source_all_fail_returns_none() -> None:
    source = TopicSource(
        FakeNewsFetcher(fail=True), FakeRedditFetcher(fail=True), rng=random.Random(0)
    )
    assert asyncio.run(source.pick()) is None


def test_topic_source_no_fetchers_returns_none() -> None:
    assert asyncio.run(TopicSource(None, None).pick()) is None


def test_interstitial_manager_disabled_never_fires() -> None:
    settings = FakeInterstitialSettings(enabled=False)
    manager = InterstitialManager(settings, TopicSource(None, None))
    assert manager.should_fire(now_monotonic=1000.0) is False


def test_interstitial_manager_respects_interval() -> None:
    settings = FakeInterstitialSettings(probability=1.0, min_interval_s=300.0)
    manager = InterstitialManager(settings, TopicSource(None, None))
    assert manager.should_fire(now_monotonic=0.0) is True
    assert manager.should_fire(now_monotonic=100.0) is False
    assert manager.should_fire(now_monotonic=301.0) is True


def test_interstitial_manager_probability_gate() -> None:
    settings = FakeInterstitialSettings(probability=0.0)
    manager = InterstitialManager(settings, TopicSource(None, None))
    assert manager.should_fire(now_monotonic=0.0) is False


def test_build_request_maps_kinds() -> None:
    manager = InterstitialManager(FakeInterstitialSettings(), TopicSource(None, None))
    from notamusicbot.content.interstitials import TopicItem

    req = manager.build_request(TopicItem(ContentKind.REDDIT, "T", "B"), "persona")
    assert req.kind is ContentKind.REDDIT and req.post_title == "T" and req.post_text == "B"
    req = manager.build_request(TopicItem(ContentKind.NEWS, "T", "A"), "persona")
    assert req.kind is ContentKind.NEWS and req.article_text == "A"


# ----------------------------------------------------------------------
# Playback glue
# ----------------------------------------------------------------------
def test_enqueue_commentary_respects_commentary_flag(tmp_path: Path) -> None:
    bot = FakeBot(tmp_path)
    bot.settings.speech.commentary_enabled = False
    bot.app.pipeline = FakePipeline(tmp_path)
    enqueued = asyncio.run(enqueue_profile_commentary(bot, 1, "goku", "hello", requester="t"))
    assert enqueued is False
    assert bot.guild_state(1).queue.empty()


def test_enqueue_commentary_unknown_profile_skips(tmp_path: Path) -> None:
    bot = FakeBot(tmp_path)
    bot.app.pipeline = FakePipeline(tmp_path)
    assert asyncio.run(enqueue_profile_commentary(bot, 1, "nope", "x", requester="t")) is False


def test_enqueue_commentary_pipeline_failure_returns_false(tmp_path: Path) -> None:
    bot = FakeBot(tmp_path, pipeline=FakePipeline(tmp_path, fail=True))
    assert asyncio.run(enqueue_profile_commentary(bot, 1, "goku", "x", requester="t")) is False


def test_fire_post_song_enqueues_interstitial(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path)
    bot = FakeBot(tmp_path, pipeline=pipeline)
    reddit_post = types.SimpleNamespace(title="A post", post_text="Body", post_id="a")
    bot.interstitials.topic_source = TopicSource(
        FakeNewsFetcher(), FakeRedditFetcher(posts=[reddit_post]), rng=random.Random(0)
    )
    assert asyncio.run(fire_post_song_content(bot, 1)) is True
    state = bot.guild_state(1)
    media = state.queue.get_nowait()
    assert "commentary" in media.title
    # Text went through the ContentProvider path (static fallback here).
    assert pipeline.calls and pipeline.calls[0].text


def test_fire_post_song_skips_silently_when_no_topic(tmp_path: Path) -> None:
    bot = FakeBot(tmp_path, pipeline=FakePipeline(tmp_path))
    bot.interstitials.topic_source = TopicSource(FakeNewsFetcher(), FakeRedditFetcher())
    assert asyncio.run(fire_post_song_content(bot, 1)) is False
    assert bot.guild_state(1).queue.empty()


def test_fire_post_song_podcast_takes_precedence(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path)
    turns = [
        types.SimpleNamespace(voice_profile_id="goku", speaker_name="Host", text="hello"),
        types.SimpleNamespace(voice_profile_id="goku", speaker_name="Guest", text="hi"),
    ]
    bot = FakeBot(tmp_path, pipeline=pipeline, podcasts=FakePodcasts(active=True, turns=turns))
    # Even with probability 0 the podcast path must fire.
    bot.interstitial_settings.probability = 0.0
    assert asyncio.run(fire_post_song_content(bot, 1)) is True
    state = bot.guild_state(1)
    first = state.queue.get_nowait()
    second = state.queue.get_nowait()
    assert "hello" in first.title and "hi" in second.title  # turn order preserved
    assert state.queue.empty()


def test_fire_post_song_never_raises(tmp_path: Path) -> None:
    bot = FakeBot(tmp_path, pipeline=FakePipeline(tmp_path))
    bot.interstitials.topic_source = TopicSource(
        FakeNewsFetcher(fail=True), FakeRedditFetcher(fail=True)
    )

    # No exception escapes even with a broken topic source.
    async def broken_pick():
        raise RuntimeError("boom")

    bot.interstitials.topic_source.pick = broken_pick  # type: ignore[method-assign]
    bot.interstitial_settings.probability = 1.0
    assert asyncio.run(fire_post_song_content(bot, 1)) is False


def test_static_provider_used_when_llm_disabled(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path)
    bot = FakeBot(tmp_path, pipeline=pipeline, provider=StaticTemplateProvider())
    reddit_post = types.SimpleNamespace(title="Title X", post_text="", post_id="a")
    bot.interstitials.topic_source = TopicSource(
        FakeNewsFetcher(), FakeRedditFetcher(posts=[reddit_post]), rng=random.Random(0)
    )
    assert asyncio.run(fire_post_song_content(bot, 1)) is True
    assert "Title X" in pipeline.calls[0].text


def test_content_request_shape_for_interstitial_is_provider_native() -> None:
    from notamusicbot.content.prompts import build_prompt

    manager = InterstitialManager(FakeInterstitialSettings(), TopicSource(None, None))
    from notamusicbot.content.interstitials import TopicItem

    req = manager.build_request(TopicItem(ContentKind.REDDIT, "T", "B"), "a host")
    prompt = build_prompt(req)
    assert "T" in prompt and "a host" in prompt
