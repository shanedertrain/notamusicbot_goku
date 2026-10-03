"""Randomized news/reddit interstitial scheduler (review blocker P0-5).

Restores the legacy ``bot_manager.generate_article_audio`` behavior through the
new abstractions: after a song finishes, with a configured probability and at
most once per minimum interval, the bot picks news or reddit at random
(``TopicSource``), turns it into a :class:`ContentRequest`, generates the
commentary text through a :class:`ContentProvider` and synthesizes it through
the :class:`SpeechPipeline` with a configured voice profile, then enqueues it
ahead of the next song. Any failure is skipped silently (logged) — playback is
never at risk. Disabled by default (``interstitial.enabled=false``).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from notamusicbot.content.base import ContentKind, ContentRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TopicItem:
    """One fetched news article or reddit post, ready for commentary."""

    kind: ContentKind  # NEWS or REDDIT
    title: str
    text: str


class TopicSource:
    """Randomly picks news vs reddit and fetches one item.

    Fetchers may be ``None`` (credentials missing / feature not configured) or
    fail; in both cases ``pick`` tries the other source, then gives up with
    ``None``. No network calls happen at construction time.
    """

    def __init__(
        self,
        news_fetcher: AnyFetch | None,
        reddit_fetcher: AnyFetch | None,
        *,
        rng: random.Random | None = None,
        news_country: str = "us",
        news_category: str | None = None,
    ) -> None:
        self._news = news_fetcher
        self._reddit = reddit_fetcher
        self._rng = rng or random.Random()
        self._news_country = news_country
        self._news_category = news_category

    async def pick(self) -> TopicItem | None:
        # Legacy: random.choice between reddit and news; skip a source that is
        # unavailable or fails and try the other one.
        prefer_reddit = self._rng.random() < 0.5
        order = [(self._reddit, True), (self._news, False)]
        if not prefer_reddit:
            order.reverse()
        for fetcher, is_reddit in order:
            if fetcher is None:
                continue
            try:
                if is_reddit:
                    post = await fetcher.get_random_post()
                    if post is not None:
                        return TopicItem(
                            kind=ContentKind.REDDIT, title=post.title, text=post.post_text
                        )
                else:
                    article = await fetcher.get_random_article(
                        self._news_country, self._news_category
                    )
                    if article is not None:
                        return TopicItem(
                            kind=ContentKind.NEWS,
                            title=article.title,
                            text=article.commentary_text,
                        )
            except Exception:
                logger.warning(
                    "%s fetch failed; skipping interstitial source",
                    "reddit" if is_reddit else "news",
                    exc_info=True,
                )
        return None


class InterstitialManager:
    """Probability- and interval-gated interstitial scheduler.

    Review Round-3 P2: last-fire timing state is **guild-scoped** (a dict keyed
    by guild id), so a cooldown fired in one guild can no longer suppress an
    interstitial in another. Callers without a guild context (legacy CLI /
    benchmark callers) share the reserved ``GLOBAL_GUILD_ID`` bucket.
    """

    #: Bucket used when no guild id is supplied (single-guild callers).
    GLOBAL_GUILD_ID = 0

    def __init__(
        self, settings: AnySettings, topic_source: TopicSource, *, rng: random.Random | None = None
    ) -> None:
        self._settings = settings
        self.topic_source = topic_source
        self._rng = rng or random.Random()
        self._last_fire_monotonic: dict[int, float] = {}

    def should_fire(
        self,
        *,
        guild_id: int = GLOBAL_GUILD_ID,
        now_monotonic: float | None = None,
    ) -> bool:
        """Decide whether an interstitial fires after the just-finished song.

        Cooldown state is tracked per guild id (``guild_id`` defaults to the
        single shared :data:`GLOBAL_GUILD_ID` bucket for callers without a
        guild context).
        """
        if not self._settings.enabled:
            return False
        now = time.monotonic() if now_monotonic is None else now_monotonic
        last = self._last_fire_monotonic.get(guild_id, float("-inf"))
        if now - last < self._settings.min_interval_s:
            return False
        if self._rng.random() >= self._settings.probability:
            return False
        self._last_fire_monotonic[guild_id] = now
        return True

    def build_request(self, topic: TopicItem, persona_description: str) -> ContentRequest:
        """Build the ContentRequest for the interstitial commentary line."""
        if topic.kind is ContentKind.REDDIT:
            return ContentRequest(
                kind=ContentKind.REDDIT,
                persona_description=persona_description,
                post_title=topic.title,
                post_text=topic.text,
            )
        return ContentRequest(
            kind=ContentKind.NEWS,
            persona_description=persona_description,
            post_title=topic.title,
            article_text=topic.text,
        )


# Structural typing note: fetchers are typed loosely (news and reddit
# fetchers expose different methods) so unit tests can pass fakes directly.
AnyFetch = Any
AnySettings = Any
