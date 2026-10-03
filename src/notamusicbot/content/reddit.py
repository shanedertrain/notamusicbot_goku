"""Reddit post fetcher — real implementation (review blocker P0-5).

Port of ``legacy_flat/reddit_scraper.py`` through PRAW in read-only mode
(REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET from ``.env``; praw is an optional
``[reddit]`` extra — when it is not installed the fetcher degrades to "no
posts" instead of crashing playback). Like the legacy scraper, only text
(self) posts are kept; batches are fetched off the event loop via
``asyncio.to_thread`` and served shuffled so posts do not repeat until the
batch is exhausted. Any failure returns ``None`` and is logged — callers skip
reddit commentary instead of crashing playback (handoff section 20).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praw.models import Front, Subreddit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedditPost:
    """A reddit text post ready for commentary."""

    post_id: str
    title: str
    url: str
    poster_name: str
    post_text: str = ""


class RedditPostFetcher:
    """Read-only reddit fetcher over praw (optional dependency).

    Credentials come from the environment (``SecretsSettings``); praw itself is
    imported lazily inside the fetch thread so the package works without the
    ``[reddit]`` extra installed.
    """

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        *,
        subreddits: list[str] | None = None,
        post_limit: int = 40,
        user_agent: str = "windows:reddit_access:v1.0 (by u/Proud-Election-4152)",
        timeout_s: float = 10.0,
        rng: random.Random | None = None,
    ) -> None:
        self._client_id = client_id or ""
        self._client_secret = client_secret or ""
        self._subreddits = [s.strip().lstrip("r/") for s in (subreddits or []) if s.strip()]
        self._post_limit = max(1, post_limit)
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._rng = rng or random.Random()
        self._pending: deque[RedditPost] = deque()
        self._missing_dep_logged = False

    @classmethod
    def from_settings(
        cls, client_id: str | None, client_secret: str | None, settings
    ) -> RedditPostFetcher:
        """Build from :class:`~notamusicbot.config.RedditSettings`."""
        return cls(
            client_id,
            client_secret,
            subreddits=list(settings.subreddits),
            post_limit=settings.post_limit,
            user_agent=settings.user_agent,
            timeout_s=settings.request_timeout_s,
        )

    async def get_random_post(self) -> RedditPost | None:
        """Return the next shuffled post, refilling the batch when empty.

        Fetches run off the event loop; every failure mode (missing praw,
        missing credentials, network/API error) yields ``None``.
        """
        if not self._pending:
            posts = await asyncio.to_thread(self._fetch_batch_sync)
            if not posts:
                return None
            self._rng.shuffle(posts)
            self._pending.extend(posts)
        return self._pending.popleft()

    # ------------------------------------------------------------------
    # Sync internals (run in a worker thread)
    # ------------------------------------------------------------------

    def _fetch_batch_sync(self) -> list[RedditPost]:
        try:
            import praw
        except ImportError:
            if not self._missing_dep_logged:
                self._missing_dep_logged = True
                logger.warning(
                    "praw is not installed (pip install 'notamusicbot-goku[reddit]'); "
                    "skipping reddit commentary"
                )
            return []
        if not self._client_id or not self._client_secret:
            logger.warning(
                "REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET are not set; skipping reddit commentary"
            )
            return []
        try:
            reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
                check_for_async=False,
            )
            reddit.read_only = True

            # praw.models is only needed for static typing; importing it at
            # runtime would break the lightweight praw test double.
            subreddit: Subreddit | Front
            if self._subreddits:
                subreddit = reddit.subreddit("+".join(self._subreddits))
            else:
                subreddit = reddit.front
            posts: list[RedditPost] = []
            for post in subreddit.hot(limit=self._post_limit):
                if not post.is_self:
                    continue
                author_name = "unknown"
                if post.author is not None and getattr(post.author, "name", None):
                    author_name = post.author.name
                posts.append(
                    RedditPost(
                        post_id=post.id,
                        title=str(post.title).encode("utf-8", "ignore").decode("utf-8"),
                        url=post.url,
                        poster_name=author_name,
                        post_text=post.selftext or "",
                    )
                )
        except Exception:
            logger.warning("Reddit fetch failed; skipping reddit commentary", exc_info=True)
            return []
        if not posts:
            logger.info(
                "No text posts found for subreddits %r; skipping reddit commentary",
                self._subreddits,
            )
        return posts
