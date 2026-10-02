"""Reddit post fetcher — stub (Phase F decision: NOT ported in this phase).

The legacy ``reddit_scraper.py`` depends on PRAW with Reddit OAuth client
credentials (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET). PRAW is not a runtime
dependency of the reworked package, and no current command path reads reddit
posts. Porting the credential flow is deferred (TODO below); article-style
commentary continues to work through the news provider and the shared
ContentProvider abstraction, so Gate F ("all content paths return text through
the same abstraction") is unaffected.

TODO(Phase G+): either port front-page text-post fetches over the public JSON
endpoint with httpx (no PRAW, off the event loop) or reintroduce praw as an
optional extra ``[project.optional-dependencies] reddit``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedditPost:
    """Shape kept for the future port; currently never produced."""

    post_id: str
    title: str
    url: str
    poster_name: str
    post_text: str = ""


class RedditPostFetcher:
    """Placeholder that always reports unavailability."""

    async def get_random_post(self) -> RedditPost | None:
        logger.info("Reddit fetching is not ported in Phase F; skipping reddit commentary")
        return None
