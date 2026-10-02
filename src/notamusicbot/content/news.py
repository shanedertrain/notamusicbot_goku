"""News fetcher: minimal port of legacy ``newsapi.py`` HTTP logic.

Only the top-headlines call and the first random pick are ported (the legacy
BeautifulSoup full-article scrape is intentionally not carried over — summaries
are generated from the article description). Requests run off the event loop
via ``asyncio.to_thread``. Any failure returns ``None``; callers skip the
commentary instead of crashing playback (handoff section 20).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class Article:
    """A news headline ready for commentary."""

    title: str
    source: str
    description: str
    url: str

    @property
    def commentary_text(self) -> str:
        return f"{self.title}. {self.description}".strip()


class NewsFetcher:
    """Fetch top headlines from newsapi.org (requires NEWS_API_KEY)."""

    def __init__(self, api_key: str | None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._api_key = api_key or ""
        self._timeout_s = timeout_s

    async def get_random_article(
        self, country: str = "us", category: str | None = None
    ) -> Article | None:
        """Fetch headlines off the event loop and pick one at random."""
        return await asyncio.to_thread(self._get_random_article_sync, country, category)

    def _get_random_article_sync(self, country: str, category: str | None) -> Article | None:
        if not self._api_key:
            logger.warning("NEWS_API_KEY is not set; skipping news commentary")
            return None
        params: dict[str, str] = {"country": country, "apiKey": self._api_key}
        if category:
            params["category"] = category
        try:
            response = httpx.get(
                "https://newsapi.org/v2/top-headlines",
                params=params,
                timeout=self._timeout_s,
            )
            data = response.json()
        except Exception:
            logger.warning("News fetch failed; skipping news commentary", exc_info=True)
            return None
        # Never log params: they contain the API key.
        if data.get("status") != "ok":
            logger.warning("News API returned non-ok status; skipping news commentary")
            return None
        articles = [a for a in data.get("articles", []) if a.get("title")]
        if not articles:
            return None
        article = random.choice(articles)
        return Article(
            title=article.get("title") or "",
            source=(article.get("source") or {}).get("name", ""),
            description=article.get("description") or "",
            url=article.get("url") or "",
        )
