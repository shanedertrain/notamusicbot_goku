"""Per-guild podcast sessions: start/stop and segment generation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from notamusicbot.content.interstitials import TopicSource
from notamusicbot.podcast.engine import PodcastEngine, PodcastTurn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PodcastSession:
    """An active podcast conversation for one guild."""

    guild_id: int
    host_profile_id: str
    guest_profile_id: str


class PodcastCoordinator:
    """Tracks which guilds are in podcast mode and generates their segments.

    ``provider_factory`` returns the ContentProvider (or ``None`` when content
    generation is disabled); ``pipeline_factory`` returns the SpeechPipeline
    (or ``None``). Both are callables so the coordinator never holds a model
    reference and never loads one itself.
    """

    def __init__(
        self,
        settings: Any,
        voice_profiles: Any,
        topic_source: TopicSource,
        *,
        provider_factory: Callable[[], Any],
        pipeline_factory: Callable[[], Any],
    ) -> None:
        self._settings = settings
        self._voice_profiles = voice_profiles
        self.topic_source = topic_source
        self._provider_factory = provider_factory
        self._pipeline_factory = pipeline_factory
        self._sessions: dict[int, PodcastSession] = {}

    def active(self, guild_id: int) -> bool:
        return guild_id in self._sessions

    def start(self, guild_id: int, host_profile_id: str, guest_profile_id: str) -> bool:
        """Start podcast mode for the guild. Returns ``False`` when refused."""
        if not self._settings.enabled:
            logger.info("Podcast mode is disabled in settings; not starting for guild %s", guild_id)
            return False
        if self._voice_profiles.get(host_profile_id) is None:
            logger.warning("Podcast host profile %r not found", host_profile_id)
            return False
        if self._voice_profiles.get(guest_profile_id) is None:
            logger.warning("Podcast guest profile %r not found", guest_profile_id)
            return False
        if guild_id in self._sessions:
            return True
        self._sessions[guild_id] = PodcastSession(guild_id, host_profile_id, guest_profile_id)
        logger.info(
            "Podcast mode started for guild %s (host=%s guest=%s)",
            guild_id,
            host_profile_id,
            guest_profile_id,
        )
        return True

    def stop(self, guild_id: int) -> bool:
        """Stop podcast mode. Returns ``True`` if a session was active."""
        session = self._sessions.pop(guild_id, None)
        if session is not None:
            logger.info("Podcast mode stopped for guild %s", guild_id)
        return session is not None

    async def segment_turns(self, guild_id: int) -> list[PodcastTurn]:
        """Generate the next dialogue segment for an active guild.

        Returns an empty list when the mode is inactive, content/speech is
        unavailable, no topic can be fetched, or generation fails — callers
        simply continue playback.
        """
        session = self._sessions.get(guild_id)
        if session is None:
            return []
        provider = self._provider_factory()
        pipeline = self._pipeline_factory()
        if provider is None or pipeline is None:
            logger.info("Content provider or speech pipeline unavailable; skipping podcast segment")
            return []
        topic = await self.topic_source.pick()
        if topic is None:
            logger.info("No podcast topic available; skipping podcast segment")
            return []
        host = self._voice_profiles.get(session.host_profile_id)
        guest = self._voice_profiles.get(session.guest_profile_id)
        if host is None or guest is None:
            logger.warning(
                "Podcast voice profiles missing; stopping podcast mode for guild %s", guild_id
            )
            self.stop(guild_id)
            return []
        engine = PodcastEngine(provider)
        return await engine.generate_segment(topic, host, guest, turns=self._settings.turns)
