"""Podcast conversation mode (review blocker P0-5 / Gate R6).

Modern port of the legacy ``legacy_flat/podcast.py`` capability through the new
abstractions: a host persona and a guest persona hold a short two-voice
conversation about a fetched news/reddit topic. Every turn's text is produced
by a :class:`~notamusicbot.content.base.ContentProvider` and every turn's audio
is synthesized through the :class:`~notamusicbot.speech.pipeline.SpeechPipeline`
with the speaker's voice profile, then enqueued in order after the song that
just finished. Simpler than the legacy infinite Gemini loop — one bounded
segment per finished song — but fully functional and fail-soft.
"""

from notamusicbot.podcast.coordinator import PodcastCoordinator
from notamusicbot.podcast.engine import PodcastEngine, PodcastTurn

__all__ = ["PodcastCoordinator", "PodcastEngine", "PodcastTurn"]
