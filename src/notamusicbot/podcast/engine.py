"""Podcast conversation engine: personas -> ContentProvider -> turn scripts.

Each turn is generated through the shared ContentProvider (no new LLM path):
the host opening reuses the reddit-post prompt shape (announce the topic title
and summarize it, exactly like legacy ``generate_article_post_opening``), and
the guest/host replies carry the previous line and the show instructions inside
the request's post text. The engine returns turn scripts only; synthesis and
queueing happen in the playback glue (``discord_bot.interlude_playback``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from notamusicbot.content.base import ContentKind, ContentProvider, ContentRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PodcastTurn:
    """One line of dialogue, bound to the speaker's voice profile."""

    voice_profile_id: str
    speaker_name: str
    text: str


def _scene_notes(previous_speaker: str, previous_line: str, instruction: str) -> str:
    return (
        "\n\n[SHOW NOTES - this part is not part of the post and is not to be read aloud]: "
        f'{previous_speaker} just said on the show: "{previous_line}". {instruction}'
    )


class PodcastEngine:
    """Generates a bounded multi-turn dialogue via a ContentProvider."""

    def __init__(self, provider: ContentProvider) -> None:
        self._provider = provider

    async def generate_segment(
        self,
        topic: Any,
        host_profile: Any,
        guest_profile: Any,
        *,
        turns: int = 2,
    ) -> list[PodcastTurn]:
        """Generate ``turns`` lines alternating host/guest, host first.

        ``topic`` is a :class:`~notamusicbot.content.interstitials.TopicItem`;
        profiles are voice profiles exposing ``id``,
        ``persona.display_name`` and ``persona.description``. Any
        provider failure aborts the segment: an empty list is returned and the
        caller continues playback.
        """
        speakers = [host_profile, guest_profile]
        try:
            script: list[PodcastTurn] = []
            # Host opening: introduce the topic (legacy opening prompt shape).
            opening = await self._provider.generate(
                ContentRequest(
                    kind=ContentKind.REDDIT,
                    persona_description=host_profile.persona.description,
                    post_title=topic.title,
                    post_text=topic.text,
                )
            )
            script.append(PodcastTurn(host_profile.id, host_profile.persona.display_name, opening))
            last_speaker, last_line = host_profile.persona.display_name, opening
            for index in range(max(1, turns)):
                speaker = speakers[(index + 1) % 2]
                reply = await self._provider.generate(
                    ContentRequest(
                        kind=ContentKind.REDDIT,
                        persona_description=speaker.persona.description,
                        post_title=topic.title,
                        post_text=topic.text
                        + _scene_notes(
                            last_speaker,
                            last_line,
                            "Respond to them directly in 1-2 sentences, in character.",
                        ),
                    )
                )
                script.append(PodcastTurn(speaker.id, speaker.persona.display_name, reply))
                last_speaker, last_line = speaker.persona.display_name, reply
            return script
        except Exception:
            logger.warning("Podcast segment generation failed; aborting the segment", exc_info=True)
            return []
