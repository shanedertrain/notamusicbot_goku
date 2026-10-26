"""ContentProvider abstraction (REWORK_HANDOFF.md Phase F).

All commentary text (song announcements, news and reddit summaries) is
produced through :class:`ContentProvider`. Implementations must never block
the event loop: heavy work belongs in ``asyncio.to_thread`` under a bounded
concurrency guard (the local LLM provider allows exactly one generation at a
time). A provider failure must never crash playback — callers fall back to
the static template provider per handoff section 20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class ContentKind(Enum):
    """Which kind of commentary is being generated."""

    SONG = auto()
    NEWS = auto()
    REDDIT = auto()
    CHAT = auto()


@dataclass(frozen=True, slots=True)
class ContentRequest:
    """Everything a content provider may need to write one commentary line.

    ``persona_description`` comes from the voice profile's persona section
    (handoff section 14.3) — never from an RVC model object. Only the fields
    relevant to ``kind`` are populated; the rest stay empty.
    """

    kind: ContentKind
    persona_description: str = ""
    # Song commentary
    song_name: str = ""
    artist_name: str = ""
    requester_name: str = ""
    requester_background: str = ""
    # Article commentary
    article_text: str = ""
    post_title: str = ""
    post_text: str = ""
    # Listen-mode chat: what the speaker just said and the recent turns,
    # pre-rendered one line per turn (oldest first).
    transcript_text: str = ""
    chat_history: tuple[str, ...] = ()
    # Persona-specific listen-mode system prompt (voice profile ``persona.chat_prompt``);
    # empty = the default Goku ``prompts.CHAT_SYSTEM_PROMPT``.
    system_prompt: str = ""


@runtime_checkable
class ContentProvider(Protocol):
    """Async content generation boundary."""

    async def generate(self, request: ContentRequest) -> str:
        """Return commentary text for ``request``."""
        ...


@dataclass(slots=True)
class StaticTemplateProvider:
    """Deterministic fallback provider (legacy pre-Phase-F behavior).

    Used directly when content generation is disabled and as the failure
    fallback when the LLM provider raises (handoff section 20).
    """

    exceptions: list[Exception] = field(default_factory=list)

    async def generate(self, request: ContentRequest) -> str:
        # Imported lazily: prompts.py imports this module.
        from notamusicbot.content import prompts

        if request.kind is ContentKind.SONG:
            return prompts.static_song_comment(
                song_name=request.song_name,
                requester_name=request.requester_name,
            )
        if request.kind is ContentKind.NEWS:
            return prompts.static_article_comment(title=request.post_title)
        return prompts.static_article_comment(title=request.post_title)
