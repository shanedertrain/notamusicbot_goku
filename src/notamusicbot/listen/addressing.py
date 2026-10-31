"""Is this line said to the character? A small LLM call for rooms with several people.

With two or more humans in the channel the bot must not answer everything. Its
name is a certain yes (checked by the caller); for the rest the listen LLM reads
the last few lines of the room and says whether the newest one is meant for the
character. Any failure or timeout counts as "no": a silent bot is a smaller
mistake than one that cuts into other people's talk.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Protocol

from notamusicbot.content.base import ContentKind, ContentRequest

logger = logging.getLogger(__name__)

# Lines of room context the judge sees.
CONTEXT_LINES = 6
# No answer in this time counts as "not addressed".
JUDGE_TIMEOUT_S = 1.5

_SYSTEM = (
    "You listen to a live voice chat between several people and a character named {name}. "
    "Decide whether the NEWEST line is said to {name}. It is only when the speaker clearly "
    "talks to {name}, calls them by name or nickname, or answers or reacts to what {name} "
    "said in the line right before it. People talking to each other, or to nobody, are not "
    "talking to {name}, even when the line is a question. When in doubt, no.\n"
    "Examples:\n"
    "Ann: should we get pizza? / Sam: what do you think about that -> no\n"
    "{name}: I would wait a day. / Sam: yeah that makes sense -> yes\n"
    "Ann: this soup is great / Sam: pass the salt -> no\n"
    "Sam: hey {name}, how is it going -> yes\n"
    "Answer with exactly one word: yes or no."
)


class _Provider(Protocol):
    async def generate(self, request: ContentRequest) -> str: ...


class AddressJudge:
    """Asks the listen LLM whether the character is being addressed."""

    def __init__(
        self, provider: _Provider, *, character: str, timeout_s: float = JUDGE_TIMEOUT_S
    ) -> None:
        self._provider = provider
        self._character = character
        self._timeout_s = timeout_s

    async def addressed(self, speaker: str, text: str, room: Sequence[str]) -> bool:
        started = time.perf_counter()
        context = tuple(room[-CONTEXT_LINES:])
        if context and context[-1].startswith(f"{self._character}:"):
            # The strongest cue for a follow-up, stated outright for a small model.
            context = (*context, f"(That was {self._character} speaking just now.)")
        request = ContentRequest(
            kind=ContentKind.CHAT,
            persona_description="",
            requester_name=speaker,
            transcript_text=text,
            chat_history=context,
            system_prompt=_SYSTEM.format(name=self._character),
        )
        try:
            answer = await asyncio.wait_for(self._provider.generate(request), self._timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure means "do not answer"
            logger.warning("Address judge failed (%s); ignoring the line", type(exc).__name__)
            return False
        verdict = answer.strip().lower().startswith("y")
        logger.info(
            "listen judge addressed=%s answer=%r latency=%.2fs",
            verdict,
            answer.strip()[:12],
            time.perf_counter() - started,
        )
        return verdict
