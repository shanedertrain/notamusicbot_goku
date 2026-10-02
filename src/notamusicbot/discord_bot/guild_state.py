"""Guild-scoped playback state.

Ownership per handoff section 12.2:

    guild -> voice client -> asyncio.Queue[PlayableMedia] -> playback task

Nothing here reaches for ``bot.voice_clients[0]``; the voice client is always
resolved from the owning guild.
"""

from __future__ import annotations

import asyncio
import logging

from notamusicbot.domain.media import PlayableMedia

logger = logging.getLogger(__name__)


class GuildState:
    """One guild's queue and playback-task handles."""

    def __init__(self, guild_id: int) -> None:
        self.guild_id = guild_id
        self.queue: asyncio.Queue[PlayableMedia] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.playback_done = asyncio.Event()
        self.player_task: asyncio.Task[None] | None = None

    def drain_queue(self) -> list[PlayableMedia]:
        """Remove and return every queued item (callers clean up temp files)."""
        drained: list[PlayableMedia] = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return drained

    def request_stop(self) -> None:
        """Signal the playback loop to exit after the current item."""
        self.stop_event.set()

    def reset(self) -> None:
        """Prepare the state for a fresh session (new stop cycle)."""
        self.stop_event.clear()
        self.playback_done.clear()


class GuildStateRegistry:
    """``guild_id -> GuildState`` map with lazy creation."""

    def __init__(self) -> None:
        self._states: dict[int, GuildState] = {}

    def get_or_create(self, guild_id: int) -> GuildState:
        state = self._states.get(guild_id)
        if state is None:
            state = GuildState(guild_id)
            self._states[guild_id] = state
            logger.debug("Created guild state for guild %s", guild_id)
        return state

    def get(self, guild_id: int) -> GuildState | None:
        return self._states.get(guild_id)

    def remove(self, guild_id: int) -> GuildState | None:
        return self._states.pop(guild_id, None)
