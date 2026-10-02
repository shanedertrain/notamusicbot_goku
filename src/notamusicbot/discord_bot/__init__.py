"""Discord layer: guild-scoped state, event-driven player, and commands.

Importing this package must NOT require discord.py (the ordinary unit suite
runs without it); ``NotAMusicBot``/``build_bot`` are imported lazily on first
attribute access. Modules that genuinely need discord import it themselves
(e.g. ``bot.py``, ``commands.py``); ``player.py`` imports it only inside its
source factory function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from notamusicbot.discord_bot.bot import NotAMusicBot, build_bot

__all__ = ["NotAMusicBot", "build_bot"]


def __getattr__(name: str) -> object:
    if name in ("NotAMusicBot", "build_bot"):
        from notamusicbot.discord_bot import bot

        return getattr(bot, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
