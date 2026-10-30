"""Round-3 command-surface tests (review section 9).

The bot prefix is ``!g``; the podcast command must therefore be registered as
``podcast`` so the public command is ``!gpodcast`` — not ``!ggpodcast``.
"""

from __future__ import annotations

import pytest

# Review Round-3 P1: the ordinary unit suite must run without the optional
# heavy runtimes installed. Skip the whole module when discord.py is absent.
pytest.importorskip("discord", reason="discord.py is an optional runtime dependency")

import discord
from discord.ext import commands

from notamusicbot.discord_bot.commands import setup_commands

PREFIX = "!g"
PUBLIC_COMMANDS = (
    "play",
    "playnow",
    "play_spotify",
    "join",
    "stop",
    "skip",
    "podcast",
    "deleteallmessages",
    "help",
)


def build_registered_bot() -> commands.Bot:
    """A bare commands.Bot with the production command registration applied.

    ``setup_commands`` only registers callbacks at setup time; command bodies
    need a real ``NotAMusicBot``, which this test does not invoke.
    """
    bot = commands.Bot(command_prefix=PREFIX, intents=discord.Intents.none())
    setup_commands(bot)
    return bot


def test_all_public_commands_resolve_under_g_prefix() -> None:
    bot = build_registered_bot()
    registered = set(bot.all_commands)
    missing = [name for name in PUBLIC_COMMANDS if name not in registered]
    assert missing == [], f"missing commands under prefix {PREFIX}: {missing}"


def test_podcast_command_is_named_podcast_not_gpodcast() -> None:
    bot = build_registered_bot()
    assert bot.get_command("podcast") is not None
    # The old double-prefixed name must be gone.
    assert bot.get_command("gpodcast") is None
    # With the configured prefix the user types '!gpodcast'.
    assert f"{PREFIX}podcast" == "!gpodcast"


def test_listen_group_is_registered_with_its_subcommands() -> None:
    bot = build_registered_bot()
    group = bot.get_command("listen")
    assert isinstance(group, commands.Group)
    assert bot.get_command("glisten") is None  # no double prefix
    # The per-user opt-in commands are gone; start/stop/status remain.
    assert {c.name for c in group.commands} == {"start", "stop", "status"}
    assert bot.get_command("listen on") is None
    assert bot.get_command("listen off") is None
