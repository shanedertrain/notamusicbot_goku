"""The Discord bot: composition between the app and the guild-scoped player.

The bot never touches ``bot.voice_clients[0]``; every voice client is resolved
from its owning guild (handoff section 12.2).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord
from discord.ext import commands

from notamusicbot.discord_bot.commands import setup_commands
from notamusicbot.discord_bot.guild_state import GuildState, GuildStateRegistry
from notamusicbot.discord_bot.player import GuildPlayer
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.media.downloader import download_audio
from notamusicbot.media.spotify import SpotifyResolver
from notamusicbot.media.youtube import YouTubeResolver

if TYPE_CHECKING:
    from notamusicbot.app import Application

logger = logging.getLogger(__name__)


class NotAMusicBot(commands.Bot):
    """commands.Bot wired to the typed application settings and media stack."""

    def __init__(self, app: Application, **kwargs: Any) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.typing = False
        intents.presences = False
        intents.members = True
        intents.voice_states = True
        super().__init__(
            command_prefix=app.settings.discord.command_prefix, intents=intents, **kwargs
        )
        self.app = app
        self.guild_states = GuildStateRegistry()
        self.resolver = YouTubeResolver.from_settings(app.settings.media)
        self.spotify_resolver = SpotifyResolver(
            client_id=app.secrets.spotify_client_id,
            client_secret=app.secrets.spotify_client_secret,
        )
        self._announce_channels: dict[int, Any] = {}

    def guild_state(self, guild_id: int) -> GuildState:
        return self.guild_states.get_or_create(guild_id)

    def set_announce_channel(self, guild_id: int, channel: Any) -> None:
        self._announce_channels[guild_id] = channel

    def _build_stream_downloader(self, guild_id: int):
        media_settings = self.app.settings.media
        clients = tuple(media_settings.yt_clients)
        cookies_file = media_settings.cookies_file

        async def _download(media: PlayableMedia) -> Path | None:
            try:
                return await download_audio(
                    media.stream_url or "",
                    build_opts=self.resolver.opts_for,
                    clients=clients,
                    cache_dir=Path(media_settings.cache_dir),
                )
            except Exception:
                logger.warning(
                    "yt-dlp download fallback failed for media %s in guild %s",
                    media.media_id,
                    guild_id,
                    exc_info=True,
                )
                return None

        if cookies_file.is_file():
            logger.debug("YouTube cookies file found (contents are never logged)")
        return _download

    def get_player(self, guild: discord.Guild) -> GuildPlayer:
        """Build (or conceptually re-bind) the guild's player.

        The player resolves the voice client from the guild on every use, so a
        fresh connection after ``!gstop`` is picked up automatically.
        """
        state: GuildState = self.guild_state(guild.id)
        return GuildPlayer(
            state,
            get_voice_client=lambda: guild.voice_client,
            stream_downloader=self._build_stream_downloader(guild.id),
            on_now_playing=self._make_now_playing_hook(guild.id),
        )

    def _make_now_playing_hook(self, guild_id: int):
        async def _announce(media: PlayableMedia) -> None:
            channel = self._announce_channels.get(guild_id)
            if channel is None or media.kind is not MediaKind.STREAM:
                return
            try:
                await channel.send(f"Now playing: {media.title} | Requester: {media.requester}")
            except Exception:
                logger.warning(
                    "Failed to send now-playing announcement in guild %s", guild_id, exc_info=True
                )

        return _announce

    async def setup_hook(self) -> None:
        # Load the persistent Qwen model off the event loop, then build the
        # shared speech pipeline that commands use for generated commentary.
        await asyncio.to_thread(self.app.load_speech_model)
        self.app.build_pipeline()
        setup_commands(self)
        logger.info("Commands registered with prefix %r", self.command_prefix)

    async def on_ready(self) -> None:
        logger.info(
            "Logged in as %s; connected to guilds: %s", self.user, [g.name for g in self.guilds]
        )


def build_bot(app: Application) -> NotAMusicBot:
    """Factory used by the composition root (keeps main import-time cheap)."""
    return NotAMusicBot(app)
