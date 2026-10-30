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

from notamusicbot.content.interstitials import InterstitialManager, TopicSource
from notamusicbot.content.news import NewsFetcher
from notamusicbot.content.reddit import RedditPostFetcher
from notamusicbot.discord_bot.commands import setup_commands
from notamusicbot.discord_bot.guild_state import GuildState, GuildStateRegistry
from notamusicbot.discord_bot.interlude_playback import fire_post_song_content
from notamusicbot.discord_bot.player import GuildPlayer, stop_guild_players
from notamusicbot.discord_bot.prefetch import CommentaryPrefetcher
from notamusicbot.domain.media import MediaKind, PlayableMedia, PlaybackOutcome
from notamusicbot.media.downloader import download_media_audio
from notamusicbot.media.spotify import SpotifyResolver
from notamusicbot.media.youtube import YouTubeResolver
from notamusicbot.podcast import PodcastCoordinator

if TYPE_CHECKING:
    from notamusicbot.app import Application
    from notamusicbot.discord_bot.listen_glue import ListenSession, ListenSessionRegistry

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
        # Voice listen mode. The registry is built on first use so listen_glue
        # (and its heavy imports) load only when a session is requested.
        self._listen_sessions: ListenSessionRegistry | None = None
        self._listen_tasks: set[asyncio.Task[None]] = set()
        # Post-song content (review P0-5): randomized news/reddit interstitials
        # and the !gpodcast conversation mode. All fetchers are lazy/fail-soft;
        # nothing connects at construction time.
        content_settings = app.settings.content
        self.news_fetcher = NewsFetcher(
            app.secrets.news_api_key,
        )
        self.reddit_fetcher = RedditPostFetcher.from_settings(
            app.secrets.reddit_client_id,
            app.secrets.reddit_client_secret,
            app.settings.reddit,
        )
        self.topic_source = TopicSource(
            self.news_fetcher,
            self.reddit_fetcher,
            news_country=app.settings.interstitial.news_country,
            news_category=app.settings.interstitial.news_category,
        )
        self.interstitials = InterstitialManager(app.settings.interstitial, self.topic_source)
        self.podcasts = PodcastCoordinator(
            app.settings.podcast,
            app.voice_profiles,
            self.topic_source,
            provider_factory=app.build_content_provider,
            pipeline_factory=lambda: app.pipeline,
        )
        # Background prefetch of post-song commentary while a STREAM song
        # plays (speech/content.prefetch_enabled). Generation goes through
        # the shared InferenceCoordinator, so it never overlaps other GPU
        # inference; see discord_bot/prefetch.py for the lifecycle.
        self.prefetch = CommentaryPrefetcher()
        if content_settings.enabled:
            logger.info(
                "Content generation enabled; interstitials=%s podcast=%s",
                app.settings.interstitial.enabled,
                app.settings.podcast.enabled,
            )

    def guild_state(self, guild_id: int) -> GuildState:
        return self.guild_states.get_or_create(guild_id)

    @property
    def listen_sessions(self) -> ListenSessionRegistry:
        if self._listen_sessions is None:
            from notamusicbot.discord_bot.listen_glue import ListenSessionRegistry

            self._listen_sessions = ListenSessionRegistry()
        return self._listen_sessions

    def make_listen_session(
        self, guild: discord.Guild, channel: discord.VoiceChannel | discord.StageChannel
    ) -> ListenSession:
        from notamusicbot.discord_bot.listen_glue import ListenSession

        return ListenSession(self.app, self, guild, channel)

    async def stop_listen_session(self, guild_id: int) -> None:
        """Stop and unregister a guild's session (idempotent; never raises).

        The session stays registered until ``stop`` returns, so music commands
        keep refusing while the voice client is being torn down.
        """
        registry = self._listen_sessions
        session = registry.get(guild_id) if registry is not None else None
        if registry is None or session is None:
            return
        try:
            await session.stop()
        except Exception:
            logger.warning("Listen session stop failed in guild %s", guild_id, exc_info=True)
        finally:
            if registry.get(guild_id) is session:
                registry.remove(guild_id)

    async def stop_all_listen_sessions(self) -> None:
        if self._listen_sessions is None:
            return
        for guild in list(self.guilds):
            await self.stop_listen_session(guild.id)

    def _spawn_listen_stop(self, guild_id: int) -> None:
        task = asyncio.create_task(self.stop_listen_session(guild_id))
        self._listen_tasks.add(task)
        task.add_done_callback(self._listen_tasks.discard)

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        """Listen mode upkeep: clear a leaver's buffered audio at once; stop when nobody is left."""
        registry = self._listen_sessions
        session = registry.get(member.guild.id) if registry is not None else None
        if session is None:
            return
        channel_id = session.channel_id
        if self.user is not None and member.id == self.user.id:
            # The bot was disconnected or moved out of the session channel.
            if after.channel is None or after.channel.id != channel_id:
                self._spawn_listen_stop(member.guild.id)
            return
        left = before.channel is not None and before.channel.id == channel_id
        if not left or (after.channel is not None and after.channel.id == channel_id):
            return
        session.on_member_left(member.id)  # synchronous buffer cleanup
        assert before.channel is not None
        if not any(not m.bot for m in before.channel.members):
            self._spawn_listen_stop(member.guild.id)

    def set_announce_channel(self, guild_id: int, channel: Any) -> None:
        self._announce_channels[guild_id] = channel

    def _build_stream_downloader(self, guild_id: int):
        media_settings = self.app.settings.media
        clients = tuple(media_settings.yt_clients)
        cookies_file = media_settings.cookies_file

        async def _download(media: PlayableMedia) -> Path | None:
            try:
                return await download_media_audio(
                    media,
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
            on_playback_complete=self._make_playback_complete_hook(guild.id),
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
            # Kick off post-song commentary prefetch while the song plays.
            # Fire-and-forget: start_prefetch only spawns a task, but guard it
            # anyway so announcement failures can never touch playback.
            try:
                self.prefetch.start_prefetch(self, guild_id, media)
            except Exception:
                logger.warning("Commentary prefetch failed to start in guild %s", guild_id)

        return _announce

    def _make_playback_complete_hook(self, guild_id: int):
        """Player completion hook (review Round-2 P0, Round-3 P0).

        Invoked by the player ONLY after a STREAM item ended NATURAL (see
        ``player.run``), with the outcome of that attempt; the bot-side caller
        double-checks ``outcome is NATURAL`` before generating post-song
        content, so skip/stop/failure never start an LLM → TTS → RVC run.
        LOCAL/generated items never reach this hook (the player gates on
        ``media.kind == STREAM``), so generated audio cannot recurse into more
        post-song content.
        """

        async def _on_complete(media: PlayableMedia, outcome: PlaybackOutcome) -> None:
            if media.kind is not MediaKind.STREAM or outcome is not PlaybackOutcome.NATURAL:
                return
            # Fast path: a prefetched artifact for THIS song that is ready is
            # enqueued immediately (near-zero wait, no second LLM/TTS run).
            # Any non-usable prefetch (still running / failed / not fired /
            # different song) is discarded here, and the at-completion
            # generation path below runs exactly as before prefetching.
            try:
                if await self.prefetch.consume(self, guild_id, media):
                    return
            except Exception:
                logger.warning(
                    "Prefetch consume failed in guild %s; falling back", guild_id, exc_info=True
                )
            try:
                await fire_post_song_content(self, guild_id)
            except Exception:
                logger.warning(
                    "Post-song content hook failed in guild %s; continuing", guild_id, exc_info=True
                )

        return _on_complete

    async def setup_hook(self) -> None:
        # Load the persistent Qwen model off the event loop, then build the
        # shared speech pipeline that commands use for generated commentary.
        await asyncio.to_thread(self.app.load_speech_model)
        self.app.build_pipeline()
        # Start the voice-conversion router (RVC worker spawn + health check)
        # and the optional non-blocking warmup; both fail soft (logged).
        await self.app.startup()
        # Local LLM (content provider) is loaded inside app.startup() — before
        # RVC start / warmup scheduling — so model loading and warmup can never
        # overlap GPU ownership; a second call here is an idempotent no-op.
        setup_commands(self)
        logger.info("Commands registered with prefix %r", self.command_prefix)

    async def close(self) -> None:
        """discord.py teardown: players first, then gateway, then application.

        Review Round-6 §3/B: guild player tasks are stopped BEFORE the
        application's converter teardown, so a player (or its post-song
        content hook) can never trigger a speech chain that races — or lazily
        restarts — the RVC worker after shutdown.
        """
        await self.stop_all_listen_sessions()
        await self._stop_guild_players()
        self.prefetch.discard_all()
        try:
            await super().close()
        finally:
            await self.app.shutdown()

    async def _stop_guild_players(self) -> None:
        """Stop every guild player task BEFORE converter teardown (§3/B)."""
        await stop_guild_players(self.guild_states.all_states())

    async def on_ready(self) -> None:
        logger.info(
            "Logged in as %s; connected to guilds: %s", self.user, [g.name for g in self.guilds]
        )


def build_bot(app: Application) -> NotAMusicBot:
    """Factory used by the composition root (keeps main import-time cheap)."""
    return NotAMusicBot(app)
