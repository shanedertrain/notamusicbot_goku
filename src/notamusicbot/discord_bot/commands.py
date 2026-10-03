"""Prefix commands (!g...) preserving legacy semantics.

Commands: play, playnow, play_spotify, join, stop, skip, deleteallmessages.
State access is guild-scoped (``ctx.guild.voice_client`` + per-guild queue);
the speech pipeline is used only for generated commentary and its failure
never blocks media playback (handoff section 20).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from discord import Guild, VoiceClient
from discord.ext import commands
from discord.ext.commands import Context

from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.discord_bot.commentary import (
    generate_commentary_media,
    resolve_commentary_text,
)
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.speech import TtsRequest
from notamusicbot.domain.users import UserProfile
from notamusicbot.domain.voice_profiles import to_conversion_profile, to_processing_profile

if TYPE_CHECKING:
    from notamusicbot.discord_bot.bot import NotAMusicBot

logger = logging.getLogger(__name__)

YOUTUBE_HOSTS = ("youtube.com", "youtu.be")


def _guild(ctx: Context) -> Guild:
    """Return the invocation guild or raise (legacy commands are guild-only).

    ``Context.guild`` is ``Guild | None`` in discord.py 2.x; every command
    below only runs in a guild, so narrow it once instead of per-attribute.
    """
    guild = ctx.guild
    if guild is None:
        raise commands.CommandError("command must be used inside a guild")
    return guild


def _requester(bot: NotAMusicBot, ctx: Context) -> UserProfile:
    user = bot.app.users.lookup(str(ctx.author.id))
    if user is not None:
        return user
    return UserProfile(user_id=str(ctx.author.id), screen_name=getattr(ctx.author, "name", ""))


async def _join_and_start(bot: NotAMusicBot, ctx: Context) -> None:
    """Legacy ``play_init_funcs``: connect to voice and ensure the player runs."""
    voice_client = _guild(ctx).voice_client
    if not voice_client:
        # In a guild the author is a Member, which carries .voice.
        voice = getattr(ctx.author, "voice", None)
        if voice is None or voice.channel is None:
            await ctx.send("You need to be in a voice channel.")
            raise commands.CommandError("invoker is not in a voice channel")
        await voice.channel.connect()
        start_sound = bot.app.settings.discord.start_sound
        if start_sound is not None and Path(start_sound).is_file():
            state = bot.guild_state(_guild(ctx).id)
            state.queue.put_nowait(
                PlayableMedia(
                    media_id="start_sound",
                    kind=MediaKind.LOCAL,
                    title="start sound",
                    guild_id=_guild(ctx).id,
                    source_path=Path(start_sound),
                )
            )
    bot.set_announce_channel(_guild(ctx).id, ctx.channel)
    bot.get_player(_guild(ctx)).start()


async def _enqueue_commentary(
    bot: NotAMusicBot,
    ctx: Context,
    user: UserProfile,
    *,
    song_name: str = "",
    artist_name: str = "",
    post_title: str = "",
    article_text: str = "",
) -> None:
    """Generate commentary via the ContentProvider, then TTS, and enqueue it.

    The provider receives the persona description from the user's voice
    profile and the requester's background from ``config/users.yaml``. Text
    generation failures fall back to the static template (handoff section 20);
    speech failures skip the commentary entirely — the caller's media is
    enqueued regardless.
    """
    speech_settings = bot.app.settings.speech
    if not speech_settings.commentary_enabled or bot.app.pipeline is None:
        return
    profile = bot.app.voice_profiles.get(user.voice_profile)
    if profile is None:
        logger.warning(
            "No voice profile %r for user %s; skipping commentary", user.voice_profile, user.user_id
        )
        return
    requester_name = user.real_name or user.screen_name
    if song_name:
        kind = ContentKind.SONG
        artist_name = artist_name or "an unknown artist"
    else:
        kind = ContentKind.NEWS
    content_request = ContentRequest(
        kind=kind,
        persona_description=profile.persona.description,
        song_name=song_name,
        artist_name=artist_name,
        requester_name=requester_name,
        requester_background=user.background,
        post_title=post_title,
        article_text=article_text,
    )
    text = await resolve_commentary_text(bot.app.build_content_provider(), content_request)
    request = TtsRequest(text=text, language=profile.tts.language, speaker=profile.tts.speaker)
    media = await generate_commentary_media(
        bot.app.pipeline,
        request,
        to_conversion_profile(profile),
        guild_id=_guild(ctx).id,
        requester=user.screen_name,
        # Per-profile audio settings (speed, target peak) — same mapping the
        # speech_test CLI uses, so the production path matches the tested one.
        processing=to_processing_profile(
            profile,
            sample_rate_hz=speech_settings.sample_rate_hz,
            channels=speech_settings.channels,
        ),
    )
    if media is not None:
        bot.guild_state(_guild(ctx).id).queue.put_nowait(media)


def setup_commands(bot: NotAMusicBot) -> None:
    @bot.command(name="play", help="Add a video or playlist to the queue and start playing")
    async def play(ctx: Context, url: str) -> None:
        try:
            await _join_and_start(bot, ctx)
            await _enqueue_youtube(bot, ctx, url, quick=False)
        except commands.CommandError:
            raise
        except Exception as exc:
            logger.error("Error processing video: %s", exc, exc_info=True)
            await ctx.send(f"Error processing video: {exc}")

    @bot.command(
        name="playnow", help="Add a video or playlist to the queue and start playing immediately"
    )
    async def playnow(ctx: Context, url: str) -> None:
        try:
            await _join_and_start(bot, ctx)
            await _enqueue_youtube(bot, ctx, url, quick=True)
        except commands.CommandError:
            raise
        except Exception as exc:
            logger.error("Error processing video: %s", exc, exc_info=True)
            await ctx.send(f"Error processing video: {exc}")

    @bot.command(name="play_spotify", help="Add spotify song to the queue and start playing")
    async def play_spotify(ctx: Context, url: str) -> None:
        try:
            await _join_and_start(bot, ctx)
            user = _requester(bot, ctx)
            track = await bot.spotify_resolver.search_track(url)
            if track is None:
                await ctx.send("Track not found on Spotify.")
                return
            media = await bot.resolver.resolve(
                f"ytsearch1:{track.search_query}", media_id=uuid.uuid4().hex[:12]
            )
            media.requester = user.screen_name
            media.guild_id = _guild(ctx).id
            first_artist = track.artists[0] if track.artists else "unknown artist"
            # Unify through media.artist (review Round-4 P1): the queued media
            # carries the artist too, not just the commentary call.
            media.artist = first_artist
            await _enqueue_commentary(
                bot,
                ctx,
                user,
                song_name=track.name,
                artist_name=first_artist,
            )
            bot.guild_state(_guild(ctx).id).queue.put_nowait(media)
        except commands.CommandError:
            raise
        except Exception as exc:
            logger.error("Error processing spotify track: %s", exc, exc_info=True)
            await ctx.send(f"Error processing spotify track: {exc}")

    @bot.command(name="join", help="Joins the voice channel of the user who invoked the command")
    async def join_channel(ctx: Context) -> None:
        await _join_and_start(bot, ctx)

    @bot.command(name="stop", help="Stops playing the audio and disconnects from the voice channel")
    async def stop(ctx: Context) -> None:
        await bot.get_player(_guild(ctx)).stop()
        await _leave(bot, ctx)

    @bot.command(name="skip", help="Skip the current song and move to the next in the playlist")
    async def skip(ctx: Context) -> None:
        voice_client = _guild(ctx).voice_client
        # VoiceClient (not the abstract VoiceProtocol) carries is_playing().
        if isinstance(voice_client, VoiceClient) and voice_client.is_playing():
            bot.get_player(_guild(ctx)).skip()
            await ctx.send("Skipping current song.")
            if bot.guild_state(_guild(ctx).id).queue.empty():
                await ctx.send("No more songs in queue!")
        else:
            await ctx.send("No song is currently playing.")

    @bot.command(
        name="podcast",
        help="Start or stop podcast mode: after each song the bot plays a short "
        "two-voice conversation (!gpodcast start|stop)",
    )
    async def podcast(ctx: Context, action: str = "start") -> None:
        action = action.lower()
        if action not in ("start", "stop"):
            await ctx.send("Usage: !gpodcast start|stop")
            return
        guild_id = ctx.guild.id if ctx.guild else 0
        if action == "stop":
            stopped = bot.podcasts.stop(guild_id)
            await ctx.send("Podcast mode stopped." if stopped else "Podcast mode was not active.")
            return
        user = _requester(bot, ctx)
        guest_profile = user.voice_profile or bot.app.settings.podcast.host_profile
        started = bot.podcasts.start(
            guild_id,
            host_profile_id=bot.app.settings.podcast.host_profile,
            guest_profile_id=guest_profile,
        )
        if started:
            await ctx.send("Podcast mode on: after each song you'll hear a short conversation.")
        else:
            await ctx.send(
                "Could not start podcast mode (disabled in settings, unknown voice "
                "profile, or already active)."
            )

    @bot.command(
        name="deleteallmessages",
        help="Deletes all messages posted by this bot in the current channel",
    )
    async def delete_bot_messages(ctx: Context) -> None:
        # In a guild the author is a Member with guild_permissions.
        permissions = getattr(ctx.author, "guild_permissions", None)
        if permissions is None or not permissions.administrator:
            return
        delay = bot.app.settings.discord.message_delete_delay_s
        async for message in ctx.channel.history(limit=None):
            if message.author == bot.user:
                await message.delete()
                await asyncio.sleep(delay)


async def _leave(bot: NotAMusicBot, ctx: Context) -> None:
    voice_client = _guild(ctx).voice_client
    if isinstance(voice_client, VoiceClient) and voice_client.is_connected():
        await voice_client.disconnect()
        await ctx.send("Disconnected from voice channel.")
    else:
        await ctx.send("The bot is not connected to a voice channel.")


async def _enqueue_youtube(bot: NotAMusicBot, ctx: Context, url: str, *, quick: bool) -> None:
    if not any(host in url for host in YOUTUBE_HOSTS):
        await ctx.send("Invalid YouTube URL.")
        return
    user = _requester(bot, ctx)
    media_id = uuid.uuid4().hex[:12]
    media = await bot.resolver.resolve(url, media_id=media_id)
    media.requester = user.screen_name
    media.guild_id = _guild(ctx).id
    if not quick:
        # Review Round-4 P1: preserve the YouTube uploader as the commentary
        # artist (legacy parity) instead of "an unknown artist".
        await _enqueue_commentary(
            bot, ctx, user, song_name=media.title, artist_name=media.artist or ""
        )
    bot.guild_state(_guild(ctx).id).queue.put_nowait(media)
