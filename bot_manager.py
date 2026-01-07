import asyncio
import os
import random
import uuid
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import discord

# Workaround for discord.py voice gateway returning an empty "modes" list.
import discord.gateway as _dg
from discord.ext import commands
from discord.ext.commands import Context
from dotenv import load_dotenv

import audio_generator as ag
import configuration as cfg
import types_playlist_items as tpi
import users
from audio_player import AudioPlayer
from rvc_cli import models

if not hasattr(_dg.DiscordVoiceWebSocket, "_patched_empty_modes"):
    _orig_initial_connection = _dg.DiscordVoiceWebSocket.initial_connection

    async def _initial_connection_safe(self, data):
        # Ensure the modes list is populated (and compatible) before passing to the original handler
        payload = dict(data.get("d") or data)
        requested_modes = payload.get("modes") or []

        # Keep only modes supported by the client; if none remain, fall back to a safe order.
        supported_modes = list(getattr(self, "_connection", None).supported_modes or [])
        preferred_order = [
            "aead_xchacha20_poly1305_rtpsize",
            "xsalsa20_poly1305",
            "xsalsa20_poly1305_suffix",
            "xsalsa20_poly1305_lite",
        ]
        modes = [m for m in requested_modes if m in supported_modes]
        if not modes:
            modes = [m for m in preferred_order if (not supported_modes or m in supported_modes)] or preferred_order

        payload["modes"] = modes

        # Extra logging to debug odd handshake payloads and close codes
        cfg.LOGGER.info(
            "Voice ready payload patched: requested=%s supported=%s chosen=%s ip=%s port=%s",
            requested_modes or [],
            supported_modes or [],
            modes,
            payload.get("ip"),
            payload.get("port"),
        )

        if "d" in data:
            new_data = dict(data)
            new_data["d"] = payload
        else:
            new_data = payload

        try:
            return await _orig_initial_connection(self, new_data)
        except Exception:
            cfg.LOGGER.exception(
                "Voice initial_connection failed: requested=%s supported=%s chosen=%s payload=%s raw=%s",
                requested_modes or [],
                supported_modes or [],
                modes,
                payload,
                data,
            )
            raise

    _dg.DiscordVoiceWebSocket.initial_connection = _initial_connection_safe
    _dg.DiscordVoiceWebSocket._patched_empty_modes = True

load_dotenv()
GUILD = os.getenv("DISCORD_GUILD")
FILEPATH_START_SOUND = cfg.FOLDER_INPUT / os.getenv("FILENAME_START_SOUND")


class BotManager:
    audio_player_task: asyncio.Task = None

    def __init__(
        self,
        command_prefix: str,
        intents: discord.Intents = discord.Intents.default(),
        output_folder: Path = cfg.FOLDER_OUTPUT,
    ):
        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self.audio_player = AudioPlayer(self.bot)
        self.output_folder = output_folder
        self.register_events()
        self.register_commands()

    def register_events(self):
        @self.bot.event
        async def on_ready():
            guild = discord.utils.find(lambda g: g.name == GUILD, self.bot.guilds)
            cfg.LOGGER.info(f"{self.bot.user} is connected to the following guild:\n{guild.name}(id: {guild.id})")
            cfg.LOGGER.info("Ready...")
            await users.get_users_from_guild(guild)

        @self.bot.event
        async def on_error(event, *args, **kwargs):
            with open(cfg.FOLDER_LOGS / "bot_errors.log", "a") as f:
                if event == "on_message":
                    f.write(f"Unhandled message: {args[0]}\n")
                else:
                    raise

    def register_commands(self):
        @self.bot.command(name="play", help="Add a video or playlist to the queue and start playing")
        async def play(ctx: Context, url: str):
            try:
                await self.play_init_funcs(ctx)
                requester = users.get_user_by_id(ctx.author.id)
                media_uid = str(uuid.uuid4())
                await self.queue_music_youtube(url, ctx, requester, media_uid)

            except Exception as e:
                cfg.LOGGER.error(f"Error processing video: {e}", exc_info=True)
                await ctx.send(f"Error processing video: {e}")

        @self.bot.command(name="playnow", help="Add a video or playlist to the queue and start playing immediately")
        async def playnow(ctx: Context, url: str):
            try:
                await self.play_init_funcs(ctx)
                requester = users.get_user_by_id(ctx.author.id)
                media_uid = str(uuid.uuid4())
                await self.queue_music_youtube(url, ctx, requester, media_uid, quick=True)

            except Exception as e:
                cfg.LOGGER.error(f"Error processing video: {e}", exc_info=True)
                await ctx.send(f"Error processing video: {e}")

        @self.bot.command(name="play_spotify", help="Add spotify song to the queue and start playing")
        async def play_spotify(ctx, url: str):
            try:
                await self.play_init_funcs(ctx)
                requester = users.get_user_by_id(ctx.author.id)
                media_uid = uuid.uuid4()
                await self.queue_music_spotify(url, ctx, requester, media_uid)

            except Exception as e:
                cfg.LOGGER.error(f"Error processing spotify track: {e}", exc_info=True)
                await ctx.send(f"Error processing spotify track: {e}")

        @self.bot.command(name="join", help="Joins the voice channel of the user who invoked the command")
        async def join_channel(ctx):
            await self.play_init_funcs(ctx)

        @self.bot.command(name="stop", help="Stops playing the audio and disconnects from the voice channel")
        async def stop(ctx):
            """
            Stops the audio player task and disconnects the bot from the voice channel.
            """

            self.audio_player.clear_playlist()
            if self.audio_player_task and not self.audio_player_task.done():
                await self.audio_player_task
            if self.audio_player_task and not self.audio_player_task.done():
                self.audio_player_task.cancel()
            await self.leave(ctx)

        @self.bot.command(name="skip", help="Skip the current song and move to the next in the playlist")
        async def skip(ctx):
            """
            Skips the current song and moves to the next in the playlist.
            """
            voice_client = self.bot.voice_clients[0] if self.bot.voice_clients else None
            if voice_client and voice_client.is_playing():
                voice_client.stop()
                await ctx.send("Skipping current song.")
                if len(self.audio_player.playlist) == 0:
                    await ctx.send("No more songs in queue!")
            else:
                await ctx.send("No song is currently playing.")

        @self.bot.command(
            name="deleteallmessages", help="Deletes all messages posted by this bot in the current channel"
        )
        async def delete_bot_messages(ctx):
            if ctx.author.guild_permissions.administrator:
                channel = ctx.channel
                async for message in channel.history(limit=None):
                    if message.author == ctx.bot.user:
                        await message.delete()
                        await asyncio.sleep(1)

    async def play_init_funcs(self, ctx: Context):
        users.USERS = users.read_users_from_json_file(cfg.FILEPATH_USERS)  # reload users from file
        models.MODELS = models.collect_models_from_folders()  # reload models

        voice_client = ctx.guild.voice_client
        if not voice_client:
            await ctx.author.voice.channel.connect()
            if not cfg.DEBUG:
                self.audio_player.add_to_playlist(tpi.Audio(filepath=FILEPATH_START_SOUND, id=0))

        # Start the audio player task if it's not already running
        if self.audio_player_task is None or self.audio_player_task.done():
            self.audio_player_task = self.bot.loop.create_task(self.audio_player.run(), name="AudioPlayer")
            await ctx.send("Audio player started.")

    async def generate_article_audio(self, requester: users.User, media_uid) -> Optional[Path]:
        # Select between reddit or news article
        if random.choice([True] + ([False] * (1 if not cfg.DEBUG else 0))):
            # reddit
            media_name = f"{media_uid}_reddit"
            media_audio_path = await ag.generate_reddit_article_audio_file(
                requester.model_name, self.output_folder, media_name
            )
        else:
            # news article
            media_name = f"{media_uid}_article"
            media_audio_path = await ag.generate_news_article_audio_file(
                requester.model_name, self.output_folder, media_name
            )

        return media_audio_path

    async def queue_music_youtube(self, url: str, ctx, requester: users.User, media_uid: str, quick=False):
        if "youtube.com" in url:
            async for video_info in tpi.extract_youtube_video_info(url):
                if video_info:
                    try:
                        video = tpi.Video(
                            id=media_uid, requester=requester, video_info=video_info, requested_channel=ctx.channel
                        )

                        if not quick:
                            if random.choice([True] + ([False] * (3 if not cfg.DEBUG else 0))):
                                media_audio_path = await self.generate_article_audio(requester, media_uid)

                                if media_audio_path:
                                    self.audio_player.add_to_playlist(
                                        tpi.Audio(filepath=media_audio_path, id=media_uid)
                                    )

                            if requester.real_name:
                                audio_file_path = await ag.generate_song_comment_audio_file(
                                    requester.model_name, video, self.output_folder, media_uid
                                )
                                if audio_file_path:
                                    self.audio_player.add_to_playlist(tpi.Audio(filepath=audio_file_path, id=media_uid))

                        self.audio_player.add_to_playlist(video)
                        # await ctx.send(f"Added to playlist: {video.video_info.title} | Duration: {td(seconds=video.video_info.duration)} | Requester: {video.requester.screen_name} ({video.requester.real_name})")
                    except Exception as e:
                        cfg.LOGGER.error(f"Error processing video: {e}", exc_info=True)
                        await ctx.send(f"Error processing video: {e}")
        else:
            await ctx.send("Invalid YouTube URL.")

    async def queue_music_spotify(self, title: str, ctx, requester: users.User, media_uid: str):
        import spotify_handler as sh

        spotify_handler = sh.SpotifyHandler()

        track = spotify_handler.search_track(title)
        if track:
            track_wrapped = tpi.SpotifyMedia(spotify_info=track, requester=requester, id=media_uid)

            if random.choice([True] + ([False] * (3 if not cfg.DEBUG else 0))):
                media_audio_path = await self.generate_article_audio(requester, media_uid)

            if requester.real_name:
                audio_file_path = await ag.generate_song_comment_audio_file(
                    requester.model_name, track_wrapped, media_uid
                )
                if audio_file_path:
                    self.audio_player.add_to_playlist(tpi.Audio(filepath=audio_file_path, id=media_uid))

            if media_audio_path:
                self.audio_player.add_to_playlist(track_wrapped)
        else:
            await ctx.send("Track not found on Spotify.")

    async def leave(self, ctx):
        """
        Disconnects the bot from the voice channel if connected.
        """
        voice_client = ctx.guild.voice_client
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
            await ctx.send("Disconnected from voice channel.")
        else:
            await ctx.send("The bot is not connected to a voice channel.")

    def run(self, token: str):
        self.bot.run(token)


if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.typing = False
    intents.presences = False
    intents.members = True
    intents.voice_states = True  # ✅ needed for voice connection

    run_folder = cfg.FOLDER_OUTPUT / "music_bot_runs" / f"run_{dt.now().strftime(cfg.DATETIME_FORMAT_FILESAFE)}"
    run_folder.mkdir(exist_ok=True, parents=True)

    bot_manager = BotManager(command_prefix="!g", intents=intents, output_folder=run_folder)
    bot_manager.run(TOKEN)
