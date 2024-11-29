from datetime import timedelta as td
from typing import Union
import asyncio

import discord
from discord.ext import commands
from configuration import LOGGER

from types_playlist_items import Video, Audio, SpotifyMedia

import subprocess
from pathlib import Path

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMPEG_OPTIONS = "-vn -filter:a volume=0.5"

class AudioPlayer:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.playlist = []
        self.voice_client = None

    async def run(self):
        while True:
            await self.play_audio()
            await asyncio.sleep(1)

    async def extract_audio_url(self, url: str) -> str:
        """Use yt-dlp to get the best audio URL."""
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-g", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"yt-dlp error: {e.stderr}")
            return ""

    async def play_audio(self):
        try:
            if len(self.playlist) > 0:
                self.voice_client = self.bot.voice_clients[0] if self.bot.voice_clients else None
                if self.voice_client and self.voice_client.is_connected():
                    media = self.playlist.pop(0)
                    LOGGER.debug(f"Popped: {media}")

                    if isinstance(media, Video):
                        audio_url = await self.extract_audio_url(media.video_info.url)
                        if not audio_url:
                            LOGGER.error(f"Failed to extract audio URL for {media.video_info.title}")
                            return

                        await media.requested_channel.send(
                            f"Now playing: {media.video_info.title} | Duration: {str(td(seconds=media.video_info.duration))} | Requester: {media.requester.screen_name} ({media.requester.real_name})"
                        )
                        self.voice_client.play(
                            discord.FFmpegPCMAudio(audio_url, before_options=FFMPEG_BEFORE_OPTIONS, options=FFMPEG_OPTIONS)
                        )

                    elif isinstance(media, Audio):
                        play_source = media.filepath
                        self.voice_client.play(discord.FFmpegPCMAudio(str(play_source)))

                    elif isinstance(media, SpotifyMedia):
                        play_source = media.spotify_info.uri
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source))

                while self.voice_client.is_playing():
                    await asyncio.sleep(3)

        except Exception as e:
            LOGGER.error(e)

    def add_to_playlist(self, media_item: Union[Video, Audio, SpotifyMedia]):
        self.playlist.append(media_item)

    async def clear_playlist(self):
        self.playlist = []
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()
