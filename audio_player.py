from typing import Union, List
from datetime import timedelta as td
import asyncio

import discord
from discord.ext import commands

import configuration as cfg
import types_playlist_items as tpi

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMEG_OPTIONS = '-vn -filter:a "volume=0.5"'
IDLE_SECONDS_MAX = 15*60  # 15 minutes

class AudioPlayer:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.playlist: List[Union[tpi.Video, tpi.Audio]] = []
        self.voice_client = None

    async def run(self):
        while True:
            await self.play_audio()

    async def play_audio(self):
        try:
            if len(self.playlist) > 0:
                media = self.playlist.pop(0)
                cfg.LOGGER.debug(f"Popped: {media}")
                self.voice_client = self.bot.voice_clients[0] if self.bot.voice_clients else None

                if isinstance(media, tpi.Video):
                    play_source = media.video_info.url
                    await media.requested_channel.send(
                        f"Now playing: {media.video_info.title} | Duration: {str(td(seconds=media.video_info.duration))} | Requester: {media.requester.screen_name} ({media.requester.real_name})"
                    )
                    if self.voice_client and self.voice_client.is_connected():
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source, before_options=FFMPEG_BEFORE_OPTIONS, options=FFMEG_OPTIONS))
                
                elif isinstance(media, tpi.Audio):
                    play_source = media.filepath
                    if self.voice_client and self.voice_client.is_connected():
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source))

                while self.voice_client.is_playing():
                    await asyncio.sleep(10)

            else:
                await asyncio.sleep(1)  # Check the playlist again after a short delay

        except Exception as e:
            cfg.LOGGER.error(e)

    def add_to_playlist(self, media_item: Union[tpi.Video, tpi.Audio]):
        self.playlist.append(media_item)

    async def clear_playlist(self):
        self.playlist = []
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()

if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True
    intents.typing = False
    intents.presences = False
    intents.members = True

    bot = commands.Bot(command_prefix='!g', intents=intents)
    audio_player = AudioPlayer(bot)
    asyncio.run(audio_player.run())