import discord
from discord.ext import commands
from types_playlist_items import Video, Audio, SpotifyMedia
from configuration import LOGGER
import asyncio
from datetime import timedelta as td

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMEG_OPTIONS = '-vn -filter:a "volume=0.5"'

class AudioPlayer:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.playlist = []
        self.voice_client = None

    async def run(self):
        while True:
            await self.play_audio()
            await asyncio.sleep(1)  # Check the playlist again after a short delay

    async def play_audio(self):
        try:
            if self.voice_client and self.voice_client.is_connected():
                if len(self.playlist) > 0:
                    media = self.playlist.pop(0)
                    LOGGER.debug(f"Popped: {media}")
                    self.voice_client = self.bot.voice_clients[0] if self.bot.voice_clients else None

                    if isinstance(media, Video):
                        play_source = media.video_info.url
                        await media.requested_channel.send(
                            f"Now playing: {media.video_info.title} | Duration: {str(td(seconds=media.video_info.duration))} | Requester: {media.requester.screen_name} ({media.requester.real_name})"
                        )
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source, before_options=FFMPEG_BEFORE_OPTIONS, options=FFMEG_OPTIONS))
                
                    elif isinstance(media, Audio):
                        play_source = media.filepath
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source))

                    elif isinstance(media, SpotifyMedia):
                        play_source = media.spotify_info.uri
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source))

                while self.voice_client.is_playing():
                    await asyncio.sleep(1)

        except Exception as e:
            LOGGER.error(e)

    def add_to_playlist(self, media_item):
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