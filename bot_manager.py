import os
import asyncio
import random
import uuid
from datetime import timedelta as td

import discord
from discord.ext import commands
from audio_player import AudioPlayer
import configuration as cfg

import types_playlist_items as tpi
import audio_generator as ag
import users
from rvc_cli import models

from dotenv import load_dotenv

load_dotenv()
GUILD = os.getenv('DISCORD_GUILD')
FILEPATH_START_SOUND = cfg.FOLDER_INPUT / os.getenv('FILENAME_START_SOUND')

class BotManager:
    audio_player_task:asyncio.Task = None
    def __init__(self, command_prefix: str, intents: discord.Intents=discord.Intents.default()):
        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self.audio_player = AudioPlayer(self.bot)
        self.register_events()
        self.register_commands()

    def register_events(self):
        @self.bot.event
        async def on_ready():
            guild = discord.utils.find(lambda g: g.name == GUILD, self.bot.guilds)
            cfg.LOGGER.info(f'{self.bot.user} is connected to the following guild:\n{guild.name}(id: {guild.id})')
            cfg.LOGGER.info("Ready...")
            await users.get_users_from_guild(guild)

        @self.bot.event
        async def on_error(event, *args, **kwargs):
            with open(cfg.FOLDER_LOGS / 'bot_errors.log', 'a') as f:
                if event == 'on_message':
                    f.write(f'Unhandled message: {args[0]}\n')
                else:
                    raise

    def register_commands(self):
        @self.bot.command(name='play', help='Add a video or playlist to the queue and start playing')
        async def play(ctx, url: str):
            users.USERS = users.read_users_from_json_file(cfg.FILEPATH_USERS) #reload users from file
            models.MODELS = models.collect_models_from_folders() #reload models 

            voice_client = ctx.guild.voice_client
            if not voice_client:
                await ctx.author.voice.channel.connect()
                if not cfg.DEBUG: self.audio_player.add_to_playlist(tpi.Audio(filepath=FILEPATH_START_SOUND, id=0))
            
            # Start the audio player task if it's not already running
            if self.audio_player_task is None or self.audio_player_task.done():
                self.audio_player_task = self.bot.loop.create_task(self.audio_player.run(), name='AudioPlayer')
                await ctx.send("Audio player started.")

            async for video_info in tpi.extract_video_info(url):
                if video_info:
                    requester = users.get_user_by_id(ctx.author.id)
                    media_uid = uuid.uuid4()

                    try:
                        if random.choice([True] + ([False] * (3 if not cfg.DEBUG else 0))):

                            #Select between reddit or news article
                            if random.choice([True] + ([False] * (1 if not cfg.DEBUG else 0))): 
                                #reddit
                                media_name = f"{media_uid}_reddit"
                                media_audio_path = await ag.generate_reddit_article_audio_file(requester.model_name, media_name)
                            else:
                                #news article
                                media_name = f"{media_uid}_article"
                                media_audio_path = await ag.generate_news_article_audio_file(requester.model_name, media_name)
                            
                            if media_audio_path:
                                self.audio_player.add_to_playlist(tpi.Audio(filepath=media_audio_path, id=media_uid))
                        
                        video = tpi.Video(id=media_uid, requester=requester, video_info=video_info, requested_channel=ctx.channel)
                        
                        if requester.real_name:
                            media_name = f"{media_uid}_preplay"
                            audio_file_path = await ag.generate_song_comment_audio_file(requester.model_name, video, media_uid)
                            if audio_file_path:
                                self.audio_player.add_to_playlist(tpi.Audio(filepath=audio_file_path, id=media_uid))
                        self.audio_player.add_to_playlist(video)
                        # await ctx.send(f"Added to playlist: {video.video_info.title} | Duration: {td(seconds=video.video_info.duration)} | Requester: {video.requester.screen_name} ({video.requester.real_name})")
                    
                    except Exception as e:
                        cfg.LOGGER.error(f"Error processing video: {e}", exc_info=True)
                        await ctx.send(f"Error processing video: {e}")

        @self.bot.command(name='stop', help='Stops playing the audio and disconnects from the voice channel')
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

        @self.bot.command(name='skip', help='Skip the current song and move to the next in the playlist')
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

        @self.bot.command(name='deleteallmessages', help='Deletes all messages posted by this bot in the current channel')
        async def delete_bot_messages(ctx):
            if ctx.author.guild_permissions.administrator:
                channel = ctx.channel
                async for message in channel.history(limit=None):
                    if message.author == ctx.bot.user:
                        await message.delete()
                        await asyncio.sleep(1)


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

if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')

    intents = discord.Intents.default()
    intents.message_content = True
    intents.typing = False
    intents.presences = False
    intents.members = True

    bot_manager = BotManager(command_prefix='!g', intents=intents)
    bot_manager.run(TOKEN)