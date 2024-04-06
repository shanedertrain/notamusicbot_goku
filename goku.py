import os
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import uuid

import discord
from discord.ext import commands

import convert_time as ct
import configuration as cfg
import video_types as vt
import users

from rvc_cli import song_comment_generator as scg
from rvc_cli import tts
from rvc_cli import audio_processor

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMEG_OPTIONS = "-vn"
GUILD = 'BigbyInTheHouse'

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False
intents.members = True

playlist:list[vt.Video] = []
audio_player_task = None

bot = commands.Bot(command_prefix='!', intents=intents)

async def generate_pre_play_audio_file(video:vt.Video) -> Path:
    song_comment = scg.generate_song_comment(video.requester.real_name, video.requester.background, video.title, video.uploader)
    tts_filepath = tts.text_to_speech(song_comment, output_path=Path(cfg.FOLDER_TTS / f"{uuid.uuid4()}.wav"))
    final_filepath = audio_processor.increase_speed_and_volume(tts_filepath, volume_modifier_db=4, speed_multiplier=1.25)
    return final_filepath

async def audio_player(bot):
    global playlist

    while True:
        if playlist:
            video = playlist.pop(0)
            voice_client = bot.voice_clients[0] if bot.voice_clients else None

            if voice_client and voice_client.is_connected():
                # Send message to the channel where the video was added
                await video.channel.send(f"Now playing: {video.title} | Duration: {ct.convert_seconds_to_minutes_seconds(video.duration)} | Requester: {video.requester.screen_name} ({video.requester.real_name})")
                
                if video.path_pre_play != None:
                    voice_client.play(discord.FFmpegPCMAudio(video.path_pre_play))
                    
                    while voice_client.is_playing():
                        await asyncio.sleep(1)

                    os.remove(video.path_pre_play)

                voice_client.play(discord.FFmpegPCMAudio(video.url, before_options=FFMPEG_BEFORE_OPTIONS, options=FFMEG_OPTIONS))
                
                while voice_client.is_playing():
                    await asyncio.sleep(1)
                    
        else:
            # If the playlist is empty, wait for a short duration and check again
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    guild = discord.utils.find(lambda g: g.name == GUILD, bot.guilds)
    print(
        f'{bot.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )
    print("Ready...")
    
    # Fetch users asynchronously
    users_dict = await get_users(guild)
    for user in users.USERS:
        user.screen_name = users_dict.get(user.id, 'Unknown')

async def get_users(guild):
    # Fetch all members in the guild
    members = []
    async for member in guild.fetch_members(limit=None):
        members.append(member)
    
    # Create a dictionary of users
    users_dict = {member.id: member.name for member in members}
    
    return users_dict

@bot.event
async def on_error(event, *args, **kwargs):
    with open('err.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

@bot.command(name='gplay', help='Add a video or playlist to the queue and start playing')
async def play(ctx, url: str):
    """
    Adds the provided link to the playlist queue and starts playing.

    Arguments:
    - url: The URL of the video or playlist to add to the queue.
    """
    global playlist
    global audio_player_task

    # Join the voice channel if the bot is not already connected
    voice_client = ctx.guild.voice_client
    if not voice_client:
        await ctx.author.voice.channel.connect()
        # await channel_join_audio()

    # Start the audio player task if it's not already running
    if audio_player_task is None or audio_player_task.done():
        audio_player_task = bot.loop.create_task(audio_player(bot))
        await ctx.send("Audio player started.")
    
    async for video in vt.extract_info(url, ctx.channel):
        video.requester = users.get_user_by_id(ctx.author.id)
        if video:
            try:
                # Generate audio file based on the real name of the requester
                if video.requester is not None and video.requester.real_name is not None:
                    audio_file_path = await generate_pre_play_audio_file(video)
                    video.path_pre_play = audio_file_path
                
                playlist.append(video)
                await video.channel.send(f"Added to playlist: {video.title} | Duration: {ct.convert_seconds_to_minutes_seconds(video.duration)} | Requester: {video.requester.screen_name} ({video.requester.real_name})")
            except Exception as e:
                await ctx.send(f"Error processing video: {e}")

@bot.command(name='stop', help='Stops playing the audio and disconnects from the voice channel')
async def stop(ctx):
    """
    Stops the audio player task and disconnects the bot from the voice channel.
    """
    global playlist
    global audio_player_task

    playlist.clear()
    if audio_player_task and not audio_player_task.done():
        await audio_player_task
    if audio_player_task and not audio_player_task.done():
        audio_player_task.cancel()
    await leave(ctx)

@bot.command(name='skip', help='Skip the current song and move to the next in the playlist')
async def skip(ctx):
    """
        Skips the current song and moves to the next in the playlist.
    """
    global playlist

    voice_client = bot.voice_clients[0] if bot.voice_clients else None
    if voice_client and voice_client.is_playing():
        voice_client.stop()
        await ctx.send("Skipping current song.")
        if len(playlist) == 0:
            leave()
    else:
        await ctx.send("No song is currently playing.")

async def channel_join_audio():
    voice_client = bot.voice_clients[0] if bot.voice_clients else None

    if voice_client and voice_client.is_connected():
        audio_file = discord.FFmpegPCMAudio(cfg.FILEPATH_START_SOUND)
        voice_client.play(audio_file)
        while voice_client.is_playing():
            await asyncio.sleep(1)

async def leave(ctx):
    """
    Disconnects the bot from the voice channel if connected.
    """
    voice_client = ctx.guild.voice_client
    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("The bot is not connected to a voice channel.")

def clear_tts_folder():
    for file in cfg.FOLDER_TTS.iterdir():
        if file.is_file():
            try:
                file.unlink()
                print(f"Deleted file: {file}")
            except Exception as e:
                print(f"Error deleting file: {file} - {e}")

if __name__ == '__main__':
    clear_tts_folder()
    bot.run(TOKEN)
