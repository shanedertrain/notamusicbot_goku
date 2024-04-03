import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import asyncio

import convert_time as ct
import configuration as cfg
import video_types as vt

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMEG_OPTIONS = "-vn"

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False

playlist:list[vt.Video] = []
audio_player_task = None

bot = commands.Bot(command_prefix='!', intents=intents)

async def audio_player(bot):
    global playlist

    while True:
        if playlist:
            video = playlist.pop(0)
            voice_client = bot.voice_clients[0] if bot.voice_clients else None

            if voice_client and voice_client.is_connected():
                # Send message to the channel where the video was added
                await video.channel.send(f"Now playing: {video.title} | Duration: {ct.convert_seconds_to_minutes_seconds(video.duration)}")
                
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
    if voice_client is None or not voice_client.is_connected():
        await ctx.author.voice.channel.connect()
        await hey_its_me_goku()

    # Start the audio player task if it's not already running
    if audio_player_task is None or audio_player_task.done():
        audio_player_task = bot.loop.create_task(audio_player(bot))
        await ctx.send("Audio player started.")
    
    async for video in vt.extract_info(url, ctx.channel):
        video.requester = vt.User(name=ctx.author.name, id=ctx.author.id)
        if video: 
            playlist.append(video)

@bot.command(name='stop', help='Stops playing the audio and disconnects from the voice channel')
async def stop(ctx):
    """
    Stops the audio player task and disconnects the bot from the voice channel.
    """
    global playlist
    global audio_player_task

    playlist.clear()
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


async def hey_its_me_goku():
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

if __name__ == '__main__':
    bot.run(TOKEN)
