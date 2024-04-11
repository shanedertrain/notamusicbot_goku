import os
from dotenv import load_dotenv
import asyncio
from pathlib import Path
import json
from typing import Union
from datetime import timedelta as td
import random
import uuid
from multiprocessing import Pool
from functools import partial

import discord
from discord.ext import commands

import convert_time as ct
import configuration as cfg
import types_playlist_items as tpi
import newsapi as na
import users

import sys
sys.path.append(str(cfg.FOLDER_ROOT / 'rvc_cli'))

from rvc_cli import comment_generator as scg
from rvc_cli import tts
from rvc_cli import audio_processor as ap
from rvc_cli import voice_converter as vc

IDLE_SECONDS_MAX = 15*60

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
PTH_PATH = os.getenv('PTH_PATH')
INDEX_PATH = os.getenv('INDEX_PATH')
GENERATOR = os.getenv('GENERATOR')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

FILEPATH_START_SOUND = cfg.FOLDER_INPUT / os.getenv('FILENAME_START_SOUND')

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMEG_OPTIONS = '-vn -filter:a "volume=0.5"'

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False
intents.members = True
BOT = commands.Bot(command_prefix='!g', intents=intents)

VC_HANDLER = vc.VoiceConverterHandler(Path(PTH_PATH), Path(INDEX_PATH), generator=GENERATOR)
NEWS_SCRAPER = na.NewsScraper(api_key=NEWS_API_KEY)

playlist:list[Union[tpi.Video, tpi.Audio]] = []
audio_player_task = None

async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    with Pool(processes=1) as pool:
        result = await loop.run_in_executor(None, partial(fn, *args))
    return result

def convert_voice_wrapper(tts_speedup_filepath, output_folder):
    # Assuming VC_HANDLER.convert_voice is the method you want to run in a separate process
    return VC_HANDLER.convert_voice(tts_speedup_filepath, output_folder)


async def generate_pre_play_audio_file(tts_module:tts.TextToSpeechConverter, video:tpi.Video, output_name:str) -> Union[Path, None]: 
    output_filepath = None
    try:
        song_comment = await asyncio.to_thread(scg.generate_song_comment, video.requester.real_name, video.requester.background, video.video_info.title, video.video_info.uploader)
        output_filepath = await generate_audio_from_text(tts_module, song_comment, output_name)
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_news_article_audio_file(tts_module:tts.TextToSpeechConverter, output_name:str) -> Union[Path, None]:
    output_filepath = None
    try:
        article = NEWS_SCRAPER.get_random_article(category=na.Category.TECHNOLOGY)
        article_text = NEWS_SCRAPER.get_article_text(article)
        generated_comment = await asyncio.to_thread(scg.generate_news_comment, article_text)
        output_filepath = await generate_audio_from_text(tts_module, generated_comment, output_name)
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_audio_from_text(tts_module:tts.TextToSpeechConverter, text:str, output_name:str) -> Union[Path, None]:
        text_without_quotes = text.replace('"', '')
        
        tts_filepath = await asyncio.to_thread(tts_module.text_to_speech, text_without_quotes, output_path=Path(cfg.FOLDER_TTS / f"{output_name}.wav"))
        tts_speedup_filepath = await asyncio.to_thread(ap.increase_speed, tts_filepath, speed_multiplier=1.0)
        
        cfg.LOGGER.debug("Starting voice conversion")
        vc_converted_filepath = await run_in_process(convert_voice_wrapper, tts_speedup_filepath, cfg.FOLDER_OUTPUT)
        cfg.LOGGER.debug("Voice conversion complete!")

        output_filepath = await asyncio.to_thread(ap.increase_volume, vc_converted_filepath, volume_modifier_db=8)
        return output_filepath

async def audio_player(bot):
    global playlist
    idle_seconds = 0
    voice_client = None
    play_source:Union[str, Path] = None #can be url or filepath

    while True:
        try:
            if len(playlist) > 0:
                media = playlist.pop(0)
                cfg.LOGGER.debug(f"Popped: {media}")
                voice_client = bot.voice_clients[0] if bot.voice_clients else None

                if type(media) == tpi.Video:
                    play_source = media.video_info.url

                    await media.requested_channel.send(f"Now playing: {media.video_info.title} | Duration: {ct.convert_seconds_to_minutes_seconds(media.video_info.duration)} | Requester: {media.requester.screen_name} ({media.requester.real_name})")
                    if voice_client and voice_client.is_connected():
                        voice_client.play(discord.FFmpegPCMAudio(play_source, before_options=FFMPEG_BEFORE_OPTIONS, options=FFMEG_OPTIONS))
                
                elif type(media) == tpi.Audio:
                    play_source = media.filepath
                    if voice_client and voice_client.is_connected():
                        voice_client.play(discord.FFmpegPCMAudio(play_source))

                while voice_client.is_playing():
                    await asyncio.sleep(1)
                    idle_seconds = 0

            else:
                # If the playlist is empty, wait for a short duration and check again
                await asyncio.sleep(1)
                idle_seconds += 1
                
                if voice_client:
                    if voice_client.is_connected():
                        if idle_seconds >= IDLE_SECONDS_MAX:
                            await leave_current_voice_channel()
        except Exception as e:
            cfg.LOGGER.error(e)

@BOT.event
async def on_ready():
    guild = discord.utils.find(lambda g: g.name == GUILD, BOT.guilds)
    cfg.LOGGER.info(
        f'{BOT.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )
    cfg.LOGGER.info("Ready...")
    
    # Fetch users asynchronously
    users_dict = await get_users(guild)
    for user in users.USERS:
        user.screen_name = users_dict.get(user.id, 'Unknown')

async def get_users(guild):
    # Fetch all members in the guild
    members = []
    async for member in guild.fetch_members(limit=None):
        members.append(member)

    if not cfg.FILEPATH_USERS.exists():
        write_users_to_json(members)
        users.USERS = users.read_users_from_json_file(cfg.FILEPATH_USERS) 

    users_dict = {member.id: member.name for member in members}
    
    return users_dict

def write_users_to_json(members):
    users_dict = {
        member.display_name: {
            "id": member.id,
            "real_name": member.display_name,
            "background": "No Background",
            "tts_type": "gtts"
        }
        for member in members
    }

    with open(cfg.FILEPATH_USERS, 'w') as f:
        json.dump(users_dict, f, indent=2)

@BOT.event
async def on_error(event, *args, **kwargs):
    with open(cfg.FOLDER_LOGS / 'bot_errors.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

@BOT.command(name='play', help='Add a video or playlist to the queue and start playing')
async def play(ctx, url: str):
    """
    Adds the provided link to the playlist queue and starts playing.

    Arguments:
    - url: The URL of the video or playlist to add to the queue.
    """
    global playlist
    global audio_player_task

    users.USERS = users.read_users_from_json_file(cfg.FILEPATH_USERS) 

    # Join the voice channel if the bot is not already connected
    voice_client = ctx.guild.voice_client
    if not voice_client:
        await ctx.author.voice.channel.connect()
        if not cfg.DEBUG: await channel_join_audio()

    # Start the audio player task if it's not already running
    if audio_player_task is None or audio_player_task.done():
        audio_player_task = BOT.loop.create_task(audio_player(BOT), name='AudioPlayer')
        await ctx.send("Audio player started.")

    async for video_info in tpi.extract_video_info(url):
        if video_info:
            requester = users.get_user_by_id(ctx.author.id)
            tts_module = tts.get_tts_module(requester.tts_type)
            media_uid = uuid.uuid4()

            try:
                #randomly generate a news article for the bot
                if random.choice([True] + ([False]*(6 if cfg.DEBUG == False else 0))):
                    article_name = f"{media_uid}_article"
                    article_audio_path = await generate_news_article_audio_file(tts_module, article_name)

                    if article_audio_path != None:
                        playlist.append(tpi.Audio(filepath=article_audio_path, id=media_uid))

                video = tpi.Video(id=media_uid, 
                                  requester=requester, 
                                  video_info=video_info, 
                                  requested_channel=ctx.channel)

                # Generate audio file based on the real name of the requester
                if requester is not None and requester.real_name is not None:
                    article_name = f"{media_uid}_preplay"
                    audio_file_path = await generate_pre_play_audio_file(tts_module, video, media_uid)
                    if audio_file_path != None:
                        playlist.append(tpi.Audio(filepath=audio_file_path, id=media_uid))
                
                playlist.append(video)
                await ctx.send(f"Added to playlist: {video.video_info.title} | Duration: {td(seconds=video.video_info.duration)} | Requester: {video.requester.screen_name} ({video.requester.real_name})")
                cfg.LOGGER.debug(f"Playlist: {playlist}")
            except Exception as e:
                cfg.LOGGER.error(f"Error processing video: {e}", exc_info=True)
                await ctx.send(f"Error processing video: {e}")


@BOT.command(name='stop', help='Stops playing the audio and disconnects from the voice channel')
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

@BOT.command(name='skip', help='Skip the current song and move to the next in the playlist')
async def skip(ctx):
    """
        Skips the current song and moves to the next in the playlist.
    """
    global playlist

    voice_client = BOT.voice_clients[0] if BOT.voice_clients else None
    if voice_client and voice_client.is_playing():
        voice_client.stop()
        await ctx.send("Skipping current song.")
        if len(playlist) == 0:
            leave(ctx)
    else:
        await ctx.send("No song is currently playing.")

# Command to delete all messages in the channel where the command is executed
@BOT.command(name='deleteallmessages', help='Deletes all messages posted by this bot in the current channel')
async def delete_bot_messages(ctx):
    # Check if the user is an admin
    if ctx.author.guild_permissions.administrator:
        # Fetches the channel where the command was executed
        channel = ctx.channel
        # Fetches all messages in the channel
        async for message in channel.history(limit=None):
            # Check if the message author is the bot itself
            if message.author == ctx.bot.user:
                await message.delete()
                await asyncio.sleep(1)  # Adjust the time as needed

async def channel_join_audio():
    voice_client = BOT.voice_clients[0] if BOT.voice_clients else None

    if voice_client and voice_client.is_connected():
        audio_file = discord.FFmpegPCMAudio(FILEPATH_START_SOUND)
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
        await ctx.send("Disconnected from voice channel.")
    else:
        await ctx.send("The bot is not connected to a voice channel.")

async def leave_current_voice_channel() -> bool:
    # Get the bot's voice client
    voice_client = discord.utils.get(BOT.voice_clients)
    if voice_client:
        await voice_client.disconnect()
        cfg.LOGGER.info(f'Idle for too long, left voice channel: {voice_client.channel.name}')
        return True
    else:
        cfg.LOGGER.error('Error: Bot is not connected to any voice channel.')
        return False

def clear_folder_contents(folder:Path):
    for file in folder.iterdir():
        if file.is_file():
            try:
                file.unlink()
                cfg.LOGGER.debug(f"Deleted file: {file}")
            except Exception as e:
                cfg.LOGGER.debug(f"Error deleting file: {file} - {e}")

if __name__ == '__main__':
    clear_folder_contents(cfg.FOLDER_TTS)
    clear_folder_contents(cfg.FOLDER_OUTPUT)
    BOT.run(TOKEN)
