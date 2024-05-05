import sys
import os
from pathlib import Path
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union

from dotenv import load_dotenv

import configuration as cfg
import newsapi as na
import reddit_scraper as rs
import types_playlist_items as tpi
import tts

sys.path.append(str(cfg.FOLDER_ROOT / 'gen_ai'))
import genai.gemini as gc

sys.path.append(str(cfg.FOLDER_ROOT / 'rvc_cli'))
from rvc_cli import audio_processor as ap
from rvc_cli import voice_converter as vc
from rvc_cli import models

load_dotenv()
GENERATOR = os.getenv('GENERATOR')

NEWS_SCRAPER = na.NewsScraper()
REDDIT_SCRAPER = rs.RedditPostFetcher()
GEMINI_CHAT = gc.GeminiChat()

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

def generate_tts_audio(tts_module:tts.TextToSpeechConverter, text:str, output_folder:Path, output_filestem:str) -> Path:
    cfg.LOGGER.debug("Starting TTS audio generation")
    text_without_quotes = text.replace('"', '')
    
    tts_filepath = tts_module.text_to_speech(text_without_quotes, output_folder=output_folder, output_filestem=output_filestem)
    # tts_filepath = ap.increase_speed(tts_filepath, speed_multiplier=1.0)
    cfg.LOGGER.debug("TTS audio generation complete!")
    return tts_filepath

def convert_voice_for_multiprocess(model:models.Model, input_audio_filepath:Path) -> Path:
    vc_handler = vc.VoiceConverterHandler(model=model, generator=GENERATOR)
    return vc_handler.convert_voice(input_audio_filepath)

async def generate_voice_converter_audio(model:models.Model, text:str, output_folder:Path, output_filestem:str) -> Optional[Path]:
    try:
        tts_module = tts.get_tts_module(model.tts_type)

        tts_audio_filepath = await asyncio.to_thread(generate_tts_audio, tts_module, text, output_folder=output_folder, output_filestem=output_filestem)

        # Run the synchronous voice conversion in a separate process
        output_filepath = await run_in_process(convert_voice_for_multiprocess, model, tts_audio_filepath)

        output_filepath = await asyncio.to_thread(ap.modify_speed, output_filepath, speed_multiplier=model.speed_multiplier)

        return output_filepath
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

async def generate_song_comment_audio_file(vc_handler_name:str, media:Union[tpi.Video, tpi.SpotifyMedia], output_folder:Path, output_filestem:str) -> Optional[Path]: 
    output_filepath = None
    model = models.get_model(vc_handler_name)
    try:
        if isinstance(media, tpi.Video):
            generated_comment = await asyncio.to_thread(GEMINI_CHAT.generate_song_comment, model.description, media.requester.real_name, media.requester.background, media.video_info.title, media.video_info.uploader)
        elif isinstance(media, tpi.SpotifyMedia):
            generated_comment = await asyncio.to_thread(GEMINI_CHAT.generate_song_comment, model.description, media.requester.real_name, media.requester.background, media.spotify_info.name, media.spotify_info.artists[0])

        if generated_comment is not False:
            output_filepath = await generate_voice_converter_audio(model, generated_comment, output_folder, output_filestem)
        else:
            raise ValueError("GEMINI_CHAT.generate_song_comment returned False instead of a comment string")
        
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_news_article_audio_file(vc_handler_name:str, output_folder:Path, output_filestem:str, category:na.Category=na.Category.TECHNOLOGY) -> Optional[Path]:
    output_filepath = None
    model = models.get_model(vc_handler_name)
    try:
        article = NEWS_SCRAPER.get_random_article(category=category)
        article_text = NEWS_SCRAPER.get_article_text(article)
        article_text_summarized = await asyncio.to_thread(GEMINI_CHAT.generate_news_comment, model.description, article_text)

        if article_text_summarized is not False:
            output_filepath = await generate_voice_converter_audio(model, article_text_summarized, output_folder, output_filestem)
        else:
            raise ValueError("GEMINI_CHAT.generate_song_comment returned False instead of a comment string")
        
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_reddit_article_audio_file(vc_handler_name: str, output_folder:Path, output_filestem:str) -> Optional[Path]:
    output_filepath = None
    article_text_summarized = None
    model = models.get_model(vc_handler_name)

    try:
        while article_text_summarized is None: #we do this because reddit can have posts gemini doesnt like
            reddit_post = next(REDDIT_SCRAPER.post_generator)
            article_text_summarized = await asyncio.to_thread(GEMINI_CHAT.generate_reddit_post_comment, model.description, reddit_post.title, reddit_post.post_text)

        output_filepath = await generate_voice_converter_audio(model, article_text_summarized, output_folder, output_filestem)
        
    except StopIteration:
        cfg.LOGGER.error("No more Reddit posts available")
    except asyncio.TimeoutError:
        cfg.LOGGER.error("Timeout occurred while fetching Reddit post")
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath