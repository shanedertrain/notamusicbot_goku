import sys
import os
from pathlib import Path
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Union

from dotenv import load_dotenv

import configuration as cfg
import newsapi as na
import reddit_scraper as rs
import types_playlist_items as tpi
import comment_generator as cg
import tts

sys.path.append(str(cfg.FOLDER_ROOT / 'rvc_cli'))
from rvc_cli import audio_processor as ap
from rvc_cli import voice_converter as vc
from rvc_cli import models

load_dotenv()
GENERATOR = os.getenv('GENERATOR')

NEWS_SCRAPER = na.NewsScraper()
REDDIT_SCRAPER = rs.RedditPostFetcher()

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

def generate_tts_audio(tts_module:tts.TextToSpeechConverter, text:str, output_name:str) -> Path:
    cfg.LOGGER.debug("Starting TTS audio generation")
    text_without_quotes = text.replace('"', '')
    
    tts_filepath = tts_module.text_to_speech(text_without_quotes, output_path=Path(cfg.FOLDER_TTS / f"{output_name}.wav"))
    # tts_filepath = ap.increase_speed(tts_filepath, speed_multiplier=1.0)
    cfg.LOGGER.debug("TTS audio generation complete!")
    return tts_filepath

def convert_voice_for_multiprocess(model:models.Model, audio_filepath:Path) -> Path:
    vc_handler = vc.VoiceConverterHandler(model=model, generator=GENERATOR)
    return vc_handler.convert_voice(audio_filepath)

async def generate_voice_converter_audio(vc_handler_name:str, text:str, output_name:str) -> Union[Path, None]:
    try:
        model = models.get_model(vc_handler_name)
        tts_module = tts.get_tts_module(model.tts_type)

        tts_audio_filepath = await asyncio.to_thread(generate_tts_audio, tts_module, text, output_name)

        # Run the synchronous voice conversion in a separate process
        output_filepath = await run_in_process(convert_voice_for_multiprocess, model, tts_audio_filepath)

        # output_filepath = await asyncio.to_thread(ap.increase_volume, output_filepath, volume_modifier_db=8)

        return output_filepath
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

async def generate_song_comment_audio_file(vc_handler_name:str, video:tpi.Video, output_name:str) -> Union[Path, None]: 
    output_filepath = None
    try:
        generated_comment = await asyncio.to_thread(cg.generate_song_comment, video.requester.real_name, video.requester.background, video.video_info.title, video.video_info.uploader)
        
        if generated_comment is not False:
            output_filepath = await generate_voice_converter_audio(vc_handler_name, generated_comment, output_name)
        else:
            raise ValueError("cg.generate_song_comment returned False instead of a comment string")
        
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_news_article_audio_file(vc_handler_name:str, output_name:str) -> Union[Path, None]:
    output_filepath = None
    try:
        article = NEWS_SCRAPER.get_random_article(category=na.Category.TECHNOLOGY)
        article_text = NEWS_SCRAPER.get_article_text(article)
        article_text_summarized = await asyncio.to_thread(cg.generate_news_comment, article_text)

        if article_text_summarized is not False:
            output_filepath = await generate_voice_converter_audio(vc_handler_name, article_text_summarized, output_name)
        else:
            raise ValueError("cg.generate_song_comment returned False instead of a comment string")
        
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_reddit_article_audio_file(vc_handler_name: str, output_name: str) -> Union[Path, None]:
    output_filepath = None
    article_text_summarized = None
    try:
        while article_text_summarized is None: #we do this because reddit can have posts gemini doesnt like
            reddit_post = await asyncio.to_thread(REDDIT_SCRAPER.get_random_reddit_post)
            article_text_summarized = await asyncio.to_thread(cg.generate_reddit_post_comment, reddit_post.title, reddit_post.post_text, reddit_post.poster_name)

        output_filepath = await generate_voice_converter_audio(vc_handler_name, article_text_summarized, output_name)
        
    except asyncio.TimeoutError:
        cfg.LOGGER.error("Timeout occurred while fetching Reddit post")
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath