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
import types_playlist_items as tpi

sys.path.append(str(cfg.FOLDER_ROOT / 'rvc_cli'))
from rvc_cli import comment_generator as cg
from rvc_cli import tts
from rvc_cli import audio_processor as ap
from rvc_cli import voice_converter as vc
from rvc_cli import models

load_dotenv()
GENERATOR = os.getenv('GENERATOR')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

NEWS_SCRAPER = na.NewsScraper(api_key=NEWS_API_KEY)

if 'VC_HANDLERS' not in globals():
    VC_HANDLERS = {
        'GOKU': vc.VoiceConverterHandler(model=models.get_model("GOKU"), generator=GENERATOR),
        'OBAMA': vc.VoiceConverterHandler(model=models.get_model("OBAMA"), generator=GENERATOR),
        'OBAMA_TRANS': vc.VoiceConverterHandler(model=models.get_model("OBAMA_TRANS"), generator=GENERATOR),
    }

def get_vc_handler(model_name:str) -> vc.VoiceConverterHandler:
    return VC_HANDLERS.get(model_name, VC_HANDLERS['OBAMA'])

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

async def run_in_process(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)

def generate_tts_audio(tts_module:tts.TextToSpeechConverter, text:str, output_name:str) -> Path:
    cfg.LOGGER.debug("Starting TTS audio generation")
    text_without_quotes = text.replace('"', '')
    
    tts_filepath = tts_module.text_to_speech(text_without_quotes, output_path=Path(cfg.FOLDER_TTS / f"{output_name}.wav"))
    tts_speedup_filepath = ap.increase_speed(tts_filepath, speed_multiplier=1.0)
    cfg.LOGGER.debug("TTS audio generation complete!")
    return tts_speedup_filepath

def convert_voice_for_multiprocess(vc_handler_name:str, audio_filepath:Path) -> Path:
    vc_handler = get_vc_handler(vc_handler_name)
    return vc_handler.convert_voice(audio_filepath)

async def generate_voice_converter_audio(vc_handler:vc.VoiceConverterHandler, text:str, output_name:str) -> Union[Path, None]:
    try:
        # Run the synchronous TTS audio generation in a separate thread
        tts_audio_filepath = await asyncio.to_thread(generate_tts_audio, vc_handler.tts_module, text, output_name)

        # Run the synchronous voice conversion in a separate process
        voice_converted_filepath = await run_in_process(convert_voice_for_multiprocess, vc_handler.model.model_name, tts_audio_filepath)

        # Run the synchronous volume increase in a separate thread
        output_filepath = await asyncio.to_thread(ap.increase_volume, voice_converted_filepath, volume_modifier_db=8)

        return output_filepath
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

async def generate_pre_play_audio_file(vc_handler:vc.VoiceConverterHandler, video:tpi.Video, output_name:str) -> Union[Path, None]: 
    output_filepath = None
    try:
        generated_comment = await asyncio.to_thread(cg.generate_song_comment, video.requester.real_name, video.requester.background, video.video_info.title, video.video_info.uploader)
        output_filepath = await generate_voice_converter_audio(vc_handler, generated_comment, output_name)
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath

async def generate_news_article_audio_file(vc_handler:vc.VoiceConverterHandler, output_name:str) -> Union[Path, None]:
    output_filepath = None
    try:
        article = NEWS_SCRAPER.get_random_article(category=na.Category.TECHNOLOGY)
        article_text = NEWS_SCRAPER.get_article_text(article)
        article_text_summarized = await asyncio.to_thread(cg.generate_news_comment, article_text)

        output_filepath = await generate_voice_converter_audio(vc_handler, article_text_summarized, output_name)
    except Exception as e:
        cfg.LOGGER.error(e, exc_info=True)

    return output_filepath