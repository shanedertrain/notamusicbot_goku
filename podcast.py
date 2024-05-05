import os
import sys
from dotenv import load_dotenv
from typing import Union
import asyncio
from datetime import datetime as dt

import discord

import configuration as cfg
from bot_manager import BotManager
import genai.genai_base as gc
import audio_generator as ag
import types_playlist_items as tpi
import reddit_scraper as rs

sys.path.append(str(cfg.FOLDER_ROOT / 'rvc_cli'))
from rvc_cli import models

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.typing = False
intents.presences = False
intents.members = True

BOT_MANAGER = BotManager(command_prefix='!g', intents=intents)
REDDIT_SCRAPER = rs.RedditPostFetcher()

def clear_folder_contents(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Deletes a file or symbolic link
        except Exception as e:
            cfg.LOGGER.error(f'Failed to delete {file_path}. Reason: {e}')

class Character(gc.GeminiChat):
    def __init__(self, initialize_prompt:str, model:models.Model):
        super().__init__()
        self.model = model

        self.send_initialization_prompt(initialize_prompt)

    def send_initialization_prompt(self, prompt) -> str:
        prompt = f"""
            You are {self.model} and remember to speak in first person.
            Stay in character. As far as the length of your response is concerned, less is more.
            Here's additional information about the setting: {prompt}
        """
        return self.chat.send_message(prompt)

    def converse(self, conversation_target:str, input_response:str) -> Union[str, bool]:
        prompt = f"""
            {conversation_target} has responded to you saying: "{input_response}". 
            What do you have to say in response? Reply to them directly.'
        """
        try:
            response = self.chat.send_message(prompt)
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

        return response.text
    
    def generate_article_post_opening(self, post_title:str, post_text:str, guest_description:str) -> Union[str, bool]:
        prompt = f"""
            You are {self.model.description} speak only as yourself and in first person.
            You are summarizing a reddit post as the host a talk show. 
            Your guest is {guest_description}.
            Introduce yourself and then ask your guest what they think about the article. 
            In your opening statement, announce the title of the post: {post_title} and the article content discussed is: {post_text}.
        """
        try:
            response = self.chat.send_message(prompt)
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

        return response.text

async def generate_conversation(character_1:Character, character_2:Character):
    conversation_folder = cfg.FOLDER_OUTPUT / 'podcasts' / f'podcast_{dt.now().strftime(cfg.DATETIME_FORMAT_FILESAFE)}'
    conversation_folder.mkdir(exist_ok=True, parents=True)

    with(open(conversation_folder / 'podcast.txt', 'w')) as f:
        f.write(f"Conversation between {character_1.model.model_name} and {character_2.model.model_name} on {dt.now().strftime(cfg.DATETIME_FORMAT)}\n\n")

    reddit_post = next(REDDIT_SCRAPER.post_generator)
    
    opening_speech = await asyncio.to_thread(character_1.generate_article_post_opening, 
                                                    reddit_post.title, 
                                                    reddit_post.post_text, 
                                                    character_2.model.description)
    
    #opening
    audio_path_charater_1 = await ag.generate_voice_converter_audio(character_1.model, opening_speech, conversation_folder, f"{character_1.model.model_name}_opening")
    with(open(conversation_folder / 'podcast.txt', 'a')) as f:
        f.write(f"{character_1.model.model_name}: {opening_speech}\n\n")
    
    audio_path_charater_1 = conversation_folder / str(audio_path_charater_1.name)
    BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(id=f"{character_1.model.model_name}_opening", filepath=audio_path_charater_1))

    response_character_2 = character_2.converse(character_1.model.model_name, opening_speech)
    with(open(conversation_folder / 'podcast.txt', 'a')) as f:
        f.write(f"{character_2.model.model_name}: {response_character_2}\n\n")
    
    audio_path_charater_2 = await ag.generate_voice_converter_audio(character_2.model, response_character_2, conversation_folder, f"{character_2.model.model_name}_opening")
    audio_path_charater_2 = conversation_folder / str(audio_path_charater_2.name)
    BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(id=f"{character_2.model.model_name}_opening", filepath=audio_path_charater_2))

    #conversation
    i = 0
    while True:
        for i in range(3):
            response_character_1 = character_1.converse(character_2.model.model_name, response_character_2)
            with(open(conversation_folder / 'podcast.txt', 'a')) as f:
                f.write(f"{character_1.model.model_name}: {response_character_1}\n\n")
            
            audio_path_charater_1 = await ag.generate_voice_converter_audio(character_1.model, response_character_1, conversation_folder, f"{character_1.model.model_name}_{i}")
            audio_path_charater_1 = conversation_folder / str(audio_path_charater_1.name)
            BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(id=f"{character_1.model.model_name}_{i}", filepath=audio_path_charater_1))

            response_character_2 = character_2.converse(character_1.model.model_name, response_character_1)
            with(open(conversation_folder / 'podcast.txt', 'a')) as f:
                f.write(f"{character_2.model.model_name}: {response_character_2}\n\n")
            
            audio_path_charater_2 = await ag.generate_voice_converter_audio(character_2.model, response_character_2, conversation_folder, f"{character_2.model.model_name}_{i}")
            audio_path_charater_2 = conversation_folder / str(audio_path_charater_2.name)
            BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(id=f"{character_2.model.model_name}_{i}", filepath=audio_path_charater_2))

            i += 1

        reddit_post = next(REDDIT_SCRAPER.post_generator)
        new_topic_prompt = f"""
            \n\n [THIS IS NOT PART OF {character_2.model.model_name}'S RESPONSE. THEY ARE INSTRUCTIONS TO YOU SPECIFICALLY]: 
            Respond to {character_2.model.model_name}'s last response and then introduce this new article into the show: 
            Title: {reddit_post.title}, Article: {reddit_post.post_text}.
            Ask {character_2.model.model_name} what they think about the article.
        """

        cfg.LOGGER.debug(new_topic_prompt)

        f.write(f"CHANGE TOPIC: {new_topic_prompt}\n\n")

        response_character_2 = response_character_2 + new_topic_prompt

if __name__ == '__main__':
    import time
    import threading
    
    clear_folder_contents(cfg.FOLDER_TTS)
    clear_folder_contents(cfg.FOLDER_OUTPUT)

    init_prompt_l = """L, as a mastermind investigator who delves deep into the intricacies of criminal minds and justice, you may find a parallel in the figure of Barack Obama, a former U.S. President renowned for his strategic leadership and intellectual rigor. 
                Like you, Obama navigated complex global and domestic landscapes with a focus on ethics and justice, often confronting formidable challenges that required both shrewd negotiation and deep understanding of human nature. 
                His presidency was marked by a dedication to larger societal issues—promoting equality, enhancing healthcare, and striving for peace—reflective of a pursuit of justice not unlike your own relentless quest to solve cases and eliminate corruption. 
                Obama's eloquent communication and ability to inspire through his words are akin to the persuasive tactics you employ in your investigations, making him a figure whose leadership and decision-making processes might resonate deeply with your analytical and strategic mindset.
                You will speak to them as if you are in the same room having a conversation.
            """
    character_l = Character(init_prompt_l, model=models.get_model('L'))

    init_prompt_obama = """President Obama, as someone who has led the United States through numerous challenges with a focus on diplomacy, justice, and equality, you may find an interesting parallel in the character of 'L' from the series Death Note. 
                    'L' is a master detective who operates within the shadows, using his intellect and keen sense of justice to track down and confront global threats. 
                    Much like your experience in the Oval Office, 'L' faces complex moral and ethical dilemmas, requiring a blend of strategic thinking, psychological insight, and an unwavering commitment to the greater good. 
                    His methods, though secretive, emphasize the importance of understanding diverse perspectives and the deep undercurrents of human behavior—themes that were also central to your presidency. 
                    This comparison might offer a unique lens through which to view your own approaches to leadership and conflict resolution.
                    You will speak to them as if you are in the same room having a conversation.
                """
    character_obama = Character(init_prompt_obama, model=models.get_model('Obama'))
    
    bot_thread = threading.Thread(target=lambda: BOT_MANAGER.run(TOKEN))
    bot_thread.start()

    voice_client = BOT_MANAGER.bot.voice_clients[0] if BOT_MANAGER.bot.voice_clients else None
    while voice_client is None:
        voice_client = BOT_MANAGER.bot.voice_clients[0] if BOT_MANAGER.bot.voice_clients else None
        cfg.LOGGER.info("Not in channel. Sleeping for 5 seconds")
        time.sleep(5)
    #conversation
    asyncio.run(generate_conversation(character_l, character_obama))
    