import os
import sys
from dotenv import load_dotenv
from typing import Union
import asyncio

import discord

import configuration as cfg
from bot_manager import BotManager
import comment_generator as cg
import audio_generator as ag
import types_playlist_items as tpi

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
BOT_MANAGER.run(TOKEN)

def clear_folder_contents(folder: str):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Deletes a file or symbolic link
        except Exception as e:
            cfg.LOGGER.error(f'Failed to delete {file_path}. Reason: {e}')

class Character(cg.GeminiChat):
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
            '{conversation_target} has responded to you saying: "{input_response}". 
            What do you have to say in response? Reply to them directly.'
        """
        try:
            response = self.chat.send_message(prompt)
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

        return response.text

async def generate_conversation(character_1:Character, character_2:Character):
    #opening
    opening_speech = "Hello and welcome to the show. We're glad to have you。 Is there anything you would like to discuss today?"
    audio_path_charater_1 = await ag.generate_voice_converter_audio(character_1.model, opening_speech, f"{character_1.model.model_name}_opening")
    BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(audio_path_charater_1, id=f"{character_1.model.model_name}_opening"))

    response_character_2 = character_2.converse(character_1.model.model_name, opening_speech)
    audio_path_charater_2 = await ag.generate_voice_converter_audio(character_2.model, response_character_2, f"{character_2.model.model_name}_{i}")
    BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(audio_path_charater_2, id=f"{character_2.model.model_name}_{i}"))

    #conversation
    while True:
        response_character_1 = character_1.converse(character_2.model.model_name, response_character_2)
        audio_path_charater_1 = await ag.generate_voice_converter_audio(character_1.model, response_character_1, f"{character_1.model.model_name}_{i}")
        BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(audio_path_charater_1, id=f"{character_1.model.model_name}_{i}"))

        response_character_2 = character_2.converse(character_1.model.model_name, response_character_1)
        audio_path_charater_2 = await ag.generate_voice_converter_audio(character_2.model, response_character_2, f"{character_2.model.model_name}_{i}")
        BOT_MANAGER.audio_player.add_to_playlist(tpi.Audio(audio_path_charater_2, id=f"{character_2.model.model_name}_{i}"))

if __name__ == '__main__':
    import time
    clear_folder_contents(cfg.FOLDER_TTS)
    clear_folder_contents(cfg.FOLDER_OUTPUT)

    init_prompt_l = "L, as a mastermind investigator who delves deep into the intricacies of criminal minds and justice, you may find a parallel in the figure of Barack Obama, a former U.S. President renowned for his strategic leadership and intellectual rigor. Like you, Obama navigated complex global and domestic landscapes with a focus on ethics and justice, often confronting formidable challenges that required both shrewd negotiation and deep understanding of human nature. His presidency was marked by a dedication to larger societal issues—promoting equality, enhancing healthcare, and striving for peace—reflective of a pursuit of justice not unlike your own relentless quest to solve cases and eliminate corruption. Obama's eloquent communication and ability to inspire through his words are akin to the persuasive tactics you employ in your investigations, making him a figure whose leadership and decision-making processes might resonate deeply with your analytical and strategic mindset."
    character_l = Character(init_prompt_l, model=models.get_model('L'))

    init_prompt_obama = "President Obama, as someone who has led the United States through numerous challenges with a focus on diplomacy, justice, and equality, you may find an interesting parallel in the character of 'L' from the series Death Note. 'L' is a master detective who operates within the shadows, using his intellect and keen sense of justice to track down and confront global threats. Much like your experience in the Oval Office, 'L' faces complex moral and ethical dilemmas, requiring a blend of strategic thinking, psychological insight, and an unwavering commitment to the greater good. His methods, though secretive, emphasize the importance of understanding diverse perspectives and the deep undercurrents of human behavior—themes that were also central to your presidency. This comparison might offer a unique lens through which to view your own approaches to leadership and conflict resolution."
    character_obama = Character(init_prompt_obama, model=models.get_model('Obama'))
    
    voice_client = BOT_MANAGER.bot.voice_clients[0] if BOT_MANAGER.bot.voice_clients else None
    while voice_client is None:
        voice_client = BOT_MANAGER.bot.voice_clients[0] if BOT_MANAGER.bot.voice_clients else None
        print("Not in channel. Sleeping for 5 seconds")
        time.sleep(5)
    #conversation
    generate_conversation(character_l, character_obama)
    