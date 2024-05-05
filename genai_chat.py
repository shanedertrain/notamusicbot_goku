import os
import textwrap
from typing import Union, Optional
from abc import ABC, abstractmethod
from IPython.display import display, Markdown
from dataclasses import dataclass

from dotenv import load_dotenv
import google.generativeai as genai

import configuration as cfg

@dataclass
class Response:
    text:str

class GenAIChat(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def send_message(self, prompt: str) -> Optional[Response]:
        """
        Abstract method to send a message and return a response.
        Subclasses must implement this method.
        """
        pass

    def to_markdown(self, text:str):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    def generate_reddit_post_comment(self, vc_description:str, post_title:str, post_text:str) -> Union[str, bool]:
        prompt = f"""
            You are {vc_description} and remember to speak in first person.
            Limit your response to maximum 3 paragraphs.
            You are summarizing a reddit post as the host of the Galactic Beats radio show. 
            Announce the title of the post: {post_title}.
            The post content to be summarized is: {post_text}.
        """
        try:
            response = self.send_message(prompt)
            display(self.to_markdown(response.text))
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

        return response.text

    def generate_news_comment(self, vc_description:str, article_text:str) -> Union[str, bool]:
        prompt = f"""
            You are {vc_description} and remember to speak in first person.
            Your response must be 2 - 3 sentences. 
            You are reading a news story as the host of the Galactic Beats radio show. 
            What you will say is a summary of the following article text:
            {article_text}.
        """
        try:
            response = self.send_message(prompt)
            display(self.to_markdown(response.text))
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

        return response.text

    def generate_song_comment(self, vc_description:str, requester_name:str, requester_background:str, song_name:str, artist_name:str) -> Union[str, bool]:
        prompt = f"""
            You are {vc_description} and remember to speak in first person.
            Your response must be 2 - 3 sentences. 
            You are announcing the song {song_name} by {artist_name} as the host of the 'Galactic Beats' radio show.
            The requester is {requester_name}, who's known for being {requester_background}. 
            Make a light-hearted joke about the requestor's background that ties into the song or its topic.
            Do not repeat jokes about the requestor's background across prompts.
        """
        try:
            response = self.send_message(prompt)
            display(self.to_markdown(response.text))
            cfg.LOGGER.debug(response.text)
        except Exception as e:
            cfg.LOGGER.error(e)
            return False

        return response.text

class GeminiChat(GenAIChat):
    def __init__(self):
        super().__init__()
        self.load_dotenv()
        genai.configure(api_key=self.get_api_key())

        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat(history=[])

    def send_message(self, prompt: str)  -> Optional[Response]:
        try:
            response = self.chat.send_message(prompt)
            return response
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

    def load_dotenv(self):
        load_dotenv()

    def get_api_key(self):
        return os.getenv('GEMENI_API_KEY')

# Example usage
if __name__ == "__main__":
    radio_host = GeminiChat()
    requester_name = "Andrew"
    requester_background = "a right-wing, conspiracy theorist who loves guns and UFOs"
    song_name = "Space Oddity"
    artist_name = "David Bowie"
    print(radio_host.generate_song_comment('L from Death Note', requester_name, requester_background, song_name, artist_name))
