import textwrap
from dotenv import load_dotenv
import os
from typing import Union

import google.generativeai as genai
from IPython.display import display, Markdown

import configuration as cfg

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

load_dotenv()
GEMENI_API_KEY = os.getenv('GEMENI_API_KEY')
genai.configure(api_key=GEMENI_API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def generate_reddit_post_comment(vc_description:str, post_title:str, post_text:str, post_submitter:str) -> Union[str, bool]:
    # Creating a scenario where Goku comments on the requester's personality and introduces the song
    prompt = f"""
        You are {vc_description} and remember to speak in first person.
        Limit your response to maximum 3 paragraphs.
        You are summarizing a reddit post by {post_submitter} as the host of the Galactic Beats radio show. 
        Announce the title of the post: {post_title}.
        The post content to be summarized is: {post_text}.
    """
    # Sending the scenario as a message to the AI model
    try:
        response = chat.send_message(prompt)

        # Displaying the AI's response
        display(to_markdown(response.text))
        
        # Logging the response
        cfg.LOGGER.debug(response.text)
    except Exception as e:
        cfg.LOGGER.error(e)
        return None

    return response.text

def generate_news_comment(vc_description:str, article_text:str) -> Union[str, bool]:
    # Creating a scenario where Goku comments on the requester's personality and introduces the song
    prompt = f"""
        You are {vc_description} and remember to speak in first person.
        Your response must be 2 - 3 sentences. 
        You are reading a news story as the host of the Galactic Beats radio show. 
        What you will say is a summary of the following article text:
        {article_text}.
    """
    # Sending the scenario as a message to the AI model
    try:
        response = chat.send_message(prompt)

        # Displaying the AI's response
        display(to_markdown(response.text))
        
        # Logging the response
        cfg.LOGGER.debug(response.text)
    except Exception as e:
        cfg.LOGGER.error(e)
        return None

    return response.text

def generate_song_comment(vc_description:str, requester_name:str, requester_background:str, song_name:str, artist_name:str) -> Union[str, bool]:
    # Creating a scenario where Goku comments on the requester's personality and introduces the song
    prompt = f"""
        You are {vc_description} and remember to speak in first person.
        Your response must be 2 - 3 sentences. 
        You are announcing the song {song_name} by {artist_name} as the host of the 'Galactic Beats' radio show.
        The requester is {requester_name}, who's known for being {requester_background}. 
        Make a light-hearted joke about the requestor's background that ties into the song or its topic.
        Do not repeat jokes about the requestor's background across prompts.
    """
    # Sending the scenario as a message to the AI model
    try:
        response = chat.send_message(prompt)

        # Displaying the AI's response
        display(to_markdown(response.text))
        
        # Logging the response
        cfg.LOGGER.debug(response.text)
    except Exception as e:
        cfg.LOGGER.error(e)
        return False

    return response.text

# Example usage
if __name__ == "__main__":
    requester_name = "Andrew"
    requester_background = "a right-wing, conspiracy theorist who loves guns and UFOs"
    song_name = "Space Oddity"
    artist_name = "David Bowie"

    print(generate_song_comment(requester_name, requester_background, song_name, artist_name))
