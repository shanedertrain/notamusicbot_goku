import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold, generation_types

from . import configuration_genai as cfg
from . import genai_base

class GeminiChat(genai_base.GenAIChat):
    # Safety config
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(self):
        super().__init__()
        gemini.configure(api_key=self.get_api_key())

        self.model = gemini.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat(history=[])

    def send_message(self, prompt: str)  -> Optional[genai_base.Response]:
        response:generation_types.GenerateContentResponse = None
        try:
            response = self.chat.send_message(prompt, safety_settings=self.safety_settings)
            return response
        except ValueError as e:
            cfg.LOGGER.error(e)

            # If the response doesn't contain text, check if the prompt was blocked.
            print(response.prompt_feedback)
            # Also check the finish reason to see if the response was blocked.
            print(response.candidates[0].finish_reason)
            # If the finish reason was SAFETY, the safety ratings have more details.
            print(response.candidates[0].safety_ratings)
            return None
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

    def get_api_key(self):
        load_dotenv()
        return os.getenv('GEMENI_API_KEY')

# Example usage
if __name__ == "__main__":
    radio_host = GeminiChat()
    requester_name = "Andrew"
    requester_background = "a right-wing, conspiracy theorist who loves guns and UFOs"
    song_name = "Space Oddity"
    artist_name = "David Bowie"
    print(radio_host.generate_song_comment('L from Death Note', requester_name, requester_background, song_name, artist_name))
