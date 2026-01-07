import os
from typing import Optional

import configuration_genai as cfg
import genai_base
from dotenv import load_dotenv
from openai import OpenAI


class ChatGPT(genai_base.GenAIChat):
    def __init__(self):
        super().__init__()

        load_dotenv()
        self.client = OpenAI(api_key=self.get_api_key())

    def send_message(self, prompt: str) -> Optional[genai_base.Response]:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                model="gpt-3.5-turbo",
            )
            print(response)
            return genai_base.Response(text=response.choices[0].text.strip())
        except Exception as e:
            cfg.LOGGER.error(e)
            return None

    def get_api_key(self):
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")


# Example usage
if __name__ == "__main__":
    radio_host = ChatGPT()
    requester_name = "Andrew"
    requester_background = "a right-wing, conspiracy theorist who loves guns and UFOs"
    song_name = "Space Oddity"
    artist_name = "David Bowie"
    print(
        radio_host.generate_song_comment(
            "L from Death Note", requester_name, requester_background, song_name, artist_name
        )
    )
