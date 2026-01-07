import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyht import Client
from pyht.client import TTSOptions

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playht_tts.playht_voices import VoiceManifest

load_dotenv()

client = Client(
    user_id=os.getenv("PLAY_HT_USER_ID"),
    api_key=os.getenv("PLAY_HT_API_KEY"),
)


class PlayHTVoiceGenerator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def generate(self, text: str, voice_manifest: VoiceManifest, output_path: Path = Path.cwd() / "output.wav") -> Path:
        options = TTSOptions(voice=voice_manifest.value)
        with open(output_path, "wb") as audio_file:
            for chunk in client.tts(text, options, voice_engine="PlayDialog-http"):
                audio_file.write(chunk)

        print(f"Audio saved as {output_path}")

        return output_path


if __name__ == "__main__":
    converter = PlayHTVoiceGenerator()
    converter.generate("Hello World", VoiceManifest.Joseph.name)
