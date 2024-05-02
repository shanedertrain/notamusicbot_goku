import sys
from pathlib import Path
import pyttsx3
from gtts import gTTS
import configuration as cfg 

sys.path.append(str(cfg.FOLDER_ROOT / 'tiktok_voice_tts'))
from tiktok_voice_tts.tiktokvoice import TikTokVoiceTTS

class TextToSpeechConverter:
    def text_to_speech(self, text: str, output_path: Path = cfg.FOLDER_OUTPUT, gender: str = 'male') -> Path:
        """
        Convert text to speech and save it as an audio file.

        Args:
        - text (str): The text to be converted to speech.
        - output_path (Path): The path of the output audio file. Default is 'output_tts.wav'.
        - gender (str): The gender of the voice. 'male' or 'female'. Default is 'male'.

        Returns:
        - output_path (Path): The path of the output audio file.
        """
        raise NotImplementedError("text_to_speech method must be implemented by subclasses")

    def list_voices(self):
        """Prints information about available voices."""
        raise NotImplementedError("list_voices method must be implemented by subclasses")

class TextToSpeechConverter_Pyttsx3(TextToSpeechConverter):
    def __init__(self):
        self.converter = pyttsx3.init()

    def text_to_speech(self, text: str, output_path: Path = cfg.FOLDER_TTS / 'test_pyttsx3.mp3', gender: str = 'male') -> Path:
        voices = self.converter.getProperty('voices')
        if gender == 'male':
            self.converter.setProperty('voice', voices[0].id)  # Use the first male voice
        else:
            self.converter.setProperty('voice', voices[1].id)  # Use the first female voice

        self.converter.save_to_file(text, str(output_path))
        self.converter.runAndWait()

        cfg.LOGGER.debug(f"Audio saved as {output_path}")

        return output_path

    def list_voices(self):
        """Prints information about available voices."""
        voices = self.converter.getProperty('voices')
        for voice in voices:
            print("Voice:")
            print("ID: %s" % voice.id)
            print("Name: %s" % voice.name)
            print("Age: %s" % voice.age)
            print("Gender: %s" % voice.gender)
            print("Languages Known: %s" % voice.languages)

class TextToSpeechConverter_gTTS(TextToSpeechConverter):
    def text_to_speech(self, text: str, output_path: Path = cfg.FOLDER_TTS / 'test_gtts.mp3', ) -> Path:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)

        cfg.LOGGER.debug(f"Audio saved as {output_path}")

        return output_path

    def list_voices(self):
        """gTTS doesn't provide voice options like pyttsx3"""
        print("gTTS doesn't provide voice options like pyttsx3")
        
class TextToSpeechConverter_TikTok(TextToSpeechConverter):
    def __init__(self):
        super().__init__()
        self.converter = TikTokVoiceTTS()

    def text_to_speech(self, text: str, output_filepath: Path = cfg.FOLDER_TTS / 'test_ttv.mp3', voice: TikTokVoiceTTS.Voices = TikTokVoiceTTS.Voices.en_us_001) -> Path:
        self.converter.tts(text, voice, output_filepath)

        cfg.LOGGER.debug(f"Audio saved as {output_filepath}")

        return output_filepath

    def list_voices(self):
        for voice_enum in self.converter.Voices.__members__.values():
            print(f"{voice_enum.value}: {voice_enum.name}")

def get_tts_module(tts_type:str) -> TextToSpeechConverter:
    tts_modules = {
        'gtts': TextToSpeechConverter_gTTS(),
        'pyttsx3': TextToSpeechConverter_Pyttsx3(),
    }
    selected_module = tts_type if tts_type in tts_modules else 'gtts'
    return tts_modules[selected_module]

if __name__ == '__main__':
    tiktok_converter = TextToSpeechConverter_TikTok()
    tiktok_converter.list_voices()

    tiktok_converter.text_to_speech(text="Hello, World!", output_filepath=cfg.FOLDER_TTS / 'test.mp3', voice=TikTokVoiceTTS.Voices.en_us_001)
