import sys
from pathlib import Path
import pyttsx3
from gtts import gTTS
import configuration as cfg 

sys.path.append(str(cfg.FOLDER_ROOT / 'tiktok_voice_tts'))
from tiktok_voice_tts.tiktokvoice import TikTokVoiceTTS
from playht_tts.playht_voice_generator import PlayHTVoiceGenerator, VoiceManifest

class TextToSpeechConverter:
    def text_to_speech(self, text: str, output_folder:Path = cfg.FOLDER_OUTPUT, output_filestem: str='test') -> Path:
        """
        Convert text to speech and save it as an audio file.

        Args:
        - text (str): The text to be converted to speech.
        - output_path (Path): The path of the output audio file. Default is 'output_tts.wav'.

        Returns:
        - output_path (Path): The path of the output audio file.
        """
        raise NotImplementedError("text_to_speech method must be implemented by subclasses")

    def list_voices(self):
        """Prints information about available voices."""
        raise NotImplementedError("list_voices method must be implemented by subclasses")

class TextToSpeechConverter_Pyttsx3(TextToSpeechConverter):
    def __init__(self, gender='male'):
        super().__init__()
        self.gender = gender
        self.converter = pyttsx3.init()

    def text_to_speech(self, text: str, output_folder: Path = cfg.FOLDER_TTS, output_filestem:str = 'test_pyttsx3') -> Path:
        output_filepath = output_folder / (output_filestem+'.mp3')
        
        voices = self.converter.getProperty('voices')
        if self.gender == 'male':
            self.converter.setProperty('voice', voices[0].id)  # Use the first male voice
        else:
            self.converter.setProperty('voice', voices[1].id)  # Use the first female voice

        self.converter.save_to_file(text, str(output_filepath))
        self.converter.runAndWait()

        cfg.LOGGER.debug(f"Audio saved as {output_filepath}")

        return output_filepath

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
    def __init__(self):
        super().__init__()

    def text_to_speech(self, text: str, output_folder: Path = cfg.FOLDER_TTS, output_filestem:str = 'test_gtts') -> Path:
        output_filepath = output_folder / (output_filestem+'.wav')

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(output_filepath))

        cfg.LOGGER.debug(f"Audio saved as {output_folder}")

        return output_filepath

    def list_voices(self):
        """gTTS doesn't provide voice options like pyttsx3"""
        print("gTTS doesn't provide voice options like pyttsx3")
        
class TextToSpeechConverter_TikTok(TextToSpeechConverter):
    def __init__(self, voice_name=TikTokVoiceTTS.Voices.en_us_001.name):
        super().__init__()
        self.converter = TikTokVoiceTTS()
        self.voice = TikTokVoiceTTS.get_voice_by_name(voice_name)

    def text_to_speech(self, text: str, output_folder:Path = cfg.FOLDER_OUTPUT, output_filestem: str = 'test_ttv') -> Path:
        output_filepath =  output_folder / (output_filestem + ".mp3")

        self.converter.tts(text, self.voice, output_filepath)

        cfg.LOGGER.debug(f"Audio saved as {output_filepath}")

        return output_filepath

    def list_voices(self):
        for voice_enum in self.converter.Voices.__members__.values():
            print(f"{voice_enum.value}: {voice_enum.name}")
        
class TextToSpeechConverter_PlayHT(TextToSpeechConverter):
    def __init__(self, voice_name=VoiceManifest.Charlotte_Narrative.name):
        super().__init__()
        self.converter = PlayHTVoiceGenerator()
        self.voice = VoiceManifest[voice_name]

    def text_to_speech(self, text: str, output_folder:Path = cfg.FOLDER_OUTPUT, output_filestem: str = 'test_playht') -> Path:
        output_filepath =  output_folder / (output_filestem + ".wav")

        self.converter.generate(text, self.voice, output_filepath)

        cfg.LOGGER.debug(f"Audio saved as {output_filepath}")

        return output_filepath

    def list_voices(self):
        for voice_enum in VoiceManifest.__members__.values():
            print(f"{voice_enum.value}: {voice_enum.name}")

def get_tts_module(tts_packed:str) -> TextToSpeechConverter:
    tts_modules = {
        'gtts': TextToSpeechConverter_gTTS,
        'pyttsx3': TextToSpeechConverter_Pyttsx3,
        'ttv': TextToSpeechConverter_TikTok,
        'pyht': TextToSpeechConverter_PlayHT,
    }

    tts_unpacked = tts_packed.split(':')

    tts_type = tts_unpacked[0]

    if len(tts_unpacked) > 1:
        tts_selection = tts_unpacked[1]
        tts_module = tts_modules.get(tts_type)(tts_selection)
    else:
        tts_module = tts_modules.get(tts_type)()
    
    return tts_module

if __name__ == '__main__':
    converter = TextToSpeechConverter_PlayHT(voice_name=VoiceManifest.Joseph.name)
    # converter.list_voices()
    converter.text_to_speech(text="My name is L...I am a famous detective.", output_folder=cfg.FOLDER_TTS, output_filestem='test')
