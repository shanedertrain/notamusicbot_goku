"""Text-to-speech backends."""

from notamusicbot.speech.tts.base import SpeechSynthesizer
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer

__all__ = ["QwenCustomVoiceSynthesizer", "SpeechSynthesizer"]
