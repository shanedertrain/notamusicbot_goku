"""Text-to-speech backends."""

from notamusicbot.speech.tts.base import SpeechSynthesizer
from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

__all__ = ["QwenCloneSynthesizer", "SpeechSynthesizer"]
