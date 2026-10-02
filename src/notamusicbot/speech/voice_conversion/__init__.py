"""Voice conversion backends."""

from notamusicbot.speech.voice_conversion.base import VoiceConverter
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter
from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

__all__ = ["PassthroughVoiceConverter", "RvcVoiceConverter", "VoiceConverter"]
