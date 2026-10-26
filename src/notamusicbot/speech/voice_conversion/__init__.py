"""Voice conversion backends."""

from notamusicbot.speech.voice_conversion.base import (
    STREAM_INPUT_RATE_HZ,
    STREAM_OUTPUT_RATE_HZ,
    StreamingVoiceConverter,
    VoiceConverter,
    VoiceStreamParams,
)
from notamusicbot.speech.voice_conversion.passthrough import PassthroughVoiceConverter
from notamusicbot.speech.voice_conversion.router import (
    RvcUnavailableError,
    UnknownBackendError,
    VoiceConverterRouter,
)
from notamusicbot.speech.voice_conversion.rvc import RvcStreamError, RvcVoiceConverter

__all__ = [
    "STREAM_INPUT_RATE_HZ",
    "STREAM_OUTPUT_RATE_HZ",
    "PassthroughVoiceConverter",
    "RvcStreamError",
    "RvcUnavailableError",
    "RvcVoiceConverter",
    "StreamingVoiceConverter",
    "UnknownBackendError",
    "VoiceConverter",
    "VoiceConverterRouter",
    "VoiceStreamParams",
]
