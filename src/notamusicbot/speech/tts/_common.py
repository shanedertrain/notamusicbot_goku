"""Shared helpers for the TTS backends."""

from __future__ import annotations

import math

# The "12Hz" speech codec runs at 12.5 frames (talker tokens) per second of
# audio (speech_tokenizer config: 24000 Hz / 1920 = 12.5). The generation bound
# gets about one extra second of slack so a cap that lands mid-word is trimmed
# (with a fade) by the caller instead of being cut dead by the model.
CODEC_TOKENS_PER_SECOND = 12.5
MAX_NEW_TOKENS_SLACK = math.ceil(CODEC_TOKENS_PER_SECOND)


def max_new_tokens_for(max_audio_seconds: float | None) -> int | None:
    """``max_new_tokens`` bound for an audio cap, or None for no bound."""
    if max_audio_seconds is None:
        return None
    return math.ceil(max_audio_seconds * CODEC_TOKENS_PER_SECOND) + MAX_NEW_TOKENS_SLACK


def normalize_text(text: str) -> str:
    """Minimal, explicit text normalization.

    Punctuation and quotes are preserved (the old pipeline stripped quotes
    indiscriminately). Only surrounding whitespace is trimmed; empty input is
    rejected by the caller before reaching the model.
    """
    return " ".join(text.split())
