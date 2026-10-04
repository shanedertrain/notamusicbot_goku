"""Naive sentence trimming for generated commentary.

The commentary prompt asks for 2-3 sentences, but the local LLM regularly
runs longer (the user's real run produced 23 s of audio). Trimming the
generated text to at most ``content.max_sentences`` sentences before TTS is
the cheap lever: TTS and RVC time scale with audio duration.
"""

from __future__ import annotations

import re

# Sentence terminator (possibly followed by a closing quote) + whitespace.
# Naive on purpose: no abbreviation awareness, but commentary text is short
# and radio-style, so the rare mid-sentence "Mr. Smith" split costs at most
# one extra sentence of headroom.
_SENTENCE_END = re.compile(r"(?<=[.!?][\"'])\s+|(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences (terminators kept, whitespace stripped)."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def trim_sentences(text: str, max_sentences: int | None) -> str:
    """Return ``text`` limited to at most ``max_sentences`` sentences.

    ``max_sentences`` is ``None`` or ``< 1`` → the text is returned unchanged
    (stripped). Sentences are kept in order, terminators preserved.
    """
    text = text.strip()
    if max_sentences is None or max_sentences < 1:
        return text
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])
