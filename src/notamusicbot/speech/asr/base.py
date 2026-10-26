"""Transcriber protocol.

Backends implement :class:`Transcriber` so the listen pipeline never depends
on a concrete engine (Whisper small.en in-process now; a Qwen3-ASR worker
later). Audio is passed in memory, never as a path.
"""

from __future__ import annotations

from typing import Protocol

from notamusicbot.domain.listen import AudioChunk, Transcript


class Transcriber(Protocol):
    async def transcribe(self, audio: AudioChunk) -> Transcript:
        """Transcribe float32 mono 16 kHz ``audio``."""
        ...

    async def shutdown(self) -> None:
        """Release the model / worker."""
        ...
