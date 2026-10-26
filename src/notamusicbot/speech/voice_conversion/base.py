"""VoiceConverter protocol.

Callers depend on this abstraction; the legacy RVC implementation is just one
backend among several.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile

STREAM_INPUT_RATE_HZ = 24000
STREAM_OUTPUT_RATE_HZ = 48000


@dataclass(frozen=True, slots=True)
class VoiceStreamParams:
    """Streaming-conversion tuning (``speech.voice_conversion.stream_*``)."""

    enabled: bool = False
    first_block_ms: int = 100
    block_ms: int = 320
    context_ms: int = 1000
    lookahead_ms: int = 100
    crossfade_ms: int = 40
    search_ms: int = 20

    @classmethod
    def from_settings(cls, s: Any) -> VoiceStreamParams:
        """Build from a ``VoiceConversionSettings``-like object (duck-typed)."""
        return cls(
            enabled=bool(s.streaming),
            first_block_ms=int(s.stream_first_block_ms),
            block_ms=int(s.stream_block_ms),
            context_ms=int(s.stream_context_ms),
            lookahead_ms=int(s.stream_lookahead_ms),
            crossfade_ms=int(s.stream_crossfade_ms),
            search_ms=int(s.stream_search_ms),
        )


class VoiceConverter(Protocol):
    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        """Convert ``source`` speech toward the target voice; PCM WAV out."""
        ...

    async def shutdown(self) -> None:
        """Release backend resources (no-op for stateless backends)."""
        ...


class StreamingVoiceConverter(Protocol):
    def supports_streaming(self, profile: VoiceConversionProfile) -> bool:
        """True when :meth:`convert_stream` can serve ``profile``."""
        ...

    def convert_stream(
        self,
        chunks: AsyncIterator[np.ndarray],
        profile: VoiceConversionProfile,
    ) -> AsyncIterator[np.ndarray]:
        """Convert a mono float32 24 kHz chunk stream to mono float32 48 kHz.

        Total output samples equal 2 x total input samples.
        """
        ...
