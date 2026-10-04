"""VoiceConverter protocol.

Callers depend on this abstraction; the legacy RVC implementation is just one
backend among several.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile


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
