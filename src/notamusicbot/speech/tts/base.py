"""SpeechSynthesizer protocol.

Backends implement :class:`SpeechSynthesizer` so callers never depend on a
concrete engine (Qwen3-TTS 0.6B Base zero-shot cloning now; VoiceDesign /
remote TTS later).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from notamusicbot.domain.speech import AudioArtifact, TtsRequest


class SpeechSynthesizer(Protocol):
    async def synthesize(
        self,
        request: TtsRequest,
        output_path: Path,
    ) -> AudioArtifact:
        """Synthesize ``request.text`` and write PCM WAV to ``output_path``."""
        ...
