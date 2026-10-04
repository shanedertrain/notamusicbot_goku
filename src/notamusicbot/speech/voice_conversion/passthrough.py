"""Passthrough voice converter: copies source audio unchanged.

Used to evaluate Qwen TTS without RVC and as the VC backend for tests and
profiles that do not require conversion.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile

logger = logging.getLogger(__name__)


class PassthroughVoiceConverter:
    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, source.path, output_path)
        logger.info("Passthrough VC: copied %s -> %s", source.path, output_path)
        return AudioArtifact(
            path=output_path,
            sample_rate_hz=source.sample_rate_hz,
            duration_s=source.duration_s,
        )

    async def shutdown(self) -> None:
        """No resources to release: passthrough copies files and exits."""
        return None
