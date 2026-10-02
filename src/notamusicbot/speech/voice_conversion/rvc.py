"""RVC voice converter adapter (stub).

The concrete implementation is filled in by the Phase D agent. Business and
Discord code must import only :class:`notamusicbot.speech.voice_conversion.base
.VoiceConverter`, never ``rvc_cli``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile

logger = logging.getLogger(__name__)


class RvcVoiceConverter:
    """Placeholder adapter for the legacy RVC backend (Phase D)."""

    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        msg = "RvcVoiceConverter is not implemented yet (Phase D); use passthrough"
        raise NotImplementedError(msg)
