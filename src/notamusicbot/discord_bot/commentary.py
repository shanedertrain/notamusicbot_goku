"""Generated commentary audio, enqueued through the speech pipeline.

The speech pipeline (TTS -> VC -> post-process) already exists; this module
only turns its :class:`AudioArtifact` output into queueable media. A failure
here must never kill the media queue: per handoff section 20 we log, skip the
commentary, and continue with the media itself.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.speech import TtsRequest, VoiceConversionProfile

logger = logging.getLogger(__name__)


async def generate_commentary_media(
    pipeline: Any,
    request: TtsRequest,
    conversion: VoiceConversionProfile,
    *,
    guild_id: int,
    requester: str = "bot",
) -> PlayableMedia | None:
    """Run one speech job and return it as queueable media, or ``None`` on failure."""
    try:
        artifact, _metrics = await pipeline.run(request, conversion)
    except Exception:
        logger.exception(
            "Commentary generation failed in guild %s; skipping commentary and continuing playback",
            guild_id,
        )
        return None
    return PlayableMedia(
        media_id=uuid.uuid4().hex[:12],
        kind=MediaKind.LOCAL,
        title=f"commentary: {request.text[:60]}",
        requester=requester,
        guild_id=guild_id,
        source_path=artifact.path,
    )
