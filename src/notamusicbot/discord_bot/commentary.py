"""Commentary text generation and its conversion to queueable audio.

Text comes from a :class:`~notamusicbot.content.base.ContentProvider` (local
LLM when enabled, otherwise the static template). A provider failure falls
back to the static template per handoff section 20: log, degrade, continue —
playback itself is never at risk.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from notamusicbot.content import prompts
from notamusicbot.content.base import ContentRequest
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.speech import TtsRequest, VoiceConversionProfile

logger = logging.getLogger(__name__)


async def resolve_commentary_text(provider: Any, request: ContentRequest) -> str:
    """Generate commentary text via the provider, falling back to the template.

    ``provider is None`` (content disabled) and provider failures both yield
    the static template text, so the result is always speakable.
    """
    if provider is None:
        return _static_for(request)
    try:
        return await provider.generate(request)
    except Exception:
        logger.warning(
            "Content provider failed (%s); using static template commentary",
            request.kind.name,
            exc_info=True,
        )
        return _static_for(request)


def _static_for(request: ContentRequest) -> str:
    if request.song_name:
        return prompts.static_song_comment(request.song_name, request.requester_name)
    return prompts.static_article_comment(request.post_title or request.article_text[:60])


async def generate_commentary_media(
    pipeline: Any,
    request: TtsRequest,
    conversion: VoiceConversionProfile,
    *,
    guild_id: int,
    requester: str = "bot",
    processing: Any | None = None,
) -> PlayableMedia | None:
    """Run one speech job and return it as queueable media, or ``None`` on failure.

    ``processing`` is the voice profile's :class:`AudioProcessingProfile`
    (speed, target peak, output format); ``None`` uses the pipeline default.
    """
    try:
        artifact, _metrics = await pipeline.run(request, conversion, processing)
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
