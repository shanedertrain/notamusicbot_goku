"""Post-song content glue: interstitials and podcast turns into the queue.

Shared production path for review blocker P0-5 / Gate R6:

    fetcher -> TopicItem -> ContentRequest -> ContentProvider
            -> SpeechPipeline -> guild queue

``host`` is duck-typed (the real :class:`~notamusicbot.discord_bot.bot.NotAMusicBot`
or a test fake): it needs ``.app`` (settings, ``voice_profiles``,
``build_content_provider``), ``.guild_state(guild_id)``, ``.interstitials`` and
``.podcasts``. Every failure is logged and swallowed — playback is never at
risk (handoff section 20).
"""

from __future__ import annotations

import logging
from typing import Any

from notamusicbot.discord_bot.commentary import (
    generate_commentary_media,
    resolve_commentary_text,
)
from notamusicbot.domain.voice_profiles import to_conversion_profile, to_processing_profile

logger = logging.getLogger(__name__)


async def enqueue_profile_commentary(
    host: Any,
    guild_id: int,
    voice_profile_id: str,
    text: str,
    *,
    requester: str,
) -> bool:
    """Synthesize ``text`` with the given voice profile and enqueue the audio.

    Returns ``True`` when a media item was enqueued. Respects the
    ``speech.commentary_enabled`` flag and the pipeline's availability.
    """
    speech_settings = host.app.settings.speech
    if not speech_settings.commentary_enabled:
        return False
    pipeline = host.app.pipeline
    if pipeline is None:
        return False
    profile = host.app.voice_profiles.get(voice_profile_id)
    if profile is None:
        logger.warning("No voice profile %r; skipping generated commentary", voice_profile_id)
        return False
    from notamusicbot.domain.speech import TtsRequest

    request = TtsRequest(text=text, language=profile.tts.language, speaker=profile.tts.speaker)
    media = await generate_commentary_media(
        pipeline,
        request,
        to_conversion_profile(profile),
        guild_id=guild_id,
        requester=requester,
        processing=to_processing_profile(
            profile,
            sample_rate_hz=speech_settings.sample_rate_hz,
            channels=speech_settings.channels,
        ),
    )
    if media is None:
        return False
    # Post-song content goes to the HEAD of the queue: the completion hook is
    # awaited before the next dequeue, so the segment plays before any song
    # that was already queued (song A → segment → song B).
    host.guild_state(guild_id).put_front(media)
    return True


async def fire_post_song_content(host: Any, guild_id: int) -> bool:
    """Fire podcast/interstitial content after a song finished.

    Podcast mode takes precedence while active; otherwise the randomized
    interstitial scheduler decides. Returns ``True`` when audio was enqueued.
    """
    try:
        if host.podcasts.active(guild_id):
            turns = await host.podcasts.segment_turns(guild_id)
            enqueued = False
            # Each turn is inserted at the head, so enqueue in reverse to keep
            # the conversation turn order.
            for turn in reversed(turns):
                if await enqueue_profile_commentary(
                    host, guild_id, turn.voice_profile_id, turn.text, requester="podcast"
                ):
                    enqueued = True
            if turns and not enqueued:
                logger.info("Podcast segment generated but no audio enqueued in guild %s", guild_id)
            return enqueued

        if not host.interstitials.should_fire(guild_id=guild_id):
            return False
        topic = await host.interstitials.topic_source.pick()
        if topic is None:
            return False
        settings = host.app.settings
        profile_id = settings.interstitial.voice_profile
        profile = host.app.voice_profiles.get(profile_id)
        persona = profile.persona.description if profile is not None else ""
        if profile is None:
            logger.warning(
                "Interstitial voice profile %r not found; using default persona", profile_id
            )
        provider = host.app.build_content_provider()
        request = host.interstitials.build_request(topic, persona)
        text = await resolve_commentary_text(provider, request)
        return await enqueue_profile_commentary(
            host, guild_id, profile_id, text, requester="interstitial"
        )
    except Exception:
        logger.warning(
            "Post-song content generation failed in guild %s; continuing playback",
            guild_id,
            exc_info=True,
        )
        return False
