"""Background prefetch of post-song commentary while a STREAM song plays.

The user-visible win: post-song commentary (``content.max_sentences``-capped
LLM text → Qwen TTS → RVC) takes ~60-80 s on this machine. Instead of
generating it AFTER the song finishes (a long silent gap before the next
dequeue), generation starts when the song starts (``on_now_playing`` hook)
and runs through the shared :class:`InferenceCoordinator` — so it serializes
with every other GPU job and never overlaps TTS/RVC inference.

Lifecycle (review-critical invariants preserved):

- ``start_prefetch`` is called from the now-playing hook for STREAM items
  only; any previous prefetch for the guild is discarded first (a new song
  starting means the previous song ended unnaturally — skip/stop/fail — or
  was already handled). LOCAL media never reaches here, so prefetched
  commentary cannot recurse into more post-song content.
- At NATURAL completion the player's completion hook calls ``consume``:
  if the prefetched artifact(s) for THIS media are ready they are enqueued
  via ``put_front`` exactly like the at-completion path (ordering unchanged:
  song A → segment → song B) and the caller skips generation entirely
  (no second LLM/TTS run). Otherwise — prefetch still running, failed, or
  the scheduler decided not to fire — the entry is discarded and the caller
  falls back to the current at-completion generation path
  (:func:`fire_post_song_content`).
- Discard semantics: a still-running generation task is marked discarded and
  cleans up its own artifacts when it finishes (per the temporary-artifact
  policy); ready-but-unconsumed artifacts are deleted immediately. A
  discarded entry can never enqueue — double generation is impossible
  because the completion hook consults the registry before falling back.

The interstitial/podcast SCHEDULING decision is made once, at prefetch time
(``podcasts.active`` first, then ``interstitials.should_fire``); at
completion a not-ready entry always falls back to the normal path, which
re-evaluates the scheduler exactly as before this module existed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from notamusicbot.discord_bot.commentary import (
    generate_commentary_media,
    resolve_commentary_text,
)
from notamusicbot.discord_bot.player import cleanup_media_item
from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.domain.voice_profiles import to_conversion_profile, to_processing_profile

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PrefetchEntry:
    media_id: str
    task: asyncio.Task[None] | None = None
    # pending → generating; ready → artifacts available; failed → generation
    # error; not_fired → scheduler decided no post-song content;
    # discarded → superseded/consumed-fallback, artifacts must be cleaned.
    status: str = "pending"
    media: list[PlayableMedia] = field(default_factory=list)


class CommentaryPrefetcher:
    """Per-guild registry of in-flight/ready post-song commentary prefetches."""

    def __init__(self) -> None:
        self._entries: dict[int, _PrefetchEntry] = {}

    # ---------------------------------------------------------------- API

    def start_prefetch(self, host: Any, guild_id: int, media: PlayableMedia) -> bool:
        """Start background generation for a just-started STREAM song.

        Fire-and-forget: spawns one asyncio task and returns immediately (the
        player awaits only the announcement itself). Returns ``True`` when a
        prefetch task was created.
        """
        settings = host.app.settings
        if not getattr(settings.content, "prefetch_enabled", False):
            return False
        if media.kind is not MediaKind.STREAM:
            return False
        # A new STREAM song starting invalidates any previous entry for this
        # guild (its song ended unnaturally or was already handled).
        self.discard(guild_id, reason="new song started")
        entry = _PrefetchEntry(media_id=media.media_id)
        self._entries[guild_id] = entry
        entry.task = asyncio.create_task(
            self._generate(host, guild_id, entry),
            name=f"commentary-prefetch-{guild_id}",
        )
        return True

    async def consume(self, host: Any, guild_id: int, media: PlayableMedia) -> bool:
        """At NATURAL completion: enqueue a ready prefetch, else give up on it.

        Returns ``True`` when prefetched audio was enqueued (caller must NOT
        generate again). Returns ``False`` when there is no usable prefetch —
        the caller falls back to the at-completion generation path.
        """
        entry = self._entries.pop(guild_id, None)
        if entry is None or entry.media_id != media.media_id:
            return False
        if entry.status == "ready" and entry.media:
            guild_state = host.guild_state(guild_id)
            for item in entry.media:
                # Same head-of-queue insertion as the at-completion path, so
                # the ordering invariant (song A → segment → song B) holds.
                guild_state.put_front(item)
            logger.info(
                "Enqueued prefetched post-song commentary in guild %s (%d item(s))",
                guild_id,
                len(entry.media),
            )
            return True
        # Not finished in time / failed / scheduler decided not to fire:
        # give up on the entry. A still-running task is marked discarded so
        # it can never enqueue and cleans its artifacts when it finishes.
        if entry.status == "pending" and entry.task is not None and not entry.task.done():
            entry.status = "discarded"
        else:
            self._cleanup_media(entry.media)
            entry.media = []
        logger.info(
            "Prefetch for guild %s not usable at completion (status=%s); falling back",
            guild_id,
            entry.status,
        )
        return False

    def discard(self, guild_id: int, *, reason: str) -> None:
        """Drop the guild's entry, cleaning any ready artifacts."""
        entry = self._entries.pop(guild_id, None)
        if entry is None:
            return
        if entry.status == "pending" and entry.task is not None and not entry.task.done():
            entry.status = "discarded"
        else:
            self._cleanup_media(entry.media)
        logger.debug("Discarded prefetch in guild %s (%s)", guild_id, reason)

    def discard_all(self) -> None:
        """Teardown path: drop every guild's entry."""
        for guild_id in list(self._entries):
            self.discard(guild_id, reason="shutdown")

    def pending_media_ids(self) -> dict[int, str]:
        """Test/diagnostic view: guild id → prefetched media id."""
        return {gid: entry.media_id for gid, entry in self._entries.items()}

    # ----------------------------------------------------------- generation

    async def _generate(self, host: Any, guild_id: int, entry: _PrefetchEntry) -> None:
        try:
            media = await self._generate_media(host, guild_id)
        except asyncio.CancelledError:
            entry.status = "discarded"
            self._cleanup_media(entry.media)
            raise
        except Exception:
            logger.warning(
                "Prefetch generation failed in guild %s; completion will fall back",
                guild_id,
                exc_info=True,
            )
            entry.status = "failed"
            return
        if entry.status == "discarded":
            # Superseded while generating: apply the artifact cleanup policy.
            self._cleanup_media(media)
            return
        entry.media = media
        entry.status = "ready" if media else "not_fired"

    async def _generate_media(self, host: Any, guild_id: int) -> list[PlayableMedia]:
        """Mirror of the fire-time decision + generation, WITHOUT enqueueing."""
        max_sentences = getattr(host.app.settings.content, "max_sentences", None)
        if host.podcasts.active(guild_id):
            turns = await host.podcasts.segment_turns(guild_id)
            media: list[PlayableMedia] = []
            # Same iteration order as fire_post_song_content: reversed turns,
            # each later put_front'd in this order keeps the turn order.
            for turn in reversed(turns):
                item = await self._synthesize(
                    host,
                    guild_id,
                    turn.voice_profile_id,
                    trim_text(turn.text, max_sentences),
                    requester="podcast",
                )
                if item is not None:
                    media.append(item)
            return media

        if not host.interstitials.should_fire(guild_id=guild_id):
            return []
        topic = await host.interstitials.topic_source.pick()
        if topic is None:
            return []
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
        text = await resolve_commentary_text(provider, request, max_sentences=max_sentences)
        item = await self._synthesize(host, guild_id, profile_id, text, requester="interstitial")
        return [item] if item is not None else []

    async def _synthesize(
        self,
        host: Any,
        guild_id: int,
        voice_profile_id: str,
        text: str,
        *,
        requester: str,
    ) -> PlayableMedia | None:
        speech_settings = host.app.settings.speech
        if not speech_settings.commentary_enabled:
            return None
        pipeline = host.app.pipeline
        if pipeline is None:
            return None
        profile = host.app.voice_profiles.get(voice_profile_id)
        if profile is None:
            logger.warning("No voice profile %r; skipping prefetched commentary", voice_profile_id)
            return None
        from notamusicbot.domain.speech import TtsRequest

        request = TtsRequest(text=text, language=profile.tts.language, speaker=profile.tts.speaker)
        return await generate_commentary_media(
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

    @staticmethod
    def _cleanup_media(media: list[PlayableMedia]) -> None:
        for item in media:
            try:
                cleanup_media_item(item)
            except Exception:
                logger.warning("Failed to clean up prefetched artifact %s", item.source_path)


def trim_text(text: str, max_sentences: int | None) -> str:
    """Trim helper indirection (keeps the trim import in one place)."""
    from notamusicbot.content.trim import trim_sentences

    return trim_sentences(text, max_sentences)
