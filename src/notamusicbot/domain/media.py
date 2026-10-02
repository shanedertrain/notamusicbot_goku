"""Media domain models: queue items consumed by the Discord playback layer.

Playback code depends only on :class:`PlayableMedia`; where an item came from
(YouTube, Spotify resolution, generated speech, a local file) is decided by the
media resolvers and the command layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class MediaKind(StrEnum):
    """How the player obtains audio for an item."""

    LOCAL = "local"  # a file on disk (download cache, speech artifact, start sound)
    STREAM = "stream"  # a remote URL handed to ffmpeg (stream headers attached)


class PlaybackOutcome(StrEnum):
    """Why one queue item's playback ended (review Round-3 P0).

    ``_play_one()`` reports this alongside the media whose file actually
    played, so the post-song content hook can distinguish a song that really
    finished from one that was skipped, stopped, or failed.
    """

    NATURAL = "natural"  # reached the end of the audio
    SKIPPED = "skipped"  # user skipped the item
    STOPPED = "stopped"  # user stopped the bot/player
    FAILED = "failed"  # source/FFmpeg/Discord playback error
    DROPPED = "dropped"  # dequeued but never played (e.g. no voice client)


@dataclass(slots=True)
class PlayableMedia:
    """One unit of work for the guild playback queue.

    ``kind=LOCAL`` requires ``source_path``; ``kind=STREAM`` requires a
    ``stream_url`` (direct media URL for ffmpeg) or at least a ``source_url``
    (canonical webpage URL) — when only the source URL is available the player
    downloads via that URL before playback. ``temp_file=True`` marks files the
    player must delete after playback (download-cache entries); generated
    speech artifacts are owned by the speech pipeline's per-job directories and
    are cleaned up by its policy.
    """

    media_id: str
    kind: MediaKind
    title: str = ""
    requester: str = ""
    guild_id: int | None = None
    source_path: Path | None = None
    stream_url: str | None = None
    source_url: str | None = None  # canonical/webpage URL used by the download fallback
    stream_headers: Mapping[str, str] = field(default_factory=dict)
    temp_file: bool = False

    def __post_init__(self) -> None:
        if self.kind is MediaKind.LOCAL and self.source_path is None:
            raise ValueError("LOCAL media requires source_path")
        if self.kind is MediaKind.STREAM and not self.stream_url and not self.source_url:
            raise ValueError("STREAM media requires stream_url or source_url")

    def as_local(self, path: Path) -> PlayableMedia:
        """Return a copy of this item backed by a downloaded local file."""
        return PlayableMedia(
            media_id=self.media_id,
            kind=MediaKind.LOCAL,
            title=self.title,
            requester=self.requester,
            guild_id=self.guild_id,
            source_path=path,
            source_url=self.source_url,
            temp_file=True,
        )
