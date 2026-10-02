"""Registry of generated speech files that are deleted after Discord playback.

The speech pipeline marks each successful job's ``final.wav`` here (unless
``keep_artifacts`` is enabled). The Discord player's ``cleanup_media`` path
consults this registry after playback and deletes the file. This keeps the
"final commentary file becomes temporary queue media" policy without requiring
every enqueue site to remember to set ``temp_file=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMP_ARTIFACTS: set[Path] = set()


def mark_temp_artifact(path: Path) -> None:
    """Register ``path`` for deletion after its playback completes."""
    _TEMP_ARTIFACTS.add(Path(path).resolve())


def is_temp_speech_artifact(path: Path) -> bool:
    """True when ``path`` was registered by the speech pipeline."""
    return Path(path).resolve() in _TEMP_ARTIFACTS


def discard_temp_speech_artifact(path: Path) -> None:
    """Unregister ``path`` (called by the cleanup path before/after deletion)."""
    _TEMP_ARTIFACTS.discard(Path(path).resolve())


def clear_registry() -> None:
    """Test helper: drop all registrations."""
    _TEMP_ARTIFACTS.clear()
