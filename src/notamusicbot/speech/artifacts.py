"""Registry of generated speech files that are deleted after Discord playback.

The speech pipeline marks each successful job's ``final.wav`` here (unless
``keep_artifacts`` is enabled). The Discord player's ``cleanup_media`` path
consults this registry after playback and deletes the file. This keeps the
"final commentary file becomes temporary queue media" policy without requiring
every enqueue site to remember to set ``temp_file=True``.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMP_ARTIFACTS: set[Path] = set()

# Marker written into a job directory when the pipeline job failed; it records
# the failure time so expired diagnostic directories can be pruned later.
FAILED_MARKER_NAME = ".failed"


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


# --------------------------------------------------------------- job lifecycle


def mark_failed_job(job_dir: Path, *, timestamp: datetime | None = None) -> Path:
    """Record the failure timestamp of ``job_dir`` in a ``.failed`` marker.

    Failed jobs keep their artifacts for diagnosis (handoff section 20), but
    the marker lets :func:`prune_expired_jobs` age them out later.
    """
    job_dir = Path(job_dir)
    marker = job_dir / FAILED_MARKER_NAME
    stamp = (timestamp or datetime.now(UTC)).isoformat()
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(stamp, encoding="utf-8")
    except OSError:
        logger.warning("Could not write failure marker in %s", job_dir, exc_info=True)
    return marker


def _job_failure_time(job_dir: Path) -> tuple[datetime | None, bool]:
    """(failure time, has marker) for a job dir; mtime fallback when unmarked."""
    marker = job_dir / FAILED_MARKER_NAME
    if marker.is_file():
        try:
            return datetime.fromisoformat(marker.read_text(encoding="utf-8").strip()), True
        except (OSError, ValueError):
            logger.debug("Unparseable failure marker in %s; falling back to mtime", job_dir)
            return None, True
    try:
        return datetime.fromtimestamp(job_dir.stat().st_mtime, tz=UTC), False
    except OSError:
        return None, False


def remove_empty_job_dir(job_dir: Path) -> bool:
    """Remove ``job_dir`` iff it exists and is empty. True when removed."""
    try:
        Path(job_dir).rmdir()
        return True
    except OSError:
        return False


def cleanup_temp_artifact(path: Path) -> None:
    """Full play-once cleanup for a registered temp artifact.

    Discards the registry entry, deletes the file, and removes the containing
    job directory when the deletion leaves it empty. Production callers
    (media.downloader) currently discard + unlink separately; this helper is
    the one-call equivalent and also used by tests.
    """
    path = Path(path)
    discard_temp_speech_artifact(path)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete temporary speech artifact %s", path, exc_info=True)
        return
    remove_empty_job_dir(path.parent)


def prune_expired_jobs(
    output_dir: Path,
    max_age_hours: float = 24.0,
    *,
    now: datetime | None = None,
) -> list[Path]:
    """Age out stale job directories under ``output_dir``.

    Removes (a) failed-job directories whose failure is older than
    ``max_age_hours`` (recursively — they keep diagnostic artifacts) and
    (b) empty job directories older than ``max_age_hours`` (left behind after
    the player deleted the final artifact). Returns the removed paths.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    current = now or datetime.now(UTC)
    removed: list[Path] = []
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            is_empty = not any(entry.iterdir())
        except OSError:
            continue
        failure_time, has_marker = _job_failure_time(entry)
        if failure_time is None:
            continue
        # Only failed jobs (marker present) are removed recursively; unmarked
        # directories are only ever dropped when already empty.
        if not has_marker and not is_empty:
            continue
        age_h = (current - failure_time).total_seconds() / 3600.0
        if age_h < max_age_hours:
            continue
        try:
            if is_empty:
                entry.rmdir()
            else:
                shutil.rmtree(entry)
            removed.append(entry)
        except OSError:
            logger.debug("Could not prune stale job directory %s", entry, exc_info=True)
    if removed:
        logger.info("Pruned %d stale speech job directories", len(removed))
    return removed
