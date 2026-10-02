"""Registry of generated speech files that are deleted after Discord playback.

The speech pipeline marks each successful job's ``final.wav`` here (unless
``keep_artifacts`` is enabled). The Discord player's ``cleanup_media`` path
consults this registry after playback and deletes the file. This keeps the
"final commentary file becomes temporary queue media" policy without requiring
every enqueue site to remember to set ``temp_file=True``.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TEMP_ARTIFACTS: set[Path] = set()

# Marker written into a job directory when the pipeline job failed; it records
# the failure time so expired diagnostic directories can be pruned later.
FAILED_MARKER_NAME = ".failed"

# Marker written into a job directory at creation (review Round-3 P1): a JSON
# blob with a timestamp and status ("pending" while running, "generated" once
# the final artifact exists and is registered for playback). If the process
# crashes after generation but before queue playback cleanup, the in-memory
# registry is lost — this on-disk marker lets retention/startup cleanup
# identify and prune the stale successful-but-unplayed temporary job.
TEMP_MARKER_NAME = ".temporary"


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


def mark_temporary_job(
    job_dir: Path, *, timestamp: datetime | None = None, status: str = "pending"
) -> Path:
    """Write the ``.temporary`` ownership marker into ``job_dir``.

    Called at speech-job creation (status ``"pending"``) and refreshed on
    success (status ``"generated"``). Never overwrites a ``.failed`` marker.
    """
    job_dir = Path(job_dir)
    marker = job_dir / TEMP_MARKER_NAME
    if (job_dir / FAILED_MARKER_NAME).is_file():
        return marker  # failed jobs keep their diagnostics marker instead
    stamp = (timestamp or datetime.now(UTC)).isoformat()
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"timestamp": stamp, "status": status}), encoding="utf-8")
    except OSError:
        logger.warning("Could not write temporary marker in %s", job_dir, exc_info=True)
    return marker


def _marker_time(job_dir: Path, name: str) -> tuple[datetime | None, bool]:
    """(timestamp, has marker) for a named JSON marker file."""
    marker = job_dir / name
    if not marker.is_file():
        return None, False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return datetime.fromisoformat(str(data["timestamp"])), True
    except (OSError, ValueError, KeyError, TypeError):
        logger.debug("Unparseable %s marker in %s; falling back to mtime", name, job_dir)
        return None, True


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
    job directory when the deletion leaves it empty (removing the job's
    ``.temporary`` marker first, since the marker itself keeps the directory
    non-empty). Production callers (the Discord player's cleanup path) use
    this helper so successful playback leaves no job directory behind.
    """
    path = Path(path)
    discard_temp_speech_artifact(path)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete temporary speech artifact %s", path, exc_info=True)
        return
    job_dir = path.parent
    try:
        (job_dir / TEMP_MARKER_NAME).unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove temporary marker in %s", job_dir, exc_info=True)
    remove_empty_job_dir(job_dir)


def prune_expired_jobs(
    output_dir: Path,
    max_age_hours: float = 24.0,
    *,
    now: datetime | None = None,
) -> list[Path]:
    """Age out stale job directories under ``output_dir``.

    Removes (a) failed-job directories whose failure is older than
    ``max_age_hours`` (recursively — they keep diagnostic artifacts), (b)
    ``.temporary``-marked job directories older than ``max_age_hours``
    (recursively — crash orphans: successful-but-unplayed temporary jobs whose
    in-memory registry was lost on restart), and (c) empty job directories
    older than ``max_age_hours`` (left behind after playback cleanup).
    Unmarked non-empty directories (e.g. a live job or ``keep_artifacts``
    debug output) are never touched. Returns the removed paths.
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
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
        except OSError:
            continue
        # .failed takes precedence: failed jobs keep diagnostics until their
        # own marker ages out, regardless of any temporary marker.
        failure_time, has_marker = _job_failure_time(entry)
        if has_marker:
            time_ref = failure_time or mtime
        else:
            temp_time, has_temp = _marker_time(entry, TEMP_MARKER_NAME)
            if has_temp:
                # Marked temporary job (crash orphan or unplayed artifact):
                # remove recursively once past the retention threshold.
                time_ref = temp_time or mtime
            elif is_empty:
                time_ref = mtime
            else:
                continue
        if time_ref is None:
            continue
        age_h = (current - time_ref).total_seconds() / 3600.0
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
