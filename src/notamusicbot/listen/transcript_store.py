"""Text transcripts of a listen session (owner decision, 2026-10-26).

Text only, never audio. Only lines the turn-taker accepted are written, plus
the bot's own replies. One file per day under a gitignored directory; files
older than the retention window are deleted when the store is created.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptStore:
    def __init__(self, directory: Path, retention_days: int) -> None:
        self._dir = directory
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._prune(retention_days)

    def write(self, who: str, text: str, *, now: datetime | None = None) -> None:
        stamp = now or datetime.now().astimezone()
        line = f"{stamp:%H:%M:%S} {who}: {' '.join(text.split())}\n"
        path = self._dir / f"{stamp:%Y-%m-%d}.log"
        try:
            with self._lock:
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as handle:
                    handle.write(line)
        except OSError:
            logger.warning("Could not write listen transcript", exc_info=True)

    def _prune(self, retention_days: int) -> None:
        cutoff = datetime.now().astimezone() - timedelta(days=retention_days)
        for path in self._dir.glob("*.log"):
            try:
                if datetime.fromtimestamp(path.stat().st_mtime).astimezone() < cutoff:
                    path.unlink()
            except OSError:
                logger.warning("Could not prune %s", path.name, exc_info=True)
