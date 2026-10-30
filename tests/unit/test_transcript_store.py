"""Opt-in listen transcripts: text only, private files, retention pruning."""

from __future__ import annotations

import os
import stat
import time
from datetime import datetime

from notamusicbot.listen.transcript_store import TranscriptStore


def test_writes_timestamped_line_to_private_daily_file(tmp_path):
    store = TranscriptStore(tmp_path / "logs", retention_days=7)
    store.write("Sam", "hello\n  there", now=datetime(2026, 10, 26, 22, 5, 9).astimezone())
    path = tmp_path / "logs" / "2026-10-26.log"
    assert path.read_text() == "22:05:09 Sam: hello there\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_prunes_files_older_than_retention(tmp_path):
    old = tmp_path / "old.log"
    new = tmp_path / "new.log"
    old.write_text("x")
    new.write_text("y")
    ten_days_ago = time.time() - 10 * 86400
    os.utime(old, (ten_days_ago, ten_days_ago))
    TranscriptStore(tmp_path, retention_days=7)
    assert not old.exists()
    assert new.exists()
