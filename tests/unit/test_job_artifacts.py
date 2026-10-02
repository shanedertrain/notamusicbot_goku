"""Round-2 P2: empty job-dir removal, failed-job markers, retention pruning."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from conftest import FakeConverter, FakeSynthesizer
from notamusicbot.domain.speech import (
    AudioProcessingProfile,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.artifacts import (
    FAILED_MARKER_NAME,
    cleanup_temp_artifact,
    clear_registry,
    mark_failed_job,
    prune_expired_jobs,
    remove_empty_job_dir,
)
from notamusicbot.speech.pipeline import SpeechPipeline

CONVERSION = VoiceConversionProfile(backend="passthrough")
PROCESSING = AudioProcessingProfile(target_sample_rate_hz=48000, channels=1)


class _ExplodingConverter(FakeConverter):
    async def convert(self, source, profile, output_path):
        raise RuntimeError("RVC worker exploded")


def _pipeline(tmp_path, synth, conv, **kwargs) -> SpeechPipeline:
    return SpeechPipeline(
        synth,
        conv,
        PROCESSING,
        output_dir=tmp_path / "speech",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


def test_successful_job_leaves_no_empty_dir_after_final_cleanup(tmp_path) -> None:
    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter())
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
    )
    job_dir = tmp_path / "speech" / metrics.job_id
    assert final.path.is_file()  # survives until playback cleanup
    # Full play-once cleanup: file removed, then the emptied job dir removed.
    cleanup_temp_artifact(final.path)
    assert not final.path.exists()
    assert not job_dir.exists()


def test_remove_empty_job_dir_only_when_empty(tmp_path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    assert remove_empty_job_dir(job_dir) is True
    assert remove_empty_job_dir(job_dir) is False  # gone
    job_dir.mkdir()
    (job_dir / "final.wav").write_bytes(b"x")
    assert remove_empty_job_dir(job_dir) is False  # not empty
    assert job_dir.exists()


def test_failed_job_keeps_artifacts_and_records_timestamp(tmp_path) -> None:
    pipeline = _pipeline(tmp_path, FakeSynthesizer(), _ExplodingConverter())
    with pytest.raises(RuntimeError, match="exploded"):
        asyncio.run(
            pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
        )
    job_dirs = list((tmp_path / "speech").iterdir())
    assert len(job_dirs) == 1
    job_dir = job_dirs[0]
    assert (job_dir / "tts_source.wav").exists()  # diagnostics retained
    marker = job_dir / FAILED_MARKER_NAME
    assert marker.is_file()
    # Marker holds a parseable ISO timestamp close to now.
    stamp = datetime.fromisoformat(marker.read_text().strip())
    assert abs((datetime.now(UTC) - stamp).total_seconds()) < 60


def test_prune_expired_jobs_removes_old_failed_and_empty_dirs(tmp_path) -> None:
    output = tmp_path / "speech"
    output.mkdir()

    now = datetime.now(UTC)
    old_failed = output / "oldfailed"
    old_failed.mkdir()
    mark_failed_job(old_failed, timestamp=now - timedelta(hours=30))
    (old_failed / "tts_source.wav").write_bytes(b"x")

    recent_failed = output / "recentfailed"
    recent_failed.mkdir()
    mark_failed_job(recent_failed, timestamp=now - timedelta(hours=1))

    old_empty = output / "oldempty"
    old_empty.mkdir()
    old_time = (now - timedelta(hours=30)).timestamp()
    import os

    os.utime(old_empty, (old_time, old_time))

    recent_busy = output / "recentbusy"  # no marker, not empty -> never touched
    recent_busy.mkdir()
    (recent_busy / "final.wav").write_bytes(b"x")

    removed = prune_expired_jobs(output, 24.0, now=now)
    names = {p.name for p in removed}
    assert names == {"oldfailed", "oldempty"}
    assert not old_failed.exists()
    assert not old_empty.exists()
    assert recent_failed.exists()
    assert recent_busy.exists()


def test_pipeline_start_prunes_expired_dirs(tmp_path) -> None:
    output = tmp_path / "speech"
    output.mkdir()
    stale = output / "stale"
    stale.mkdir()
    mark_failed_job(stale, timestamp=datetime.now(UTC) - timedelta(hours=48))
    (stale / "converted.wav").write_bytes(b"x")

    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter())
    asyncio.run(
        pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
    )
    assert not stale.exists()  # pruned opportunistically at job start


def test_keep_artifacts_mode_never_prunes_or_removes(tmp_path) -> None:
    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter(), keep_artifacts=True)
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
    )
    job_dir = tmp_path / "speech" / metrics.job_id
    assert (job_dir / "tts_source.wav").exists()
    assert (job_dir / "converted.wav").exists()
    assert final.path.is_file()
    assert job_dir.exists()


# ------------------------------------------------- Round-3 P1 crash orphans


def test_temporary_marker_written_and_refreshed_on_success(tmp_path) -> None:
    from notamusicbot.speech.artifacts import TEMP_MARKER_NAME

    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter())
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
    )
    marker = tmp_path / "speech" / metrics.job_id / TEMP_MARKER_NAME
    assert marker.is_file()
    data = json.loads(marker.read_text())
    assert data["status"] == "generated"  # refreshed from "pending" on success
    assert abs((datetime.now(UTC) - datetime.fromisoformat(data["timestamp"])).total_seconds()) < 60
    assert final.path.is_file()


def test_full_cleanup_removes_file_marker_and_job_dir(tmp_path) -> None:
    """(a) normal generated playback cleanup: file gone AND job dir gone."""
    from notamusicbot.speech.artifacts import TEMP_MARKER_NAME

    pipeline = _pipeline(tmp_path, FakeSynthesizer(), FakeConverter())
    final, metrics = asyncio.run(
        pipeline.run(TtsRequest(text="hello", language="English", speaker="Aiden"), CONVERSION)
    )
    job_dir = tmp_path / "speech" / metrics.job_id
    assert job_dir.is_dir()  # marker keeps it alive until playback cleanup
    cleanup_temp_artifact(final.path)
    assert not final.path.exists()
    assert not (job_dir / TEMP_MARKER_NAME).exists()
    assert not job_dir.exists()


def test_prune_removes_stale_successful_temporary_crash_orphan(tmp_path) -> None:
    """(b) simulated crash: a successful-but-unplayed temporary job with an
    old marker is pruned per the retention policy; recent/unmarked survive."""
    from notamusicbot.speech.artifacts import mark_temporary_job

    output = tmp_path / "speech"
    output.mkdir()
    now = datetime.now(UTC)

    stale = output / "crashedjob"
    stale.mkdir()
    mark_temporary_job(stale, timestamp=now - timedelta(hours=30), status="generated")
    (stale / "final.wav").write_bytes(b"x")

    recent = output / "livejob"
    recent.mkdir()
    mark_temporary_job(recent, status="pending")
    (recent / "tts_source.wav").write_bytes(b"x")

    unmarked = output / "keepartifacts"  # no marker, non-empty -> never touched
    unmarked.mkdir()
    (unmarked / "final.wav").write_bytes(b"x")

    removed = prune_expired_jobs(output, 24.0, now=now)
    assert {p.name for p in removed} == {"crashedjob"}
    assert not stale.exists()
    assert recent.exists()
    assert unmarked.exists()


def test_prune_takes_failed_marker_precedence_over_temporary(tmp_path) -> None:
    from notamusicbot.speech.artifacts import TEMP_MARKER_NAME, mark_temporary_job

    output = tmp_path / "speech"
    output.mkdir()
    now = datetime.now(UTC)
    mixed = output / "mixed"
    mixed.mkdir()
    mark_temporary_job(mixed, status="pending")
    mark_failed_job(mixed, timestamp=now - timedelta(hours=1))
    (mixed / "tts_source.wav").write_bytes(b"x")
    # Recent failure (inside retention): kept even though a temporary marker
    # with an older default mtime exists — diagnostics policy wins.
    removed = prune_expired_jobs(output, 24.0, now=now)
    assert removed == []
    assert mixed.exists()
    assert (mixed / TEMP_MARKER_NAME).is_file() or (mixed / FAILED_MARKER_NAME).is_file()


def test_cancelled_speech_job_is_deterministic_not_swallowed(tmp_path) -> None:
    """(c) cancelled speech job: CancelledError propagates (never swallowed)
    and the job dir keeps its .temporary marker for retention pruning."""

    class _HangingSynth(FakeSynthesizer):
        async def synthesize(self, request, output_path):
            await asyncio.Event().wait()  # never resolves until cancelled

    pipeline = _pipeline(tmp_path, _HangingSynth(), FakeConverter())

    async def _run() -> None:
        task = asyncio.ensure_future(
            pipeline.run(TtsRequest(text="hi", language="English", speaker="Aiden"), CONVERSION)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    job_dirs = list((tmp_path / "speech").iterdir())
    assert len(job_dirs) == 1
    marker = job_dirs[0] / "marker_placeholder"
    from notamusicbot.speech.artifacts import TEMP_MARKER_NAME

    marker = job_dirs[0] / TEMP_MARKER_NAME
    assert marker.is_file()
    assert json.loads(marker.read_text())["status"] == "pending"
    # And retention cleanup removes it deterministically once stale.
    from notamusicbot.speech.artifacts import mark_temporary_job

    stale_time = datetime.now(UTC) - timedelta(hours=48)
    mark_temporary_job(job_dirs[0], timestamp=stale_time, status="pending")
    assert prune_expired_jobs(tmp_path / "speech", 24.0) == [job_dirs[0]]
    assert not job_dirs[0].exists()
