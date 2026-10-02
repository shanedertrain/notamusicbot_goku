"""Round-2 P2: empty job-dir removal, failed-job markers, retention pruning."""

from __future__ import annotations

import asyncio
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
