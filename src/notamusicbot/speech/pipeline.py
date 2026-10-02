"""Speech pipeline: TTS -> voice conversion -> post-processing -> final WAV.

Discord and content code never performs these stages manually; they call
:attr:`SpeechPipeline.run`. Intermediates are PCM/WAV only.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Protocol

import soundfile as sf

from notamusicbot.domain.speech import (
    AudioArtifact,
    AudioProcessingProfile,
    SpeechJobMetrics,
    TtsRequest,
    VoiceConversionProfile,
)
from notamusicbot.speech.artifacts import (
    mark_failed_job,
    mark_temp_artifact,
    prune_expired_jobs,
    remove_empty_job_dir,
)
from notamusicbot.speech.coordinator import InferenceCoordinator
from notamusicbot.speech.postprocess import postprocess
from notamusicbot.speech.tts.base import SpeechSynthesizer
from notamusicbot.speech.voice_conversion.base import VoiceConverter

logger = logging.getLogger(__name__)


class PipelineClock(Protocol):
    """Injectable clock so tests can control timing without sleeping."""

    def perf_counter(self) -> float: ...


class SpeechPipeline:
    """Owns the full TTS -> VC -> postprocess chain for one request."""

    def __init__(
        self,
        synthesizer: SpeechSynthesizer,
        converter: VoiceConverter,
        processing: AudioProcessingProfile | None = None,
        coordinator: InferenceCoordinator | None = None,
        output_dir: Path = Path("var/speech"),
        *,
        keep_artifacts: bool = False,
        failed_job_retention_hours: float = 24.0,
    ) -> None:
        self._synthesizer = synthesizer
        self._converter = converter
        self._processing = processing or AudioProcessingProfile()
        self._coordinator = coordinator or InferenceCoordinator()
        self._output_dir = output_dir
        # Artifact cleanup policy (review P1-11):
        #   tts_source.wav  deleted after VC succeeds
        #   converted.wav   deleted after post-processing succeeds
        #   final.wav       registered as play-once queue media; the Discord
        #                   player deletes it after playback (see
        #                   speech/artifacts.py + media.downloader.cleanup_media)
        # A failed stage retains what it produced so failures stay diagnosable;
        # ``keep_artifacts=True`` (debug/benchmark mode) retains everything.
        self._keep_artifacts = keep_artifacts
        self._failed_job_retention_hours = failed_job_retention_hours

    def _paths(self, job_id: str) -> tuple[Path, Path, Path]:
        job_dir = self._output_dir / job_id
        return (
            job_dir / "tts_source.wav",
            job_dir / "converted.wav",
            job_dir / "final.wav",
        )

    async def run(
        self,
        request: TtsRequest,
        conversion: VoiceConversionProfile,
        processing: AudioProcessingProfile | None = None,
    ) -> tuple[AudioArtifact, SpeechJobMetrics]:
        """Execute one speech job and return the final artifact plus metrics."""
        processing = processing or self._processing
        job_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        queue_start = time.perf_counter()

        # Opportunistic housekeeping (review Round-2 P2): age out failed-job
        # dirs past the retention window and empty dirs left behind after
        # playback cleanup. Best-effort: never blocks or fails the job.
        if not self._keep_artifacts:
            try:
                await asyncio.to_thread(
                    prune_expired_jobs, self._output_dir, self._failed_job_retention_hours
                )
            except Exception:
                logger.debug("Job-dir pruning failed", exc_info=True)

        job_dir = self._output_dir / job_id
        try:
            async with self._coordinator.slot():
                queue_wait = time.perf_counter() - queue_start
                tts_path, vc_path, final_path = self._paths(job_id)
                tts_start = time.perf_counter()
                source = await self._synthesizer.synthesize(request, tts_path)
                tts_s = time.perf_counter() - tts_start

                vc_start = time.perf_counter()
                converted = await self._converter.convert(source, conversion, vc_path)
                vc_s = time.perf_counter() - vc_start
                if not self._keep_artifacts:
                    tts_path.unlink(missing_ok=True)

                pp_start = time.perf_counter()
                final = await postprocess(converted, final_path, processing)
                pp_s = time.perf_counter() - pp_start
                if not self._keep_artifacts:
                    vc_path.unlink(missing_ok=True)
                    mark_temp_artifact(final.path)
        except Exception:
            # Failed jobs keep their artifacts for diagnosis; record when the
            # failure happened so retention pruning can age the dir out later.
            mark_failed_job(job_dir)
            logger.exception("Speech job %s failed; artifacts kept in %s", job_id, job_dir)
            raise

        # Review Round-2 P2: drop the per-job directory when nothing is left in
        # it (final.wav normally survives until playback cleanup; the periodic
        # prune then removes the emptied dir).
        if not self._keep_artifacts:
            remove_empty_job_dir(job_dir)

        total_s = time.perf_counter() - start
        out_duration = final.duration_s or _probe_duration(final.path)
        metrics = SpeechJobMetrics(
            job_id=job_id,
            queue_wait_s=queue_wait,
            tts_s=tts_s,
            voice_conversion_s=vc_s,
            postprocess_s=pp_s,
            total_s=total_s,
            output_duration_s=out_duration,
            tts_rtf=tts_s / out_duration if out_duration > 0 else 0.0,
            voice_conversion_rtf=vc_s / out_duration if out_duration > 0 else 0.0,
            total_rtf=total_s / out_duration if out_duration > 0 else 0.0,
        )
        logger.info(
            "Speech job %s done tts=%.2fs vc=%.2fs pp=%.2fs total=%.2fs rtf=%.2f",
            job_id,
            tts_s,
            vc_s,
            pp_s,
            total_s,
            metrics.total_rtf,
        )
        return final, metrics


def _probe_duration(path: Path) -> float:
    frames, rate = sf.info(path).frames, sf.info(path).samplerate
    return frames / rate if rate else 0.0
