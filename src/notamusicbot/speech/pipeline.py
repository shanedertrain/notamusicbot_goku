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
    mark_temporary_job,
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
        """Execute one speech job and return the final artifact plus metrics.

        Review Round-4 P0 (cancellation safety): the ENTIRE TTS -> VC ->
        postprocess chain runs through ``coordinator.run_detached`` — the chain
        coroutine executes inside a detached holder task that owns the GPU
        lease. The lease is held from before the chain starts until it
        actually finishes, even if the awaiting caller (e.g. the player's
        post-song task) is cancelled mid-flight: cancellation detaches the
        caller, and the lease stays with the holder until the RVC worker reply
        has been consumed and the artifacts are written. No later job can
        enter the coordinator while the cancelled job's GPU work is still
        running.
        """
        processing = processing or self._processing
        job_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()

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
        # Persist ownership on disk (review Round-3 P1): the .temporary marker
        # carries a timestamp+status so a crash after successful generation —
        # when the in-memory registry is lost — still leaves an identifiable
        # stale temporary job for retention/startup cleanup to prune.
        if not self._keep_artifacts:
            mark_temporary_job(job_dir, status="pending")
        try:
            # Round-5 §7/E: the submission timestamp is taken BEFORE
            # submitting to the coordinator; the chain records the lease
            # entry when the detached holder actually starts running (i.e.
            # after the GPU lease was acquired). queue_wait_s is then
            # lease_entry - submission, i.e. real queue wait, not synthesis
            # duration.
            submitted_at = time.perf_counter()
            final, timings = await self._coordinator.run_detached(
                self._run_chain, request, conversion, processing, job_id, submitted_at
            )
        except asyncio.CancelledError:
            # Cancellation is not an error (review Round-3 P1): never swallow
            # it. The caller returns promptly; the detached holder keeps the
            # GPU lease until the chain really finishes (Round-4 P0). The job
            # keeps its .temporary marker (status "pending") and whatever
            # partial artifacts exist, so retention pruning removes the
            # directory deterministically once past the retention window.
            logger.warning("Speech job %s cancelled; artifacts left under temporary marker", job_id)
            raise
        except Exception:
            # Failed jobs keep their artifacts for diagnosis; record when the
            # failure happened so retention pruning can age the dir out later.
            mark_failed_job(job_dir)
            logger.exception("Speech job %s failed; artifacts kept in %s", job_id, job_dir)
            raise

        # Review Round-2 P2: drop the per-job directory when nothing is left in
        # it (final.wav normally survives until playback cleanup; the periodic
        # prune then removes the emptied dir). Refresh the temporary marker to
        # status "generated" so a crash-before-playback is identifiable as a
        # stale successful temporary job (Round-3 P1).
        if not self._keep_artifacts:
            mark_temporary_job(job_dir, status="generated")
            remove_empty_job_dir(job_dir)

        total_s = time.perf_counter() - start
        out_duration = final.duration_s or _probe_duration(final.path)
        metrics = SpeechJobMetrics(
            job_id=job_id,
            queue_wait_s=timings["queue_wait"],
            tts_s=timings["tts"],
            voice_conversion_s=timings["vc"],
            postprocess_s=timings["pp"],
            total_s=total_s,
            output_duration_s=out_duration,
            tts_rtf=timings["tts"] / out_duration if out_duration > 0 else 0.0,
            voice_conversion_rtf=timings["vc"] / out_duration if out_duration > 0 else 0.0,
            total_rtf=total_s / out_duration if out_duration > 0 else 0.0,
        )
        logger.info(
            "Speech job %s done tts=%.2fs vc=%.2fs pp=%.2fs total=%.2fs rtf=%.2f",
            job_id,
            timings["tts"],
            timings["vc"],
            timings["pp"],
            total_s,
            metrics.total_rtf,
        )
        return final, metrics

    async def _run_chain(
        self,
        request: TtsRequest,
        conversion: VoiceConversionProfile,
        processing: AudioProcessingProfile,
        job_id: str,
        submitted_at: float,
    ) -> tuple[AudioArtifact, dict[str, float]]:
        """TTS -> VC -> postprocess chain — the detached lease-holder body.

        The RVC adapter's own worker-reply wait thread lives inside this chain
        coroutine, so a subprocess conversion can never outlive the lease:
        caller cancellation detaches the holder but the lease is released only
        when this coroutine (reply wait included) actually finishes.
        """
        # Review Round-3 P1: the shared slot is deliberately held across
        # TTS -> VC -> postprocess. The RVC backend runs in a separate
        # process no in-process lock can reach, so keeping the slot for the
        # whole VC round-trip is what prevents the RVC subprocess from
        # overlapping main-process LLM/TTS inference on the same GPU.
        # Round-4 P0: the lease is now owned by the coordinator's detached
        # holder around this whole chain coroutine, so it survives caller
        # cancellation.
        # Round-5 §7/E: this coroutine only runs inside the detached holder
        # AFTER the coordinator's semaphore was acquired, so "now" is the
        # lease-entry timestamp. queue_wait = lease entry - submission.
        lease_entry = time.perf_counter()
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
        timings = {
            "queue_wait": lease_entry - submitted_at,
            "tts": tts_s,
            "vc": vc_s,
            "pp": pp_s,
        }
        return final, timings


def _probe_duration(path: Path) -> float:
    frames, rate = sf.info(path).frames, sf.info(path).samplerate
    return frames / rate if rate else 0.0
