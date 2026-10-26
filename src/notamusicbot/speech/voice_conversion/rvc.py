"""RVC voice converter adapter.

Bridges the generic :class:`VoiceConverter` interface to the isolated legacy
RVC runtime (``legacy/rvc_worker.py``) over line-based JSON IPC on
stdin/stdout. This module never imports ``rvc_cli`` or its dependency graph:
the worker subprocess is the only bridge, and no shell command is ever built
from untrusted text — paths and parameters travel as JSON fields.

One worker process holds one persistent RVC model *directory* (the weights)
and one YAML config variant selected from it. Requesting a different model
directory restarts the worker; requesting a different YAML variant of the
already-loaded directory (e.g. ``obama`` vs ``obama_trans`` in ``Obama/``,
which share ``obama.pth``) only hot-swaps the in-worker configuration —
no process restart, no weight reload.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import json
import logging
import os
import select
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.base import (
    STREAM_INPUT_RATE_HZ,
    STREAM_OUTPUT_RATE_HZ,
    VoiceStreamParams,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_WORKER_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_WORKER_SCRIPT = REPO_ROOT / "legacy" / "rvc_worker.py"


class WorkerError(RuntimeError):
    """Base class for RVC worker failures."""


class WorkerStartError(WorkerError):
    """The worker could not be started or did not announce readiness in time."""


class WorkerTimeoutError(WorkerError, TimeoutError):
    """A request to the worker did not receive a reply within its timeout."""


class WorkerRequestError(WorkerError):
    """The worker replied with ``status: "error"`` for a request."""

    def __init__(self, action: str, message: str, traceback: str | None = None) -> None:
        self.action = action
        self.traceback = traceback
        super().__init__(f"worker {action} failed: {message}")


class WorkerDiedError(WorkerError):
    """The worker process exited while a request was in flight."""


class RvcStreamError(WorkerError):
    """A streaming conversion failed (worker error reply or protocol violation)."""


# Wire limit of one ``stream_push`` (2 s of 24 kHz mono).
_MAX_PUSH_SAMPLES = 2 * STREAM_INPUT_RATE_HZ
# How long a second stream waits for the active one before failing.
_STREAM_LOCK_TIMEOUT_S = 5.0
# Budget for the shielded abort (``stream_close`` flush=false) on cancel.
_STREAM_ABORT_TIMEOUT_S = 2.0
# Bounded prefetch between the TTS chunk producer and the push loop.
_STREAM_QUEUE_MAXSIZE = 8
_STREAM_END = object()


def encode_request(request_id: int, action: str, **fields: Any) -> str:
    """Encode one protocol request as a single JSON line (no trailing newline)."""
    payload: dict[str, Any] = {"id": request_id, "action": action, **fields}
    return json.dumps(payload, ensure_ascii=False)


def parse_reply(line: str) -> dict[str, Any]:
    """Parse one protocol reply line; malformed replies raise ``WorkerError``."""
    try:
        reply = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"malformed worker reply: {line!r}"
        raise WorkerError(msg) from exc
    if not isinstance(reply, dict):
        msg = f"worker reply is not a JSON object: {line!r}"
        raise WorkerError(msg)
    return reply


class _StderrDrainer:
    """Logs worker stderr lines from a daemon thread (legacy banners/logs)."""

    def __init__(self, stream: Any, source: str) -> None:
        self._stream = stream
        self._source = source

    def start(self) -> None:
        thread = threading.Thread(target=self._drain, name="rvc-worker-stderr", daemon=True)
        thread.start()

    def _drain(self) -> None:
        try:
            for line in self._stream:
                text = line.rstrip()
                if text:
                    logger.debug("[%s] %s", self._source, text)
        except (ValueError, OSError):  # stream closed during shutdown
            pass


def _quality_fields(profile: VoiceConversionProfile) -> dict[str, Any]:
    """Wire fields shared by ``convert`` and ``stream_open``; only those set."""
    fields: dict[str, Any] = {}
    if profile.use_index:
        fields["use_index"] = True
        if profile.index_rate is not None:
            fields["index_rate"] = float(profile.index_rate)
        if profile.index_nprobe is not None:
            fields["index_nprobe"] = int(profile.index_nprobe)
    if profile.protect is not None:
        fields["protect"] = float(profile.protect)
    if profile.f0_post:
        fields["f0_post"] = dict(profile.f0_post)
    return fields


def _stream_quality_fields(profile: VoiceConversionProfile) -> dict[str, Any]:
    """Extra ``stream_open`` fields (no rms_mix, no auto pitch in streams)."""
    fields = _quality_fields(profile)
    if profile.stream_f0_algo != "pm":
        fields["f0_algo"] = profile.stream_f0_algo
    return fields


class RvcVoiceConverter:
    """Voice converter backed by one persistent legacy RVC worker subprocess.

    The adapter is lazily started: the first :meth:`convert` spawns the worker,
    performs the health check, and loads the profile's model. Call
    :meth:`start` / :meth:`shutdown` explicitly to control the worker
    lifecycle (e.g. from application startup/teardown).
    """

    def __init__(
        self,
        *,
        worker_python: Path | str | None = None,
        worker_script: Path | str | None = None,
        start_timeout_s: float | None = None,
        load_timeout_s: float | None = None,
        convert_timeout_s: float | None = None,
        stream_params: VoiceStreamParams | None = None,
        stream_timeout_s: float | None = None,
        stream_open_timeout_s: float | None = None,
    ) -> None:
        self._worker_python = Path(
            worker_python or os.environ.get("RVC_WORKER_PYTHON") or DEFAULT_WORKER_PYTHON
        )
        self._worker_script = Path(
            worker_script or os.environ.get("RVC_WORKER_SCRIPT") or DEFAULT_WORKER_SCRIPT
        )
        self._start_timeout = float(
            start_timeout_s or os.environ.get("RVC_WORKER_START_TIMEOUT", "30")
        )
        self._load_timeout = float(
            load_timeout_s or os.environ.get("RVC_WORKER_LOAD_TIMEOUT", "300")
        )
        self._convert_timeout = float(
            convert_timeout_s or os.environ.get("RVC_WORKER_CONVERT_TIMEOUT", "120")
        )
        self._stream_params = stream_params
        self._stream_timeout = float(
            stream_timeout_s or os.environ.get("RVC_WORKER_STREAM_TIMEOUT", "10")
        )
        self._stream_open_timeout = float(
            stream_open_timeout_s or os.environ.get("RVC_WORKER_STREAM_OPEN_TIMEOUT", "30")
        )
        # One stream per worker: concurrent streams queue here (see
        # ``convert_stream``). Distinct from ``_transaction_lock``, which is
        # held per request, so ``convert``/``health`` stay legal mid-stream.
        self._stream_lock = asyncio.Lock()
        self._process: subprocess.Popen[str] | None = None
        # Live-shutdown hardening (user report: a second Ctrl+C bypasses the
        # async close path and would orphan the worker subprocess, which keeps
        # VRAM allocated). atexit runs on ANY interpreter exit — clean,
        # KeyboardInterrupt-forced, or os._exit-excepted — and reaps the worker.
        atexit.register(self._kill_worker_at_exit)
        self._next_request_id = 1
        # Loaded-model cache identity: (resolved model directory, yaml
        # variant). Two variants of one directory are DISTINCT identities
        # even though they share the weights.
        self._loaded_key: tuple[str, str | None] | None = None
        self._io_lock = threading.Lock()
        # Review Round-5 P0 (§2/A): ADAPTER-LEVEL TRANSACTION SERIALIZATION.
        # ``_request`` allocates the request id, writes the line, and waits
        # for the matching reply while holding this lock, so no second
        # request can write to the worker until the preceding request has
        # received its reply, timed out (worker terminated), or been
        # otherwise conclusively retired. Primitive choice: ``asyncio.Lock``.
        # The adapter is asyncio-facing (convert/start/shutdown are coroutines
        # and the tests/tools drive it from an event loop); the lock is held
        # ACROSS the ``asyncio.to_thread`` awaits, which is legal for an
        # asyncio.Lock — it is owned by the coroutine on the event loop, and
        # the threads it spawns never touch the lock themselves. A
        # ``threading.Lock`` would be wrong here: holding it across ``await``
        # would block the loop, and thread-based callers do not exist on the
        # public surface.
        self._transaction_lock = asyncio.Lock()
        # Review Round-6 §2/A: WORKER LIFECYCLE SINGLE-FLIGHT. Spawn → ready
        # negotiation → health check, and shutdown, all run under this lock so
        # they can never interleave: two concurrent ``start()`` calls spawn
        # exactly one worker (the loser re-checks ``is_running`` and returns),
        # and ``shutdown()`` cannot touch the protocol while startup is still
        # negotiating it. LOCK ORDER (documented invariant): ``_lifecycle_lock``
        # is ALWAYS acquired OUTSIDE ``_transaction_lock`` —
        # ``shutdown`` = lifecycle → transaction; ``_request`` takes only the
        # transaction lock; ``start`` holds only the lifecycle lock and its
        # health check enters ``_request`` (which then takes the transaction
        # lock inside). The reverse order (transaction → lifecycle) is never
        # taken anywhere, so no deadlock cycle exists.
        self._lifecycle_lock = asyncio.Lock()
        self._stderr_drainer: _StderrDrainer | None = None
        self._shutting_down = False
        # Review Round-4 P0 (reworked Round-5 §3/B): request ids whose
        # awaiting coroutine was cancelled BEFORE their reader thread could
        # consume the reply. Because the reply may then be written by the
        # worker with nobody owning it, a LATER reader must discard it
        # instead of treating it as a mismatch. The set is only consulted for
        # replies whose id differs from the reader's own request id — a
        # reader NEVER discards its own terminal reply (Round-5 §3/B fix);
        # when the cancelled reader thread does consume its own reply, the id
        # is removed from the set, so the reply is simply dropped by the
        # cancelled future, never re-consumed. The subprocess conversion
        # itself keeps running under the pipeline's GPU lease (the whole
        # TTS -> VC -> postprocess chain runs inside ``coordinator
        # .run_detached``), so the lease is never released while the worker is
        # still converting.
        self._orphaned_ids: set[int] = set()

    # ------------------------------------------------------------------ state

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def loaded_model(self) -> str | None:
        """Resolved model directory currently loaded, if any."""
        return self._loaded_key[0] if self._loaded_key else None

    @property
    def loaded_variant(self) -> str | None:
        """YAML config variant currently loaded (None = directory default)."""
        return self._loaded_key[1] if self._loaded_key else None

    # ------------------------------------------------------------------ async

    async def start(self) -> None:
        """Spawn the worker, wait for its ready line, and run a health check.

        Review Round-6 §2/A: the whole lifecycle is single-flight under
        ``_lifecycle_lock``. Concurrent callers serialize; the second caller
        re-checks ``is_running`` after acquiring the lock and returns without
        spawning again — two concurrent ``start()`` calls therefore spawn
        exactly one worker and both succeed. ``shutdown()`` acquires the SAME
        lock, so it cannot interleave with startup (it runs strictly after a
        concurrent startup completes or fails). On any startup failure the
        partially created worker is terminated/reaped and the protocol state
        is cleaned before the error propagates, so a failed start never
        leaves a live, untracked worker behind.
        """
        async with self._lifecycle_lock:
            if self.is_running:
                return
            self._shutting_down = False
            self._loaded_key = None
            # A freshly spawned worker means a fresh protocol stream: replies
            # from a previous process can never appear again, so stale orphan
            # marks from before a restart are dropped (Round-5 §3/B: stale-reply
            # handling must stay safe across a worker restart).
            self._orphaned_ids.clear()
            try:
                await asyncio.to_thread(self._spawn_and_wait_ready)
                # Health check via the public request path: this coroutine
                # holds only the lifecycle lock, so ``_request`` acquiring the
                # transaction lock cannot self-deadlock (lock order:
                # lifecycle → transaction, never the reverse).
                await self._request("health", timeout=self._start_timeout)
            except BaseException:
                await self._cleanup_failed_start()
                raise
            logger.info(
                "RVC worker started (python=%s, script=%s)",
                self._worker_python,
                self._worker_script,
            )

    async def _cleanup_failed_start(self) -> None:
        """Reap a partially started worker and reset protocol state.

        Called when ``start()`` fails after spawn (health check failed, timed
        out, or the caller was cancelled): no live worker may survive a failed
        startup, and ``_process`` / ``_loaded_key`` must be left clean.
        """
        process = self._process
        self._process = None
        self._loaded_key = None
        self._orphaned_ids.clear()
        if process is not None:
            logger.warning("RVC startup failed; terminating partial worker")
            await asyncio.to_thread(self._terminate, process)

    async def shutdown(self) -> None:
        """Ask the worker to shut down cleanly, then reap the process.

        Review Round-6 §2/A: shutdown acquires the SAME ``_lifecycle_lock`` as
        ``start()``, so it can never interleave with startup — a concurrent
        shutdown strictly follows a startup (which by then has completed ready
        negotiation and the health check), and a concurrent startup strictly
        follows a shutdown. LOCK ORDER: lifecycle → transaction.

        Review Round-5 §2/A shutdown semantics: shutdown WAITS for the active
        transaction. The transaction lock below is held until the in-flight
        ``convert`` has received its reply, so the request is never corrupted
        by shutdown; it either succeeds or fails on its own timeout (which
        itself terminates the worker and releases the lock). Once both locks
        are held, the shutdown request is sent via the lock-internal
        ``_request_locked`` path to avoid self-deadlock.
        """
        async with self._lifecycle_lock:
            async with self._transaction_lock:
                process = self._process
                if process is None:
                    return
                self._shutting_down = True
                try:
                    if process.poll() is None:
                        await self._request_locked("shutdown", timeout=10.0)
                except WorkerError as exc:
                    logger.warning("RVC worker shutdown request failed: %s", exc)
                await asyncio.to_thread(self._terminate, process)
                self._process = None
                self._loaded_key = None

    async def convert(
        self,
        source: AudioArtifact,
        profile: VoiceConversionProfile,
        output_path: Path,
    ) -> AudioArtifact:
        """Convert ``source`` toward ``profile.model`` via the worker."""
        if not self.is_running:
            await self.start()
        try:
            await self._ensure_model(profile.model, variant=profile.variant)
        except WorkerDiedError:
            # The worker died while (re)loading: one clean restart, then retry.
            await self._restart()
            await self._ensure_model(profile.model, variant=profile.variant)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_fields: dict[str, Any] = {
            "input_wav_path": str(Path(source.path).resolve()),
            "output_wav_path": str(Path(output_path).resolve()),
            "pitch_shift": int(profile.pitch_shift),
        }
        # Optional f0 extraction method override (config-selectable): only
        # sent when the profile carries one, so the model yaml value applies
        # otherwise (shipped default: harvest).
        if profile.f0_method:
            request_fields["f0_method"] = str(profile.f0_method)
        # RVC output-quality fields (docs/RVC_QUALITY_PLAN.md 3.2): sent only
        # when set, so an unconfigured request is unchanged.
        request_fields.update(_quality_fields(profile))
        if profile.rms_mix_rate is not None:
            request_fields["rms_mix_rate"] = float(profile.rms_mix_rate)
        if profile.auto_pitch_target_hz is not None:
            request_fields["auto_pitch"] = {
                "target_median_hz": float(profile.auto_pitch_target_hz),
                "max_abs": int(profile.auto_pitch_max),
                "strength": float(profile.auto_pitch_strength),
            }
        try:
            reply = await self._request("convert", timeout=self._convert_timeout, **request_fields)
        except WorkerDiedError:
            # Worker died mid-conversion: restart once and retry the request.
            await self._restart()
            await self._ensure_model(profile.model, variant=profile.variant)
            reply = await self._request("convert", timeout=self._convert_timeout, **request_fields)

        output_path = Path(reply.get("output_wav_path") or output_path)
        if not output_path.is_file():
            msg = f"worker reported success but output is missing: {output_path}"
            raise WorkerError(msg)
        return await asyncio.to_thread(self._probe_output, output_path)

    # -------------------------------------------------------------- streaming

    def supports_streaming(self, profile: VoiceConversionProfile) -> bool:
        """True when streaming is enabled and ``profile`` is an RVC profile."""
        params = self._stream_params
        return bool(params and params.enabled and profile.backend == "rvc" and profile.model)

    async def warm_stream(self, profile: VoiceConversionProfile) -> bool:
        """Run a short synthetic stream so CUDA kernels are hot; never raises."""
        if not self.supports_streaming(profile):
            return False
        piece = STREAM_INPUT_RATE_HZ // 4
        t = np.arange(2 * piece, dtype=np.float32) / STREAM_INPUT_RATE_HZ
        tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

        async def chunks() -> AsyncIterator[np.ndarray]:
            for start in range(0, tone.size, piece):
                yield tone[start : start + piece]

        try:
            async with contextlib.aclosing(self.convert_stream(chunks(), profile)) as outputs:
                async for _ in outputs:
                    pass
        except Exception:
            logger.warning("RVC stream warm-up failed", exc_info=True)
            return False
        return True

    async def convert_stream(
        self,
        chunks: AsyncIterator[np.ndarray],
        profile: VoiceConversionProfile,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Convert a mono f32 24 kHz chunk stream to mono f32 48 kHz arrays.

        Total output samples equal 2 x total input samples. No mid-stream
        retry: worker death/timeouts propagate; worker error replies become
        :class:`RvcStreamError`. Leaving the generator early (``aclose``,
        cancellation, error) aborts the worker-side stream.
        """
        params = self._stream_params
        if params is None or not self.supports_streaming(profile):
            msg = "streaming voice conversion is not enabled for this profile"
            raise RvcStreamError(msg)
        try:
            await asyncio.wait_for(self._stream_lock.acquire(), _STREAM_LOCK_TIMEOUT_S)
        except TimeoutError as exc:
            msg = "another stream is active"
            raise RvcStreamError(msg) from exc
        producer: asyncio.Task[None] | None = None
        stream_id: str | None = None
        try:
            if not self.is_running:
                await self.start()
            try:
                await self._ensure_model(profile.model, variant=profile.variant)
            except WorkerDiedError:
                await self._restart()
                await self._ensure_model(profile.model, variant=profile.variant)
            if profile.f0_method not in (None, "pm"):
                logger.info("RVC streams force f0 method pm (profile asked %s)", profile.f0_method)
            sid = uuid.uuid4().hex
            # Set before the request: a cancel while the open is in flight must
            # still abort the worker-side stream (close of an unknown id is a no-op).
            stream_id = sid
            await self._request(
                "stream_open",
                timeout=self._stream_open_timeout,
                stream_id=sid,
                pitch_shift=int(profile.pitch_shift),
                f0_method="pm",
                **_stream_quality_fields(profile),
                first_block_ms=params.first_block_ms,
                block_ms=params.block_ms,
                context_ms=params.context_ms,
                lookahead_ms=params.lookahead_ms,
                crossfade_ms=params.crossfade_ms,
                search_ms=params.search_ms,
            )
            queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
            producer = asyncio.create_task(self._pump_chunks(chunks, queue))
            input_samples = 0
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, BaseException):
                    raise item
                pcm = np.ascontiguousarray(item, dtype="<f4").reshape(-1)
                for start in range(0, pcm.size, _MAX_PUSH_SAMPLES):
                    piece = pcm[start : start + _MAX_PUSH_SAMPLES]
                    input_samples += piece.size
                    out = await self._stream_call(
                        "stream_push",
                        stream_id,
                        pcm_b64=base64.b64encode(piece.tobytes()).decode("ascii"),
                        samples=int(piece.size),
                    )
                    if out.size:
                        yield out
            tail, close_reply = await self._stream_close_flush(stream_id)
            stream_id = None
            if tail.size:
                yield tail
            in_total = int(close_reply.get("input_total", input_samples))
            out_total = int(close_reply.get("output_total", -1))
            if out_total != 2 * in_total or in_total != input_samples:
                logger.warning(
                    "RVC stream totals off: in=%d (sent %d) out=%d (expected %d)",
                    in_total,
                    input_samples,
                    out_total,
                    2 * input_samples,
                )
            logger.info(
                "RVC stream done: audio=%.2fs compute=%.0fms steps=%s",
                out_total / STREAM_OUTPUT_RATE_HZ,
                float(close_reply.get("compute_ms_total", 0.0)),
                close_reply.get("steps"),
            )
        finally:
            try:
                if producer is not None:
                    producer.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await producer
                if stream_id is not None:
                    await self._abort_stream(stream_id)
            finally:
                self._stream_lock.release()

    @staticmethod
    async def _pump_chunks(chunks: AsyncIterator[np.ndarray], queue: asyncio.Queue[Any]) -> None:
        """Pull TTS chunks into ``queue`` so TTS keeps generating during pushes."""
        try:
            async for chunk in chunks:
                await queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(exc)
            return
        await queue.put(_STREAM_END)

    @staticmethod
    def _decode_pcm(reply: dict[str, Any]) -> np.ndarray:
        raw = base64.b64decode(reply.get("pcm_b64") or "")
        out = np.frombuffer(raw, dtype="<f4").copy()
        if out.size != int(reply.get("samples", out.size)):
            msg = f"stream reply samples {reply.get('samples')} != decoded {out.size}"
            raise RvcStreamError(msg)
        return out

    async def _stream_call(self, action: str, stream_id: str, **fields: Any) -> np.ndarray:
        """One push; returns the decoded audio. Died/timeout errors propagate."""
        try:
            reply = await self._request(
                action, timeout=self._stream_timeout, stream_id=stream_id, **fields
            )
        except WorkerRequestError as exc:
            raise RvcStreamError(str(exc)) from exc
        return self._decode_pcm(reply)

    async def _stream_close_flush(self, stream_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            reply = await self._request(
                "stream_close", timeout=self._stream_timeout, stream_id=stream_id, flush=True
            )
        except WorkerRequestError as exc:
            raise RvcStreamError(str(exc)) from exc
        return self._decode_pcm(reply), reply

    async def _abort_stream(self, stream_id: str) -> None:
        """Shielded ``stream_close`` flush=false inside a 2 s budget; best effort."""
        if not self.is_running:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(
                    self._request(
                        "stream_close",
                        timeout=_STREAM_ABORT_TIMEOUT_S,
                        stream_id=stream_id,
                        flush=False,
                    )
                ),
                _STREAM_ABORT_TIMEOUT_S,
            )
        except Exception:
            logger.warning("RVC stream abort failed", exc_info=True)

    # ------------------------------------------------------------ worker mgmt

    def _spawn_and_wait_ready(self) -> None:
        argv = [str(self._worker_python), str(self._worker_script)]
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            msg = f"failed to spawn RVC worker {argv!r}: {exc}"
            raise WorkerStartError(msg) from exc
        self._process = process
        self._stderr_drainer = _StderrDrainer(process.stderr, "rvc-worker")
        self._stderr_drainer.start()

        if not self._wait_for_ready(process, self._start_timeout):
            detail = self._exit_detail(process)
            self._terminate(process)
            self._process = None
            msg = f"RVC worker did not signal ready within {self._start_timeout}s. {detail}"
            raise WorkerStartError(msg)

    def _wait_for_ready(self, process: subprocess.Popen[str], timeout: float) -> bool:
        """Wait for the worker's ``action: ready`` line with a HARD timeout.

        Uses :func:`select.select` on the raw stdout fd plus a deadline so a
        worker that stays alive but emits nothing fails at the deadline
        instead of blocking forever inside ``readline()``. Reading via
        ``os.read`` bypasses the buffered ``stdout`` wrapper; the worker
        emits only the ready line during startup, so no protocol bytes are
        lost (everything it prints otherwise goes to stderr).
        """
        if process.stdout is None:
            return False
        try:
            fd = process.stdout.fileno()
        except (OSError, ValueError):
            return False
        buf = b""
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    reply = parse_reply(line)
                except WorkerError:
                    continue
                if reply.get("status") == "ok" and reply.get("action") == "ready":
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                continue  # loop re-checks the deadline
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return False
            if not chunk:  # EOF: worker exited during startup
                return False
            buf += chunk

    async def _restart(self) -> None:
        logger.warning("RVC worker died; restarting it once")
        if self._process is not None:
            await asyncio.to_thread(self._terminate, self._process)
        self._process = None
        self._loaded_key = None
        await self.start()

    def _kill_worker_at_exit(self) -> None:
        """atexit last-resort: reap the worker subprocess on any exit path.

        The async ``shutdown()`` is the clean path; this catches the unclean
        ones (second Ctrl+C, ``os._exit``, unhandled crash) so the worker
        cannot outlive the application holding GPU memory.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.kill()
        except (OSError, ValueError):
            return
        try:
            process.wait(timeout=5)
            logger.info("RVC worker reaped at interpreter exit")
        except subprocess.TimeoutExpired:
            logger.warning("RVC worker did not die after exit kill; it may be orphaned")

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except (OSError, ValueError):
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    @staticmethod
    def _exit_detail(process: subprocess.Popen[str]) -> str:
        if process.poll() is None:
            return "(worker still running)"
        return f"worker exited with code {process.returncode}."

    # ------------------------------------------------------------------ model

    @staticmethod
    def _resolve_model_name(model: str) -> str:
        """Resolve a profile model name to the on-disk model directory.

        Legacy profiles store uppercased names (e.g. ``GOKU``) while the
        model directories use mixed case (``Goku``). Resolve
        case-insensitively against ``rvc_cli/models``; fall back to the
        given name (the worker will produce a precise error if it is
        genuinely unknown).
        """
        models_dir = Path(__file__).resolve().parents[4] / "rvc_cli" / "models"
        if models_dir.is_dir():
            for entry in sorted(models_dir.iterdir()):
                if entry.is_dir() and entry.name.lower() == model.lower():
                    return entry.name
        return model

    async def _ensure_model(self, model: str | None, *, variant: str | None = None) -> None:
        if model is None:
            msg = "RVC voice conversion requires a model name in the profile"
            raise WorkerError(msg)
        # Resolve the on-disk model name FIRST: profiles may carry legacy
        # aliases (e.g. "GOKU" for the "Goku" directory), and comparing the
        # raw profile name against the previously loaded canonical name
        # would needlessly restart the worker on every utterance.
        model = self._resolve_model_name(model)
        variant = variant.strip() if isinstance(variant, str) and variant.strip() else None
        key = (model, variant)
        if self._loaded_key == key:
            return
        if self._loaded_key is not None and self._loaded_key[0] != model:
            # Different model directory: the weights must be swapped, which
            # requires a worker restart (one weight set per process).
            await self._restart()
        # Same directory with a different (or first) variant: NO restart.
        # The worker keeps the loaded weights and selects the requested
        # YAML config in-process (or performs the initial load).
        fields: dict[str, Any] = {"name": model}
        if variant is not None:
            fields["variant"] = variant
        reply = await self._request("load-model", timeout=self._load_timeout, **fields)
        self._loaded_key = (
            str(reply.get("model") or model),
            str(reply["variant"]) if reply.get("variant") else variant,
        )
        logger.info(
            "RVC model %s (variant %s) loaded on %s (target_sample_rate=%s, index_rate=%s)",
            self._loaded_key[0],
            self._loaded_key[1],
            reply.get("device"),
            reply.get("target_sample_rate"),
            reply.get("index_rate"),
        )

    # --------------------------------------------------------------- protocol

    async def _request(self, action: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
        """Send one request and await its matching reply line.

        Review Round-5 §2/A: the WHOLE transaction — id allocation, write,
        and matching-reply read — is serialized under the adapter-level
        ``_transaction_lock``. A second request cannot write to the worker
        until the preceding one has received its reply, timed out (which
        terminates the worker), or otherwise retired. The lock is held across
        the ``to_thread`` awaits (legal for an ``asyncio.Lock`` owned by this
        coroutine; the threads never touch the lock).
        """
        async with self._transaction_lock:
            return await self._request_locked(action, timeout=timeout, **fields)

    async def _request_locked(
        self, action: str, *, timeout: float, **fields: Any
    ) -> dict[str, Any]:
        """Transaction body; caller must hold ``_transaction_lock``."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            msg = "RVC worker is not running"
            raise WorkerError(msg)
        request_id = self._next_request_id
        self._next_request_id += 1
        line = encode_request(request_id, action, **fields)
        try:
            await asyncio.to_thread(self._write_line, process, line)
            reply = await asyncio.wait_for(
                asyncio.to_thread(self._read_reply, process, request_id), timeout
            )
        except asyncio.CancelledError:
            # Caller cancelled while the reader may not yet have consumed the
            # reply. Record the id so a LATER reader discards the stale reply
            # (correct use of orphan tracking). If the reader thread for THIS
            # request does consume its own reply, ``_read_reply`` removes the
            # id and the result is simply dropped by the cancelled future —
            # the reader terminates after its own reply either way (Round-5
            # §3/B). The underlying subprocess conversion keeps running; with
            # the pipeline's run_detached design the GPU lease stays held
            # until that reply thread finishes regardless of this
            # cancellation.
            self._orphaned_ids.add(request_id)
            logger.warning(
                "RVC %s request %d cancelled while awaiting reply; late reply will be discarded",
                action,
                request_id,
            )
            raise
        except TimeoutError as exc:
            # The reply thread is stuck on a blocking readline; the worker is
            # unresponsive, so kill it (it will be restarted on next use).
            logger.error("RVC worker %s timed out after %.1fs; killing worker", action, timeout)
            if self._process is not None:
                await asyncio.to_thread(self._terminate, self._process)
                self._process = None
            msg = f"RVC worker {action} timed out after {timeout}s"
            raise WorkerTimeoutError(msg) from exc
        if reply.get("status") != "ok":
            raise WorkerRequestError(
                action, str(reply.get("error", "unknown error")), reply.get("traceback")
            )
        return reply

    @staticmethod
    def _write_line(process: subprocess.Popen[str], line: str) -> None:
        stdin = process.stdin
        if stdin is None:
            msg = "RVC worker stdin is closed"
            raise WorkerDiedError(msg)
        try:
            stdin.write(line + "\n")
            stdin.flush()
        except (OSError, ValueError) as exc:
            msg = f"failed writing to RVC worker stdin: {exc}"
            raise WorkerDiedError(msg) from exc

    def _read_reply(self, process: subprocess.Popen[str], request_id: int) -> dict[str, Any]:
        """Blocking read of the next reply line, matched by id.

        Round-4 P0: a reply whose id was orphaned (its awaiting coroutine was
        cancelled before its reader consumed the reply) is discarded here by
        a LATER reader so a stale reply can never be consumed as the reply to
        a different request.

        Round-5 §3/B: this reader's OWN terminal reply is never discarded.
        Even if the awaiting coroutine was cancelled, the reader returns as
        soon as it sees ``id == request_id`` (dropping the result by way of
        the cancelled future) instead of looping on and consuming the next
        request's reply. Ownership rule:
        ``reply.id == request_id`` → this reader owns it (consume, clear any
        orphan mark, return);
        ``reply.id in _orphaned_ids`` (and not ours) → stale, discard;
        otherwise → mismatch (should not happen under full serialization),
        log and keep waiting.
        """
        stdout = process.stdout
        if stdout is None:
            msg = "RVC worker stdout is closed"
            raise WorkerDiedError(msg)
        with self._io_lock:
            if process.poll() is not None:
                msg = f"RVC worker exited with code {process.returncode}"
                raise WorkerDiedError(msg)
            while True:
                line = stdout.readline()
                if not line:
                    msg = f"RVC worker stdout closed (exit code {process.returncode})"
                    raise WorkerDiedError(msg)
                line = line.strip()
                if not line:
                    continue
                reply = parse_reply(line)
                if reply.get("id") is None and reply.get("status") == "error":
                    # Malformed-notice from the worker; log and keep waiting.
                    logger.warning("RVC worker protocol notice: %s", reply.get("error"))
                    continue
                reply_id = reply.get("id")
                if reply_id == request_id:
                    # OUR reply: consume it unconditionally and terminate.
                    # Clear the orphan mark (if the caller was cancelled, the
                    # reply has now been consumed and must not be re-dropped
                    # by a later reader's stale check).
                    self._orphaned_ids.discard(request_id)
                    return reply
                if reply_id in self._orphaned_ids:
                    # Stale reply for an earlier cancelled request (Round-4
                    # P0): drop it and keep waiting for our own reply.
                    self._orphaned_ids.discard(reply_id)
                    logger.info("RVC discarded orphaned reply for cancelled request %s", reply_id)
                    continue
                logger.warning("RVC worker reply id mismatch: %s", reply)

    # ------------------------------------------------------------------ audio

    @staticmethod
    def _probe_output(path: Path) -> AudioArtifact:
        info = sf.info(path)
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        return AudioArtifact(path=path, sample_rate_hz=info.samplerate, duration_s=duration)
