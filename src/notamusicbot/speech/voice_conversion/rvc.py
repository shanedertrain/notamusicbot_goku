"""RVC voice converter adapter.

Bridges the generic :class:`VoiceConverter` interface to the isolated legacy
RVC runtime (``legacy/rvc_worker.py``) over line-based JSON IPC on
stdin/stdout. This module never imports ``rvc_cli`` or its dependency graph:
the worker subprocess is the only bridge, and no shell command is ever built
from untrusted text — paths and parameters travel as JSON fields.

One worker process holds one persistent RVC model. Requesting a different
model restarts the worker (the legacy runtime cannot hot-swap models).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile

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
    ) -> None:
        self._worker_python = Path(
            worker_python or os.environ.get("RVC_WORKER_PYTHON", DEFAULT_WORKER_PYTHON)
        )
        self._worker_script = Path(
            worker_script or os.environ.get("RVC_WORKER_SCRIPT", DEFAULT_WORKER_SCRIPT)
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
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._loaded_model: str | None = None
        self._io_lock = threading.Lock()
        self._stderr_drainer: _StderrDrainer | None = None
        self._shutting_down = False

    # ------------------------------------------------------------------ state

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    # ------------------------------------------------------------------ async

    async def start(self) -> None:
        """Spawn the worker, wait for its ready line, and run a health check."""
        if self.is_running:
            return
        self._shutting_down = False
        self._loaded_model = None
        await asyncio.to_thread(self._spawn_and_wait_ready)
        await self._request("health", timeout=self._start_timeout)
        logger.info(
            "RVC worker started (python=%s, script=%s)",
            self._worker_python,
            self._worker_script,
        )

    async def shutdown(self) -> None:
        """Ask the worker to shut down cleanly, then reap the process."""
        process = self._process
        if process is None:
            return
        self._shutting_down = True
        try:
            if process.poll() is None:
                await self._request("shutdown", timeout=10.0)
        except WorkerError as exc:
            logger.warning("RVC worker shutdown request failed: %s", exc)
        await asyncio.to_thread(self._terminate, process)
        self._process = None
        self._loaded_model = None

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
            await self._ensure_model(profile.model)
        except WorkerDiedError:
            # The worker died while (re)loading: one clean restart, then retry.
            await self._restart()
            await self._ensure_model(profile.model)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_fields: dict[str, Any] = {
            "input_wav_path": str(Path(source.path).resolve()),
            "output_wav_path": str(Path(output_path).resolve()),
            "pitch_shift": int(profile.pitch_shift),
        }
        try:
            reply = await self._request("convert", timeout=self._convert_timeout, **request_fields)
        except WorkerDiedError:
            # Worker died mid-conversion: restart once and retry the request.
            await self._restart()
            await self._ensure_model(profile.model)
            reply = await self._request("convert", timeout=self._convert_timeout, **request_fields)

        output_path = Path(reply.get("output_wav_path") or output_path)
        if not output_path.is_file():
            msg = f"worker reported success but output is missing: {output_path}"
            raise WorkerError(msg)
        return await asyncio.to_thread(self._probe_output, output_path)

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
        """Block until the worker's ``action: ready`` line; False on timeout/EOF."""
        if process.stdout is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:  # EOF: worker exited during startup
                return False
            line = line.strip()
            if not line:
                continue
            try:
                reply = parse_reply(line)
            except WorkerError:
                continue
            if reply.get("status") == "ok" and reply.get("action") == "ready":
                return True
        return False

    async def _restart(self) -> None:
        logger.warning("RVC worker died; restarting it once")
        if self._process is not None:
            await asyncio.to_thread(self._terminate, self._process)
        self._process = None
        self._loaded_model = None
        await self.start()

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

    async def _ensure_model(self, model: str | None) -> None:
        if model is None:
            msg = "RVC voice conversion requires a model name in the profile"
            raise WorkerError(msg)
        if self._loaded_model == model:
            return
        model = self._resolve_model_name(model)
        if self._loaded_model is not None:
            # One model per worker: restart to switch voices.
            await self._restart()
        reply = await self._request("load-model", timeout=self._load_timeout, name=model)
        self._loaded_model = str(reply.get("model") or model)
        logger.info(
            "RVC model %s loaded on %s (target_sample_rate=%s)",
            self._loaded_model,
            reply.get("device"),
            reply.get("target_sample_rate"),
        )

    # --------------------------------------------------------------- protocol

    async def _request(self, action: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
        """Send one request and await its matching reply line."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            msg = "RVC worker is not running"
            raise WorkerError(msg)
        request_id = self._next_request_id
        self._next_request_id += 1
        line = encode_request(request_id, action, **fields)
        await asyncio.to_thread(self._write_line, process, line)
        try:
            reply = await asyncio.wait_for(
                asyncio.to_thread(self._read_reply, process, request_id), timeout
            )
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
        """Blocking read of the next reply line, matched by id."""
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
                if reply.get("id") != request_id:
                    logger.warning("RVC worker reply id mismatch: %s", reply)
                    continue
                return reply

    # ------------------------------------------------------------------ audio

    @staticmethod
    def _probe_output(path: Path) -> AudioArtifact:
        info = sf.info(path)
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        return AudioArtifact(path=path, sample_rate_hz=info.samplerate, duration_s=duration)
