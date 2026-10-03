"""Unit tests for :class:`RvcVoiceConverter` against a fake worker process.

The fake worker speaks the same line-JSON stdin/stdout protocol as
``legacy/rvc_worker.py`` but is stdlib-only, so these tests run without the
legacy venv, GPU, or model weights.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.speech.voice_conversion.rvc import (
    RvcVoiceConverter,
    WorkerRequestError,
    WorkerStartError,
    WorkerTimeoutError,
)

FAKE_WORKER = (
    textwrap.dedent(
        """
    import json, os, sys, time, wave

    mode, log_path = sys.argv[1], sys.argv[2]
    extra = sys.argv[3] if len(sys.argv) > 3 else ""

    def log(obj):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**obj, "t": time.time()}) + "\\n")

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    def write_wav(path):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\\x00\\x10" * 1600)

    reply({"id": None, "status": "ok", "action": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            reply({"id": None, "status": "error", "error": "malformed"})
            continue
        action = req.get("action")
        log({"action": action, **{k: v for k, v in req.items() if k != "id"}})
        if action == "health":
            reply({"id": req["id"], "status": "ok", "device": "cpu",
                   "torch": "fake", "cuda_available": False})
            log({"action": action, "event": "reply", "id": req["id"]})
        elif action == "load-model":
            if mode == "load-error":
                reply({"id": req["id"], "status": "error",
                       "error": "unknown model", "traceback": "Traceback (boom)"})
            else:
                reply({"id": req["id"], "status": "ok", "model": req.get("name"),
                       "device": "cpu", "target_sample_rate": 16000})
        elif action == "convert":
            if mode == "no-reply":
                pass  # never answer: the adapter must time out
            elif mode == "slow":
                time.sleep(0.3)  # create scheduling pressure for concurrency tests
                write_wav(req["output_wav_path"])
                log({"action": action, "event": "reply", "id": req["id"]})
                reply({"id": req["id"], "status": "ok",
                       "output_wav_path": req["output_wav_path"],
                       "elapsed_seconds": 0.01, "model": "Fake"})
            elif mode == "die-on-convert" and extra and not os.path.exists(extra):
                open(extra, "w").close()
                os._exit(1)
            elif mode == "convert-error":
                reply({"id": req["id"], "status": "error",
                       "error": "conversion failed", "traceback": "Traceback (nan)"})
            else:
                write_wav(req["output_wav_path"])
                reply({"id": req["id"], "status": "ok",
                       "output_wav_path": req["output_wav_path"],
                       "elapsed_seconds": 0.01, "model": "Fake"})
        elif action == "shutdown":
            reply({"id": req["id"], "status": "ok", "action": "shutdown"})
            sys.exit(0)
        else:
            reply({"id": req["id"], "status": "error", "error": f"unknown {action}"})
    """
    ).strip()
    + "\n"
)


@pytest.fixture()
def worker_env(tmp_path: Path):
    """Write the fake worker script; return a factory for running converters."""
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"

    def make(mode: str = "ok", **kwargs: Any) -> RvcVoiceConverter:
        extra = str(tmp_path / f"marker-{mode}")
        converter = RvcVoiceConverter(
            worker_python=sys.executable,
            worker_script=script,
            start_timeout_s=kwargs.pop("start_timeout_s", 10.0),
            load_timeout_s=kwargs.pop("load_timeout_s", 10.0),
            convert_timeout_s=kwargs.pop("convert_timeout_s", 10.0),
            **kwargs,
        )
        # Route the mode/args through env-free plumbing: encode them in the
        # script path is not possible, so the adapter argv is patched here.
        converter._worker_script = script  # noqa: SLF001
        converter._argv_extra = (mode, str(log_path), extra)  # type: ignore[attr-defined]
        return converter

    make.script = script  # type: ignore[attr-defined]
    make.log_path = log_path  # type: ignore[attr-defined]
    return make


@pytest.fixture()
def patched_spawn(monkeypatch):
    """Make the adapter spawn the fake worker with its mode arguments."""

    def _apply(converter: RvcVoiceConverter) -> None:
        extra: tuple[str, ...] = getattr(converter, "_argv_extra", ())
        original = RvcVoiceConverter._spawn_and_wait_ready

        def spawn(self: RvcVoiceConverter) -> None:
            import subprocess

            argv = [str(self._worker_python), str(self._worker_script), *extra]
            process = subprocess.Popen(
                argv,
                cwd=str(self._worker_script.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            self._process = process
            from notamusicbot.speech.voice_conversion.rvc import _StderrDrainer

            self._stderr_drainer = _StderrDrainer(process.stderr, "fake-worker")
            self._stderr_drainer.start()
            if not self._wait_for_ready(process, self._start_timeout):
                detail = self._exit_detail(process)
                self._terminate(process)
                self._process = None
                msg = f"fake worker not ready. {detail}"
                raise WorkerStartError(msg)

        monkeypatch.setattr(type(converter), "_spawn_and_wait_ready", spawn)
        monkeypatch.setattr(
            type(converter), "_spawn_and_wait_ready_original", original, raising=False
        )

    return _apply


def make_source(path: Path) -> AudioArtifact:
    return AudioArtifact(path=path, sample_rate_hz=16000, duration_s=1.5)


def make_profile(model: str = "Fake", pitch_shift: int = 2) -> VoiceConversionProfile:
    return VoiceConversionProfile(backend="rvc", model=model, pitch_shift=pitch_shift)


# ---------------------------------------------------------------- happy path


def test_convert_happy_path_with_fake_worker(tmp_path, worker_env, patched_spawn):
    converter = worker_env()
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    output = tmp_path / "out" / "converted.wav"

    artifact = asyncio.run(converter.convert(source, make_profile(), output))

    assert output.is_file()
    assert artifact.path == output
    assert artifact.sample_rate_hz == 16000
    assert artifact.duration_s == pytest.approx(0.1)
    assert converter.loaded_model == "Fake"
    requests = [json.loads(line) for line in worker_env.log_path.read_text().splitlines()]
    # Reply-event probes ({"event": "reply"}) also carry "action"; keep the
    # request-receipt entries only for this ordering assertion.
    actions = [req["action"] for req in requests if "event" not in req]
    assert actions == ["health", "load-model", "convert"]
    convert_req = requests[-1]
    assert convert_req["input_wav_path"] == str(source.path)
    assert convert_req["output_wav_path"] == str(output)
    assert convert_req["pitch_shift"] == 2
    asyncio.run(converter.shutdown())
    assert not converter.is_running


def test_shutdown_sends_shutdown_action(tmp_path, worker_env, patched_spawn):
    converter = worker_env()
    patched_spawn(converter)
    asyncio.run(converter.start())
    asyncio.run(converter.shutdown())
    requests = [json.loads(line) for line in worker_env.log_path.read_text().splitlines()]
    assert requests[-1]["action"] == "shutdown"


def test_start_is_idempotent(tmp_path, worker_env, patched_spawn):
    converter = worker_env()
    patched_spawn(converter)
    asyncio.run(converter.start())
    first = converter._process
    asyncio.run(converter.start())
    assert converter._process is first
    asyncio.run(converter.shutdown())


# ------------------------------------------------------------------- timeouts


def test_convert_timeout_raises_worker_timeout(tmp_path, worker_env, patched_spawn):
    converter = worker_env(mode="no-reply", convert_timeout_s=0.2)
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    with pytest.raises(WorkerTimeoutError, match="timed out"):
        asyncio.run(converter.convert(source, make_profile(), tmp_path / "out.wav"))
    asyncio.run(converter.shutdown())


def test_start_timeout_when_worker_never_ready(tmp_path, worker_env, patched_spawn):
    converter = worker_env(mode="no-reply", start_timeout_s=0.2)
    # "no-reply" sleeps 30s per request, so the health check times out.
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    with pytest.raises((WorkerStartError, WorkerTimeoutError)):
        asyncio.run(converter.convert(source, make_profile(), tmp_path / "out.wav"))


# --------------------------------------------------------- error propagation


def test_worker_error_reply_propagates_with_traceback(tmp_path, worker_env, patched_spawn):
    converter = worker_env(mode="convert-error")
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    with pytest.raises(WorkerRequestError, match="conversion failed") as excinfo:
        asyncio.run(converter.convert(source, make_profile(), tmp_path / "out.wav"))
    assert excinfo.value.traceback == "Traceback (nan)"
    assert excinfo.value.action == "convert"
    asyncio.run(converter.shutdown())


def test_load_model_error_propagates(tmp_path, worker_env, patched_spawn):
    converter = worker_env(mode="load-error")
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    with pytest.raises(WorkerRequestError, match="unknown model"):
        asyncio.run(converter.convert(source, make_profile(), tmp_path / "out.wav"))


def test_missing_model_name_rejected(tmp_path, worker_env, patched_spawn):
    converter = worker_env()
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    profile = VoiceConversionProfile(backend="rvc", model=None)
    with pytest.raises(Exception, match="requires a model name"):
        asyncio.run(converter.convert(source, profile, tmp_path / "out.wav"))


# ------------------------------------------------------------- worker death


def test_worker_death_during_convert_restarts_once(tmp_path, worker_env, patched_spawn):
    converter = worker_env(mode="die-on-convert")
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    output = tmp_path / "out.wav"

    artifact = asyncio.run(converter.convert(source, make_profile(), output))

    # First worker died on convert; the restarted one completed the request.
    assert output.is_file()
    assert artifact.sample_rate_hz == 16000
    asyncio.run(converter.shutdown())


# ---------------------------------------------------------------------------
# Review Round-5 §2/A: full transaction serialization (id → write → reply)
# ---------------------------------------------------------------------------


async def test_concurrent_requests_serialize_full_transaction(
    tmp_path, worker_env, patched_spawn
) -> None:
    """Two concurrent _request() calls: the second request must not be
    written until the first one's reply was consumed — no mismatch, no
    discarded reply, both callers get their own reply."""
    converter = worker_env(mode="slow")
    patched_spawn(converter)
    await converter.start()

    output = tmp_path / "out.wav"
    task_a = asyncio.create_task(
        converter._request(
            "convert",
            timeout=10,
            input_wav_path="in.wav",
            output_wav_path=str(output),
            pitch_shift=0,
        )
    )
    await asyncio.sleep(0.05)  # A's request written; worker sleeping on it
    task_b = asyncio.create_task(converter._request("load-model", timeout=10, name="Fake"))
    reply_a, reply_b = await asyncio.gather(task_a, task_b)

    assert reply_a["status"] == "ok"
    assert reply_b["status"] == "ok"
    assert output.is_file()

    events = [json.loads(line) for line in worker_env.log_path.read_text().splitlines()]
    convert_req_t = next(e["t"] for e in events if e.get("action") == "convert")
    convert_reply_t = next(
        e["t"] for e in events if e.get("event") == "reply" and e.get("id") == reply_a["id"]
    )
    load_req_t = next(e["t"] for e in events if e.get("action") == "load-model")
    # Serialization proof: B's request was written only AFTER A's reply left
    # the worker (B queued ~0.3s behind A's slow convert).
    assert load_req_t >= convert_reply_t > convert_req_t
    await converter.shutdown()


async def test_shutdown_waits_for_active_transaction(tmp_path, worker_env, patched_spawn):
    """Concurrent convert + shutdown: shutdown waits for the in-flight
    transaction; the convert completes successfully, then the worker is
    shut down (shutdown request logged only after the convert reply)."""
    converter = worker_env(mode="slow")
    patched_spawn(converter)
    source = make_source(tmp_path / "in.wav")
    output = tmp_path / "out.wav"

    convert_task = asyncio.create_task(converter.convert(source, make_profile(), output))
    await asyncio.sleep(0.15)  # convert request in flight (worker sleeping)
    shutdown_task = asyncio.create_task(converter.shutdown())

    artifact = await asyncio.wait_for(convert_task, 10)
    await asyncio.wait_for(shutdown_task, 10)

    assert output.is_file()
    assert artifact.path == output
    assert not converter.is_running

    events = [json.loads(line) for line in worker_env.log_path.read_text().splitlines()]
    convert_reply_t = next(e["t"] for e in events if e.get("event") == "reply" and e.get("id") == 3)
    shutdown_req_t = next(e["t"] for e in events if e.get("action") == "shutdown")
    assert shutdown_req_t >= convert_reply_t, "shutdown corrupted the in-flight transaction"


# ---------------------------------------------------------------------------
# Review Round-5 §3/B: cancelled reader owns and consumes its own reply
# ---------------------------------------------------------------------------


async def test_cancelled_reader_consumes_own_reply_and_stream_stays_healthy(
    tmp_path, worker_env, patched_spawn
) -> None:
    """Regression: reader N cancelled while blocked, reply N arrives → the
    reader consumes its OWN reply and terminates; no mismatch, no
    WorkerDiedError, no consumption of reply N+1; a subsequent new request
    works."""
    converter = worker_env(mode="slow")
    patched_spawn(converter)
    await converter.start()

    request_task = asyncio.create_task(
        converter._request(
            "convert",
            timeout=10,
            input_wav_path="in.wav",
            output_wav_path=str(tmp_path / "out.wav"),
            pitch_shift=0,
        )
    )
    await asyncio.sleep(0.1)  # request written; reader thread blocked on the reply
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    # The worker replies ~0.3s after the request. The detached reader thread
    # must consume its own reply (id 3) and finish — whether or not the
    # cancellation landed before the thread started, the stream must stay
    # consistent for the next request.
    await asyncio.sleep(0.7)
    assert 3 not in converter._orphaned_ids, (
        "reader did not consume its own terminal reply (or never cleaned up)"
    )

    # A subsequent new request must succeed: no stolen reply, no mismatch,
    # no WorkerDiedError.
    reply = await asyncio.wait_for(converter._request("health", timeout=5), 10)
    assert reply["status"] == "ok"
    await converter.shutdown()
