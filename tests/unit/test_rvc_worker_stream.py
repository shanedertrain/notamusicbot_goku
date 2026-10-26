"""Streaming actions of legacy/rvc_worker.py, driven over real line-JSON in fake mode.

No GPU, no rvc_cli, no torch: RVC_WORKER_FAKE=1 swaps in canned model info and a
tiny 2x-repeat engine.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[2] / "legacy" / "rvc_worker.py"
TIMEOUT_S = 10.0


class WorkerProc:
    def __init__(self) -> None:
        env = dict(os.environ, RVC_WORKER_FAKE="1")
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._n = 0
        self.ready = self._read()

    def _read(self) -> dict:
        box: list = []
        t = threading.Thread(target=lambda: box.append(self.proc.stdout.readline()), daemon=True)
        t.start()
        t.join(TIMEOUT_S)
        if t.is_alive() or not box or not box[0]:
            self.proc.kill()
            raise AssertionError("worker did not reply within timeout")
        return json.loads(box[0])

    def call(self, action: str, **fields) -> dict:
        self._n += 1
        req = {"id": self._n, "action": action, **fields}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        assert reply["id"] == self._n
        return reply

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=TIMEOUT_S)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def b64(arr) -> str:
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def unb64(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


@pytest.fixture()
def w():
    worker = WorkerProc()
    yield worker
    worker.close()


@pytest.fixture()
def loaded(w):
    assert w.call("load-model", name="Goku")["status"] == "ok"
    return w


def open_stream(w, sid="s-1", **kw):
    reply = w.call("stream_open", stream_id=sid, **kw)
    assert reply["status"] == "ok", reply
    return reply


def push(w, pcm, sid="s-1", **kw):
    pcm = np.asarray(pcm, dtype=np.float32)
    return w.call("stream_push", stream_id=sid, pcm_b64=b64(pcm), samples=pcm.size, **kw)


def test_ready_and_load_model(w):
    assert w.ready["status"] == "ok" and w.ready["action"] == "ready"
    info = w.call("load-model", name="Goku")
    assert info["status"] == "ok"
    assert info["model"] == "Goku" and info["device"] == "fake"
    assert info["target_sample_rate"] == 40000 and info["config_switched"] is False


def test_open_requires_model(w):
    reply = w.call("stream_open", stream_id="s-1")
    assert reply["status"] == "error"
    assert "no model loaded" in reply["error"]
    assert reply["stream_closed"] is False


def test_happy_path(loaded):
    opened = open_stream(loaded, pitch_shift=2, block_ms=320)
    assert opened["input_sample_rate"] == 24000 and opened["output_sample_rate"] == 48000
    assert opened["sample_format"] == "f32le" and opened["channels"] == 1
    assert opened["min_input_samples_for_first_output"] == 5760
    assert opened["replaced_stream"] is False
    assert opened["model"] == "Goku" and opened["f0_method"] == "pm"

    pcm = np.linspace(-0.5, 0.5, 6000, dtype=np.float32)
    reply = push(loaded, pcm)
    assert reply["status"] == "ok" and reply["stream_id"] == "s-1"
    assert reply["samples"] == 12000
    assert reply["input_total"] == 6000 and reply["output_total"] == 12000
    assert reply["compute_ms"] >= 0
    np.testing.assert_array_equal(unb64(reply["pcm_b64"]), np.repeat(pcm, 2))

    closed = loaded.call("stream_close", stream_id="s-1", flush=True)
    assert closed["status"] == "ok" and closed["closed"] is True
    assert closed["samples"] == 0 and closed["pcm_b64"] == ""
    assert closed["input_total"] == 6000 and closed["output_total"] == 12000
    assert closed["output_total"] == 2 * closed["input_total"]
    assert closed["steps"] == 1 and closed["compute_ms_total"] >= 0

    assert (
        loaded.call("stream_push", stream_id="s-1", pcm_b64="", samples=0)["error"]
        == "no open stream"
    )


def test_empty_push_ok(loaded):
    open_stream(loaded)
    reply = loaded.call("stream_push", stream_id="s-1", pcm_b64="", samples=0)
    assert reply["status"] == "ok" and reply["pcm_b64"] == "" and reply["samples"] == 0


def test_replaced_stream(loaded):
    open_stream(loaded, "a")
    second = open_stream(loaded, "b")
    assert second["replaced_stream"] is True
    mismatch = push(loaded, np.zeros(10), sid="a")
    assert mismatch["status"] == "error" and mismatch["error"] == "stream id mismatch"
    assert mismatch["stream_closed"] is False
    assert push(loaded, np.zeros(10), sid="b")["status"] == "ok"


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "a/b", 5, None])
def test_bad_stream_id(loaded, bad):
    reply = loaded.call("stream_open", stream_id=bad)
    assert reply["status"] == "error" and reply["stream_closed"] is False


@pytest.mark.parametrize("method", ["harvest", "dio", "crepe", "rmvpe", 3])
def test_non_pm_rejected(loaded, method):
    reply = loaded.call("stream_open", stream_id="s-1", f0_method=method)
    assert reply["status"] == "error" and "pm" in reply["error"]
    assert loaded.call("health")["stream"]["open"] is False


def test_pm_explicit_and_bad_fields(loaded):
    assert open_stream(loaded, f0_method="pm")["f0_method"] == "pm"
    assert loaded.call("stream_open", stream_id="s-2", pitch_shift="2")["status"] == "error"
    assert loaded.call("stream_open", stream_id="s-2", block_ms=320.5)["status"] == "error"
    assert loaded.call("stream_open", stream_id="s-2", search_ms=True)["status"] == "error"


def stream_dead(w):
    return (
        w.call("stream_push", stream_id="s-1", pcm_b64="", samples=0)["error"] == "no open stream"
    )


def test_bad_base64(loaded):
    open_stream(loaded)
    reply = loaded.call("stream_push", stream_id="s-1", pcm_b64="!!!notbase64", samples=4)
    assert reply["status"] == "error" and reply["stream_closed"] is True
    assert stream_dead(loaded)


def test_length_not_multiple_of_four(loaded):
    open_stream(loaded)
    reply = loaded.call(
        "stream_push", stream_id="s-1", pcm_b64=base64.b64encode(b"abcdef").decode(), samples=1
    )
    assert reply["status"] == "error" and reply["stream_closed"] is True
    assert stream_dead(loaded)


def test_samples_mismatch(loaded):
    open_stream(loaded)
    reply = loaded.call("stream_push", stream_id="s-1", pcm_b64=b64(np.zeros(10)), samples=11)
    assert reply["status"] == "error" and reply["stream_closed"] is True
    assert stream_dead(loaded)


def test_oversize_push(loaded):
    open_stream(loaded)
    reply = push(loaded, np.zeros(48001))
    assert reply["status"] == "error" and reply["stream_closed"] is True
    assert stream_dead(loaded)
    open_stream(loaded)
    assert push(loaded, np.zeros(48000))["samples"] == 96000


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input(loaded, bad):
    open_stream(loaded)
    pcm = np.zeros(100, dtype=np.float32)
    pcm[50] = bad
    reply = push(loaded, pcm)
    assert reply["status"] == "error" and reply["stream_closed"] is True
    assert stream_dead(loaded)


def test_fake_boom(loaded):
    open_stream(loaded)
    reply = push(loaded, np.full(100, 1234.5, dtype=np.float32))
    assert reply["status"] == "error" and "fake boom" in reply["error"]
    assert reply["stream_closed"] is True
    assert stream_dead(loaded)
    assert loaded.call("health")["stream"] == {"open": False, "stream_id": None}


def test_push_without_stream(loaded):
    reply = loaded.call("stream_push", stream_id="s-1", pcm_b64="", samples=0)
    assert reply["status"] == "error" and reply["error"] == "no open stream"
    assert reply["stream_closed"] is False


def test_close_unknown_is_idempotent(loaded):
    for _ in range(2):
        reply = loaded.call("stream_close", stream_id="nope", flush=True)
        assert reply["status"] == "ok" and reply["closed"] is False and reply["samples"] == 0
    open_stream(loaded)
    other = loaded.call("stream_close", stream_id="other", flush=False)
    assert other["closed"] is False
    assert loaded.call("health")["stream"]["open"] is True  # untouched


def test_close_abort(loaded):
    open_stream(loaded)
    push(loaded, np.zeros(100))
    reply = loaded.call("stream_close", stream_id="s-1", flush=False)
    assert reply["status"] == "ok" and reply["closed"] is True
    assert reply["samples"] == 0 and reply["pcm_b64"] == ""
    again = loaded.call("stream_close", stream_id="s-1", flush=False)
    assert again["closed"] is False


def test_close_requires_bool_flush(loaded):
    reply = loaded.call("stream_close", stream_id="s-1")
    assert reply["status"] == "error"


def test_health_stream_key(loaded):
    assert loaded.call("health")["stream"] == {"open": False, "stream_id": None}
    open_stream(loaded, "h-1")
    health = loaded.call("health")
    assert health["status"] == "ok"
    assert health["stream"] == {"open": True, "stream_id": "h-1"}


def test_load_model_aborts_stream(loaded):
    open_stream(loaded)
    assert loaded.call("load-model", name="Goku")["status"] == "ok"
    assert loaded.call("health")["stream"]["open"] is False
    assert stream_dead(loaded)


def test_convert_regression_missing_paths(loaded):
    reply = loaded.call("convert")
    assert reply["status"] == "error"
    assert reply["error"] == "convert requires input_wav_path and output_wav_path strings"
    assert "stream_closed" not in reply


def test_convert_legal_during_stream(loaded):
    open_stream(loaded)
    reply = loaded.call(
        "convert", input_wav_path="/nonexistent/in.wav", output_wav_path="/tmp/x.wav"
    )
    assert reply["status"] == "error" and reply["error"].startswith("convert failed:")
    assert loaded.call("health")["stream"]["open"] is True


def test_unknown_action_unchanged(w):
    reply = w.call("bogus")
    assert reply["status"] == "error" and reply["error"] == "unknown action: 'bogus'"
    assert "stream_closed" not in reply
