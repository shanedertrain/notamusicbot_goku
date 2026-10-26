"""Quality fields of legacy/rvc_worker.py (convert / stream_open / health), in fake mode.

Fake mode needs no torch, rvc_cli or GPU. The optional GPU smoke at the bottom drives the real
worker in the legacy venv and is opt-in (RVC_QUALITY_GPU_TESTS=1), run under the shared GPU lock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from test_rvc_worker_stream import WorkerProc

REPO = Path(__file__).resolve().parents[2]
SAMPLE = Path("/home/cbash23/.cache/listen-speed/qwen-fast/samples/fast_nonstream_s0.wav")


@pytest.fixture()
def w():
    worker = WorkerProc()
    yield worker
    worker.close()


@pytest.fixture()
def loaded(w):
    assert w.call("load-model", name="Obama")["status"] == "ok"
    return w


@pytest.fixture()
def wav(tmp_path) -> Path:
    path = tmp_path / "in.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes((np.sin(np.arange(2400) * 0.1) * 8000).astype("<i2").tobytes())
    return path


def convert(w, wav, tmp_path, **kw):
    return w.call(
        "convert", input_wav_path=str(wav), output_wav_path=str(tmp_path / "out.wav"), **kw
    )


def test_health_has_new_flags(w):
    reply = w.call("health")
    assert reply["status"] == "ok"
    assert isinstance(reply["rmvpe_available"], bool) and isinstance(reply["faiss"], bool)


def test_convert_defaults_unchanged(loaded, wav, tmp_path):
    reply = convert(loaded, wav, tmp_path)
    assert reply["status"] == "ok", reply
    assert reply["pitch_shift"] == 0 and reply["pitch_shift_applied"] == 0
    assert reply["index_used"] is False and reply["index_path"] is None
    assert reply["source_f0_median_hz"] is None and reply["f0_method"] == "harvest"
    assert (tmp_path / "out.wav").is_file()


def test_convert_accepts_and_echoes_quality_fields(loaded, wav, tmp_path):
    reply = convert(
        loaded,
        wav,
        tmp_path,
        pitch_shift=-3,
        f0_method="rmvpe",
        use_index=True,
        index_rate=0.5,
        index_nprobe=16,
        protect=0.33,
        rms_mix_rate=0.25,
        f0_post={"median_width": 5, "octave_repair": True, "clamp_min_hz": 50, "clamp_max_hz": 500},
        auto_pitch={"target_median_hz": 108.0, "max_abs": 8, "strength": 0.5},
    )
    assert reply["status"] == "ok", reply
    assert reply["f0_method"] == "rmvpe" and reply["pitch_shift"] == -3
    assert reply["pitch_shift_applied"] == -3  # fake mode cannot measure the source median
    assert reply["index_used"] is True and reply["index_path"]


def test_use_index_with_zero_rate_is_not_used(loaded, wav, tmp_path):
    reply = convert(loaded, wav, tmp_path, use_index=True, index_rate=0.0)
    assert reply["status"] == "ok" and reply["index_used"] is False


@pytest.mark.parametrize(
    "fields",
    [
        {"use_index": 1},
        {"use_index": "yes"},
        {"index_rate": 1.5},
        {"index_rate": True},
        {"index_rate": "0.5"},
        {"index_nprobe": 0},
        {"index_nprobe": 257},
        {"index_nprobe": True},
        {"index_nprobe": 2.5},
        {"protect": 0.6},
        {"protect": -0.01},
        {"protect": True},
        {"rms_mix_rate": 1.1},
        {"f0_method": "bogus"},
        {"f0_post": 3},
        {"f0_post": {"nope": 1}},
        {"f0_post": {"median_width": 4}},
        {"f0_post": {"median_width": 1}},
        {"f0_post": {"octave_repair": 1}},
        {"f0_post": {"clamp_min_hz": 500, "clamp_max_hz": 100}},
        {"f0_post": {"fill_gap_frames": -1}},
        {"auto_pitch": {}},
        {"auto_pitch": {"target_median_hz": 0}},
        {"auto_pitch": {"target_median_hz": 100, "max_abs": 99}},
        {"auto_pitch": {"target_median_hz": 100, "strength": -1}},
        {"auto_pitch": {"target_median_hz": 100, "extra": 1}},
        {"auto_pitch": "108"},
    ],
)
def test_convert_rejects_bad_fields(loaded, wav, tmp_path, fields):
    reply = convert(loaded, wav, tmp_path, **fields)
    assert reply["status"] == "error", reply


def test_convert_bad_input_path_still_convert_failed(loaded, tmp_path):
    reply = loaded.call(
        "convert", input_wav_path="/nonexistent/in.wav", output_wav_path=str(tmp_path / "o.wav"),
        use_index=True,
    )  # fmt: skip
    assert reply["status"] == "error" and reply["error"].startswith("convert failed:")


# -- stream_open ---------------------------------------------------------------------------------


def test_stream_open_f0_method_rmvpe_still_error(loaded):
    reply = loaded.call("stream_open", stream_id="s-1", f0_method="rmvpe")
    assert reply["status"] == "error" and "pm" in reply["error"]
    assert reply["stream_closed"] is False


def test_stream_open_defaults_reply(loaded):
    reply = loaded.call("stream_open", stream_id="s-1")
    assert reply["status"] == "ok"
    assert reply["f0_method"] == "pm" and reply["f0_algo"] == "pm"
    assert reply["index_used"] is False and reply["protect"] is None


def test_stream_open_new_fields_echoed(loaded):
    reply = loaded.call(
        "stream_open", stream_id="s-1", f0_algo="rmvpe", use_index=True, index_rate=0.4,
        index_nprobe=8, protect=0.33, f0_post={"median_width": 5}, pitch_shift=-3,
    )  # fmt: skip
    assert reply["status"] == "ok", reply
    assert reply["f0_method"] == "pm" and reply["f0_algo"] == "rmvpe"
    assert reply["index_used"] is True and reply["protect"] == 0.33


@pytest.mark.parametrize(
    "fields",
    [
        {"f0_algo": "harvest"},
        {"f0_algo": 1},
        {"protect": 0.9},
        {"protect": True},
        {"use_index": "true"},
        {"index_rate": 2},
        {"index_nprobe": 0},
        {"f0_post": {"bogus": 1}},
        {"f0_post": "x"},
    ],
)
def test_stream_open_rejects_bad_fields(loaded, fields):
    reply = loaded.call("stream_open", stream_id="s-1", **fields)
    assert reply["status"] == "error", reply
    assert reply["stream_closed"] is False


# -- real-worker GPU smoke (legacy venv; opt-in) --------------------------------------------------

LEGACY_PY = REPO / ".venv" / "bin" / "python"
GPU_LOCK = Path("/home/cbash23/.cache/listen-speed/gpu.lock")


@pytest.mark.skipif(
    os.environ.get("RVC_QUALITY_GPU_TESTS") != "1",
    reason="set RVC_QUALITY_GPU_TESTS=1 (GPU, legacy venv, Obama model)",
)
@pytest.mark.skipif(
    not (SAMPLE.is_file() and LEGACY_PY.is_file() and (REPO / "rvc_cli/models/Obama").is_dir()),
    reason="sample wav, legacy venv or Obama model missing",
)
def test_gpu_index_actually_loads_and_changes_output(tmp_path):  # pragma: no cover - GPU
    requests = [
        {"id": 1, "action": "load-model", "name": "Obama"},
        {"id": 2, "action": "convert", "input_wav_path": str(SAMPLE),
         "output_wav_path": str(tmp_path / "plain.wav"), "use_index": False},
        {"id": 3, "action": "convert", "input_wav_path": str(SAMPLE),
         "output_wav_path": str(tmp_path / "idx.wav"), "use_index": True, "index_rate": 0.5,
         "index_nprobe": 16, "protect": 0.33},
        {"id": 4, "action": "shutdown"},
    ]  # fmt: skip
    cmd = [str(LEGACY_PY), str(REPO / "legacy" / "rvc_worker.py")]
    if shutil.which("flock"):
        cmd = ["flock", str(GPU_LOCK), *cmd]
    proc = subprocess.run(
        cmd,
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(tmp_path),
    )
    replies = {r["id"]: r for r in map(json.loads, proc.stdout.splitlines()) if r.get("id")}
    assert replies[1]["status"] == "ok", replies
    assert replies[2]["status"] == "ok" and replies[2]["index_used"] is False
    assert replies[3]["status"] == "ok", replies[3]
    assert replies[3]["index_used"] is True and replies[3]["index_path"].endswith(".index")

    import soundfile as sf

    a, _ = sf.read(str(tmp_path / "plain.wav"))
    b, _ = sf.read(str(tmp_path / "idx.wav"))
    n = min(len(a), len(b))
    assert n > 0 and not np.allclose(a[:n], b[:n], atol=1e-4)


def test_rmvpe_unavailable_is_an_error_not_a_silent_pm_fallback(monkeypatch, wav, tmp_path):
    """Tripwire: asking for rmvpe without the weights must raise, never quietly use pm."""
    import sys

    monkeypatch.setenv("RVC_WORKER_FAKE", "1")
    monkeypatch.syspath_prepend(str(REPO / "legacy"))
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    import rvc_worker

    worker = rvc_worker.Worker()
    worker.load_model("Obama")
    monkeypatch.setattr(rvc_worker, "_rmvpe_available", lambda: False)
    with pytest.raises(ValueError, match="rmvpe"):
        worker.convert(str(wav), str(tmp_path / "o.wav"), None, "rmvpe")
    # pm is still fine and is what the reply reports when pm was asked for
    info = worker.convert(str(wav), str(tmp_path / "o2.wav"), None, "pm")
    assert info["f0_method"] == "pm"
