"""Integration tests for the legacy RVC worker (legacy/rvc_worker.py).

Skipped cleanly unless the legacy assets are present:

- repo-root ``.venv/bin/python``  (Python 3.10 legacy env with fairseq/torch cu121)
- ``hubert_base.pt`` at the repo root
- at least one complete model under ``rvc_cli/models/`` (e.g. Goku)

Run for real with the MAIN venv's pytest (test process) while the worker
subprocess itself runs under the legacy ``.venv``:

    .venv-main/bin/python -m pytest tests/integration/test_rvc_worker.py \
        -m "rvc and integration" -s

The test process only needs stdlib + numpy/soundfile; all heavy lifting
happens inside the worker subprocess on GPU.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "legacy" / "rvc_worker.py"
LEGACY_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
HUBERT = REPO_ROOT / "hubert_base.pt"
MODELS_DIR = REPO_ROOT / "rvc_cli" / "models"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

pytestmark = [
    pytest.mark.rvc,
    pytest.mark.integration,
]

# Preferred verification targets, in order.
CANDIDATE_MODELS = ["Goku", "Obama", "Biden"]


def _available_model() -> str | None:
    if not MODELS_DIR.is_dir():
        return None
    for name in CANDIDATE_MODELS:
        d = MODELS_DIR / name
        if (
            d.is_dir()
            and list(d.glob("*.pth"))
            and list(d.glob("*.index"))
            and list(d.glob("*.yaml"))
        ):
            return name
    return None


def _assets_available() -> bool:
    return (
        LEGACY_PYTHON.is_file()
        and WORKER.is_file()
        and HUBERT.is_file()
        and _available_model() is not None
    )


pytestmark_skip = pytest.mark.skipif(
    not _assets_available(),
    reason="legacy RVC assets missing (.venv, hubert_base.pt or rvc_cli/models/<name>)",
)


class WorkerProcess:
    """Small line-JSON IPC client mirroring what the production adapter does."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [str(LEGACY_PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
        ready = self._recv()
        assert ready is not None and ready.get("action") == "ready", ready

    def _recv(self):
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read()
            raise RuntimeError(f"worker died. stderr tail:\n{stderr[-2000:]}")
        return json.loads(line)

    def request(self, timeout_s: float = 180, **payload) -> dict:
        start = time.monotonic()
        self.proc.stdin.write(json.dumps({"id": payload.pop("id", 1), **payload}) + "\n")
        self.proc.stdin.flush()
        while time.monotonic() - start < timeout_s:
            reply = self._recv()
            # The worker emits exactly one reply per request; any line it
            # emits is the reply (incidental output goes to stderr).
            return reply
        raise TimeoutError(f"no reply within {timeout_s}s for {payload}")

    def shutdown(self) -> None:
        try:
            self.request(timeout_s=10, action="shutdown")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


@pytest.fixture(scope="module")
def fixture_wav() -> Path:
    """Small real WAV: 2 s of a 220 Hz sine with an amplitude envelope."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "sine_2s.wav"
    if not path.is_file():
        sr = 24000
        t = np.arange(sr * 2) / sr
        envelope = 0.5 + 0.4 * np.sin(2 * math.pi * 0.5 * t)
        audio = (0.3 * envelope * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
        sf.write(path, audio, sr)
    assert path.is_file() and path.stat().st_size > 44
    return path


@pytest.mark.skipif(not _assets_available(), reason="legacy RVC assets missing")
def test_worker_health_and_convert_real_model(fixture_wav):
    model_name = _available_model()
    assert model_name is not None
    output_path = FIXTURES / f"converted_{model_name.lower()}.wav"
    if output_path.is_file():
        output_path.unlink()

    with WorkerProcess() as w:
        health = w.request(action="health")
        assert health["status"] == "ok"
        assert health["cuda_available"] is True, "GPU expected for the legacy RVC stack"

        t0 = time.monotonic()
        loaded = w.request(action="load-model", name=model_name)
        assert loaded["status"] == "ok", loaded
        assert loaded["model"] == model_name
        assert loaded["target_sample_rate"] > 0
        load_seconds = time.monotonic() - t0

        t0 = time.monotonic()
        reply = w.request(
            action="convert",
            input_wav_path=str(fixture_wav),
            output_wav_path=str(output_path),
            pitch_shift=0,
        )
        convert_seconds = time.monotonic() - t0
        assert reply["status"] == "ok", reply
        assert reply["elapsed_seconds"] > 0

    # Output assertions (worker has exited; nothing can rewrite it now).
    assert output_path.is_file() and output_path.stat().st_size > 44
    data, sr = sf.read(output_path, dtype="float64")
    assert data.size > 0
    duration = data.size / sr
    assert duration > 0.5, f"output too short: {duration:.3f}s"
    assert math.isfinite(float(data.min())) and math.isfinite(float(data.max())), (
        "NaN/Inf in output"
    )
    peak = float(np.abs(data).max())
    assert 0.0 < peak <= 1.0, f"insane peak: {peak}"

    print(
        f"\nmodel={model_name} load={load_seconds:.2f}s convert={convert_seconds:.2f}s "
        f"out_dur={duration:.2f}s sr={sr} peak={peak:.3f}"
    )

    # Record latency for the performance baseline (section 35 of the handoff).
    latency_note = FIXTURES / "rvc_worker_last_latency.txt"
    latency_note.write_text(
        f"model={model_name}\nload_seconds={load_seconds:.3f}\n"
        f"convert_seconds={convert_seconds:.3f}\noutput_duration={duration:.3f}\npeak={peak:.4f}\n"
    )


@pytest.mark.skipif(not _assets_available(), reason="legacy RVC assets missing")
def test_worker_rejects_bad_requests(fixture_wav):
    with WorkerProcess() as w:
        r = w.request(
            action="convert",
            input_wav_path=str(fixture_wav),
            output_wav_path=str(FIXTURES / "never.wav"),
        )
        assert r["status"] == "error" and "no model loaded" in r["error"]

        r = w.request(action="load-model", name="../escape")
        assert r["status"] == "error"

        r = w.request(action="load-model", name="definitely_not_a_model")
        assert r["status"] == "error" and "unknown model" in r["error"]

        r = w.request(action="bogus")
        assert r["status"] == "error" and "unknown action" in r["error"]
