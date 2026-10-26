#!/usr/bin/env python
"""Legacy RVC voice-conversion worker.

Long-running process that must be executed with the legacy RVC environment
(repo-root ``.venv``, Python 3.12) because it imports the ``rvc_cli``
submodule, which needs torch 2.5.0+cu121 (HuBERT via rvc_cli/hubert_loader.py, no fairseq).

It loads ONE RVC model persistently and serves line-based JSON requests on
stdin / responses on stdout.  See legacy/README.md for the protocol.

Stdlib-only: the worker itself imports nothing outside the standard library
plus the rvc_cli stack that is already installed in the legacy venv.

This component is explicitly exempt from the "no code may import rvc_cli"
rule (REWORK_HANDOFF.md sections 4.3 / 10.3): it is the isolated legacy
runtime boundary.  Nothing outside ``legacy/`` should import or spawn it
except through this JSON IPC contract.
"""

from __future__ import annotations

import json
import math
import base64
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_CLI_DIR = REPO_ROOT / "rvc_cli"

# The worker is a legacy isolated component: it manages its own sys.path and
# working directory instead of relying on the caller's environment.
sys.path.insert(0, str(RVC_CLI_DIR))
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # voice_converter resolves "hubert_base.state_dict.pt" relative to cwd

CONVERT_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_CONVERT_TIMEOUT", "120"))
LOAD_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_LOAD_TIMEOUT", "300"))

STREAM_OPEN_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_STREAM_OPEN_TIMEOUT", "30"))
STREAM_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_STREAM_TIMEOUT", "10"))

# Streaming wire constants (see legacy/README.md, docs/RVC_STREAMING_DESIGN.md section 5).
STREAM_INPUT_SAMPLE_RATE = 24000
STREAM_OUTPUT_SAMPLE_RATE = 48000
STREAM_MAX_PUSH_SAMPLES = 48000  # 2.0 s at 24 kHz
STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STREAM_MS_FIELDS = (
    "first_block_ms",
    "block_ms",
    "context_ms",
    "lookahead_ms",
    "crossfade_ms",
    "search_ms",
)


def _fake_mode() -> bool:
    """RVC_WORKER_FAKE=1: no torch / rvc_cli; canned model info and a fake stream engine."""
    return os.environ.get("RVC_WORKER_FAKE") == "1"


class _FakeEngine:
    """Stand-in stream engine for fake mode: 2x sample repeat, no model."""

    min_input_samples_for_first_output = 5760

    def __init__(self) -> None:
        self.input_total = 0
        self.output_total = 0
        self.steps = 0
        self.compute_s = 0.0

    def push(self, pcm24):  # noqa: ANN001
        import numpy as np

        pcm24 = np.asarray(pcm24, dtype=np.float32)
        if pcm24.size and pcm24[0] == np.float32(1234.5):
            raise RuntimeError("fake boom")
        out = np.repeat(pcm24, 2)
        self.input_total += int(pcm24.size)
        self.output_total += int(out.size)
        self.steps += 1
        return out

    def flush(self):
        import numpy as np

        return np.zeros(0, dtype=np.float32)

    def abort(self) -> None:
        return None


# f0 extraction methods rvc_cli's vc_infer_pipeline supports. Requests may
# override the model yaml's f0_method with one of these (config-selectable).
# "rmvpe" is accepted for convert only (needs legacy/rvc_f0.py + the weights);
# stream_open keeps f0_method pm-only and selects rmvpe through ``f0_algo``.
ALLOWED_F0_METHODS = {"pm", "harvest", "dio", "crepe", "crepe-tiny", "rmvpe"}

_F0_POST_KEYS = ("median_width", "octave_repair", "clamp_min_hz", "clamp_max_hz", "fill_gap_frames")
_CONVERT_QUALITY_KEYS = (
    "use_index",
    "index_rate",
    "index_nprobe",
    "protect",
    "rms_mix_rate",
    "f0_post",
    "auto_pitch",
)
_STREAM_QUALITY_KEYS = ("use_index", "index_rate", "index_nprobe", "protect", "f0_post")


def _rmvpe_available() -> bool:
    """True when legacy/rvc_f0.py is importable and reports the RMVPE weights present."""
    if _fake_mode():
        return True  # fake mode has no model; accept the field and echo it
    try:
        import rvc_f0  # legacy/rvc_f0.py; the script directory is on sys.path

        return bool(rvc_f0.rmvpe_available())
    except Exception:  # noqa: BLE001 - import or weights problem
        return False


def _faiss_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("faiss") is not None
    except Exception:  # noqa: BLE001
        return False


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_num(name: str, value: Any, lo: float, hi: float) -> None:
    if not _is_num(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a number in {lo}..{hi}, got {value!r}")


def _check_int(name: str, value: Any, lo: int, hi: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}, got {value!r}")


def _validate_f0_post(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("f0_post must be an object")
    unknown = sorted(set(value) - set(_F0_POST_KEYS))
    if unknown:
        raise ValueError(f"unknown f0_post keys: {unknown}")
    out: Dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if key == "median_width":
            _check_int(key, item, 0, 101)
            if item != 0 and (item < 3 or item % 2 == 0):
                raise ValueError("f0_post.median_width must be 0 or an odd integer >= 3")
        elif key == "octave_repair":
            if not isinstance(item, bool):
                raise ValueError("f0_post.octave_repair must be a boolean")
        elif key in ("clamp_min_hz", "clamp_max_hz"):
            _check_num("f0_post." + key, item, 1.0, 4000.0)
        else:  # fill_gap_frames
            _check_int("f0_post.fill_gap_frames", item, 0, 100)
        out[key] = item
    lo, hi = out.get("clamp_min_hz"), out.get("clamp_max_hz")
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("f0_post.clamp_min_hz must be < clamp_max_hz")
    return out


def _validate_auto_pitch(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("auto_pitch must be an object")
    unknown = sorted(set(value) - {"target_median_hz", "max_abs", "strength"})
    if unknown:
        raise ValueError(f"unknown auto_pitch keys: {unknown}")
    if "target_median_hz" not in value:
        raise ValueError("auto_pitch requires target_median_hz")
    _check_num("auto_pitch.target_median_hz", value["target_median_hz"], 30.0, 1000.0)
    out: Dict[str, Any] = {"target_median_hz": float(value["target_median_hz"])}
    if value.get("max_abs") is not None:
        _check_int("auto_pitch.max_abs", value["max_abs"], 0, 24)
        out["max_abs"] = value["max_abs"]
    if value.get("strength") is not None:
        _check_num("auto_pitch.strength", value["strength"], 0.0, 2.0)
        out["strength"] = float(value["strength"])
    return out


def _validate_quality(fields: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
    """Validate the optional quality fields; returns only the non-null ones. Raises ValueError."""
    allowed = _STREAM_QUALITY_KEYS if stream else _CONVERT_QUALITY_KEYS
    out: Dict[str, Any] = {}
    for key in allowed:
        value = fields.get(key)
        if value is None:
            continue
        if key == "use_index":
            if not isinstance(value, bool):
                raise ValueError("use_index must be a boolean")
        elif key == "index_rate":
            _check_num(key, value, 0.0, 1.0)
        elif key == "index_nprobe":
            _check_int(key, value, 1, 256)
        elif key == "protect":
            _check_num(key, value, 0.0, 0.5)
        elif key == "rms_mix_rate":
            _check_num(key, value, 0.0, 1.0)
        elif key == "f0_post":
            value = _validate_f0_post(value)
        elif key == "auto_pitch":
            value = _validate_auto_pitch(value)
        out[key] = value
    return out

# Set to the protocol stream at startup in main(); falls back to stdout.
_PROTOCOL_OUT = sys.stdout


def _ensure_ffmpeg_python_shim() -> None:
    """Provide a minimal stdlib stand-in for the missing ``ffmpeg-python`` package.

    ``rvc_cli/my_utils.py`` uses exactly one ffmpeg-python call pattern:

        ffmpeg.input(file, threads=0).output("-", format="f32le",
        acodec="pcm_f32le", ac=1, ar=sr).run(cmd=["ffmpeg", "-nostdin"],
        capture_stdout=True, capture_stderr=True)

    The shim reproduces that invocation byte-for-byte using the ffmpeg CLI
    (a documented system prerequisite) so the legacy venv needs no new
    packages.  If the real ``ffmpeg-python`` package is importable, it is
    used instead and the shim is a no-op.
    """
    try:
        import ffmpeg  # noqa: F401

        return
    except ImportError:
        pass

    import subprocess

    class _Node:
        def __init__(self, args: list):
            self._args = args

        def output(self, *outputs, **kwargs):
            args = list(self._args)
            for key, value in kwargs.items():
                flag = "-f" if key == "format" else "-" + key.rstrip("_")
                if value is True:
                    args.append(flag)
                else:
                    args.extend([flag, str(value)])
            args.extend(outputs)
            return _Node(args)

        def run(self, cmd=None, capture_stdout=False, capture_stderr=False, **_ignored):
            argv = list(cmd or ["ffmpeg"])
            argv += self._args
            proc = subprocess.run(argv, capture_output=capture_stdout or capture_stderr)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])
            return proc.stdout, proc.stderr

        def __getitem__(self, _):  # unused; ffmpeg-python chain compatibility
            return self

    class _FFmpegShim:
        @staticmethod
        def input(*args, **kwargs):
            node_args = ["-i"] + [str(a) for a in args]
            for key, value in kwargs.items():
                node_args.extend(["-" + key.rstrip("_"), str(value)])
            return _Node(node_args)

        # API surface used elsewhere in rvc_cli would fail loudly if hit.
        def __getattr__(self, name):
            raise NotImplementedError(f"ffmpeg shim does not implement {name!r}")

    sys.modules["ffmpeg"] = _FFmpegShim()


class ConvertTimeout(RuntimeError):
    pass


class StreamError(RuntimeError):
    """Stream failure; ``closed`` says whether the stream was dropped."""

    def __init__(self, message: str, closed: bool) -> None:
        super().__init__(message)
        self.closed = closed


class _timeout:
    """SIGALRM-based hard timeout (Linux). Raises ConvertTimeout on expiry."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._old_handler = None
        self._old_alarm = 0

    def _on_alarm(self, signum, frame):  # noqa: ANN001
        raise ConvertTimeout(f"operation exceeded {self._seconds}s timeout")

    def __enter__(self):
        import signal

        self._signal = signal
        self._old_handler = signal.signal(signal.SIGALRM, self._on_alarm)
        self._old_alarm = signal.alarm(max(1, int(math.ceil(self._seconds))))
        return self

    def __exit__(self, exc_type, exc, tb):
        import signal

        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old_handler)
        if exc_type is not None and self._old_alarm:
            # Restore a previously pending alarm, at least its remainder.
            signal.alarm(self._old_alarm)
        return False


def _reply(request_id: Any, status: str, **payload: Any) -> None:
    """Write one JSON line to the protocol stream and flush immediately."""
    line = json.dumps({"id": request_id, "status": status, **payload}, default=str)
    _PROTOCOL_OUT.write(line + "\n")
    _PROTOCOL_OUT.flush()


def _fail(request_id: Any, message: str, **extra: Any) -> None:
    _reply(request_id, "error", error=message, **extra)


class Worker:
    """Holds one loaded RVC model and performs conversions."""

    def __init__(self) -> None:
        self._vc = None
        self._model = None
        self._vc_args = None
        # Open stream: (stream_id, engine, params) or None. One per worker.
        self._stream: Optional[tuple] = None
        # Tests / fake mode may inject a factory (worker, fields) -> engine.
        self._engine_factory: Optional[Callable[["Worker", dict], Any]] = None

    # -- model loading ----------------------------------------------------

    def load_model(self, name: str, variant: Optional[str] = None) -> Dict[str, Any]:
        self._drop_stream(abort=True)
        if _fake_mode():
            import types

            self._vc = object()
            self._vc_args = types.SimpleNamespace(f0_pitch=0, f0_method="harvest", index_rate=0.0, sid=0)
            self._model = type("_FakeModel", (), {"model_name": name})()
            return {
                "model": name,
                "variant": variant or name.lower(),
                "config_yaml": (variant or name.lower()) + ".yaml",
                "config_switched": False,
                "device": "fake",
                "target_sample_rate": 40000,
                "version": "v1",
                "tts_type": "fake",
                "index_rate": 0.0,
                "f0_method": "harvest",
                "default_f0_pitch": 0,
            }
        _ensure_ffmpeg_python_shim()

        # torch 2.5.0+cu121 on this GPU raises "cuDNN Frontend error: No
        # execution plans support the graph" inside HuBERT's scaled-dot-product
        # attention.  Routing SDP away from cuDNN preserves the previous
        # (working) math-path behavior.
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)

        import models as rvc_models  # rvc_cli/models.py
        import voice_converter as rvc_vc  # rvc_cli/voice_converter.py

        model_dir = RVC_CLI_DIR / "models" / name
        if not model_dir.is_dir():
            raise ValueError(f"unknown model {name!r}: no directory {model_dir}")

        found = rvc_models.get_models_from_directory(model_dir)
        if not found:
            raise ValueError(f"model directory {model_dir} has no complete model (pth/index/yaml)")
        # get_models_from_directory returns one entry per yaml; select the
        # EXACT requested variant (yaml stem) — never found[0]. When no
        # variant is given, default to the yaml matching the directory name,
        # falling back to the single yaml if the directory has exactly one.
        model = self._select_variant(found, variant, model_dir)

        # In-process variant switch: if the weights (pth) of the requested
        # model are already resident in this worker, only the YAML-derived
        # configuration needs to change — no VoiceConverter re-init, no
        # weight reload, no process restart.
        if (
            self._vc is not None
            and self._model is not None
            and Path(self._model.pth_filepath) == Path(model.pth_filepath)
        ):
            self._model = model
            self._vc_args = model.vc_args
            return {
                "model": name,
                "variant": model.model_name,
                "config_yaml": model.model_name + ".yaml",
                "config_switched": True,
                "device": str(getattr(self._vc.config, "device", "unknown")),
                "target_sample_rate": int(self._vc.tgt_sr),
                "version": str(getattr(self._vc, "version", "")),
                "tts_type": str(model.tts_type),
                "index_rate": float(model.vc_args.index_rate),
                "f0_method": str(model.vc_args.f0_method),
                "default_f0_pitch": int(model.vc_args.f0_pitch),
            }

        with _timeout(LOAD_TIMEOUT_SECONDS):
            vc = rvc_vc.VoiceConverter(
                model.pth_filepath,
                model.index_filepath,
                generator=os.environ.get("RVC_WORKER_DEVICE", "GPU"),
            )

        self._vc = vc
        self._model = model
        self._vc_args = model.vc_args

        return {
            "model": name,
            "variant": model.model_name,
            "config_yaml": model.model_name + ".yaml",
            "config_switched": False,
            "device": str(getattr(vc.config, "device", "unknown")),
            "target_sample_rate": int(vc.tgt_sr),
            "version": str(getattr(vc, "version", "")),
            "tts_type": str(model.tts_type),
            "index_rate": float(model.vc_args.index_rate),
            "f0_method": str(model.vc_args.f0_method),
            "default_f0_pitch": int(model.vc_args.f0_pitch),
        }

    @staticmethod
    def _select_variant(found: list, variant: str | None, model_dir: Path):
        """Pick the Model whose yaml stem matches ``variant`` exactly.

        Errors loudly when the requested variant yaml does not exist in the
        model directory — silently falling back to ``found[0]`` is exactly
        the round-3 P0 defect this replaces.
        """
        by_stem = {m.model_name: m for m in found}
        if variant:
            for stem, model in by_stem.items():
                if stem.lower() == variant.strip().lower():
                    return model
            raise ValueError(
                f"variant {variant!r} not found in {model_dir}: "
                f"available yaml configs: {sorted(by_stem)}"
            )
        # Default selection: the yaml matching the directory's name
        # ("Obama" -> obama.yaml) when present, else the single yaml.
        for stem, model in by_stem.items():
            if stem.lower() == model_dir.name.lower():
                return model
        if len(by_stem) == 1:
            return next(iter(by_stem.values()))
        raise ValueError(
            f"model directory {model_dir} has multiple yaml configs "
            f"({sorted(by_stem)}); specify a variant explicitly"
        )

    def health(self) -> Dict[str, Any]:
        stream_info = {
            "open": self._stream is not None,
            "stream_id": self._stream[0] if self._stream else None,
        }
        if _fake_mode():
            return {
                "loaded_model": self._model.model_name if self._model else None,
                "torch": None,
                "cuda_available": False,
                "device": "fake" if self._vc else None,
                "stream": stream_info,
                "rmvpe_available": _rmvpe_available(),
                "faiss": _faiss_available(),
            }
        import torch

        info: Dict[str, Any] = {
            "loaded_model": self._model.model_name if self._model else None,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(getattr(self._vc.config, "device", None)) if self._vc else None,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        info["stream"] = stream_info
        info["rmvpe_available"] = _rmvpe_available()
        info["faiss"] = _faiss_available()
        return info

    # -- streaming ------------------------------------------------------------

    def _drop_stream(self, abort: bool = True) -> bool:
        """Forget the open stream (aborting its engine). True if one was open."""
        stream, self._stream = self._stream, None
        if stream is None:
            return False
        if abort:
            try:
                stream[1].abort()
            except Exception:  # noqa: BLE001 - best effort
                pass
        return True

    def _make_engine(self, fields: dict) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory(self, fields)
        if _fake_mode():
            return _FakeEngine()
        import rvc_stream  # legacy/rvc_stream.py; script dir is on sys.path

        params = rvc_stream.StreamParams.from_fields(fields)
        return rvc_stream.open_engine(self._vc, self._vc_args, params)

    def stream_open(self, stream_id: str, fields: dict) -> Dict[str, Any]:
        if self._vc is None:
            raise RuntimeError("no model loaded; send load-model first")
        replaced = self._drop_stream(abort=True)
        with _timeout(STREAM_OPEN_TIMEOUT_SECONDS):
            engine = self._make_engine(fields)
        index_used = False
        if fields.get("use_index"):
            rate = fields.get("index_rate")
            if rate is None:
                rate = getattr(self._vc_args, "index_rate", 0.0)
            wanted = bool(rate)
            if _fake_mode() or self._engine_factory is not None:
                index_used = wanted
            else:
                index_used = wanted and getattr(self._vc, "_index", None) is not None
                if wanted and not index_used:
                    try:
                        engine.abort()
                    except Exception:  # noqa: BLE001
                        pass
                    raise RuntimeError("use_index requested but the faiss index could not be loaded")
        self._stream = (stream_id, engine, fields)
        return {
            "stream_id": stream_id,
            "input_sample_rate": STREAM_INPUT_SAMPLE_RATE,
            "output_sample_rate": STREAM_OUTPUT_SAMPLE_RATE,
            "sample_format": "f32le",
            "channels": 1,
            "min_input_samples_for_first_output": int(engine.min_input_samples_for_first_output),
            "replaced_stream": replaced,
            "model": self._model.model_name,
            "f0_method": "pm",
            "f0_algo": fields.get("f0_algo", "pm"),
            "index_used": index_used,
            "protect": fields.get("protect"),
        }

    @staticmethod
    def _encode_pcm(arr: Any) -> tuple:
        import numpy as np

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size and not bool(np.isfinite(arr).all()):
            raise RuntimeError("stream output contains NaN/Inf samples")
        data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        return base64.b64encode(data).decode("ascii"), int(arr.size)

    def stream_push(self, stream_id: str, pcm_b64: Any, samples: Any) -> Dict[str, Any]:
        """Raises StreamError(closed=...) on failure."""
        import numpy as np

        if self._stream is None:
            raise StreamError("no open stream", closed=False)
        if self._stream[0] != stream_id:
            raise StreamError("stream id mismatch", closed=False)
        engine = self._stream[1]
        try:
            if not isinstance(samples, int) or isinstance(samples, bool):
                raise ValueError("samples must be an integer")
            if not 0 <= samples <= STREAM_MAX_PUSH_SAMPLES:
                raise ValueError(f"samples must be within 0..{STREAM_MAX_PUSH_SAMPLES}")
            if not isinstance(pcm_b64, str):
                raise ValueError("pcm_b64 must be a base64 string")
            raw = base64.b64decode(pcm_b64, validate=True)
            if len(raw) % 4:
                raise ValueError("pcm byte length is not a multiple of 4")
            if len(raw) // 4 != samples:
                raise ValueError(f"samples={samples} does not match decoded length {len(raw) // 4}")
            pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            if pcm.size and not bool(np.isfinite(pcm).all()):
                raise ValueError("input contains NaN/Inf samples")
            start = time.monotonic()
            with _timeout(STREAM_TIMEOUT_SECONDS):
                out = engine.push(pcm)
            compute_ms = (time.monotonic() - start) * 1000.0
            out_b64, out_n = self._encode_pcm(out)
        except Exception as exc:
            self._drop_stream(abort=True)
            raise StreamError(f"stream_push failed: {exc}", closed=True) from exc
        return {
            "stream_id": stream_id,
            "pcm_b64": out_b64,
            "samples": out_n,
            "input_total": int(engine.input_total),
            "output_total": int(engine.output_total),
            "compute_ms": round(compute_ms, 3),
        }

    def stream_close(self, stream_id: str, flush: bool) -> Dict[str, Any]:
        if self._stream is None or self._stream[0] != stream_id:
            return {
                "stream_id": stream_id,
                "closed": False,
                "pcm_b64": "",
                "samples": 0,
                "input_total": 0,
                "output_total": 0,
                "compute_ms_total": 0.0,
                "steps": 0,
            }
        engine = self._stream[1]
        out_b64, out_n = "", 0
        if flush:
            try:
                with _timeout(STREAM_TIMEOUT_SECONDS):
                    out = engine.flush()
                out_b64, out_n = self._encode_pcm(out)
            except Exception as exc:
                self._drop_stream(abort=True)
                raise StreamError(f"stream_close failed: {exc}", closed=True) from exc
        result = {
            "stream_id": stream_id,
            "closed": True,
            "pcm_b64": out_b64,
            "samples": out_n,
            "input_total": int(engine.input_total),
            "output_total": int(engine.output_total),
            "compute_ms_total": round(float(engine.compute_s) * 1000.0, 3),
            "steps": int(engine.steps),
        }
        self._drop_stream(abort=not flush)
        return result

    # -- conversion ---------------------------------------------------------

    def convert(
        self,
        input_wav_path: str,
        output_wav_path: str,
        pitch_shift: Optional[int] = None,
        f0_method: Optional[str] = None,
        **opts: Any,
    ) -> Dict[str, Any]:
        """Convert one file. ``opts`` = the optional quality fields (see README, ``convert``):
        use_index, index_rate, index_nprobe, protect, rms_mix_rate, f0_post, auto_pitch."""
        if self._vc is None:
            raise RuntimeError("no model loaded; send load-model first")
        unknown = sorted(set(opts) - set(_CONVERT_QUALITY_KEYS))
        if unknown:
            raise ValueError(f"unknown convert options: {unknown}")
        quality = _validate_quality(opts)

        input_path = Path(input_wav_path)
        output_path = Path(output_wav_path)
        if not input_path.is_file():
            raise ValueError(f"input file not found: {input_path}")
        if not input_path.resolve().is_absolute() or not output_path.resolve().is_absolute():
            raise ValueError("input_wav_path and output_wav_path must be absolute paths")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        vc_args = self._vc_args
        if pitch_shift is not None or f0_method is not None:
            # Do not mutate the shared dataclass: build a shallow copy.
            import copy

            vc_args = copy.copy(vc_args)
        if pitch_shift is not None:
            vc_args.f0_pitch = int(pitch_shift)
        if f0_method is not None:
            method = str(f0_method)
            if method not in ALLOWED_F0_METHODS:
                raise ValueError(
                    f"invalid f0_method {method!r}; expected one of {sorted(ALLOWED_F0_METHODS)}"
                )
            vc_args.f0_method = method
        if str(vc_args.f0_method) == "rmvpe" and not _rmvpe_available():
            raise ValueError(
                "f0_method 'rmvpe' is unavailable: legacy/rvc_f0.py or the RMVPE weights are missing "
                "(health reports rmvpe_available)"
            )
        base_shift = int(vc_args.f0_pitch)
        eff_rate = quality.get("index_rate")
        if eff_rate is None:
            eff_rate = float(getattr(vc_args, "index_rate", 0.0) or 0.0)
        want_index = bool(quality.get("use_index")) and bool(eff_rate)

        # rvc_cli's VoiceConverter.convert_voice writes "<stem>_converted.wav"
        # next to the input; use a scratch copy of the input in a temp dir so
        # the caller's directories are never written to unexpectedly, then move
        # the result to the requested output path.
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rvc_worker_") as tmp:
            tmp_in = Path(tmp) / ("in" + input_path.suffix)
            shutil.copyfile(input_path, tmp_in)

            if _fake_mode():
                produced = Path(tmp) / "fake_converted.wav"
                shutil.copyfile(tmp_in, produced)
                info = {
                    "index_used": want_index,
                    "index_path": "fake.index" if want_index else None,
                    "pitch_shift_applied": base_shift,
                    "source_f0_median_hz": None,
                }
            else:
                call_kw = {k: v for k, v in quality.items() if k != "use_index" or v}
                with _timeout(CONVERT_TIMEOUT_SECONDS):
                    produced = self._vc.convert_voice(tmp_in, vc_args, **call_kw)
                last = getattr(self._vc, "last_info", None)
                info = dict(last) if isinstance(last, dict) and call_kw else {}
                info.setdefault("index_used", False)
                info.setdefault("index_path", None)
                info.setdefault("pitch_shift_applied", base_shift)
                info.setdefault("source_f0_median_hz", None)
            elapsed = time.monotonic() - start

            if produced is None:
                raise RuntimeError("rvc_cli VoiceConverter.convert_voice returned None (see worker stderr)")
            if want_index and not info["index_used"]:
                raise RuntimeError("use_index requested but the faiss index could not be loaded")

            produced = Path(produced)
            if not produced.is_file():
                raise RuntimeError(f"conversion did not produce expected file {produced}")

            shutil.move(str(produced), output_path)

        if not _fake_mode():
            self._validate_output(output_path)

        return {
            "output_wav_path": str(output_path),
            "elapsed_seconds": round(elapsed, 3),
            "model": self._model.model_name,
            "variant": self._model.model_name,
            "f0_method": str(vc_args.f0_method),
            "pitch_shift": (
                int(pitch_shift) if pitch_shift is not None else int(self._vc_args.f0_pitch)
            ),
            "pitch_shift_applied": int(info["pitch_shift_applied"]),
            "source_f0_median_hz": info["source_f0_median_hz"],
            "index_used": bool(info["index_used"]),
            "index_path": info["index_path"],
        }

    @staticmethod
    def _validate_output(path: Path) -> None:
        """Fail loudly on NaN/Inf or degenerate audio."""
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float64", always_2d=False)
        if data.size == 0:
            raise RuntimeError("output audio is empty")
        if not math.isfinite(float(data.min())) or not math.isfinite(float(data.max())):
            raise RuntimeError("output audio contains NaN/Inf samples")
        peak = float(abs(data).max())
        if peak == 0.0:
            raise RuntimeError("output audio is digital silence")
        # Output duration must roughly match a non-trivial conversion.
        duration = data.shape[0] / sr
        if duration <= 0:
            raise RuntimeError("output audio has non-positive duration")
        return None


def render_file(worker: Worker, input_wav: str, output_wav: str, **convert_fields: Any) -> Dict[str, Any]:
    """Engine entry for in-process callers: ``worker.convert`` with the optional quality fields."""
    return worker.convert(input_wav, output_wav, **convert_fields)


def _handle(worker: Worker, request: Dict[str, Any]) -> None:
    request_id = request.get("id")
    action = request.get("action")

    if action == "load-model":
        name = request.get("name")
        if not isinstance(name, str) or "/" in name or "\\" in name or name in ("", ".", ".."):
            _fail(request_id, f"invalid model name: {name!r}")
            return
        variant = request.get("variant")
        if variant is not None and (
            not isinstance(variant, str)
            or "/" in variant
            or "\\" in variant
            or variant in ("", ".", "..")
        ):
            _fail(request_id, f"invalid variant: {variant!r}")
            return
        try:
            info = worker.load_model(name, variant)
        except Exception as exc:  # noqa: BLE001 - report across IPC
            _fail(request_id, f"load-model failed: {exc}", traceback=traceback.format_exc(limit=8))
            return
        _reply(request_id, "ok", **info)
        return

    if action == "convert":
        input_wav = request.get("input_wav_path")
        output_wav = request.get("output_wav_path")
        pitch = request.get("pitch_shift")
        if not isinstance(input_wav, str) or not isinstance(output_wav, str):
            _fail(request_id, "convert requires input_wav_path and output_wav_path strings")
            return
        if pitch is not None and not isinstance(pitch, int):
            _fail(request_id, "pitch_shift must be an integer (semitones)")
            return
        f0_method = request.get("f0_method")
        if f0_method is not None and (
            not isinstance(f0_method, str) or f0_method not in ALLOWED_F0_METHODS
        ):
            _fail(
                request_id,
                f"invalid f0_method: {f0_method!r}; expected one of {sorted(ALLOWED_F0_METHODS)}",
            )
            return
        try:
            quality = _validate_quality(request)
        except ValueError as exc:
            _fail(request_id, f"invalid convert field: {exc}")
            return
        try:
            info = worker.convert(input_wav, output_wav, pitch, f0_method, **quality)
        except Exception as exc:  # noqa: BLE001 - report across IPC
            _fail(request_id, f"convert failed: {exc}", traceback=traceback.format_exc(limit=8))
            return
        _reply(request_id, "ok", **info)
        return

    if action == "health":
        try:
            info = worker.health()
        except Exception as exc:  # noqa: BLE001
            _fail(request_id, f"health failed: {exc}")
            return
        _reply(request_id, "ok", **info)
        return

    if action == "stream_open":
        stream_id = request.get("stream_id")
        if not isinstance(stream_id, str) or not STREAM_ID_RE.match(stream_id):
            _fail(request_id, f"invalid stream_id: {stream_id!r}", stream_closed=False)
            return
        fields: Dict[str, Any] = {}
        pitch = request.get("pitch_shift")
        if pitch is not None:
            if not isinstance(pitch, int) or isinstance(pitch, bool):
                _fail(request_id, "pitch_shift must be an integer (semitones)", stream_closed=False)
                return
            fields["pitch_shift"] = pitch
        f0_method = request.get("f0_method")
        if f0_method is not None and f0_method != "pm":
            _fail(
                request_id,
                f"invalid f0_method: {f0_method!r}; streaming supports only 'pm'",
                stream_closed=False,
            )
            return
        fields["f0_method"] = "pm"
        f0_algo = request.get("f0_algo")
        if f0_algo is not None:
            if f0_algo not in ("pm", "rmvpe"):
                _fail(request_id, f"invalid f0_algo: {f0_algo!r}; expected 'pm' or 'rmvpe'", stream_closed=False)
                return
            if f0_algo == "rmvpe" and not _rmvpe_available():
                _fail(
                    request_id,
                    "f0_algo 'rmvpe' is unavailable: legacy/rvc_f0.py or the RMVPE weights are missing",
                    stream_closed=False,
                )
                return
            fields["f0_algo"] = f0_algo
        try:
            fields.update(_validate_quality(request, stream=True))
        except ValueError as exc:
            _fail(request_id, f"invalid stream_open field: {exc}", stream_closed=False)
            return
        for key in STREAM_MS_FIELDS:
            value = request.get(key)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                _fail(request_id, f"{key} must be an integer (ms)", stream_closed=False)
                return
            fields[key] = value
        try:
            info = worker.stream_open(stream_id, fields)
        except Exception as exc:  # noqa: BLE001 - report across IPC
            _fail(
                request_id,
                f"stream_open failed: {exc}",
                traceback=traceback.format_exc(limit=8),
                stream_closed=worker._vc is not None,  # any other failure leaves no open stream
            )
            return
        _reply(request_id, "ok", **info)
        return

    if action == "stream_push":
        try:
            info = worker.stream_push(
                request.get("stream_id"), request.get("pcm_b64"), request.get("samples")
            )
        except StreamError as exc:
            _fail(request_id, str(exc), stream_closed=exc.closed)
            return
        _reply(request_id, "ok", **info)
        return

    if action == "stream_close":
        stream_id = request.get("stream_id")
        flush = request.get("flush")
        if not isinstance(stream_id, str) or not isinstance(flush, bool):
            _fail(
                request_id,
                "stream_close requires stream_id string and flush boolean",
                stream_closed=False,
            )
            return
        try:
            info = worker.stream_close(stream_id, flush)
        except StreamError as exc:
            _fail(request_id, str(exc), stream_closed=exc.closed)
            return
        _reply(request_id, "ok", **info)
        return

    _fail(request_id, f"unknown action: {action!r}")


def main() -> int:
    global _PROTOCOL_OUT
    # Legacy rvc_cli code prints logs/banners to stdout; keep fd 1 clean for
    # the JSON protocol and push all incidental output to stderr.
    _PROTOCOL_OUT = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    worker = Worker()
    _reply(None, "ok", action="ready", note="worker started; send load-model")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except Exception as exc:  # noqa: BLE001
            _fail(None, f"malformed request: {exc}")
            continue

        action = request.get("action")
        if action == "shutdown":
            _reply(request.get("id"), "ok", action="shutdown")
            return 0

        try:
            _handle(worker, request)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            _fail(request.get("id"), f"internal error: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
