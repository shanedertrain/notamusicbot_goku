#!/usr/bin/env python
"""Legacy RVC voice-conversion worker.

Long-running process that must be executed with the legacy Python 3.10
environment (repo-root ``.venv``) because it imports the ``rvc_cli``
submodule, which needs fairseq 0.12.2 / torch 2.5.0+cu121.

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
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_CLI_DIR = REPO_ROOT / "rvc_cli"

# The worker is a legacy isolated component: it manages its own sys.path and
# working directory instead of relying on the caller's environment.
sys.path.insert(0, str(RVC_CLI_DIR))
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # voice_converter resolves "hubert_base.pt" relative to cwd

CONVERT_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_CONVERT_TIMEOUT", "120"))
LOAD_TIMEOUT_SECONDS = float(os.environ.get("RVC_WORKER_LOAD_TIMEOUT", "300"))

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

    # -- model loading ----------------------------------------------------

    def load_model(self, name: str, variant: Optional[str] = None) -> Dict[str, Any]:
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
        import torch

        info: Dict[str, Any] = {
            "loaded_model": self._model.model_name if self._model else None,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(getattr(self._vc.config, "device", None)) if self._vc else None,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        return info

    # -- conversion ---------------------------------------------------------

    def convert(
        self,
        input_wav_path: str,
        output_wav_path: str,
        pitch_shift: Optional[int] = None,
        f0_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._vc is None:
            raise RuntimeError("no model loaded; send load-model first")

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

        # rvc_cli's VoiceConverter.convert_voice writes "<stem>_converted.wav"
        # next to the input; use a scratch copy of the input in a temp dir so
        # the caller's directories are never written to unexpectedly, then move
        # the result to the requested output path.
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rvc_worker_") as tmp:
            tmp_in = Path(tmp) / ("in" + input_path.suffix)
            shutil.copyfile(input_path, tmp_in)

            with _timeout(CONVERT_TIMEOUT_SECONDS):
                produced = self._vc.convert_voice(tmp_in, vc_args)
            elapsed = time.monotonic() - start

            if produced is None:
                raise RuntimeError("rvc_cli VoiceConverter.convert_voice returned None (see worker stderr)")

            produced = Path(produced)
            if not produced.is_file():
                raise RuntimeError(f"conversion did not produce expected file {produced}")

            shutil.move(str(produced), output_path)

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
            info = worker.convert(input_wav, output_wav, pitch, f0_method)
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
