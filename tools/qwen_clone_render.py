"""Render voice-clone samples from reference audio (evidence tool, not production).

Drives Qwen3-TTS Base zero-shot cloning from a JSON manifest so the same harness
can be reused for any character voice:

    {
      "refs":  [{"name": "obama_auto_14s", "path": "var/.../ref.wav", "text": "..."}],
      "lines": [{"id": "L1", "text": "Look, let me be clear. ..."}],
      "runs":  [{"ref": "obama_auto_14s", "line": "L1", "modes": ["xvec", "icl"]}]
    }

``xvec`` = speaker-embedding-only clone (``ref_text`` not needed); ``icl`` =
in-context clone (reference audio + transcript; better fidelity, copies more of
the reference prosody). Prompts are built once per (ref, mode) and reused, which
is the documented production pattern.

Outputs ``<out>/<ref>_<line>_<mode>.wav``, ``timings.json`` and ``index.json``.
With ``--asr`` each render is round-tripped through the locally cached Parakeet
model so truncation/hallucination shows up as text, not as a guess.

    .venv-main/bin/python tools/qwen_clone_render.py \
        --manifest var/obama_voice/manifest.json --out var/obama_voice/renders --asr
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def load_24k_mono(path: Path) -> np.ndarray:
    """Decode audio to mono float32 24 kHz (the clone model's reference rate)."""
    import soxr

    wav, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if np.asarray(wav).ndim > 1:
        wav = np.asarray(wav).mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr != 24000:
        wav = soxr.resample(wav, sr, 24000)
    return np.ascontiguousarray(wav, dtype=np.float32)


def resolve_model_path(model_id: str) -> str:
    """Local snapshot path only (the runtime must not download)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=True)


class _Asr:
    """Optional Parakeet round-trip for render verification."""

    def __init__(self) -> None:
        import onnx_asr
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        files = [
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "nemo128.onnx",
            "vocab.txt",
            "config.json",
        ]
        snap = snapshot_download(
            "istupakov/parakeet-tdt-0.6b-v2-onnx", local_files_only=True, allow_patterns=files
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self._model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            snap,
            quantization="int8",
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, path: Path) -> str:
        import soxr

        wav, sr = sf.read(str(path), always_2d=False, dtype="float32")
        wav = (
            np.asarray(wav).reshape(-1)
            if np.asarray(wav).ndim == 1
            else np.asarray(wav).mean(axis=1)
        )
        wav16 = soxr.resample(np.asarray(wav, dtype=np.float32), sr, 16000)
        return str(self._model.recognize(wav16, sample_rate=16000)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--asr", action="store_true", help="round-trip renders through Parakeet")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    refs = {r["name"]: r for r in manifest["refs"]}
    lines = {line["id"]: line for line in manifest["lines"]}
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from qwen_tts import Qwen3TTSModel

    model_path = resolve_model_path(args.model)
    print(f"loading {model_path}", flush=True)
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.2f}s", flush=True)

    asr = _Asr() if args.asr else None
    prompts: dict[tuple[str, str], tuple[object, float]] = {}
    rows: list[dict[str, object]] = []

    def prompt_for(ref_name: str, mode: str) -> tuple[object, float]:
        key = (ref_name, mode)
        if key in prompts:
            return prompts[key], 0.0
        ref = refs[ref_name]
        wav = load_24k_mono(Path(ref["path"]))
        started = time.perf_counter()
        if mode == "xvec":
            items = model.create_voice_clone_prompt(ref_audio=(wav, 24000), x_vector_only_mode=True)
        else:
            items = model.create_voice_clone_prompt(
                ref_audio=(wav, 24000),
                ref_text=str(ref["text"]),
                x_vector_only_mode=False,
            )
        elapsed = time.perf_counter() - started
        prompts[key] = (items, elapsed)
        return prompts[key], elapsed

    for run in manifest["runs"]:
        for mode in run["modes"]:
            (items, _), build_s = prompt_for(run["ref"], mode)
            text = str(lines[run["line"]]["text"])
            name = f"{run['ref']}_{run['line']}_{mode}"
            started = time.perf_counter()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=items,  # type: ignore[arg-type]
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.perf_counter() - started
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            dur = len(wav) / int(sr)
            path = out / f"{name}.wav"
            sf.write(str(path), wav, int(sr), subtype="PCM_16")
            row: dict[str, object] = {
                "name": name,
                "ref": run["ref"],
                "line": run["line"],
                "mode": mode,
                "prompt_build_s": round(build_s, 2),
                "gen_s": round(elapsed, 2),
                "audio_s": round(dur, 2),
                "rtf": round(elapsed / dur, 2) if dur else None,
            }
            if asr is not None:
                row["asr"] = asr(path)
            rows.append(row)
            print(f"{name}: {row['gen_s']}s for {row['audio_s']}s (rtf {row['rtf']})", flush=True)

    summary = {
        "model": args.model,
        "model_load_s": round(load_s, 2),
        "refs": {name: {"path": r["path"], "text": r.get("text", "")} for name, r in refs.items()},
        "lines": {name: line["text"] for name, line in lines.items()},
        "runs": rows,
    }
    (out / "timings.json").write_text(json.dumps(summary, indent=2))
    (out / "index.json").write_text(json.dumps(rows, indent=2))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
