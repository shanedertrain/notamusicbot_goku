"""Probe: Qwen3-TTS Base zero-shot voice cloning vs the current CustomVoice path.

Question this answers (owner, 2026-10-30): can the existing Qwen voice model
replace the RVC post-conversion stage by cloning the target voice from a short
reference clip? This tool is evidence, not production code: it downloads the
Base checkpoint if missing, builds a voice-clone prompt from a reference clip,
renders a few sentences with and without the reference transcript (ICL), and
renders the same sentences with the installed CustomVoice model for a timing
baseline.

Outputs (gitignored): ``var/qwen_clone_probe/*.wav`` and ``timings.json``.

    .venv-main/bin/python tools/qwen_clone_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "var" / "qwen_clone_probe"
BASE_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CV_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
ASR_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx"

TEXTS = {
    "a": "Hey, it's Goku. Welcome back to Galactic Beats, where the playlist never stops.",
    "b": "That last track was insane. Let me tell you what I think about it.",
}


def _load_24k_mono(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 24 kHz (soxr; falls back to ffmpeg)."""
    import soxr

    try:
        wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
    except Exception:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    tmp.name,
                ],
                check=True,
            )
            wav, sr = sf.read(tmp.name, always_2d=False, dtype="float32")
    if sr != 24000:
        wav = soxr.resample(np.asarray(wav, dtype=np.float32), sr, 24000)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) or 1.0
    return (wav / peak * 0.95).astype(np.float32)


def _transcribe(path: Path, sr: int = 16000) -> str:
    """Best-effort reference transcript via the locally cached Parakeet int8 model."""
    import onnx_asr
    import onnxruntime as ort
    import soxr
    from huggingface_hub import snapshot_download

    files = [
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
        "nemo128.onnx",
        "vocab.txt",
        "config.json",
    ]
    snap = snapshot_download(ASR_ID, local_files_only=True, allow_patterns=files)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v2",
        snap,
        quantization="int8",
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    wav, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav16 = soxr.resample(np.asarray(wav, dtype=np.float32), file_sr, sr)
    return str(model.recognize(wav16, sample_rate=sr)).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    print("[1/6] Base checkpoint", flush=True)
    t0 = time.perf_counter()
    snapshot_download(BASE_ID)
    print(f"      downloaded/verified in {time.perf_counter() - t0:.1f}s", flush=True)

    import torch
    from qwen_tts import Qwen3TTSModel

    print("[2/6] load Base 0.6B", flush=True)
    t0 = time.perf_counter()
    base = Qwen3TTSModel.from_pretrained(
        BASE_ID, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    load_base_s = time.perf_counter() - t0
    print(f"      {load_base_s:.1f}s", flush=True)

    # Reference candidates: the accepted Goku render kept by the RVC quality lab and
    # the original source clip shipped with the repo.
    refs = {
        "refA": Path(
            "/home/cbash23/.cache/listen-speed/rvc-quality/samples/HeyGoku/"
            "00_REFERENCE_e2e_goku.wav"
        ),
        "refB": REPO_ROOT / "input" / "heyitsmegoku.mp3",
    }
    ref_wavs: dict[str, np.ndarray] = {}
    for name, path in refs.items():
        if not path.is_file():
            print(f"      !! missing {path}", flush=True)
            continue
        wav = _load_24k_mono(path)
        ref_wavs[name] = wav
        sf.write(OUT / f"{name}_24k.wav", wav, 24000, subtype="PCM_16")
        print(f"      {name}: {len(wav) / 24000:.2f}s from {path.name}", flush=True)

    print("[3/6] reference transcripts (Parakeet int8 CPU)", flush=True)
    ref_texts: dict[str, str] = {}
    for name in refs:
        if name not in ref_wavs:
            continue
        try:
            text = _transcribe(OUT / f"{name}_24k.wav")
            ref_texts[name] = text
            print(f"      {name}: {text!r}", flush=True)
        except Exception as exc:  # noqa: BLE001 - probe must continue
            print(f"      {name}: ASR unavailable ({type(exc).__name__}: {exc})", flush=True)

    timings: dict[str, object] = {
        "base_load_s": round(load_base_s, 2),
        "refs": {k: round(len(v) / 24000, 2) for k, v in ref_wavs.items()},
        "ref_texts": ref_texts,
        "runs": [],
    }

    def render(name: str, fn) -> None:
        t0 = time.perf_counter()
        wavs, sr = fn()
        elapsed = time.perf_counter() - t0
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        dur = len(wav) / sr
        sf.write(OUT / f"{name}.wav", wav, int(sr), subtype="PCM_16")
        row = {
            "name": name,
            "seconds": round(elapsed, 2),
            "audio_s": round(dur, 2),
            "rtf": round(elapsed / dur, 2) if dur else None,
            "sample_rate": int(sr),
        }
        timings["runs"].append(row)  # type: ignore[union-attr]
        print(f"      {name}: {elapsed:.2f}s for {dur:.2f}s audio (rtf {row['rtf']})", flush=True)

    print("[4/6] clone prompts", flush=True)
    prompts = {}
    for name, wav in ref_wavs.items():
        t0 = time.perf_counter()
        try:
            prompts[name] = base.create_voice_clone_prompt(
                ref_audio=(wav, 24000), x_vector_only_mode=True
            )
            print(f"      {name} xvec prompt in {time.perf_counter() - t0:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"      {name} xvec prompt failed: {type(exc).__name__}: {exc}", flush=True)
    # ICL prompts need the reference transcript.
    icl_prompts = {}
    for name, wav in ref_wavs.items():
        text = ref_texts.get(name)
        if not text:
            continue
        try:
            icl_prompts[name] = base.create_voice_clone_prompt(
                ref_audio=(wav, 24000), ref_text=text, x_vector_only_mode=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"      {name} icl prompt failed: {type(exc).__name__}: {exc}", flush=True)

    print("[5/6] clone renders", flush=True)
    for ref_name, prompt in prompts.items():
        for text_name, text in TEXTS.items():
            render(
                f"clone_{ref_name}_xvec_{text_name}",
                lambda p=prompt, t=text, m=base: m.generate_voice_clone(
                    text=t, language="English", voice_clone_prompt=p, max_new_tokens=400
                ),
            )
        if ref_name in icl_prompts:
            render(
                f"clone_{ref_name}_icl_a",
                lambda p=icl_prompts[ref_name], m=base: m.generate_voice_clone(
                    text=TEXTS["a"], language="English", voice_clone_prompt=p, max_new_tokens=400
                ),
            )

    del base
    torch.cuda.empty_cache()

    print("[6/6] CustomVoice baseline (installed checkpoint)", flush=True)
    t0 = time.perf_counter()
    cv = Qwen3TTSModel.from_pretrained(
        CV_ID, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    print(f"      load {time.perf_counter() - t0:.1f}s", flush=True)
    for text_name, text in TEXTS.items():
        render(
            f"customvoice_ryan_{text_name}",
            lambda t=text: cv.generate_custom_voice(
                text=t, language="English", speaker="Ryan", max_new_tokens=400
            ),
        )

    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
