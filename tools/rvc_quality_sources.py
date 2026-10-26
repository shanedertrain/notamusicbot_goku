#!/usr/bin/env python
"""Generate alternative Qwen3-TTS source-speaker clips for the RVC quality lab.

Run with the qwen-fast venv (offline, GPU, so under the shared GPU lock)::

    HF_HUB_OFFLINE=1 flock /home/cbash23/.cache/listen-speed/gpu.lock \\
        /home/cbash23/.cache/listen-speed/qwen-fast/venv/bin/python \\
        tools/rvc_quality_sources.py --speakers aiden,eric,dylan

Writes ``<out>/sources/<speaker>/s{0..5}.wav`` (24 kHz mono PCM16) from the six sentences of
``qwen-fast/scripts/common.py`` (SENT), using the same call as ``scripts/fast.py``
(``FasterQwen3TTS.generate_custom_voice``, language English, seed 1234 per sentence). The Ryan
clips already exist as ``qwen-fast/samples/fast_nonstream_s{0..5}.wav`` and are used as speaker
``ryan`` directly by the lab, so ``ryan`` is not generated here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

QWEN_BASE = Path("/home/cbash23/.cache/listen-speed/qwen-fast")
DEFAULT_OUT = Path("/home/cbash23/.cache/listen-speed/rvc-quality")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SEED = 1234


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speakers", default="aiden,eric,dylan")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    sys.path.insert(0, str(QWEN_BASE / "scripts"))
    import numpy as np
    import soundfile as sf
    import torch
    from common import SENT
    from faster_qwen3_tts import FasterQwen3TTS
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, local_files_only=True)
    model = FasterQwen3TTS.from_pretrained(
        path,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_seq_len=1024,
    )
    known = sorted(model.model.model.config.talker_config.spk_id)
    print("supported speakers:", ", ".join(known), flush=True)
    wanted = [s.strip().lower() for s in args.speakers.split(",") if s.strip()]
    todo = []
    for spk in wanted:
        if spk not in known:
            print(f"skip unknown speaker {spk!r}", flush=True)
        else:
            todo.append(spk)
    if not todo:
        return 0

    model.warmup()
    model.generate_custom_voice(
        text="Warmup, one two three.", speaker=todo[0], language="English", max_new_tokens=60
    )
    for spk in todo:
        out_dir = Path(args.out) / "sources" / spk
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, sentence in enumerate(SENT):
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            wav, sr = model.generate_custom_voice(
                text=sentence,
                speaker=spk,
                language="English",
                max_new_tokens=args.max_new_tokens,
            )
            wav = np.asarray(wav[0], dtype=np.float32)
            dest = out_dir / f"s{i}.wav"
            sf.write(str(dest), wav, int(sr), subtype="PCM_16")
            print(
                f"{spk} s{i} {len(wav) / sr:.2f}s sr={sr} "
                f"in {time.perf_counter() - t0:.2f}s -> {dest}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
