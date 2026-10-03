"""Review Round-3 P1: real concurrent GPU arbitration test.

With all three models resident (local Qwen3-4B content LLM, Qwen3-TTS 0.6B
CustomVoice, RVC worker subprocess with the Goku model), submits an LLM
generation and a TTS -> RVC generation as concurrent asyncio tasks through the
SHARED :class:`InferenceCoordinator` and asserts — over recorded per-stage
start/end timestamps, repeated across >= 3 iterations — that the main-process
GPU sections never overlapped.

Run for real:

    .venv-main/bin/python -m pytest tests/integration/test_gpu_arbitration.py \
        -m "gpu and integration" -s
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
HUBERT = REPO_ROOT / "hubert_base.pt"
GOKU_DIR = REPO_ROOT / "rvc_cli" / "models" / "Goku"
ITERATIONS = 3


def _requirements_met() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(
        torch.cuda.is_available()
        and LEGACY_PYTHON.is_file()
        and HUBERT.is_file()
        and GOKU_DIR.is_dir()
    )


pytest.importorskip("torch", reason="torch not installed")
if not _requirements_met():
    pytest.skip(
        "GPU arbitration requirements not met (CUDA / caches / legacy RVC assets)",
        allow_module_level=True,
    )


def test_shared_coordinator_serializes_llm_and_tts_rvc(tmp_path) -> None:
    import torch

    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.domain.speech import TtsRequest, VoiceConversionProfile
    from notamusicbot.speech.coordinator import InferenceCoordinator
    from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
    from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

    torch.cuda.reset_peak_memory_stats()
    coord = InferenceCoordinator()  # the shared slot under test
    timeline: list[tuple[str, float, float]] = []  # (stage, start, end)

    async def scenario() -> None:
        llm = LocalQwenLlmProvider(max_new_tokens=60, gpu_coordinator=coord)
        llm.load()
        synth = QwenCustomVoiceSynthesizer.from_settings(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device="cuda",
            dtype="bfloat16",
            attn_implementation="sdpa",
        )
        converter = RvcVoiceConverter()
        try:
            for i in range(ITERATIONS):
                prompt = ContentRequest(
                    kind=ContentKind.SONG, song_name=f"Arbitration {i}", requester_name="Tester"
                )
                tts_req = TtsRequest(
                    text=f"Arbitration iteration {i}.", language="English", speaker="Aiden"
                )
                conv = VoiceConversionProfile(backend="rvc", model="Goku")

                async def llm_task(prompt=prompt, i=i) -> None:
                    s = time.perf_counter()
                    text = await llm.generate(prompt)
                    e = time.perf_counter()
                    assert text.strip()
                    timeline.append((f"iter{i}:llm", s, e))

                async def tts_vc_task(i=i, tts_req=tts_req, conv=conv) -> None:
                    async with coord.slot():
                        s = time.perf_counter()
                        src = await synth.synthesize(tts_req, tmp_path / f"tts{i}.wav")
                        tts_e = time.perf_counter()
                        out = await converter.convert(src, conv, tmp_path / f"rvc{i}.wav")
                        e = time.perf_counter()
                        assert out.duration_s > 0
                    timeline.append((f"iter{i}:tts", s, tts_e))
                    timeline.append((f"iter{i}:rvc", tts_e, e))

                await asyncio.gather(llm_task(), tts_vc_task())
        finally:
            await converter.shutdown()

    asyncio.run(scenario())

    print(f"\n=== shared-GPU-coordinator arbitration timeline ({ITERATIONS} iterations) ===")
    for name, s, e in sorted(timeline, key=lambda t: t[1]):
        print(f"  {name:14s} start={s:.4f} end={e:.4f} dur={e - s:.4f}s")

    # Core assertion: no two main-process GPU sections overlap in time.
    ordered = sorted(timeline, key=lambda t: t[1])
    for (_n1, _s1, e1), (n2, s2, _e2) in zip(ordered, ordered[1:], strict=False):
        assert s2 >= e1 - 1e-4, (
            f"GPU sections overlapped: {n2} started at {s2:.4f} before previous ended at {e1:.4f}"
        )
    # Every iteration actually produced an LLM, TTS and RVC section.
    assert {n.split(":")[1] for n, _s, _e in timeline} == {"llm", "tts", "rvc"}
    assert len(timeline) == ITERATIONS * 3
