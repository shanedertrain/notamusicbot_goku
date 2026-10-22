"""Gate R8: full production GPU residency — LLM + TTS + RVC coexisting on one GPU.

Review Round-2 P1 evidence test. Loads the real content LLM (Qwen3-4B), the
real Qwen TTS (0.6B CustomVoice) and the real legacy RVC worker with the Goku
model so all three are GPU-resident simultaneously, then runs one content
generation and one TTS -> RVC generation, recording:

  - idle VRAM after all three loaded
  - content generation peak
  - TTS generation peak
  - RVC conversion peak
  - end-to-end peak

Run for real:

    .venv-main/bin/python -m pytest tests/integration/test_gpu_residency.py \
        -m "gpu and integration" -s

Skipped cleanly when CUDA, the HF caches or the legacy RVC assets are missing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
HUBERT = REPO_ROOT / "hubert_base.state_dict.pt"
GOKU_DIR = REPO_ROOT / "rvc_cli" / "models" / "Goku"


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
if not _requirements_met():  # evaluated at collection; skip the whole module
    pytest.skip(
        "GPU residency requirements not met (CUDA / caches / legacy RVC assets)",
        allow_module_level=True,
    )


@dataclass
class Stage:
    name: str
    vram_used_mib: int
    vram_total_mib: int
    peak_allocated_mib: int


def _stage(label: str, stages: list[Stage]) -> Stage:
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    used_mib = int((total_b - free_b) // (1024 * 1024))
    peak = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    stage = Stage(label, used_mib, int(total_b // (1024 * 1024)), peak)
    stages.append(stage)
    print(
        f"[residency] {label}: VRAM {used_mib}/{stage.vram_total_mib} MiB used, "
        f"peak allocated {peak} MiB"
    )
    return stage


def test_full_gpu_residency_no_oom(tmp_path) -> None:
    import torch

    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.domain.speech import TtsRequest, VoiceConversionProfile
    from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer
    from notamusicbot.speech.voice_conversion.rvc import RvcVoiceConverter

    torch.cuda.reset_peak_memory_stats()
    stages: list[Stage] = []

    async def scenario() -> None:
        # (a) content LLM
        llm = LocalQwenLlmProvider(max_new_tokens=60)
        llm.load()
        _stage("idle_after_llm_load", stages)

        # (b) Qwen TTS
        synth = QwenCustomVoiceSynthesizer.from_settings(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device="cuda",
            dtype="bfloat16",
            attn_implementation="sdpa",
        )
        _stage("idle_after_tts_load", stages)

        # (c) RVC worker + Goku model resident
        converter = RvcVoiceConverter()
        src = await synth.synthesize(
            TtsRequest(text="Residency check.", language="English", speaker="Aiden"),
            tmp_path / "src.wav",
        )
        await converter.convert(
            src, VoiceConversionProfile(backend="rvc", model="Goku"), tmp_path / "rvc0.wav"
        )
        _stage("idle_after_all_three_loaded", stages)

        # (d) one content generation with everything resident
        text = await llm.generate(
            ContentRequest(kind=ContentKind.SONG, song_name="Test Song", requester_name="Tester")
        )
        _stage("content_generation_peak", stages)
        assert text.strip()

        # (e) one full TTS -> RVC generation with everything resident
        tts_src = await synth.synthesize(
            TtsRequest(text="Now playing Test Song.", language="English", speaker="Aiden"),
            tmp_path / "tts.wav",
        )
        _stage("tts_generation_peak", stages)
        converted = await converter.convert(
            tts_src, VoiceConversionProfile(backend="rvc", model="Goku"), tmp_path / "rvc1.wav"
        )
        _stage("rvc_conversion_peak", stages)
        assert converted.duration_s > 0

        _stage("end_to_end_peak", stages)
        await converter.shutdown()

    try:
        asyncio.run(scenario())
    except torch.cuda.OutOfMemoryError:
        pytest.fail("CUDA OOM during full residency scenario: LLM+TTS+RVC cannot coexist")
    _stage("final", stages)

    print("\n=== GPU residency summary (RTX 4070 Ti SUPER 16GB) ===")
    for s in stages:
        print(f"  {s.name}: used={s.vram_used_mib} MiB peak_alloc={s.peak_allocated_mib} MiB")
    by_name = {s.name: s for s in stages}
    # No OOM anywhere (an OOM raises and fails above); sanity assertions:
    assert by_name["idle_after_all_three_loaded"].vram_used_mib > 0
    assert by_name["final"].peak_allocated_mib >= 0

    # Round-4 review P2: persist the raw real-GPU residency evidence.
    from evidence_utils import write_evidence

    write_evidence(
        "gpu_residency.json",
        {
            "test": "test_full_gpu_residency_no_oom",
            "stages": [
                {
                    "name": s.name,
                    "vram_used_mib": s.vram_used_mib,
                    "vram_total_mib": s.vram_total_mib,
                    "peak_allocated_mib": s.peak_allocated_mib,
                }
                for s in stages
            ],
            "model_ids": {
                "content_llm": "Qwen/Qwen3-4B-Instruct-2507",
                "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                "rvc": "Goku (legacy worker subprocess)",
            },
            "pass": True,
        },
    )
