"""Gate R8: full production GPU residency — LLM + TTS clone model on one GPU.

Review Round-2 P1 evidence test. Loads the real content LLM (Qwen3-4B) and the
real Qwen3-TTS 0.6B Base clone model so both are GPU-resident simultaneously,
then runs one content generation and one clone-TTS generation (with a real
voice-clone prompt), recording:

  - idle VRAM after both loaded
  - content generation peak
  - TTS generation peak
  - end-to-end peak

Run for real:

    .venv-main/bin/python -m pytest tests/integration/test_gpu_residency.py \
        -m "gpu and integration" -s

Skipped cleanly when CUDA, the HF caches or the voice reference are missing.
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
REFERENCE = REPO_ROOT / "config" / "voices" / "refs" / "obama.wav"


def _requirements_met() -> bool:
    try:
        import torch

        from notamusicbot.content.llm import resolve_local_model_path
    except ImportError:
        return False
    if not torch.cuda.is_available() or not REFERENCE.is_file():
        return False
    try:
        resolve_local_model_path("Qwen/Qwen3-4B-Instruct-2507")
        resolve_local_model_path("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    except Exception:
        return False
    return True


pytest.importorskip("torch", reason="torch not installed")
if not _requirements_met():  # evaluated at collection; skip the whole module
    pytest.skip(
        "GPU residency requirements not met (CUDA / model caches / voice reference)",
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


def _clone_request(text: str):
    from notamusicbot.config import load_app_settings
    from notamusicbot.domain.voice_profiles import load_voice_profiles, tts_request_for

    settings = load_app_settings()
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles.get("obama") or next(iter(profiles.values()))
    return tts_request_for(profile, text)


def test_full_gpu_residency_no_oom(tmp_path) -> None:
    import torch

    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

    torch.cuda.reset_peak_memory_stats()
    stages: list[Stage] = []

    async def scenario() -> None:
        # (a) content LLM
        llm = LocalQwenLlmProvider(max_new_tokens=60)
        llm.load()
        _stage("idle_after_llm_load", stages)

        # (b) Qwen TTS clone model
        synth = QwenCloneSynthesizer.from_settings(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device="cuda",
            dtype="bfloat16",
            attn_implementation="sdpa",
        )
        _stage("idle_after_tts_load", stages)

        # (c) one clone prompt built + one generation with both models resident
        src = await synth.synthesize(_clone_request("Residency check."), tmp_path / "src.wav")
        _stage("idle_after_clone_prompt_built", stages)
        assert src.duration_s > 0

        # (d) one content generation with everything resident
        text = await llm.generate(
            ContentRequest(kind=ContentKind.SONG, song_name="Test Song", requester_name="Tester")
        )
        _stage("content_generation_peak", stages)
        assert text.strip()

        # (e) one more clone generation (prompt cache hit) with everything resident
        clone = await synth.synthesize(
            _clone_request("Now playing Test Song."), tmp_path / "tts.wav"
        )
        _stage("tts_generation_peak", stages)
        assert clone.duration_s > 0

        _stage("end_to_end_peak", stages)
        await synth.shutdown()

    try:
        asyncio.run(scenario())
    except torch.cuda.OutOfMemoryError:
        pytest.fail("CUDA OOM during full residency scenario: LLM+TTS cannot coexist")
    _stage("final", stages)

    print("\n=== GPU residency summary (RTX 4070 Ti SUPER 16GB) ===")
    for s in stages:
        print(f"  {s.name}: used={s.vram_used_mib} MiB peak_alloc={s.peak_allocated_mib} MiB")
    by_name = {s.name: s for s in stages}
    # No OOM anywhere (an OOM raises and fails above); sanity assertions:
    assert by_name["idle_after_tts_load"].vram_used_mib > 0
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
                "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            },
            "voice_reference": str(REFERENCE),
            "pass": True,
        },
    )
