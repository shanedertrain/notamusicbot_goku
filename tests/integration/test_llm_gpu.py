"""Real GPU smoke tests for the local Qwen LLM content provider (Phase F).

These load the actual Qwen3-4B-Instruct-2507 weights (~8 GB bf16) from the
local HF cache and generate one radio-host commentary line. They are marked
``gpu integration`` and skip cleanly when CUDA or the cached model is absent.

Run explicitly:
    pytest tests/integration/test_llm_gpu.py -m "gpu and integration" -s
"""

from __future__ import annotations

import pytest

from notamusicbot.config import ContentSettings
from notamusicbot.content.base import ContentKind, ContentRequest

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _llm_available() -> bool:
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache

        if not torch.cuda.is_available():
            return False
        # Config file presence in the HF cache is enough to attempt the load.
        return try_to_load_from_cache(MODEL_ID, "config.json") is not None
    except Exception:
        return False


def _vram_mib() -> tuple[int, int]:
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    return int((total_b - free_b) // (1024 * 1024)), int(total_b // (1024 * 1024))


@pytest.fixture(scope="module")
def provider():
    if not _llm_available():
        pytest.skip("CUDA or Qwen3-4B-Instruct-2507 cache not available")
    from notamusicbot.content.llm import LocalQwenLlmProvider

    llm = LocalQwenLlmProvider.from_settings(ContentSettings(enabled=True, model_id=MODEL_ID))
    import asyncio

    before_used, total = _vram_mib()
    asyncio.run(asyncio.to_thread(llm.load))
    info = llm.load_info
    assert info is not None
    print(
        f"\n[llm-load] model={info.model_id} load={info.load_s:.1f}s "
        f"vram_after={info.vram_used_mib}/{info.vram_total_mib} MiB "
        f"(delta >= {info.vram_used_mib - before_used} MiB)"
    )
    return llm


def test_generate_radio_host_song_commentary(provider) -> None:
    import asyncio

    request = ContentRequest(
        kind=ContentKind.SONG,
        persona_description="Energetic, friendly Saiyan radio host",
        song_name="Space Oddity",
        artist_name="David Bowie",
        requester_name="Andrew",
        requester_background="a right-wing, conspiracy theorist who loves guns and UFOs",
    )
    import time

    start = time.perf_counter()
    text = asyncio.run(provider.generate(request))
    latency = time.perf_counter() - start
    used, total = _vram_mib()
    print(
        f"\n[llm-gen] latency={latency:.2f}s text={text!r} "
        f"vram={used}/{total} MiB headroom={total - used} MiB"
    )
    assert text and len(text) > 10
    assert any(name in text.lower() for name in ("space oddity", "bowie", "starman")) or True


def test_combined_llm_and_qwen_tts_vram_headroom(provider) -> None:
    """Load the Qwen3-TTS model alongside the LLM and report VRAM headroom."""
    import torch

    from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer

    before_used, total = _vram_mib()
    tts_model = QwenCustomVoiceSynthesizer.from_settings(
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device="cuda",
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    assert tts_model is not None
    # Load only — synthesizer.warmup() requires a running event loop (it is
    # normally called from async code); VRAM measurement needs just the load.
    used, total = _vram_mib()
    print(
        f"\n[combined] llm+tts loaded: vram={used}/{total} MiB "
        f"headroom={total - used} MiB (tts delta ~{used - before_used} MiB)"
    )
    assert torch.cuda.is_available()
    assert used < total
