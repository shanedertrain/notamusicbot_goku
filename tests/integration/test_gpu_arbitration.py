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
HUBERT = REPO_ROOT / "hubert_base.state_dict.pt"
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


class _LeaseRecorder(set):
    """Records exact lease-hold intervals from the coordinator holder registry.

    ``InferenceCoordinator`` adds a task to ``_holders`` immediately after it
    acquires the GPU semaphore and discards it immediately before releasing, so
    each add/discard pair is the precise lease interval, for every entry point
    (``slot()``, ``run_blocking``, ``run_detached``). The previous approach
    sampled the registry every 1 ms; a fast run could miss the release and
    re-acquire between two iterations, merge two busy periods and fail the
    "at least one busy period per iteration" assertion on scheduler timing.
    """

    def __init__(self, intervals: list[tuple[float, float]]) -> None:
        super().__init__()
        self._intervals = intervals
        self._started: dict[object, float] = {}

    def add(self, task: object) -> None:
        super().add(task)  # type: ignore[arg-type]
        self._started[task] = time.perf_counter()

    def discard(self, task: object) -> None:
        started = self._started.pop(task, None)
        super().discard(task)  # type: ignore[arg-type]
        if started is not None:
            self._intervals.append((started, time.perf_counter()))

    def clear(self) -> None:
        now = time.perf_counter()
        for task in list(self._started):
            self._intervals.append((self._started.pop(task), now))
        super().clear()


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
    lease_busy: list[tuple[float, float]] = []  # exact lease-hold intervals
    # Round-4 P0 note: per-stage wall spans include QUEUE WAIT (a stage's
    # submission timestamp precedes its lease acquisition), so wall spans of
    # concurrently submitted jobs legitimately overlap. Exclusivity is asserted
    # on the recorded lease-hold intervals instead — matching the
    # logical-concurrency approach of the unit suite.
    coord._holders = _LeaseRecorder(lease_busy)  # type: ignore[assignment]

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
    print("(stage spans include queue wait; lease intervals are recorded at acquire/release)")
    for name, s, e in sorted(timeline, key=lambda t: t[1]):
        print(f"  {name:14s} submit/enter={s:.4f} end={e:.4f} dur={e - s:.4f}s")
    print(f"  lease-hold intervals ({len(lease_busy)}):")
    for s, e in lease_busy:
        print(f"    busy {s:.4f} -> {e:.4f} dur={e - s:.4f}s")

    # Core assertion (Round-4 P0 semantics): the recorded lease-hold intervals
    # never overlap — at most one task held the shared GPU slot at any instant.
    for (_s1, e1), (s2, _e2) in zip(lease_busy, lease_busy[1:], strict=False):
        assert s2 >= e1 - 1e-4, f"lease-hold intervals overlap: {e1} > {s2}"
    # The lease actually arbitrated: at least one recorded hold per iteration.
    assert len(lease_busy) >= ITERATIONS
    # Every iteration actually produced an LLM, TTS and RVC section.
    assert {n.split(":")[1] for n, _s, _e in timeline} == {"llm", "tts", "rvc"}
    assert len(timeline) == ITERATIONS * 3

    # Round-4 review P2: persist the raw real-GPU arbitration evidence.
    from evidence_utils import write_evidence

    write_evidence(
        "gpu_arbitration.json",
        {
            "test": "test_shared_coordinator_serializes_llm_and_tts_rvc",
            "iterations": ITERATIONS,
            "timeline": [
                {"stage": name, "start": s, "end": e, "duration_s": e - s}
                for name, s, e in sorted(timeline, key=lambda t: t[1])
            ],
            "lease_busy_intervals": [
                {"start": s, "end": e, "duration_s": e - s} for s, e in lease_busy
            ],
            "timeline_note": (
                "stage spans include queue wait; lease_busy_intervals are exact "
                "acquire/release intervals recorded from the holder registry"
            ),
            "max_new_tokens": 60,
            "pass": True,
        },
    )
