"""Review Round-3 P1: real concurrent GPU arbitration test.

With both models resident (local Qwen3-4B content LLM and the Qwen3-TTS 0.6B
Base clone model), submits an LLM generation and a clone-TTS generation as
concurrent asyncio tasks through the SHARED :class:`InferenceCoordinator` and
asserts — over recorded lease-hold intervals, repeated across >= 3 iterations —
that the main-process GPU sections never overlapped.

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
ITERATIONS = 3


def _requirements_met() -> bool:
    try:
        import torch

        from notamusicbot.content.llm import resolve_local_model_path
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if not (REPO_ROOT / "config" / "voices" / "refs" / "obama.wav").is_file():
        return False
    try:
        resolve_local_model_path("Qwen/Qwen3-4B-Instruct-2507")
        resolve_local_model_path("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    except Exception:
        return False
    return True


pytest.importorskip("torch", reason="torch not installed")
if not _requirements_met():
    pytest.skip(
        "GPU arbitration requirements not met (CUDA / model caches / voice reference)",
        allow_module_level=True,
    )


class _LeaseRecorder(set):
    """Records exact lease-hold intervals from the coordinator holder registry.

    ``InferenceCoordinator`` adds a task to ``_holders`` immediately after it
    acquires the GPU semaphore and discards it immediately before releasing, so
    each add/discard pair is the precise lease interval, for every entry point
    (``slot()``, ``run_blocking``, ``run_detached``).
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


def _clone_request():
    from notamusicbot.config import load_app_settings
    from notamusicbot.domain.voice_profiles import load_voice_profiles, tts_request_for

    settings = load_app_settings()
    profiles = load_voice_profiles(Path(settings.speech.voices_dir))
    profile = profiles.get("obama") or next(iter(profiles.values()))
    return tts_request_for(profile, "Arbitration iteration.")


def test_shared_coordinator_serializes_llm_and_tts(tmp_path) -> None:
    import torch

    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.content.llm import LocalQwenLlmProvider
    from notamusicbot.speech.coordinator import InferenceCoordinator
    from notamusicbot.speech.tts.qwen_clone import QwenCloneSynthesizer

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
        synth = QwenCloneSynthesizer.from_settings(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device="cuda",
            dtype="bfloat16",
            attn_implementation="sdpa",
        )
        try:
            for i in range(ITERATIONS):
                prompt = ContentRequest(
                    kind=ContentKind.SONG, song_name=f"Arbitration {i}", requester_name="Tester"
                )
                tts_req = _clone_request()

                async def llm_task(prompt=prompt, i=i) -> None:
                    s = time.perf_counter()
                    text = await llm.generate(prompt)
                    e = time.perf_counter()
                    assert text.strip()
                    timeline.append((f"iter{i}:llm", s, e))

                async def tts_task(i=i, tts_req=tts_req) -> None:
                    async with coord.slot():
                        s = time.perf_counter()
                        artifact = await synth.synthesize(tts_req, tmp_path / f"tts{i}.wav")
                        e = time.perf_counter()
                        assert artifact.duration_s > 0
                    timeline.append((f"iter{i}:tts", s, e))

                await asyncio.gather(llm_task(), tts_task())
        finally:
            await synth.shutdown()

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
    # Every iteration actually produced an LLM and a TTS section.
    assert {n.split(":")[1] for n, _s, _e in timeline} == {"llm", "tts"}
    assert len(timeline) == ITERATIONS * 2

    # Round-4 review P2: persist the raw real-GPU arbitration evidence.
    from evidence_utils import write_evidence

    write_evidence(
        "gpu_arbitration.json",
        {
            "test": "test_shared_coordinator_serializes_llm_and_tts",
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
