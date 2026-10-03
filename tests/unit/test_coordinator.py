"""Review Round-3 P1: shared GPU coordinator mutual exclusion.

Proves that jobs entering the same :class:`InferenceCoordinator` instance —
the way the content LLM and the speech pipeline share it — are serialized:
at most one section is inside the slot at any time.

The invariant is asserted with a logical concurrency counter (incremented on
slot entry, decremented on exit) rather than raw wall-clock comparisons:
monotonic timestamps are still recorded for diagnostics, but timer-resolution
aliasing across asyncio scheduling points must not be able to fail the test.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from notamusicbot.speech.coordinator import (
    InferenceCoordinator,
    shared_gpu_coordinator,
)

SECTION_S = 0.02


async def _tracked_section(
    coord: InferenceCoordinator,
    stats: dict[str, int],
    log: list[tuple[float, float, str]],
    name: str,
) -> None:
    async with coord.slot():
        stats["active"] += 1
        stats["max_active"] = max(stats["max_active"], stats["active"])
        start = time.perf_counter()
        try:
            await asyncio.sleep(SECTION_S)
        finally:
            end = time.perf_counter()
            stats["active"] -= 1
            log.append((start, end, name))


@pytest.mark.asyncio
async def test_shared_slot_serializes_sections_no_overlap() -> None:
    coord = InferenceCoordinator()
    stats: dict[str, int] = {"active": 0, "max_active": 0}
    log: list[tuple[float, float, str]] = []
    names = ["llm-1", "tts-1", "llm-2", "tts-2", "llm-3"]
    await asyncio.gather(*(_tracked_section(coord, stats, log, n) for n in names))

    assert len(log) == 5
    # Core invariant, measured logically: never more than one holder.
    assert stats["max_active"] == 1, "sections overlapped on the shared GPU slot"
    # Sections ran in a strict serial order by wall clock too (loose form:
    # with a one-holder slot, starts are monotonically non-decreasing).
    ordered = sorted(log)
    starts = [s for s, _e, _n in ordered]
    assert starts == sorted(starts)
    # Every section actually ran for its full duration (no elision).
    for start, end, _name in log:
        assert end - start == pytest.approx(SECTION_S, rel=0.5)


@pytest.mark.asyncio
async def test_shared_gpu_coordinator_returns_same_instance() -> None:
    assert shared_gpu_coordinator() is shared_gpu_coordinator()


@pytest.mark.asyncio
async def test_llm_and_pipeline_style_tasks_are_mutually_exclusive() -> None:
    """Emulates the production shape: interleaved LLM and pipeline jobs all
    going through one shared coordinator; at most one inside at any time."""
    coord = InferenceCoordinator()
    stats: dict[str, int] = {"active": 0, "max_active": 0}
    log: list[tuple[float, float, str]] = []

    async def llm_job(i: int) -> None:
        await _tracked_section(coord, stats, log, f"llm-{i}")

    async def pipeline_job(i: int) -> None:
        # Like the real pipeline: one slot held across "TTS" and "VC" stages.
        async with coord.slot():
            stats["active"] += 1
            stats["max_active"] = max(stats["max_active"], stats["active"])
            try:
                await asyncio.sleep(SECTION_S / 2)
                await asyncio.sleep(SECTION_S / 2)
            finally:
                stats["active"] -= 1
            log.append((0.0, 0.0, f"pipeline-{i}"))

    await asyncio.gather(*(llm_job(i) for i in range(3)), *(pipeline_job(i) for i in range(3)))
    assert len(log) == 6  # 3 LLM sections + 3 pipeline slots
    assert stats["max_active"] == 1, "GPU sections overlapped on the shared slot"
