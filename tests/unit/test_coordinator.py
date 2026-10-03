"""Review Round-3 P1: shared GPU coordinator mutual exclusion.

Proves that jobs entering the same :class:`InferenceCoordinator` instance —
the way the content LLM and the speech pipeline share it — are serialized:
sections never overlap in time, measured with recorded monotonic timestamps.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from notamusicbot.speech.coordinator import (
    InferenceCoordinator,
    shared_gpu_coordinator,
)

SECTION_S = 0.05


async def _timed_section(
    coord: InferenceCoordinator,
    log: list[tuple[float, float, str]],
    name: str,
) -> None:
    async with coord.slot():
        start = time.perf_counter()
        await asyncio.sleep(SECTION_S)
        end = time.perf_counter()
        log.append((start, end, name))


@pytest.mark.asyncio
async def test_shared_slot_serializes_sections_no_overlap() -> None:
    coord = InferenceCoordinator()
    log: list[tuple[float, float, str]] = []
    names = ["llm-1", "tts-1", "llm-2", "tts-2", "llm-3"]
    await asyncio.gather(*(_timed_section(coord, log, n) for n in names))

    assert len(log) == 5
    ordered = sorted(log)
    for (_s1, e1, n1), (s2, _e2, n2) in zip(ordered, ordered[1:], strict=False):
        assert s2 >= e1, f"sections {n1} and {n2} overlapped on the shared GPU slot"
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
    log: list[tuple[float, float, str]] = []

    async def llm_job(i: int) -> None:
        await _timed_section(coord, log, f"llm-{i}")

    async def pipeline_job(i: int) -> None:
        # Like the real pipeline: one slot held across "TTS" and "VC" stages.
        async with coord.slot():
            tts_start = time.perf_counter()
            await asyncio.sleep(SECTION_S / 2)
            vc_start = time.perf_counter()
            await asyncio.sleep(SECTION_S / 2)
            log.append((tts_start, vc_start, f"tts-{i}"))
            log.append((vc_start, time.perf_counter(), f"vc-{i}"))

    await asyncio.gather(*(llm_job(i) for i in range(3)), *(pipeline_job(i) for i in range(3)))
    assert len(log) == 9  # 3 LLM sections + 3 pipeline x (tts, vc)
    ordered = sorted(log)
    for (_s1, e1, n1), (s2, _e2, n2) in zip(ordered, ordered[1:], strict=False):
        assert s2 >= e1 - 1e-6, f"GPU sections {n1} and {n2} overlapped"
