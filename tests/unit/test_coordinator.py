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
    CoordinatorClosedError,
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


# ---------------------------------------------------------------------------
# Review Round-6 §3/B: inference shutdown — close()/drain()
# ---------------------------------------------------------------------------


async def test_close_drains_started_job_cancels_queued_rejects_new() -> None:
    """close(): the started job is awaited to real completion, the queued
    job never starts, and new submissions are rejected afterwards."""
    coord = InferenceCoordinator(max_concurrent=1, queue_size=4)
    release = asyncio.Event()
    started = {"started_job": False, "queued_job": False}

    async def job(name: str) -> str:
        started[name] = True
        await release.wait()
        return name

    started_task = asyncio.create_task(coord.run_detached(job, "started_job"))
    while not started["started_job"]:
        await asyncio.sleep(0)
    queued_task = asyncio.create_task(coord.run_detached(job, "queued_job"))
    # Deterministic: wait until the queued job is admitted into the
    # coordinator's holder registry (still waiting for the semaphore).
    while len(coord._holder_started) < 2:  # noqa: SLF001
        await asyncio.sleep(0)
    assert not started["queued_job"]

    close_task = asyncio.create_task(coord.close())
    # Drain cannot finish while the started job is still parked on `release`.
    await asyncio.sleep(0)
    assert not close_task.done()

    release.set()
    assert await asyncio.wait_for(started_task, 5) == "started_job"
    await asyncio.wait_for(close_task, 5)

    # Queued job: cancelled by the drain, never started.
    with pytest.raises(asyncio.CancelledError):
        await queued_task
    assert started["queued_job"] is False
    assert coord.closed is True

    # New submissions are rejected with a clear error (both entry points).
    with pytest.raises(CoordinatorClosedError):
        await coord.run_detached(job, "late_job")
    with pytest.raises(CoordinatorClosedError):
        await coord.run_blocking(lambda: None)
    with pytest.raises(CoordinatorClosedError):
        async with coord.slot():
            pass  # pragma: no cover - never reached


async def test_drain_is_alias_of_close_and_close_is_idempotent() -> None:
    coord = InferenceCoordinator()
    release = asyncio.Event()
    done: list[str] = []

    async def job() -> None:
        await release.wait()
        done.append("job")

    task = asyncio.create_task(coord.run_detached(job))
    while not coord._holder_started:  # noqa: SLF001
        await asyncio.sleep(0)
    drain_task = asyncio.create_task(coord.drain())
    await asyncio.sleep(0)
    assert not drain_task.done()
    release.set()
    await asyncio.wait_for(drain_task, 5)
    await asyncio.wait_for(task, 5)
    assert done == ["job"]
    await coord.drain()  # idempotent: second drain returns immediately
    assert coord.closed is True


@pytest.mark.asyncio
async def test_close_rejects_caller_blocked_on_admission() -> None:
    """Review Round-7 §2 P0: a submission already WAITING on admission when
    close() begins must be rejected after admission is later granted — it must
    never execute. (max_concurrent=1, queue_size=0: B blocks on admission.)"""
    coord = InferenceCoordinator(max_concurrent=1, queue_size=0)
    release = asyncio.Event()
    ran: list[str] = []

    async def job(tag: str) -> str:
        ran.append(tag)
        await release.wait()
        return tag

    task_a = asyncio.create_task(coord.run_detached(job, "a"))
    while not coord._holder_started:  # noqa: SLF001
        await asyncio.sleep(0)

    task_b = asyncio.create_task(coord.run_detached(job, "b"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task_b.done()
    assert ran == ["a"]  # A started and is parked; B has not begun

    close_task = asyncio.create_task(coord.close())
    await asyncio.sleep(0)
    release.set()  # A finishes; its admission is released
    await asyncio.wait_for(task_a, 5)
    with pytest.raises(CoordinatorClosedError):
        await asyncio.wait_for(task_b, 5)
    await asyncio.wait_for(close_task, 5)
    assert ran == ["a"]  # B's function NEVER executed
    assert coord.closed is True


@pytest.mark.asyncio
async def test_slot_rejects_caller_blocked_on_admission_after_close() -> None:
    """Same invariant for the slot() context-manager API: a caller waiting on
    admission when close() begins gets CoordinatorClosedError, never the GPU."""
    coord = InferenceCoordinator(max_concurrent=1, queue_size=0)
    release = asyncio.Event()

    async def holder_job() -> None:
        async with coord.slot():
            await release.wait()

    task_a = asyncio.create_task(holder_job())
    while not coord._holders:  # noqa: SLF001
        await asyncio.sleep(0)

    close_task = asyncio.create_task(coord.close())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(task_a, 5)
    with pytest.raises(CoordinatorClosedError):
        async with coord.slot():
            pytest.fail("slot must not be granted after close")
    await asyncio.wait_for(close_task, 5)
    assert coord.closed is True
