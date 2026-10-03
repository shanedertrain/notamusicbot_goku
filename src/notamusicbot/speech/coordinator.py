"""Bounded inference coordinator (shared GPU arbitration — review Round-3 P1).

Async requests enter through :meth:`InferenceCoordinator.slot`, which caps how
many jobs can touch the GPU concurrently (default: one). Queue wait is part of
job metrics, and a bounded ``asyncio.Queue``-shaped semaphore prevents
unbounded task buildup.

Review Round-3 P1: one *shared* coordinator arbitrates ALL main-process GPU
inference — the local content LLM (:mod:`notamusicbot.content.llm`) and the
speech pipeline (TTS -> VC -> postprocess). The speech pipeline holds the
shared slot through the voice-conversion stage, so the RVC worker subprocess
can never overlap main-process LLM/TTS inference on the same GPU. (The RVC
worker itself runs out of process — no in-process lock can cover it — which is
exactly why the pipeline must keep the slot for the whole VC round-trip.)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self

logger = logging.getLogger(__name__)


class InferenceCoordinator:
    """Serialize GPU inference with a semaphore plus a bounded admission queue."""

    def __init__(self, max_concurrent: int = 1, queue_size: int = 8) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._admission = asyncio.Semaphore(max_concurrent + queue_size)
        self._waiting = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire permission to run one inference; may wait when saturated."""
        await self._admission.acquire()
        self._waiting += 1
        try:
            await self._semaphore.acquire()
            try:
                yield
            finally:
                self._semaphore.release()
        finally:
            self._waiting -= 1
            self._admission.release()

    @property
    def pending(self) -> int:
        """Number of jobs currently admitted or waiting for admission."""
        return self._waiting

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


_shared_coordinator: InferenceCoordinator | None = None


def shared_gpu_coordinator() -> InferenceCoordinator:
    """Return the process-wide shared GPU arbitration coordinator.

    The local content LLM provider falls back to this when it is not handed an
    explicit coordinator, guaranteeing that LLM generation and speech-pipeline
    inference (TTS/RVC) are serialized against each other even when the two
    components are composed independently (e.g. CLI tools and benchmarks).
    """
    global _shared_coordinator
    if _shared_coordinator is None:
        _shared_coordinator = InferenceCoordinator()
    return _shared_coordinator
