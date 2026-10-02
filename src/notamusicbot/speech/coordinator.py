"""Bounded inference coordinator.

Async requests enter through :meth:`InferenceCoordinator.slot`, which caps how
many jobs can touch the GPU concurrently (default: one). Queue wait is part of
job metrics, and a bounded ``asyncio.Queue``-shaped semaphore prevents
unbounded task buildup.
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
