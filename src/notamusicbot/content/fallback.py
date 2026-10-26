"""Spark-first chat with a lazy in-process fallback (plan section 17).

``FallbackChatProvider`` asks the Spark first. If the Spark is unreachable,
times out, answers 5xx or returns no text, the reply comes from the in-process
Qwen3-4B (``LazyLocalProvider``: built and loaded only on first use) when a
fallback is configured; otherwise the error propagates and the caller drops
the turn. While the Spark is healthy the in-process model is never loaded for
listen mode, and ``ensure_spark_preferred`` unloads it if it sits idle.

Only ``ContentKind.CHAT`` is handled here; other kinds never fall back and
are delegated to ``other`` (whatever the app used before) or rejected.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any

from notamusicbot.content.base import ContentKind, ContentProvider, ContentRequest
from notamusicbot.content.spark_chat import (
    TRANSPORT_ERRORS,
    EmptyModelText,
    SparkChatProvider,
    SparkHTTPError,
    SparkUnavailableError,
    UnsupportedContentKind,
)
from notamusicbot.speech.coordinator import InferenceCoordinator

logger = logging.getLogger(__name__)


def _free_model(provider: Any) -> None:
    """Drop the model weights held by ``provider`` and the CUDA cache."""
    unload = getattr(provider, "unload", None)
    if callable(unload):
        unload()
    else:  # LocalQwenLlmProvider keeps weights in these attributes
        provider._model = None
        provider._tokenizer = None
        provider.load_info = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class LazyLocalProvider:
    """Builds the in-process provider on first use; can free it again."""

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        gpu_coordinator: InferenceCoordinator | None = None,
    ) -> None:
        self._factory = factory
        self._gpu = gpu_coordinator
        self._provider: Any | None = None
        self._inflight = 0

    @property
    def loaded(self) -> bool:
        provider = self._provider
        return provider is not None and bool(provider.is_loaded())

    @property
    def idle(self) -> bool:
        return self._inflight == 0

    async def generate(self, request: ContentRequest) -> str:
        # The provider loads its model inside its own coordinator-guarded
        # thread on the first generate (the existing LocalQwen pattern).
        if self._provider is None:
            self._provider = self._factory()
        self._inflight += 1
        try:
            return await self._provider.generate(request)
        finally:
            self._inflight -= 1

    async def unload(self) -> bool:
        """Free the model if it is loaded and idle; True if it was freed."""
        provider = self._provider
        if provider is None or not self.loaded or not self.idle:
            return False
        if self._gpu is not None:
            await self._gpu.run_blocking(_free_model, provider)
        else:
            await asyncio.to_thread(_free_model, provider)
        logger.info("Unloaded the in-process listen LLM (Spark is preferred)")
        return True


def fallback_eligible(exc: BaseException) -> bool:
    """Failures that justify asking the local model instead."""
    if isinstance(exc, SparkHTTPError):
        return exc.status >= 500
    return isinstance(exc, (*TRANSPORT_ERRORS, TimeoutError, EmptyModelText, SparkUnavailableError))


class FallbackChatProvider:
    """Spark first; in-process model only on a Spark failure (if configured)."""

    def __init__(
        self,
        primary: SparkChatProvider,
        fallback: LazyLocalProvider | None = None,
        health_interval_s: float = 30.0,
        *,
        other: ContentProvider | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._health_interval_s = health_interval_s
        self._other = other
        # After a Spark failure, go straight to the local model until then.
        self._down_until = 0.0

    @property
    def primary(self) -> SparkChatProvider:
        return self._primary

    @property
    def fallback(self) -> LazyLocalProvider | None:
        return self._fallback

    def _spark_marked_down(self) -> bool:
        return self._fallback is not None and time.monotonic() < self._down_until

    def _warn_fallback(self, exc: BaseException) -> None:
        self._down_until = time.monotonic() + self._health_interval_s
        logger.warning(
            "Spark chat failed (%s: %s); using the in-process local model "
            "(a cold load may be slow)",
            type(exc).__name__,
            exc,
        )

    async def _other_generate(self, request: ContentRequest) -> str:
        if self._other is None:
            raise UnsupportedContentKind(f"No provider for {request.kind.name} here")
        return await self._other.generate(request)

    async def generate(self, request: ContentRequest) -> str:
        if request.kind is not ContentKind.CHAT:
            return await self._other_generate(request)
        if not self._spark_marked_down():
            try:
                text = await self._primary.generate(request)
            except Exception as exc:
                if self._fallback is None or not fallback_eligible(exc):
                    raise
                self._warn_fallback(exc)
            else:
                self._down_until = 0.0
                return text
        assert self._fallback is not None
        return await self._fallback.generate(request)

    async def stream(self, request: ContentRequest) -> AsyncIterator[str]:
        """Text deltas from the Spark; one whole-reply chunk from the fallback.

        Once any Spark text has been yielded a later failure is re-raised
        (switching models mid-reply would splice two different answers).
        """
        if request.kind is not ContentKind.CHAT:
            yield await self._other_generate(request)
            return
        if not self._spark_marked_down():
            yielded = False
            try:
                # aclosing: an early stop by the consumer closes the Spark
                # request now, not whenever the generator is garbage-collected.
                async with aclosing(self._primary.stream(request)) as deltas:
                    async for delta in deltas:
                        yielded = True
                        yield delta
            except Exception as exc:
                if yielded or self._fallback is None or not fallback_eligible(exc):
                    raise
                self._warn_fallback(exc)
            else:
                self._down_until = 0.0
                return
        assert self._fallback is not None
        yield await self._fallback.generate(request)

    async def ensure_spark_preferred(self) -> bool:
        """Unload an idle local model when the Spark is healthy.

        Called at listen start. Returns True if a model was unloaded.
        """
        if self._fallback is None:
            return False
        healthy = await self._primary.health()
        if healthy:
            self._down_until = 0.0
        if not healthy or not self._fallback.loaded:
            return False
        return await self._fallback.unload()
