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

Review Round-4 P0 — cancellation safety (Option B design)
---------------------------------------------------------

``asyncio.to_thread`` inside ``async with coord.slot()`` has a defect: when the
awaiting coroutine is cancelled, the ``async with`` exits and the semaphore is
released immediately, while the worker thread keeps running on the GPU. A next
task can then acquire the slot during live GPU work.

Two cancellation-safe entry points fix this. Both own BOTH the GPU lease and
the work:

1. :meth:`run_blocking(fn, *args, **kwargs)` — for purely blocking functions
   (the content LLM's ``_generate_sync``). The holder task acquires the
   semaphore first and only THEN submits ``fn`` to the default executor, so
   the blocking work never starts without the lease. The holder awaits the
   executor future through ``asyncio.shield`` and releases the semaphore ONLY
   after the future actually completed (returned or raised) — the future is
   even waited out when the holder itself is cancelled at event-loop shutdown.
   Caller cancellation raises ``CancelledError`` in the caller immediately,
   but the detached holder keeps the lease until the thread work ends.

2. :meth:`run_detached(async_fn, *args)` — for async jobs whose stages mix
   awaited worker-reply threads with async glue (the speech pipeline's
   TTS -> VC -> postprocess chain). The coroutine runs inside the detached
   holder task itself; the caller awaits through ``asyncio.shield``. Caller
   cancellation detaches the holder, which keeps the lease until the chain
   coroutine actually finished (including the RVC worker-reply wait). Unlike
   ``run_blocking``, a holder cancelled at event-loop shutdown releases the
   lease as its coroutine unwinds — deterministic teardown, pinned by the
   cancelled-job artifact tests.

Both keep admission bounded: the ``_admission`` semaphore (max_concurrent +
queue_size) is acquired before the holder task is created and released only
after the work completes, so pending submissions queue exactly like ``slot()``
callers and cannot grow without bound.

The result/exception of the work is delivered to the caller whenever the
caller is still awaiting; a caller that was cancelled simply never sees it
(the holder stores the outcome and always returns normally, so no "exception
was never retrieved" warnings are emitted).

Double-acquire guard: neither entry point may be called from a task that
already holds the slot (that would deadlock on a one-slot coordinator). The
coordinator records which tasks currently hold the semaphore and raises
``RuntimeError`` if the calling task is already a holder.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InferenceCoordinator:
    """Serialize GPU inference with a semaphore plus a bounded admission queue."""

    def __init__(self, max_concurrent: int = 1, queue_size: int = 8) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._admission = asyncio.Semaphore(max_concurrent + queue_size)
        self._waiting = 0
        # Tasks currently holding the semaphore (double-acquire guard).
        self._holders: set[asyncio.Task[None]] = set()
        # Strong references to detached holder tasks so a cancelled caller's
        # lease holder cannot be garbage-collected mid-flight.
        # Task[Any]: holders return the (ok, outcome) pair unpacked below.
        self._detached: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire permission to run one inference; may wait when saturated."""
        await self._admission.acquire()
        self._waiting += 1
        try:
            await self._semaphore.acquire()
            task = asyncio.current_task()
            assert task is not None
            self._holders.add(task)
            try:
                yield
            finally:
                self._holders.discard(task)
                self._semaphore.release()
        finally:
            self._waiting -= 1
            self._admission.release()

    async def run_blocking(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run blocking ``fn`` under the GPU lease, cancellation-safely.

        The lease is held from before ``fn`` starts on its worker thread until
        ``fn`` actually returns or raises — NOT until the awaiting coroutine
        finishes. If the awaiting coroutine is cancelled, it raises
        ``CancelledError`` immediately, while the detached holder keeps the
        lease until the thread-level work really ends; no other job can enter
        the coordinator during that window. Even if the holder task itself is
        cancelled (event-loop shutdown), it keeps waiting on the shielded
        executor future before releasing, so the lease can never be dropped
        while ``fn`` runs on its thread.

        Raises ``RuntimeError`` when called from a task that already holds the
        slot (double-acquire deadlock guard).
        """

        async def _invoke() -> Any:
            call = lambda: fn(*args, **kwargs)  # noqa: E731 - executor target
            return await asyncio.get_running_loop().run_in_executor(None, call)

        return await self._run_detached(_invoke, thread_shield=True)

    async def run_detached(self, async_fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``async_fn(*args)`` under the GPU lease, caller-cancellation-safe.

        The coroutine runs inside a DETACHED holder task on the running loop;
        the caller awaits it through ``asyncio.shield``. If the caller is
        cancelled, it raises ``CancelledError`` immediately while the holder
        keeps the lease until the coroutine actually finished — so awaited
        worker-reply threads (e.g. the RVC adapter) can never outlive the
        lease. Unlike :meth:`run_blocking` (whose thread is shielded even
        against loop shutdown), a holder cancelled at event-loop shutdown
        releases the lease when its coroutine unwinds — the cancellation then
        propagates into any in-flight await, which is the deterministic
        teardown behavior the cancelled-job artifact tests pin.

        Raises ``RuntimeError`` when called from a task that already holds the
        slot (double-acquire deadlock guard).
        """
        return await self._run_detached(lambda: async_fn(*args), thread_shield=False)

    async def _run_detached(self, invoke: Callable[[], Any], *, thread_shield: bool) -> Any:
        """Shared detached-holder machinery for run_blocking/run_detached."""
        caller = asyncio.current_task()
        if caller is not None and caller in self._holders:
            msg = (
                "run_blocking/run_detached called from a task that already "
                "holds the coordinator slot; this would deadlock"
            )
            raise RuntimeError(msg)
        await self._admission.acquire()
        self._waiting += 1
        finished = asyncio.Event()
        holder = asyncio.get_running_loop().create_task(
            self._hold_and_run(invoke, finished, thread_shield),
            name="inference-coordinator-detached-holder",
        )
        self._detached.add(holder)
        holder.add_done_callback(self._detached.discard)
        try:
            # shield: caller cancellation must NOT cancel the holder; the
            # holder alone owns the lease lifetime.
            ok, outcome = await asyncio.shield(holder)
        except asyncio.CancelledError:
            # Detached holder keeps the lease until the work really finishes.
            raise
        finally:
            if holder.done():
                self._detached.discard(holder)
        if not ok:
            raise outcome
        return outcome

    async def _hold_and_run(
        self,
        invoke: Callable[[], Any],
        finished: asyncio.Event,
        thread_shield: bool,
    ) -> tuple[bool, Any]:
        """Hold the lease while the work runs to real completion.

        Always returns ``(ok, outcome)`` — never raises — so a detached
        (caller-cancelled) holder never produces unretrieved exceptions.
        """
        outcome: Any = None
        ok = True
        try:
            await self._semaphore.acquire()
            task = asyncio.current_task()
            assert task is not None
            self._holders.add(task)
            try:
                if thread_shield:
                    outcome = await self._await_thread_shielded(invoke)
                else:
                    outcome = await invoke()
            except BaseException as exc:  # noqa: BLE001 - relayed to the caller
                ok, outcome = False, exc
            finally:
                self._holders.discard(task)
                self._semaphore.release()
        except asyncio.CancelledError as exc:
            # Cancelled while still queued for the semaphore (or, for
            # coroutine jobs, while the job already unwound): no GPU work of
            # ours is running anymore, so releasing the lease here is safe.
            ok, outcome = False, exc
        finally:
            self._waiting -= 1
            self._admission.release()
            finished.set()
        return ok, outcome

    @staticmethod
    async def _await_thread_shielded(invoke: Callable[[], Any]) -> Any:
        """Await the work, waiting it out even if we are cancelled.

        Used by :meth:`run_blocking`: ``invoke()`` returns a coroutine that
        awaits the executor future running ``fn``. We wrap it in a task and
        shield it; if this holder task is cancelled while the thread future is
        in flight (event-loop shutdown), we still wait for the task so the
        lease is released only after the thread work really ended.
        """
        future = asyncio.ensure_future(invoke())
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await future  # wait the thread out; its outcome no longer matters
            except BaseException:  # noqa: BLE001 - outcome relayed upstream
                pass
            raise

    @property
    def pending(self) -> int:
        """Number of jobs currently admitted or waiting for admission."""
        return self._waiting

    async def __aenter__(self) -> Any:
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
