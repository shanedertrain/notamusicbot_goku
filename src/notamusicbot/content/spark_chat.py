"""Listen-mode reply text from the owner's Spark system (plan section 17).

``SparkChatProvider`` streams an OpenAI-style ``/chat/completions`` response
over the loopback SSH tunnel (default ``http://127.0.0.1:8000/v1``) and yields
only the visible ``content`` deltas. This is the ONE listen-related module
allowed to import ``httpx`` (see ``tests/unit/test_listen_local_only.py``), and
it is loopback-only: the base URL is checked by the config validator, again in
the constructor and again at every call before any socket is opened.

The Spark serves reasoning models. With thinking on, ``content`` stays null
and the reply is lost, so every request disables thinking under BOTH template
spellings (``thinking`` and ``enable_thinking``). A stream that ends with no
content text raises :class:`EmptyModelText`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import httpx

from notamusicbot.config import ListenLlmSettings, assert_loopback_url
from notamusicbot.content import prompts
from notamusicbot.content.base import ContentKind, ContentRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.7
HEALTH_TIMEOUT_S = 2.0
# Connect/read/write failures; exported so fallback.py needs no httpx import.
TRANSPORT_ERRORS: tuple[type[Exception], ...] = (httpx.TransportError,)


class SparkChatError(RuntimeError):
    """Base class for Spark chat failures."""


class SparkHTTPError(SparkChatError):
    """The Spark answered with a non-success HTTP status."""

    def __init__(self, status: int, what: str) -> None:
        super().__init__(f"Spark {what} returned HTTP {status}")
        self.status = status


class SparkUnavailableError(SparkChatError):
    """The Spark answered but serves no model."""


class EmptyModelText(ValueError):  # noqa: N818 - name fixed by the task spec
    """The reply stream finished without any content text (e.g. thinking ate it)."""


class _HedgedEmptyText(EmptyModelText):  # noqa: N818 - internal marker
    """Both the primary and the hedge came back empty; do not retry a third time."""


class UnsupportedContentKind(ValueError):
    """Only ``ContentKind.CHAT`` is served by the Spark provider."""


def _clean_base(base_url: str) -> str:
    return assert_loopback_url(base_url).rstrip("/")


class SparkChatProvider:
    """``ContentProvider`` for ``ContentKind.CHAT`` backed by the Spark."""

    def __init__(
        self,
        settings: ListenLlmSettings,
        *,
        max_new_tokens: int = 60,
        temperature: float = DEFAULT_TEMPERATURE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Raises before any client exists if the URL is not loopback.
        _clean_base(settings.base_url)
        self._settings = settings
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._transport = transport
        self._model: str | None = None if settings.model == "auto" else settings.model

    # ------------------------------------------------------------- plumbing

    def _guard(self) -> str:
        """Re-parse the URL at call time; raises before any socket is opened."""
        return _clean_base(self._settings.base_url)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(self._settings.timeout_s),
            follow_redirects=False,
            # Ignore HTTP(S)_PROXY / ALL_PROXY: a proxy would carry the loopback
            # request to a non-loopback host.
            trust_env=False,
        )

    @staticmethod
    async def _within(awaitable: Awaitable[Any], deadline: float) -> Any:
        async with asyncio.timeout_at(deadline):
            return await awaitable

    async def _resolve_model(self, client: httpx.AsyncClient, base: str, deadline: float) -> str:
        if self._model is not None:
            return self._model
        response = await self._within(client.get(f"{base}/models"), deadline)
        if response.status_code >= 400:
            raise SparkHTTPError(response.status_code, "/models")
        try:
            models = response.json()["data"]
            model = str(models[0]["id"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SparkUnavailableError("Spark /models lists no model") from exc
        self._model = model
        logger.info("Spark chat model resolved to %s", model)
        return model

    def _body(
        self, request: ContentRequest, model: str, *, force_off: bool = False
    ) -> dict[str, Any]:
        temperature = getattr(request, "temperature", None)
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompts.chat_system_prompt(request)},
                {"role": "user", "content": prompts.build_chat_prompt(request)},
            ],
            "stream": True,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_new_tokens,
        }
        if self._settings.thinking == "low" and not force_off:
            # Reasoning tokens share max_tokens: leave room or content stays null.
            body["max_tokens"] = self._max_new_tokens + self._settings.thinking_token_budget
            body["reasoning_effort"] = "low"
        else:
            # Reasoning model: thinking on => content null. Send both spellings.
            body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        return body

    @staticmethod
    def _delta_text(line: str) -> str | None:
        """Content text of one SSE line; ``None`` for [DONE]; ``""`` for no text."""
        if not line.startswith("data:"):
            return ""
        payload = line[5:].strip()
        if payload == "[DONE]":
            return None
        try:
            choices = json.loads(payload).get("choices") or []
            content = (choices[0].get("delta") or {}).get("content") if choices else None
        except (ValueError, AttributeError, IndexError, TypeError):
            return ""
        return content if isinstance(content, str) else ""

    # ----------------------------------------------------------------- API

    async def health(self) -> bool:
        """True if ``GET /models`` answers 2xx within 2 s."""
        base = self._guard()
        try:
            async with self._client() as client:
                async with asyncio.timeout(HEALTH_TIMEOUT_S):
                    response = await client.get(f"{base}/models")
        except (httpx.HTTPError, TimeoutError, OSError):
            return False
        return response.is_success

    async def stream(self, request: ContentRequest) -> AsyncIterator[str]:
        """Yield content text deltas as they arrive.

        With ``thinking: low`` a stream that ends with no content (the reasoning
        ate the budget) is retried once with thinking off, so one empty reply
        does not drop the Spark for the in-process fallback.

        The first-token timeout covers model resolution, connect and the wait
        for the first content delta (reasoning deltas do not count); the
        overall timeout covers the whole reply. Deadlines are applied per
        await, never across a ``yield``, so a slow consumer is not cancelled.
        """
        if request.kind is not ContentKind.CHAT:
            raise UnsupportedContentKind(f"Spark chat serves CHAT only, not {request.kind.name}")
        retry = self._settings.thinking == "low"
        first_pass = self._stream_hedged(request) if self._hedging() else self._stream_once(request)
        try:
            async with aclosing(first_pass) as deltas:
                async for delta in deltas:
                    yield delta
        except _HedgedEmptyText:
            raise
        except EmptyModelText:
            if not retry:
                raise
            logger.warning("Spark returned no text with thinking low; retrying once with it off")
        else:
            return
        async with aclosing(self._stream_once(request, force_off=True)) as deltas:
            async for delta in deltas:
                yield delta

    def _hedging(self) -> bool:
        after = self._settings.hedge_after_s
        return self._settings.thinking == "low" and after is not None and after > 0

    async def _stream_hedged(self, request: ContentRequest) -> AsyncIterator[str]:
        """Race a thinking-low request against a late thinking-off hedge.

        The hedge starts only if the primary has produced no content delta after
        ``hedge_after_s``. The first request to yield a content delta wins; ONLY
        its stream is yielded and the other is cancelled and closed at once.
        """
        after = self._settings.hedge_after_s
        assert after is not None
        gens: dict[asyncio.Task[str], AsyncIterator[str]] = {}
        primary_gen = self._stream_once(request, force_off=False)
        hedge_gen: AsyncIterator[str] | None = None
        primary_error: Exception | None = None
        hedge_error: Exception | None = None

        async def first_delta(gen: AsyncIterator[str]) -> str:
            return await anext(gen)

        def launch(gen: AsyncIterator[str]) -> None:
            gens[asyncio.create_task(first_delta(gen))] = gen

        all_gens: list[AsyncIterator[str]] = [primary_gen]
        winner: AsyncIterator[str] | None = None
        first = ""
        launch(primary_gen)
        try:
            timeout: float | None = after
            while gens and winner is None:
                done, _ = await asyncio.wait(
                    gens, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    hedge_gen = self._stream_once(request, force_off=True)
                    all_gens.append(hedge_gen)
                    launch(hedge_gen)
                    timeout = None
                    logger.info("Spark hedge started after %.2f s with no content", after)
                    continue
                for task in done:
                    gen = gens.pop(task)
                    try:
                        text = task.result()
                    except StopAsyncIteration:
                        error: Exception = EmptyModelText("Spark stream ended with no content text")
                    except Exception as exc:
                        error = exc
                    else:
                        if winner is None:
                            winner, first = gen, text
                        continue
                    if gen is primary_gen:
                        primary_error = error
                        if hedge_gen is None:
                            raise error  # failed before the hedge: behave as unhedged
                    else:
                        hedge_error = error
            if winner is None:
                assert primary_error is not None
                if isinstance(primary_error, EmptyModelText):
                    raise _HedgedEmptyText(str(primary_error)) from hedge_error
                raise primary_error
            logger.info(
                "Spark hedge race won by the %s request",
                "primary" if winner is primary_gen else "hedge",
            )
        finally:
            for task in gens:
                task.cancel()
            await asyncio.gather(*gens, return_exceptions=True)
            for gen in all_gens:
                if gen is not winner:
                    await gen.aclose()  # type: ignore[attr-defined]
        assert winner is not None
        async with aclosing(winner) as deltas:  # type: ignore[type-var]
            yield first
            async for delta in deltas:
                yield delta

    async def _stream_once(
        self, request: ContentRequest, *, force_off: bool = False
    ) -> AsyncIterator[str]:
        base = self._guard()
        loop = asyncio.get_running_loop()
        start = loop.time()
        overall = start + self._settings.timeout_s
        first = min(overall, start + self._settings.first_token_timeout_s)
        got_text = False
        total_chars = 0

        async with self._client() as client:
            for attempt in (0, 1):
                model = await self._resolve_model(client, base, first)
                http_request = client.build_request(
                    "POST",
                    f"{base}/chat/completions",
                    json=self._body(request, model, force_off=force_off),
                )
                response = await self._within(client.send(http_request, stream=True), first)
                try:
                    if (
                        response.status_code == 404
                        and attempt == 0
                        and self._settings.model == "auto"
                    ):
                        logger.warning("Spark model %r not found; re-resolving once", model)
                        self._model = None
                        continue
                    if response.status_code >= 400:
                        raise SparkHTTPError(response.status_code, "/chat/completions")
                    lines = response.aiter_lines().__aiter__()
                    while True:
                        deadline = overall if got_text else first
                        try:
                            line = await self._within(anext(lines), deadline)
                        except StopAsyncIteration:
                            break
                        delta = self._delta_text(line)
                        if delta is None:
                            break
                        if delta:
                            got_text = True
                            total_chars += len(delta.strip())
                            yield delta
                finally:
                    await response.aclose()
                break
        if total_chars == 0:
            raise EmptyModelText("Spark stream ended with no content text")

    async def generate(self, request: ContentRequest) -> str:
        """Join the stream into the full reply text."""
        async with aclosing(self.stream(request)) as deltas:
            parts = [delta async for delta in deltas]
        return "".join(parts).strip()
