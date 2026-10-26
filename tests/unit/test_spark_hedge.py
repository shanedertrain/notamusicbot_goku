"""Hedged Spark request: a late thinking-off request races a slow thinking-low one."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from test_spark_chat import Server, _delta, _request, _sse

from notamusicbot.config import ListenLlmSettings
from notamusicbot.content.base import ContentRequest
from notamusicbot.content.spark_chat import EmptyModelText, SparkChatProvider, SparkHTTPError

HEDGE = 0.1


def _is_primary(req: httpx.Request) -> bool:
    return "reasoning_effort" in json.loads(req.content)


class Racer:
    """Chat handler with a per-role delay; tracks closed response streams."""

    def __init__(self, primary, hedge) -> None:
        self.primary = primary
        self.hedge = hedge
        self.closed: dict[str, bool] = {"primary": False, "hedge": False}

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        role = "primary" if _is_primary(req) else "hedge"
        delay, body = self.primary if role == "primary" else self.hedge
        if isinstance(body, Exception):
            await asyncio.sleep(delay)
            raise body
        return httpx.Response(200, stream=_Stream(delay, body, self.closed, role))


class _Stream(httpx.AsyncByteStream):
    def __init__(self, delay: float, body: bytes, closed: dict[str, bool], role: str) -> None:
        self._delay, self._body, self._closed, self._role = delay, body, closed, role

    async def __aiter__(self):
        await asyncio.sleep(self._delay)
        yield self._body

    async def aclose(self) -> None:
        self._closed[self._role] = True


def _provider(server: Server, **settings) -> SparkChatProvider:
    settings = {"thinking": "low", "hedge_after_s": HEDGE, "model": "m"} | settings
    return SparkChatProvider(
        ListenLlmSettings(**settings), max_new_tokens=60, transport=server.transport
    )


def _chat_count(server: Server) -> int:
    return sum(p.endswith("/chat/completions") for p in server.paths())


def _ok(text: str) -> bytes:
    return _sse(_delta(text), "[DONE]")


async def _collect(provider: SparkChatProvider, request: ContentRequest) -> list[str]:
    return [d async for d in provider.stream(request)]


def test_hedge_disabled_by_default() -> None:
    assert ListenLlmSettings().hedge_after_s is None
    assert ListenLlmSettings(hedge_after_s=0).hedge_after_s == 0


async def test_fast_primary_makes_exactly_one_request() -> None:
    racer = Racer((0.0, _ok("fast")), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["fast"]
    assert _chat_count(server) == 1


async def test_slow_primary_fast_hedge_streams_only_hedge_and_closes_primary() -> None:
    racer = Racer((5.0, _ok("slow")), (0.0, _ok("hedge text")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["hedge text"]
    assert _chat_count(server) == 2
    assert racer.closed["primary"] is True
    assert racer.closed["hedge"] is True


async def test_primary_delivering_before_deadline_makes_no_hedge() -> None:
    racer = Racer((HEDGE / 4, _ok("mine")), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    assert await _collect(_provider(server, hedge_after_s=0.5), _request()) == ["mine"]
    assert _chat_count(server) == 1


async def test_primary_wins_after_hedge_started_and_hedge_is_closed() -> None:
    racer = Racer((HEDGE * 2, _ok("primary")), (5.0, _ok("hedge")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["primary"]
    assert _chat_count(server) == 2
    assert racer.closed["hedge"] is True


async def test_reasoning_deltas_do_not_stop_the_hedge() -> None:
    reasoning_only = _sse(_delta(None, reasoning="thinking"), _delta("late"), "[DONE]")
    racer = Racer((5.0, reasoning_only), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["hedge"]


async def test_primary_error_after_hedge_started_hedge_continues() -> None:
    racer = Racer((HEDGE * 2, httpx.ConnectError("boom")), (HEDGE * 4, _ok("hedge ok")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["hedge ok"]


async def test_primary_error_before_hedge_raises_without_hedging() -> None:
    server = Server(chat=lambda req: httpx.Response(503))
    with pytest.raises(SparkHTTPError):
        await _collect(_provider(server, hedge_after_s=5.0), _request())
    assert _chat_count(server) == 1


async def test_both_fail_raises_primary_error() -> None:
    racer = Racer((HEDGE * 2, httpx.ConnectError("primary")), (0.0, httpx.ReadError("hedge")))
    server = Server(chat=racer)
    with pytest.raises(httpx.ConnectError):
        await _collect(_provider(server), _request())
    assert _chat_count(server) == 2


async def test_empty_primary_content_with_hedge_ok_yields_hedge_text() -> None:
    empty = _sse(_delta(None, reasoning="hmm"), "[DONE]")
    racer = Racer((HEDGE * 2, empty), (HEDGE * 4, _ok("hedge words")))
    server = Server(chat=racer)
    assert await _collect(_provider(server), _request()) == ["hedge words"]
    assert _chat_count(server) == 2


async def test_both_empty_raises_empty_model_text_without_third_request() -> None:
    empty = _sse("[DONE]")
    racer = Racer((HEDGE * 2, empty), (HEDGE * 3, empty))
    server = Server(chat=racer)
    with pytest.raises(EmptyModelText):
        await _collect(_provider(server), _request())
    assert _chat_count(server) == 2


async def test_consumer_cancellation_during_race_closes_both() -> None:
    racer = Racer((5.0, _ok("a")), (5.0, _ok("b")))
    server = Server(chat=racer)
    task = asyncio.create_task(_collect(_provider(server), _request()))
    await asyncio.sleep(HEDGE * 3)
    assert _chat_count(server) == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert racer.closed == {"primary": True, "hedge": True}
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert leftovers == []


async def test_consumer_closing_stream_after_winner_closes_it() -> None:
    racer = Racer((5.0, _ok("a")), (0.0, _ok("b")))
    server = Server(chat=racer)
    stream = _provider(server).stream(_request())
    assert await anext(stream) == "b"
    await stream.aclose()  # type: ignore[attr-defined]
    assert racer.closed == {"primary": True, "hedge": True}


async def test_thinking_off_never_hedges() -> None:
    racer = Racer((HEDGE * 3, _ok("off")), (HEDGE * 3, _ok("off")))
    server = Server(chat=racer)
    provider = _provider(server, thinking="off")
    assert await _collect(provider, _request()) == ["off"]
    assert _chat_count(server) == 1


async def test_hedge_bodies_differ_only_in_thinking_keys_and_max_tokens() -> None:
    racer = Racer((5.0, _ok("slow")), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    await _collect(_provider(server, thinking_token_budget=300), _request())
    bodies = [json.loads(r.content) for r in server.requests if r.url.path.endswith("/completions")]
    primary = next(b for b in bodies if "reasoning_effort" in b)
    hedge = next(b for b in bodies if "reasoning_effort" not in b)
    assert primary["max_tokens"] == 360
    assert hedge["max_tokens"] == 60
    assert hedge["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
    drop = {"reasoning_effort", "chat_template_kwargs", "max_tokens"}
    assert {k: v for k, v in primary.items() if k not in drop} == {
        k: v for k, v in hedge.items() if k not in drop
    }


async def test_hedge_reuses_cached_model_id() -> None:
    racer = Racer((5.0, _ok("slow")), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    provider = _provider(server, model="auto")
    await _collect(provider, _request())
    assert server.paths().count("/v1/models") == 1


async def test_hedge_is_refused_by_the_loopback_guard_when_base_url_turns_non_loopback() -> None:
    """The guard runs per request: a hedge to a non-loopback host never reaches the transport."""
    settings = ListenLlmSettings(thinking="low", hedge_after_s=HEDGE, model="m")
    flipped = False

    async def chat(req: httpx.Request) -> httpx.Response:
        nonlocal flipped
        if not flipped:
            flipped = True
            settings.base_url = "http://10.0.0.5:8000/v1"  # changes before the hedge fires
        return httpx.Response(200, stream=_Stream(HEDGE * 3, _ok("primary"), {}, "primary"))

    server = Server(chat=chat)
    provider = SparkChatProvider(settings, max_new_tokens=60, transport=server.transport)
    assert await _collect(provider, _request()) == ["primary"]
    assert _chat_count(server) == 1  # the hedge was refused before any request was made


async def test_zero_hedge_after_means_off() -> None:
    racer = Racer((0.0, _ok("mine")), (0.0, _ok("hedge")))
    server = Server(chat=racer)
    provider = _provider(server, hedge_after_s=0.0)
    assert provider._hedging() is False
    assert await _collect(provider, _request()) == ["mine"]
    assert _chat_count(server) == 1


class _GatedStream(httpx.AsyncByteStream):
    """Streams two deltas once ``gate`` is set; records close."""

    def __init__(self, gate: asyncio.Event, tag: str, closed: dict[str, bool], role: str) -> None:
        self._gate, self._tag, self._closed, self._role = gate, tag, closed, role

    async def __aiter__(self):
        await self._gate.wait()
        yield _sse(_delta(f"{self._tag}1"))
        yield _sse(_delta(f"{self._tag}2"), "[DONE]")

    async def aclose(self) -> None:
        self._closed[self._role] = True


async def test_simultaneous_first_deltas_yield_one_stream_and_close_the_other() -> None:
    """Both requests deliver in the same wake-up: one winner, no splice, loser closed."""
    gate = asyncio.Event()
    closed = {"primary": False, "hedge": False}

    async def chat(req: httpx.Request) -> httpx.Response:
        role = "primary" if _is_primary(req) else "hedge"
        tag = "P" if role == "primary" else "H"
        return httpx.Response(200, stream=_GatedStream(gate, tag, closed, role))

    server = Server(chat=chat)
    provider = _provider(server)
    task = asyncio.create_task(_collect(provider, _request()))
    await asyncio.sleep(HEDGE * 3)
    assert _chat_count(server) == 2
    gate.set()
    out = await task
    assert out in (["P1", "P2"], ["H1", "H2"])
    assert closed == {"primary": True, "hedge": True}
