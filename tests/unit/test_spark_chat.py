"""SparkChatProvider: SSE parsing, request shape, timeouts, loopback guard (no real network)."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from notamusicbot.config import ListenLlmSettings, ListenSettings
from notamusicbot.content import prompts
from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.spark_chat import (
    EmptyModelText,
    SparkChatProvider,
    SparkHTTPError,
    UnsupportedContentKind,
)

BASE = "http://127.0.0.1:8000/v1"


def _sse(*chunks: dict | str) -> bytes:
    lines = []
    for chunk in chunks:
        lines.append("data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk)))
        lines.append("")
    return "\n".join(lines).encode()


def _delta(content=None, **extra) -> dict:
    delta = dict(extra)
    if content is not None or "content" in extra:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta}]}


def _models(*ids: str) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})


class Server:
    """Mock Spark: records requests; routes /models and /chat/completions."""

    def __init__(self, chat=None, models=None) -> None:
        self.requests: list[httpx.Request] = []
        self.chat = chat or (lambda req: httpx.Response(200, content=_sse(_delta("Hi"), "[DONE]")))
        self.models = models or (lambda req: _models("spark-model-x"))

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            result = self.models(request)
        else:
            result = self.chat(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def _request(**kw) -> ContentRequest:
    base = {
        "kind": ContentKind.CHAT,
        "persona_description": "Goku, a cheerful saiyan",
        "requester_name": "Sam",
        "transcript_text": "What is your favorite food?",
    }
    return ContentRequest(**(base | kw))


def _provider(server: Server, **settings) -> SparkChatProvider:
    return SparkChatProvider(
        ListenLlmSettings(**settings), max_new_tokens=60, transport=server.transport
    )


async def _collect(provider: SparkChatProvider, request: ContentRequest) -> list[str]:
    return [d async for d in provider.stream(request)]


async def test_stream_yields_only_content_deltas_and_ignores_reasoning() -> None:
    body = _sse(
        _delta(None, reasoning="thinking about food"),
        _delta(None, reasoning_content="more"),
        {"choices": []},
        _delta("I love "),
        _delta(None, role="assistant"),
        _delta("ramen!"),
        "{not json",
        "[DONE]",
    )
    server = Server(chat=lambda req: httpx.Response(200, content=body))
    provider = _provider(server)
    assert await _collect(provider, _request()) == ["I love ", "ramen!"]
    assert await provider.generate(_request()) == "I love ramen!"


async def test_request_body_shape_and_model_from_models_endpoint() -> None:
    server = Server()
    provider = _provider(server)
    await provider.generate(_request())
    assert server.paths() == ["/v1/models", "/v1/chat/completions"]
    chat = server.requests[1]
    assert chat.method == "POST"
    body = json.loads(chat.content)
    assert body["model"] == "spark-model-x"  # from the mocked /models, never hardcoded
    assert body["stream"] is True
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 60
    assert body["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
    assert body["messages"] == [
        {"role": "system", "content": prompts.CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.build_chat_prompt(_request())},
    ]
    # The model is cached: a second reply does not hit /models again.
    await provider.generate(_request())
    assert server.paths().count("/v1/models") == 1


async def test_explicit_model_skips_models_endpoint() -> None:
    server = Server()
    await _provider(server, model="pinned-model").generate(_request())
    assert server.paths() == ["/v1/chat/completions"]
    assert json.loads(server.requests[0].content)["model"] == "pinned-model"


@pytest.mark.parametrize(
    "body",
    [
        _sse(_delta(None, reasoning="only thoughts"), "[DONE]"),
        _sse(_delta(None, role="assistant"), _delta(""), "[DONE]"),
        _sse(_delta("   "), "[DONE]"),
        b"",
    ],
)
async def test_empty_content_stream_raises_empty_model_text(body: bytes) -> None:
    server = Server(chat=lambda req: httpx.Response(200, content=body))
    with pytest.raises(EmptyModelText):
        await _provider(server).generate(_request())
    assert isinstance(EmptyModelText("x"), ValueError)


async def test_first_token_timeout_raises_timeout_error_quickly() -> None:
    async def slow(req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(2.0)
        return httpx.Response(200, content=_sse(_delta("late"), "[DONE]"))

    server = Server(chat=slow)
    provider = _provider(server, first_token_timeout_s=0.05, timeout_s=5.0)
    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        await provider.generate(_request())
    assert time.perf_counter() - start < 0.5


async def test_http_5xx_and_other_errors_raise_spark_http_error() -> None:
    server = Server(chat=lambda req: httpx.Response(503))
    with pytest.raises(SparkHTTPError) as err:
        await _provider(server).generate(_request())
    assert err.value.status == 503
    server = Server(chat=lambda req: httpx.Response(400))
    with pytest.raises(SparkHTTPError) as err:
        await _provider(server).generate(_request())
    assert err.value.status == 400


async def test_404_triggers_exactly_one_model_re_resolve() -> None:
    models = iter(["old-model", "new-model", "never-used"])
    seen: list[str] = []

    def chat(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        seen.append(model)
        if model == "old-model":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, content=_sse(_delta("ok"), "[DONE]"))

    server = Server(chat=chat, models=lambda req: _models(next(models)))
    assert await _provider(server).generate(_request()) == "ok"
    assert seen == ["old-model", "new-model"]
    assert server.paths() == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/models",
        "/v1/chat/completions",
    ]


async def test_second_404_raises_and_does_not_loop() -> None:
    server = Server(chat=lambda req: httpx.Response(404))
    with pytest.raises(SparkHTTPError) as err:
        await _provider(server).generate(_request())
    assert err.value.status == 404
    assert server.paths().count("/v1/models") == 2
    assert server.paths().count("/v1/chat/completions") == 2


async def test_explicit_model_404_is_not_re_resolved() -> None:
    server = Server(chat=lambda req: httpx.Response(404))
    with pytest.raises(SparkHTTPError):
        await _provider(server, model="pinned").generate(_request())
    assert server.paths() == ["/v1/chat/completions"]


async def test_non_chat_kind_is_rejected_without_a_request() -> None:
    server = Server()
    with pytest.raises(UnsupportedContentKind):
        await _provider(server).generate(ContentRequest(kind=ContentKind.SONG))
    assert server.requests == []


@pytest.mark.parametrize("url", ["http://10.0.0.5:8000/v1", "http://example.com/v1"])
async def test_non_loopback_url_rejected_by_config_constructor_and_call(url: str) -> None:
    with pytest.raises(ValueError, match="not loopback"):
        ListenSettings(llm={"base_url": url})
    server = Server()
    # Bypass the config validator (model_construct) -> the constructor still refuses.
    bad = ListenLlmSettings.model_construct(**ListenLlmSettings().model_dump() | {"base_url": url})
    with pytest.raises(ValueError, match="not loopback"):
        SparkChatProvider(bad, transport=server.transport)
    # Settings mutated after construction -> the call-time guard refuses.
    settings = ListenLlmSettings()
    provider = SparkChatProvider(settings, transport=server.transport)
    settings.base_url = url
    with pytest.raises(ValueError, match="not loopback"):
        await provider.generate(_request())
    with pytest.raises(ValueError, match="not loopback"):
        await provider.health()
    assert server.requests == []  # no request ever reached the transport


def test_client_ignores_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    # With trust_env a proxy env var would route the loopback request off-box.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "all_proxy"):
        monkeypatch.setenv(var, "http://10.9.9.9:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    client = SparkChatProvider(ListenLlmSettings())._client()
    assert client.trust_env is False
    assert client._mounts == {}  # no proxy transport mounted for any URL


async def test_loopback_variants_are_accepted() -> None:
    for url in ("http://localhost:8000/v1", "http://[::1]:8000/v1", "http://127.0.0.1:1/v1/"):
        server = Server()
        assert await _provider(server, base_url=url).generate(_request()) == "Hi"


async def test_health_true_false_and_connect_error() -> None:
    assert await _provider(Server()).health() is True
    assert await _provider(Server(models=lambda req: httpx.Response(500))).health() is False

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert await _provider(Server(models=boom)).health() is False


async def test_connect_error_propagates_from_stream() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        await _provider(Server(chat=boom)).generate(_request())


def _spark_up() -> bool:
    try:
        return httpx.get("http://127.0.0.1:8000/v1/models", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _spark_up(), reason="Spark tunnel not answering on 127.0.0.1:8000")
async def test_live_spark_streams_a_goku_reply(capsys) -> None:
    provider = SparkChatProvider(ListenLlmSettings(), max_new_tokens=60)
    request = _request(transcript_text="Hey Goku, what do you like to eat after a big fight?")
    start = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    async for delta in provider.stream(request):
        if first is None:
            first = time.perf_counter() - start
        parts.append(delta)
    total = time.perf_counter() - start
    text = "".join(parts).strip()
    with capsys.disabled():
        print(
            f"\nSPARK LIVE model={provider._model} first_delta_s={first:.3f} "
            f"total_s={total:.3f} chars={len(text)} text={text!r}"
        )
    assert text
    assert first is not None


async def test_thinking_low_sends_reasoning_effort_and_widens_max_tokens() -> None:
    server = Server()
    provider = _provider(server, thinking="low", thinking_token_budget=300)
    await provider.generate(_request())
    body = json.loads(server.requests[1].content)
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 60 + 300  # reasoning shares the limit
    assert "chat_template_kwargs" not in body  # must not force thinking off


async def test_thinking_off_is_the_default_and_sends_no_reasoning_effort() -> None:
    server = Server()
    provider = _provider(server)
    await provider.generate(_request())
    body = json.loads(server.requests[1].content)
    assert "reasoning_effort" not in body
    assert body["max_tokens"] == 60


async def test_thinking_low_empty_stream_retries_once_with_thinking_off() -> None:
    calls: list[dict] = []

    def chat(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        if len(calls) == 1:  # reasoning ate the budget: no content at all
            return httpx.Response(200, content=_sse(_delta(None, reasoning="hmm"), "[DONE]"))
        return httpx.Response(200, content=_sse(_delta("Yum, ramen!"), "[DONE]"))

    provider = _provider(Server(chat=chat), thinking="low")
    assert "".join(await _collect(provider, _request())) == "Yum, ramen!"
    assert calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in calls[1]
    assert calls[1]["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}


async def test_thinking_low_empty_twice_still_raises_and_off_never_retries() -> None:
    empty = lambda req: httpx.Response(200, content=_sse("[DONE]"))  # noqa: E731
    server = Server(chat=empty)
    with pytest.raises(EmptyModelText):
        await _collect(_provider(server, thinking="low"), _request())
    assert sum(p.endswith("/chat/completions") for p in server.paths()) == 2
    server = Server(chat=empty)
    with pytest.raises(EmptyModelText):
        await _collect(_provider(server), _request())
    assert sum(p.endswith("/chat/completions") for p in server.paths()) == 1
