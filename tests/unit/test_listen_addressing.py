"""AddressJudge: reads the model's yes/no and fails closed."""

from __future__ import annotations

import asyncio

from notamusicbot.content.base import ContentRequest
from notamusicbot.listen.addressing import AddressJudge


class Provider:
    def __init__(self, answer: str = "yes", delay: float = 0.0, fail: bool = False) -> None:
        self.answer, self.delay, self.fail = answer, delay, fail
        self.requests: list[ContentRequest] = []

    async def generate(self, request: ContentRequest) -> str:
        self.requests.append(request)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("down")
        return self.answer


async def test_yes_and_no_are_read_from_the_first_word() -> None:
    assert await AddressJudge(Provider("Yes"), character="Obama").addressed("A", "hm", [])
    assert not await AddressJudge(Provider("no."), character="Obama").addressed("A", "hm", [])


async def test_the_request_carries_context_and_the_character() -> None:
    provider = Provider("no")
    await AddressJudge(provider, character="Obama").addressed("Sam", "nice", ["Ann: hi"])
    request = provider.requests[0]
    assert "Obama" in request.system_prompt
    # The last history line re-anchors the one-word answer format (see addressing.py).
    assert request.chat_history[:-1] == ("Ann: hi",)
    assert "one word" in request.chat_history[-1]
    assert (request.requester_name, request.transcript_text) == ("Sam", "nice")


async def test_failure_and_timeout_mean_not_addressed() -> None:
    assert not await AddressJudge(Provider(fail=True), character="O").addressed("A", "hey", [])
    slow = AddressJudge(Provider(delay=1.0), character="O", timeout_s=0.01)
    assert not await slow.addressed("A", "hey", [])
