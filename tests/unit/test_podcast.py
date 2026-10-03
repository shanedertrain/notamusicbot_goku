"""Unit tests for the podcast conversation mode (fakes only, no GPU/network)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.interstitials import TopicItem
from notamusicbot.podcast import PodcastCoordinator, PodcastEngine


@dataclass
class ScriptedProvider:
    """ContentProvider fake returning scripted lines in order."""

    replies: list[str] = field(default_factory=list)
    requests: list[ContentRequest] = field(default_factory=list)
    fail_on_index: int | None = None

    async def generate(self, request: ContentRequest) -> str:
        self.requests.append(request)
        index = len(self.requests) - 1
        if self.fail_on_index is not None and index >= self.fail_on_index:
            raise RuntimeError("llm exploded")
        return self.replies.pop(0)


@dataclass
class FakePersona:
    display_name: str
    description: str = "a personality"


@dataclass
class FakeProfile:
    id: str
    persona: FakePersona = field(default=None)

    def __post_init__(self) -> None:
        if self.persona is None:
            self.persona = FakePersona(self.id)


TOPIC = TopicItem(kind=ContentKind.REDDIT, title="A great post", text="Post body")


# ----------------------------------------------------------------------
# PodcastEngine
# ----------------------------------------------------------------------
def test_segment_alternates_host_then_guest() -> None:
    provider = ScriptedProvider(["host opening", "guest reply", "host rejoinder"])
    engine = PodcastEngine(provider)
    host = FakeProfile("obama", FakePersona("Obama", "former president"))
    guest = FakeProfile("l", FakePersona("L", "detective"))
    turns = asyncio.run(engine.generate_segment(TOPIC, host, guest, reply_turns=2))
    assert [t.text for t in turns] == ["host opening", "guest reply", "host rejoinder"]
    assert [t.voice_profile_id for t in turns] == ["obama", "l", "obama"]
    assert [t.speaker_name for t in turns] == ["Obama", "L", "Obama"]
    # All turns went through the shared ContentProvider abstraction.
    assert all(r.kind is ContentKind.REDDIT for r in provider.requests)
    assert provider.requests[0].persona_description == "former president"
    assert provider.requests[1].persona_description == "detective"


def test_guest_reply_carries_previous_line_and_topic() -> None:
    provider = ScriptedProvider(["opening", "reply"])
    engine = PodcastEngine(provider)
    host = FakeProfile("h", FakePersona("Host", "d1"))
    guest = FakeProfile("g", FakePersona("Guest", "d2"))
    asyncio.run(engine.generate_segment(TOPIC, host, guest, reply_turns=1))
    guest_request = provider.requests[1]
    assert "A great post" in guest_request.post_title
    assert "opening" in guest_request.post_text  # previous host line included
    assert "Respond" in guest_request.post_text


def test_provider_failure_aborts_segment() -> None:
    provider = ScriptedProvider(["opening"], fail_on_index=1)
    engine = PodcastEngine(provider)
    turns = asyncio.run(
        engine.generate_segment(TOPIC, FakeProfile("h"), FakeProfile("g"), reply_turns=2)
    )
    assert turns == []  # segment stopped, no partial audio


# ----------------------------------------------------------------------
# PodcastCoordinator
# ----------------------------------------------------------------------
@dataclass
class FakePodcastSettings:
    enabled: bool = True
    host_profile: str = "obama"
    reply_turns: int = 2


@dataclass
class FakeTopicSource:
    topic: TopicItem | None = TOPIC

    async def pick(self) -> TopicItem | None:
        return self.topic


class FakeProfiles:
    def __init__(self, ids: list) -> None:
        self._ids = set(ids)

    def get(self, pid):
        return FakeProfile(pid, FakePersona(pid)) if pid in self._ids else None


def _coordinator(**kwargs) -> PodcastCoordinator:
    settings = kwargs.pop("settings", FakePodcastSettings())
    profiles = kwargs.pop("profiles", FakeProfiles(["obama", "goku"]))
    topic_source = kwargs.pop("topic_source", FakeTopicSource())
    return PodcastCoordinator(
        settings,
        profiles,
        topic_source,
        provider_factory=kwargs.pop("provider_factory", lambda: ScriptedProvider(["a", "b", "c"])),
        pipeline_factory=kwargs.pop("pipeline_factory", lambda: object()),
    )


def test_start_stop_lifecycle() -> None:
    coordinator = _coordinator()
    assert coordinator.active(1) is False
    assert coordinator.start(1, "obama", "goku") is True
    assert coordinator.active(1) is True
    assert coordinator.stop(1) is True
    assert coordinator.active(1) is False
    assert coordinator.stop(1) is False


def test_start_refused_when_disabled_or_unknown_profile() -> None:
    coordinator = _coordinator(settings=FakePodcastSettings(enabled=False))
    assert coordinator.start(1, "obama", "goku") is False
    coordinator = _coordinator()
    assert coordinator.start(1, "obama", "ghost") is False
    assert coordinator.active(1) is False


def test_start_is_idempotent() -> None:
    coordinator = _coordinator()
    assert coordinator.start(1, "obama", "goku") is True
    assert coordinator.start(1, "obama", "goku") is True
    assert coordinator.stop(1) is True


def test_segment_turns_inactive_returns_empty() -> None:
    coordinator = _coordinator()
    assert asyncio.run(coordinator.segment_turns(1)) == []


def test_segment_turns_full_path() -> None:
    coordinator = _coordinator()
    coordinator.start(1, "obama", "goku")
    turns = asyncio.run(coordinator.segment_turns(1))
    assert [t.voice_profile_id for t in turns] == ["obama", "goku", "obama"]


def test_segment_turns_without_topic_stops_segment() -> None:
    coordinator = _coordinator(topic_source=FakeTopicSource(topic=None))
    coordinator.start(1, "obama", "goku")
    assert asyncio.run(coordinator.segment_turns(1)) == []


def test_segment_turns_without_provider_stops_segment() -> None:
    coordinator = _coordinator(provider_factory=lambda: None)
    coordinator.start(1, "obama", "goku")
    assert asyncio.run(coordinator.segment_turns(1)) == []


def test_segment_turns_stops_mode_on_missing_profiles() -> None:
    coordinator = _coordinator(profiles=FakeProfiles(["obama"]))
    coordinator.start(1, "obama", "goku")
    assert asyncio.run(coordinator.segment_turns(1)) == []
    assert coordinator.active(1) is False  # session cleaned up


def test_provider_failure_in_full_path_is_swallowed() -> None:
    coordinator = _coordinator(provider_factory=lambda: ScriptedProvider(fail_on_index=0))
    coordinator.start(1, "obama", "goku")
    assert asyncio.run(coordinator.segment_turns(1)) == []
    assert coordinator.active(1) is True  # mode continues after a failed segment


@pytest.mark.parametrize("reply_turns", [1, 3])
def test_reply_turns_setting_controls_segment_length(reply_turns: int) -> None:
    provider = ScriptedProvider(["line"] * (reply_turns + 1))
    engine = PodcastEngine(provider)
    out = asyncio.run(
        engine.generate_segment(TOPIC, FakeProfile("h"), FakeProfile("g"), reply_turns=reply_turns)
    )
    assert len(out) == reply_turns + 1  # opening + N replies, alternating
