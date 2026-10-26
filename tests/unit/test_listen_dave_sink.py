"""DAVE decrypt helper and the reader-thread sink (consent before decrypt)."""

from __future__ import annotations

from types import SimpleNamespace

import davey
import pytest

from notamusicbot.discord_bot.listen_glue import ListenDaveSink, dave_decrypt
from notamusicbot.domain.listen import ConsentState

BOT_ID = 999


class FakeDaveSession:
    """Records decrypt calls; ``result`` is returned, an Exception is raised."""

    def __init__(self, result: bytes | Exception = b"plain") -> None:
        self.result = result
        self.calls: list[tuple[int, object, bytes]] = []

    def decrypt(self, user_id: int, media_type: object, opus: bytes) -> bytes:
        self.calls.append((user_id, media_type, opus))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeIngress:
    def __init__(self) -> None:
        self.packets: list[tuple[int | None, bytes]] = []

    def accept_packet(self, user_id: int | None, opus: bytes) -> None:
        self.packets.append((user_id, opus))


def test_decrypt_ok_returns_decrypted_payload() -> None:
    session = FakeDaveSession(b"plain")
    assert dave_decrypt(session, 1, 42, b"enc") == b"plain"
    assert session.calls == [(42, davey.MediaType.audio, b"enc")]


def test_passthrough_error_uses_the_raw_payload() -> None:
    session = FakeDaveSession(ValueError("Decryption failed: UnencryptedWhenPassthroughDisabled"))
    assert dave_decrypt(session, 1, 42, b"raw") == b"raw"


@pytest.mark.parametrize(
    "error", [ValueError("InvalidKey"), RuntimeError("UnencryptedWhenPassthroughDisabled")]
)
def test_any_other_decrypt_error_drops_the_frame(error: Exception) -> None:
    assert dave_decrypt(FakeDaveSession(error), 1, 42, b"raw") is None


def test_protocol_zero_is_raw_and_never_decrypts() -> None:
    session = FakeDaveSession()
    assert dave_decrypt(session, 0, 42, b"raw") == b"raw"
    assert session.calls == []


def test_no_session_is_raw() -> None:
    assert dave_decrypt(None, 1, 42, b"raw") == b"raw"


def _sink(
    session: FakeDaveSession, *, opted: tuple[int, ...] = (1,), version: int = 1
) -> tuple[ListenDaveSink, FakeIngress]:
    consent = ConsentState(bot_user_id=BOT_ID)
    for user in opted:
        consent.opt_in(user)
    ingress = FakeIngress()
    conn = SimpleNamespace(dave_session=session, dave_protocol_version=version)
    sink = ListenDaveSink(consent, ingress, BOT_ID, lambda: conn)  # type: ignore[arg-type]
    return sink, ingress


def _data(opus: bytes | None = b"enc") -> SimpleNamespace:
    return SimpleNamespace(opus=opus)


def test_sink_wants_opus() -> None:
    sink, _ = _sink(FakeDaveSession())
    assert sink.wants_opus() is True


def test_sink_decrypts_then_hands_the_payload_to_the_ingress() -> None:
    session = FakeDaveSession(b"plain")
    sink, ingress = _sink(session)
    sink.write(SimpleNamespace(id=1), _data())  # type: ignore[arg-type]
    assert ingress.packets == [(1, b"plain")]
    assert len(session.calls) == 1


@pytest.mark.parametrize("user", [None, SimpleNamespace(id=2), SimpleNamespace(id=BOT_ID)])
def test_sink_drops_unconsented_none_and_bot_before_decrypt(user: object) -> None:
    session = FakeDaveSession()
    sink, ingress = _sink(session, opted=(1,))
    # The bot id is dropped even when a consent state somehow lists it.
    sink._consent._opted_in.add(BOT_ID)  # type: ignore[attr-defined]
    sink.write(user, _data())  # type: ignore[arg-type]
    assert session.calls == []
    assert ingress.packets == []


def test_sink_drops_a_user_who_opted_out() -> None:
    session = FakeDaveSession()
    sink, ingress = _sink(session)
    sink._consent.opt_out(1)  # type: ignore[attr-defined]
    sink.write(SimpleNamespace(id=1), _data())  # type: ignore[arg-type]
    assert session.calls == []
    assert ingress.packets == []


def test_sink_counts_and_drops_undecryptable_frames() -> None:
    sink, ingress = _sink(FakeDaveSession(ValueError("InvalidKey")))
    sink.write(SimpleNamespace(id=1), _data())  # type: ignore[arg-type]
    sink.write(SimpleNamespace(id=1), _data())  # type: ignore[arg-type]
    assert ingress.packets == []
    assert sink.dave_dropped == 2


def test_sink_passthrough_frame_reaches_the_ingress_raw() -> None:
    session = FakeDaveSession(ValueError("UnencryptedWhenPassthroughDisabled"))
    sink, ingress = _sink(session)
    sink.write(SimpleNamespace(id=1), _data(b"raw"))  # type: ignore[arg-type]
    assert ingress.packets == [(1, b"raw")]
    assert sink.dave_dropped == 0


def test_sink_skips_empty_opus() -> None:
    session = FakeDaveSession()
    sink, ingress = _sink(session)
    sink.write(SimpleNamespace(id=1), _data(None))  # type: ignore[arg-type]
    assert session.calls == []
    assert ingress.packets == []
