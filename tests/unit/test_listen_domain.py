"""Listen-mode domain models and the per-session ConsentState."""

from __future__ import annotations

import threading

import numpy as np

from notamusicbot.domain.listen import (
    LISTEN_SAMPLE_RATE_HZ,
    AudioChunk,
    ConsentState,
    UtteranceEvent,
)

BOT_ID = 999


def test_audio_chunk_duration() -> None:
    chunk = AudioChunk(samples=np.zeros(24000, dtype=np.float32))
    assert chunk.sample_rate_hz == LISTEN_SAMPLE_RATE_HZ == 16000
    assert chunk.duration_s == 1.5


def test_utterance_event_duration() -> None:
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32))
    event = UtteranceEvent(user_id=1, audio=chunk, t_start=10.0, t_eos=11.25)
    assert event.duration_s == 1.25


def test_consent_opt_in_and_allowed() -> None:
    consent = ConsentState(bot_user_id=BOT_ID)
    assert consent.allowed(1) is False
    assert consent.opt_in(1) is True
    assert consent.allowed(1) is True
    assert consent.allowed(2) is False
    assert consent.opted_in() == frozenset({1})


def test_consent_never_allows_bot_or_none() -> None:
    consent = ConsentState(bot_user_id=BOT_ID)
    assert consent.opt_in(BOT_ID) is False
    assert consent.allowed(BOT_ID) is False
    assert consent.allowed(None) is False
    assert consent.opted_in() == frozenset()


def test_opt_out_clears_synchronously_after_removal() -> None:
    consent = ConsentState(bot_user_id=BOT_ID)
    seen: list[tuple[int, bool]] = []
    # The hook observes the user as already disallowed.
    consent.add_clear_hook(lambda uid: seen.append((uid, consent.allowed(uid))))
    consent.opt_in(1)
    consent.opt_in(2)

    assert consent.opt_out(1) is True
    assert seen == [(1, False)]
    assert consent.allowed(1) is False
    assert consent.allowed(2) is True

    # Opting out a user who never opted in still runs the hooks.
    assert consent.opt_out(3) is False
    assert seen == [(1, False), (3, False)]


def test_clear_opts_everyone_out_and_runs_hooks() -> None:
    consent = ConsentState(bot_user_id=BOT_ID)
    cleared: list[int] = []
    consent.add_clear_hook(cleared.append)
    for uid in (3, 1, 2):
        consent.opt_in(uid)

    consent.clear()
    assert cleared == [1, 2, 3]
    assert consent.opted_in() == frozenset()
    assert not any(consent.allowed(uid) for uid in (1, 2, 3))


def test_consent_thread_safety_smoke() -> None:
    consent = ConsentState(bot_user_id=BOT_ID)
    errors: list[BaseException] = []
    start = threading.Barrier(5)

    def writer(base: int) -> None:
        try:
            start.wait()
            for i in range(2000):
                uid = base + i % 50
                consent.opt_in(uid)
                consent.opt_out(uid)
            consent.opt_in(base)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for i in range(10000):
                consent.allowed(i % 500)
                consent.opted_in()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(b,)) for b in (100, 200, 300, 400)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert not any(t.is_alive() for t in threads)
    assert consent.opted_in() == frozenset({100, 200, 300, 400})


# ------------------------------------------------------------ fakes (conftest)


async def test_listen_fakes_follow_their_scripts() -> None:
    from conftest import (
        FakeClock,
        FakeContentProvider,
        FakePacketSource,
        FakeTranscriber,
        FakeVad,
    )
    from notamusicbot.content.base import ContentKind, ContentRequest
    from notamusicbot.speech.vad import FRAME_SAMPLES

    assert FRAME_SAMPLES == 512
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    vad = FakeVad([0.9, 0.1], default=0.0)
    assert [vad.process(frame) for _ in range(3)] == [0.9, 0.1, 0.0]

    transcriber = FakeTranscriber(["a b", "c d"])
    chunk = AudioChunk(samples=frame)
    texts = [(await transcriber.transcribe(chunk)).text for _ in range(3)]
    assert texts == ["a b", "c d", "c d"]
    assert len(transcriber.calls) == 3

    provider = FakeContentProvider(["one", RuntimeError("llm down")])
    request = ContentRequest(kind=ContentKind.SONG)
    assert await provider.generate(request) == "one"
    try:
        await provider.generate(request)
    except RuntimeError as exc:
        assert str(exc) == "llm down"
    else:  # pragma: no cover
        raise AssertionError("expected the scripted exception")

    received: list[tuple[int | None, bytes]] = []
    source = FakePacketSource.interleaved([1, 2, None], count=2)
    assert source.deliver(lambda uid, data: received.append((uid, data))) == 6
    assert [uid for uid, _ in received] == [1, 2, None, 1, 2, None]

    clock = FakeClock(start=5.0)
    await clock.sleep(0.5)
    clock.advance(1.0)
    assert clock() == clock.monotonic() == 6.5
    assert clock.sleeps == [0.5]
