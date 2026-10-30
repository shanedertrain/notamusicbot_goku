"""Listen-mode domain models: captured audio and utterance events.

The per-session opt-in ConsentState is gone; the surviving domain
models must not grow it back.
"""

from __future__ import annotations

import numpy as np

from notamusicbot.domain.listen import (
    LISTEN_SAMPLE_RATE_HZ,
    AudioChunk,
    UtteranceEvent,
)


def test_audio_chunk_duration() -> None:
    chunk = AudioChunk(samples=np.zeros(24000, dtype=np.float32))
    assert chunk.sample_rate_hz == LISTEN_SAMPLE_RATE_HZ == 16000
    assert chunk.duration_s == 1.5


def test_utterance_event_duration() -> None:
    chunk = AudioChunk(samples=np.zeros(16000, dtype=np.float32))
    event = UtteranceEvent(user_id=1, audio=chunk, t_start=10.0, t_eos=11.25)
    assert event.duration_s == 1.25


def test_consent_state_is_gone() -> None:
    """The per-session opt-in set was removed from the domain, not merely unused."""
    import notamusicbot.domain.listen as domain

    assert not hasattr(domain, "ConsentState")


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
