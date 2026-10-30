from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeClock
from notamusicbot.config import ListenVadSettings
from notamusicbot.domain.listen import SpeechStarted, UtteranceEvent
from notamusicbot.listen.endpointer import Endpointer, EndpointerThread
from notamusicbot.speech.vad.base import FRAME_SAMPLES
from notamusicbot.speech.vad.energy import EnergyVad

SR = 16000
FRAME_S = FRAME_SAMPLES / SR


def _tone(seconds: float, dbfs: float = -20.0) -> np.ndarray:
    n = int(seconds * SR)
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * 300 * np.arange(n) / SR)).astype(np.float32)


def _sil(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _pad(x: np.ndarray) -> np.ndarray:
    extra = (-len(x)) % FRAME_SAMPLES
    return np.concatenate([x, np.zeros(extra, dtype=np.float32)])


def _make(settings: ListenVadSettings | None = None):
    clock = FakeClock()
    events: list = []
    ep = Endpointer(EnergyVad, settings or ListenVadSettings(), clock, events.append)
    return ep, clock, events


def _run(ep: Endpointer, clock: FakeClock, user: int, signal: np.ndarray) -> None:
    for i in range(0, len(signal), FRAME_SAMPLES):
        clock.advance(FRAME_S)
        ep.feed(user, signal[i : i + FRAME_SAMPLES])


def test_defaults_match_plan() -> None:
    s = ListenVadSettings()
    assert (s.min_speech_ms, s.end_silence_ms, s.pre_roll_ms, s.max_utterance_s) == (
        250,
        600,
        300,
        15.0,
    )


def test_one_utterance_with_preroll() -> None:
    ep, clock, events = _make()
    sig = _pad(np.concatenate([_sil(0.3), _tone(1.0), _sil(0.8)]))
    _run(ep, clock, 7, sig)
    started = [e for e in events if isinstance(e, SpeechStarted)]
    utts = [e for e in events if isinstance(e, UtteranceEvent)]
    assert len(started) == 1 and len(utts) == 1
    assert events.index(started[0]) < events.index(utts[0])
    u = utts[0]
    assert u.user_id == 7 and started[0].user_id == 7
    assert 1.0 <= u.t_eos - u.t_start <= 1.35
    assert abs(len(u.audio.samples) - int((0.3 + 1.0) * SR)) <= 512
    assert u.audio.sample_rate_hz == SR
    assert started[0].t_start == u.t_start


def test_short_blip_below_min_speech_is_ignored() -> None:
    ep, clock, events = _make()
    _run(ep, clock, 1, _pad(np.concatenate([_sil(0.3), _tone(0.1), _sil(1.0)])))
    assert events == []


def test_max_utterance_cut_at_15s() -> None:
    ep, clock, events = _make()
    _run(ep, clock, 1, _pad(_tone(20.0)))
    utts = [e for e in events if isinstance(e, UtteranceEvent)]
    assert len(utts) >= 1
    first = utts[0]
    assert first.t_eos - first.t_start == pytest.approx(15.0, abs=0.05)
    assert abs(len(first.audio.samples) - 15 * SR) <= 512


def test_two_users_interleaved_keep_separate_state() -> None:
    ep, clock, events = _make()
    a = _pad(np.concatenate([_sil(0.3), _tone(1.0), _sil(0.8)]))
    b = _pad(np.concatenate([_tone(3.0), _sil(1.0)]))
    n = max(len(a), len(b))
    a = np.concatenate([a, np.zeros(n - len(a), dtype=np.float32)])
    b = np.concatenate([b, np.zeros(n - len(b), dtype=np.float32)])
    for i in range(0, n, FRAME_SAMPLES):
        clock.advance(FRAME_S)
        ep.feed(1, a[i : i + FRAME_SAMPLES])
        ep.feed(2, b[i : i + FRAME_SAMPLES])
    utts = {e.user_id: e for e in events if isinstance(e, UtteranceEvent)}
    assert set(utts) == {1, 2}
    # User 1 finishing (and its silence) did not end user 2's 3 s utterance.
    assert utts[2].t_eos - utts[2].t_start == pytest.approx(3.0, abs=0.1)
    assert 1.0 <= utts[1].t_eos - utts[1].t_start <= 1.35
    order = [e.user_id for e in events if isinstance(e, UtteranceEvent)]
    assert order == [1, 2]


def test_arbitrary_chunking_gives_same_result() -> None:
    ep, clock, events = _make()
    sig = _pad(np.concatenate([_sil(0.3), _tone(1.0), _sil(0.8)]))
    for i in range(0, len(sig), 960):  # 60 ms chunks, not frame aligned
        clock.advance(960 / SR)
        ep.feed(1, sig[i : i + 960])
    utts = [e for e in events if isinstance(e, UtteranceEvent)]
    assert len(utts) == 1
    assert abs(len(utts[0].audio.samples) - int(1.3 * SR)) <= 2 * 512


def test_tick_ends_utterance_when_audio_stops() -> None:
    ep, clock, events = _make()
    _run(ep, clock, 1, _tone(1.0))
    assert not any(isinstance(e, UtteranceEvent) for e in events)
    clock.advance(0.7)
    ep.tick()
    assert sum(isinstance(e, UtteranceEvent) for e in events) == 1


def test_discard_drops_open_utterance() -> None:
    ep, clock, events = _make()
    _run(ep, clock, 1, _tone(1.0))
    ep.discard(1)
    clock.advance(5.0)
    ep.tick()
    assert not any(isinstance(e, UtteranceEvent) for e in events)


class _Ingress:
    """Ingress double: known users come from the buffers the ingress holds."""

    def __init__(self) -> None:
        self.queues: dict[int, list[np.ndarray]] = {}
        self.drained: list[int] = []
        self.users: set[int] = set()

    def known_user_ids(self) -> frozenset[int]:
        return frozenset(self.users)

    def drain(self, user_id: int) -> np.ndarray:
        self.drained.append(user_id)
        q = self.queues.get(user_id, [])
        if not q:
            return np.zeros(0, dtype=np.float32)
        return q.pop(0)

    def forget(self, user_id: int) -> None:
        """What ``ListenIngress.remove_user`` does when a member leaves."""
        self.users.discard(user_id)
        self.queues.pop(user_id, None)


def test_thread_discards_a_user_forgotten_by_the_ingress() -> None:
    ep, clock, events = _make()
    ingress = _Ingress()
    ingress.users.add(1)
    ingress.queues[1] = [_tone(1.0)]
    th = EndpointerThread(ingress, ep)
    clock.advance(1.0)
    assert th.poll_once() == SR
    assert ingress.drained == [1]
    # The member left: the ingress forgot the buffer and the open utterance
    # must vanish silently (no UtteranceEvent, and the user is not polled again).
    ingress.forget(1)
    clock.advance(5.0)
    th.poll_once()
    assert ingress.drained == [1]
    assert not any(isinstance(e, UtteranceEvent) for e in events)


def test_thread_start_stop_delivers_events() -> None:
    import threading

    got: list = []
    done = threading.Event()

    def emit(ev) -> None:
        got.append(ev)
        if isinstance(ev, UtteranceEvent):
            done.set()

    import time

    ep = Endpointer(EnergyVad, ListenVadSettings(), time.monotonic, emit)
    ingress = _Ingress()
    ingress.users.add(3)
    ingress.queues[3] = [_pad(np.concatenate([_tone(1.0), _sil(0.8)]))]
    th = EndpointerThread(ingress, ep, poll_interval_s=0.001)
    th.start()
    try:
        assert done.wait(5.0)
    finally:
        th.stop()
    assert isinstance(got[0], SpeechStarted)
    assert got[-1].user_id == 3
